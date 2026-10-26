#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import re
import gc
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> tuple[list[str], str | None]:
    parser = argparse.ArgumentParser(
        description="Calculate retrieval metrics from existing experiment result files."
    )
    parser.add_argument(
        "--datasets",
        required=True,
        help="Comma-separated dataset names: hupd, tender, ledgar (e.g. tender,ledgar).",
    )
    parser.add_argument(
        "--experiment",
        help="Optional experiment-name filter, e.g. dense_parent_child. Omit to process all experiments.",
    )
    args = parser.parse_args()
    selected = [name.strip().lower() for name in args.datasets.split(",") if name.strip()]
    allowed = {"hupd", "tender", "ledgar"}
    if not selected or len(selected) != len(set(selected)) or set(selected) - allowed:
        parser.error("--datasets must contain unique names from hupd,tender,ledgar")
    return selected, args.experiment.strip() if args.experiment else None


SELECTED_DATASETS, EXPERIMENT_FILTER = parse_args()
if len(SELECTED_DATASETS) > 1:
    script_path = Path(__file__).resolve()
    failed = []
    for dataset in SELECTED_DATASETS:
        print(f"\n===== AlignScore analysis: {dataset} =====", flush=True)
        result = subprocess.run(
            [sys.executable, str(script_path), "--datasets", dataset]
            + (["--experiment", EXPERIMENT_FILTER] if EXPERIMENT_FILTER else []),
            check=False,
        )
        if result.returncode:
            failed.append((dataset, result.returncode))
    if failed:
        print(f"AlignScore analyses failed: {failed}", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(0)

# Preload Triton before PyTorch/Transformers and before any dependency
# initializes TensorFlow/oneDNN.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import triton
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


try:
    from IPython.display import display
except Exception:
    def display(obj):
        print(obj)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = SELECTED_DATASETS[0]
DATASET_DIR = PROJECT_ROOT / f"{DATASET_NAME}_dataset"


# Configure the English AlignScore and multilingual NLI backends.

STANDARD_ALIGNSCORE_CKPT_PATH = str(
    Path(__file__).resolve().parent / "checkpoints" / "AlignScore-base.ckpt"
)

STANDARD_ALIGNSCORE_MODEL = "roberta-base"
STANDARD_ALIGNSCORE_EVALUATION_MODE = "nli_sp"
STANDARD_ALIGNSCORE_BATCH_SIZE = 16
STANDARD_ALIGNSCORE_CLAIM_UNIT_MAX_TOKENS = 150

MULTILINGUAL_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
MULTILINGUAL_NLI_BATCH_SIZE = 16

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CONTEXT_CHUNK_TOKENS = 350
MAX_SEQUENCE_TOKENS = 512
SPECIAL_TOKEN_BUFFER = 12
CLAIM_UNIT_MAX_TOKENS = (
    MAX_SEQUENCE_TOKENS
    - CONTEXT_CHUNK_TOKENS
    - SPECIAL_TOKEN_BUFFER
)

OUTPUT_SUFFIX = "faithfulness_context_recall"


# Handle dataset paths and result files.

def safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name)).strip("_")


def infer_dataset_kind_from_path(dataset_dir: Path) -> str:
    folder_name = dataset_dir.name.lower()

    if "tender" in folder_name:
        return "tender"

    if "ledgar" in folder_name:
        return "ledgar"

    if "hupd" in folder_name:
        return "hupd"

    raise ValueError(
        f"Could not infer dataset kind from folder name: {dataset_dir}. "
        "Expected a folder name containing tender, ledgar, or hupd."
    )


def get_results_dir(dataset_dir: Path) -> Path:
    results_dir = dataset_dir / "results"

    if not results_dir.exists():
        raise FileNotFoundError(f"Results folder does not exist: {results_dir}")

    return results_dir


def get_output_dir(dataset_dir: Path) -> Path:
    output_dir = dataset_dir / "results_test" / "retrieval_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def find_result_files(dataset_dir: Path):
    results_dir = get_results_dir(dataset_dir)

    result_files = sorted(
        list(results_dir.glob("*.json")) +
        list(results_dir.glob("*.jsonl"))
    )

    result_files = [path for path in result_files if path.is_file()]

    if not result_files:
        raise FileNotFoundError(
            f"No .json or .jsonl result files found in: {results_dir}"
        )

    return result_files


def load_json_or_jsonl(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as file:
        first_char = file.read(1)
        file.seek(0)

        if first_char == "[":
            data = json.load(file)
        else:
            data = [json.loads(line) for line in file if line.strip()]

    return pd.DataFrame(data)


# Derive method, top-k, and test-case metadata from the result files.

def infer_method_from_experiment_name(experiment_name: str) -> str:
    experiment_name = str(experiment_name or "").strip()
    return re.sub(r"_top_\d+$", "", experiment_name)


def infer_top_k_from_experiment_name(experiment_name: str):
    match = re.search(r"_top_(\d+)$", str(experiment_name or "").strip())

    if match:
        return int(match.group(1))

    return np.nan


def get_single_or_joined_value(df: pd.DataFrame, column: str, fallback: str) -> str:
    values = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = sorted([value for value in values.unique() if value])

    if not values:
        return fallback

    if len(values) == 1:
        return safe_filename(values[0])

    joined = "__".join(safe_filename(value) for value in values)

    if len(joined) > 180:
        return f"multiple_{column}"

    return joined


def extract_trailing_number(value):
    if pd.isna(value):
        return pd.NA

    match = re.search(r"(\d+)$", str(value).strip())

    if not match:
        return pd.NA

    return int(match.group(1))


def infer_test_case_id(row):
    dataset_value = str(row.get("dataset", "")).strip().lower()
    test_id_value = row.get("test_id", pd.NA)
    block_id_value = row.get("block_id", pd.NA)

    if "tender" in dataset_value:
        tender_id = extract_trailing_number(test_id_value)

        if not pd.isna(tender_id):
            return tender_id

        fallback_tender_id = extract_trailing_number(block_id_value)

        if not pd.isna(fallback_tender_id):
            return fallback_tender_id

        return str(test_id_value).strip() if not pd.isna(test_id_value) else pd.NA

    if "ledgar" in dataset_value or "hupd" in dataset_value:
        if not pd.isna(block_id_value) and str(block_id_value).strip():
            return str(block_id_value).strip()

        if not pd.isna(test_id_value) and str(test_id_value).strip():
            return str(test_id_value).strip()

        return pd.NA

    if not pd.isna(block_id_value) and str(block_id_value).strip():
        return str(block_id_value).strip()

    if not pd.isna(test_id_value) and str(test_id_value).strip():
        return str(test_id_value).strip()

    return pd.NA


# Merge retrieved context entries while removing empty values and duplicates.

def get_context_text_from_contexts_field(value) -> str:
    if not isinstance(value, list):
        return ""

    cleaned_contexts = []
    seen = set()

    for context in value:
        text = "" if context is None else str(context).strip()
        text = re.sub(r"\s+", " ", text)

        if text and text.lower() != "nan" and text not in seen:
            cleaned_contexts.append(text)
            seen.add(text)

    return "\n\n".join(cleaned_contexts)


# Aggregate valid metric values without failing on missing scores.

def safe_mean(values):
    valid_values = [
        value for value in values
        if value is not None and not np.isnan(value)
    ]

    if not valid_values:
        return np.nan

    return float(np.mean(valid_values))


def balanced_token_spans(number_of_tokens: int, max_tokens: int):
    """
    Return non-overlapping integer token spans that never exceed max_tokens.

    Full max-sized chunks are kept whenever possible. If a final remainder would
    create a short tail, the last max-sized block and that remainder are split
    as evenly as possible into two integer-sized spans. No token is dropped.

    Examples for max_tokens=150:
        150 -> [150]
        151 -> [75, 76]
        200 -> [100, 100]
        299 -> [149, 150]
        300 -> [150, 150]
        301 -> [150, 75, 76]
    """
    if number_of_tokens <= 0:
        return []

    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero.")

    if number_of_tokens <= max_tokens:
        return [(0, number_of_tokens)]

    full_chunks, remainder = divmod(number_of_tokens, max_tokens)

    if remainder == 0:
        return [
            (start, start + max_tokens)
            for start in range(0, number_of_tokens, max_tokens)
        ]

    spans = []

    # Keep all earlier full chunks. The final full chunk and remainder are
    # rebalanced together so that no tiny trailing unit is created.
    prefix_full_chunks = max(0, full_chunks - 1)

    for chunk_index in range(prefix_full_chunks):
        start = chunk_index * max_tokens
        spans.append((start, start + max_tokens))

    tail_start = prefix_full_chunks * max_tokens
    tail_length = number_of_tokens - tail_start

    left_length = tail_length // 2
    right_length = tail_length - left_length

    if left_length > max_tokens or right_length > max_tokens:
        raise RuntimeError(
            "Balanced token splitting produced a span larger than max_tokens."
        )

    middle = tail_start + left_length
    spans.append((tail_start, middle))
    spans.append((middle, number_of_tokens))

    return spans


# Evaluate English LEDGAR and HUPD texts with standard AlignScore.

class StandardAlignScoreRunner:
    def __init__(
        self,
        ckpt_path: str = STANDARD_ALIGNSCORE_CKPT_PATH,
        model: str = STANDARD_ALIGNSCORE_MODEL,
        batch_size: int = STANDARD_ALIGNSCORE_BATCH_SIZE,
        device: str = DEVICE,
        evaluation_mode: str = STANDARD_ALIGNSCORE_EVALUATION_MODE,
        claim_unit_max_tokens: int = STANDARD_ALIGNSCORE_CLAIM_UNIT_MAX_TOKENS,
    ):
        checkpoint = Path(ckpt_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "AlignScore checkpoint not found: "
                f"{checkpoint}. Download it using the instructions in README.md."
            )

        from alignscore import AlignScore

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.claim_unit_max_tokens = claim_unit_max_tokens

        self.scorer = AlignScore(
            model=model,
            batch_size=batch_size,
            device=device,
            ckpt_path=str(checkpoint),
            evaluation_mode=evaluation_mode,
        )

        print("Loaded standard AlignScore model:", model)
        print("Maximum claim-unit tokens:", self.claim_unit_max_tokens)

    def split_overlong_sentence_by_tokens(self, sentence: str):
        # Split overlong sentences into balanced, non-overlapping token units.
        # Full 150-token units are retained where possible; a short final tail
        # is rebalanced with the preceding block (e.g. 301 -> 150 + 75 + 76).
        sentence = "" if sentence is None else str(sentence).strip()

        if not sentence:
            return []

        token_ids = self.tokenizer.encode(
            sentence,
            add_special_tokens=False,
            truncation=False,
        )

        if not token_ids:
            return []

        if len(token_ids) <= self.claim_unit_max_tokens:
            return [sentence]

        units = []
        spans = balanced_token_spans(
            number_of_tokens=len(token_ids),
            max_tokens=self.claim_unit_max_tokens,
        )

        for start, end in spans:
            unit_ids = token_ids[start:end]

            unit_text = self.tokenizer.decode(
                unit_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            if unit_text:
                units.append(unit_text)

        return units

    def split_claim_into_sentence_groups(self, claim: str):
        # Keep normal sentences intact and retain split units under their original sentence.
        claim = "" if claim is None else str(claim).strip()

        if not claim or claim.lower() == "nan":
            return []

        claim = re.sub(r"\s+", " ", claim)
        sentences = re.split(
            r"(?<=[.!?])\s+(?=[\"'\(\[]?[A-Z0-9])",
            claim,
        )
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        sentence_groups = []

        for sentence in sentences:
            sentence_units = self.split_overlong_sentence_by_tokens(sentence)

            if sentence_units:
                sentence_groups.append(sentence_units)

        return sentence_groups

    @staticmethod
    def normalize_scores(scores):
        # Convert AlignScore outputs to a consistent one-dimensional NumPy array.
        if isinstance(scores, np.ndarray):
            return scores.astype(float).reshape(-1)

        if isinstance(scores, (list, tuple)):
            return np.asarray(scores, dtype=float).reshape(-1)

        return np.asarray([float(scores)], dtype=float)

    def score_context_to_claim(self, context: str, claim: str) -> float:
        # Score all claim units and restore equal weighting across the original sentences.
        context = "" if context is None else str(context).strip()
        claim = "" if claim is None else str(claim).strip()

        if not context or not claim or claim.lower() == "nan":
            return np.nan

        sentence_groups = self.split_claim_into_sentence_groups(claim)

        if not sentence_groups:
            return np.nan

        claim_units = [
            unit
            for sentence_group in sentence_groups
            for unit in sentence_group
        ]

        scores = self.scorer.score(
            contexts=[context] * len(claim_units),
            claims=claim_units,
        )
        unit_scores = self.normalize_scores(scores)

        if len(unit_scores) != len(claim_units):
            raise ValueError(
                "AlignScore returned an unexpected number of scores: "
                f"{len(unit_scores)} scores for {len(claim_units)} claim units."
            )

        sentence_scores = []
        score_index = 0

        for sentence_group in sentence_groups:
            number_of_units = len(sentence_group)
            group_scores = unit_scores[
                score_index:score_index + number_of_units
            ]
            sentence_scores.append(float(np.mean(group_scores)))
            score_index += number_of_units

        return safe_mean(sentence_scores)


# Evaluate German Tender texts with multilingual NLI and AlignScore-style aggregation.

class MultilingualNLIAlignScoreStyleRunner:
    def __init__(
        self,
        model_name: str = MULTILINGUAL_NLI_MODEL,
        batch_size: int = MULTILINGUAL_NLI_BATCH_SIZE,
        device: str = DEVICE,
        context_chunk_tokens: int = CONTEXT_CHUNK_TOKENS,
        claim_unit_max_tokens: int = CLAIM_UNIT_MAX_TOKENS,
        max_sequence_tokens: int = MAX_SEQUENCE_TOKENS,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.context_chunk_tokens = context_chunk_tokens
        self.claim_unit_max_tokens = claim_unit_max_tokens
        self.max_sequence_tokens = max_sequence_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()

        self.entailment_label_id = self.find_entailment_label_id()

        print("Loaded multilingual NLI model:", model_name)
        print("Device:", device)
        print("Model labels:", self.model.config.id2label)
        print("Entailment label id:", self.entailment_label_id)
        print("Context chunk tokens:", self.context_chunk_tokens)
        print("Claim unit max tokens:", self.claim_unit_max_tokens)
        print("Max pair sequence tokens:", self.max_sequence_tokens)

    def find_entailment_label_id(self) -> int:
        for label_id, label_name in self.model.config.id2label.items():
            if "entail" in str(label_name).lower():
                return int(label_id)

        raise ValueError(
            f"Could not find entailment label in model labels: {self.model.config.id2label}"
        )

    def token_count(self, text: str) -> int:
        text = "" if text is None else str(text).strip()

        if not text:
            return 0

        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,
            )
        )

    def split_context_into_chunks(self, context: str):
        context = "" if context is None else str(context).strip()

        if not context or context.lower() == "nan":
            return []

        token_ids = self.tokenizer.encode(
            context,
            add_special_tokens=False,
            truncation=False,
        )

        if not token_ids:
            return []

        chunks = []

        for start in range(0, len(token_ids), self.context_chunk_tokens):
            end = start + self.context_chunk_tokens
            chunk_ids = token_ids[start:end]

            chunk_text = self.tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            if chunk_text:
                chunks.append(chunk_text)

            if end >= len(token_ids):
                break

        return chunks

    def split_overlong_sentence_by_tokens(self, sentence: str):
        # Use the same balanced token splitting as the standard AlignScore path.
        # No short remainder is discarded; all tokens are retained.
        sentence = "" if sentence is None else str(sentence).strip()

        if not sentence:
            return []

        token_ids = self.tokenizer.encode(
            sentence,
            add_special_tokens=False,
            truncation=False,
        )

        if not token_ids:
            return []

        if len(token_ids) <= self.claim_unit_max_tokens:
            return [sentence]

        sub_units = []
        spans = balanced_token_spans(
            number_of_tokens=len(token_ids),
            max_tokens=self.claim_unit_max_tokens,
        )

        for start, end in spans:
            sub_ids = token_ids[start:end]

            sub_text = self.tokenizer.decode(
                sub_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            if sub_text:
                sub_units.append(sub_text)

        return sub_units

    def split_claim_into_sentence_groups(self, claim: str):
        # Preserve the original sentence structure. If one original sentence is
        # too long for the model input budget, split only that sentence into
        # balanced technical sub-units and keep those units grouped together.
        # This allows the sub-unit scores to be averaged back to one score for
        # the original sentence before averaging across the complete claim.
        claim = "" if claim is None else str(claim).strip()

        if not claim or claim.lower() == "nan":
            return []

        claim = re.sub(r"\s+", " ", claim)

        sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9])", claim)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        sentence_groups = []

        for sentence in sentences:
            sentence_units = self.split_overlong_sentence_by_tokens(sentence)

            if sentence_units:
                sentence_groups.append(sentence_units)

        return sentence_groups

    def entailment_scores_for_pairs(self, premises, hypotheses):
        if not premises:
            return np.array([], dtype=float)

        all_scores = []

        for start in range(0, len(premises), self.batch_size):
            batch_premises = premises[start:start + self.batch_size]
            batch_hypotheses = hypotheses[start:start + self.batch_size]

            inputs = self.tokenizer(
                batch_premises,
                batch_hypotheses,
                padding=True,
                truncation="only_first",
                max_length=self.max_sequence_tokens,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                entailment_scores = probabilities[:, self.entailment_label_id]

            all_scores.append(entailment_scores.detach().cpu().numpy())

        return np.concatenate(all_scores, axis=0)

    def score_context_to_claim(self, context: str, claim: str) -> float:
        # Mirror the standard AlignScore weighting logic: each technical
        # sub-unit is scored against all context chunks, the best context
        # support is retained for that sub-unit, sub-units belonging to the
        # same original sentence are averaged back to one sentence score, and
        # the final claim score is the mean across original sentences.
        context_chunks = self.split_context_into_chunks(context)
        sentence_groups = self.split_claim_into_sentence_groups(claim)

        if not context_chunks or not sentence_groups:
            return np.nan

        sentence_scores = []

        for sentence_group in sentence_groups:
            unit_support_scores = []

            for claim_unit in sentence_group:
                premises = context_chunks
                hypotheses = [claim_unit] * len(context_chunks)

                pair_scores = self.entailment_scores_for_pairs(
                    premises=premises,
                    hypotheses=hypotheses,
                )

                if pair_scores.size == 0:
                    unit_support_scores.append(np.nan)
                else:
                    unit_support_scores.append(float(np.max(pair_scores)))

            sentence_scores.append(safe_mean(unit_support_scores))

        return safe_mean(sentence_scores)


# Select the metric backend from the dataset folder name.

def load_metric_runner(dataset_kind: str):
    if dataset_kind == "tender":
        print("Using multilingual NLI with AlignScore-style aggregation for German Tender.")
        return MultilingualNLIAlignScoreStyleRunner(), "multilingual_mdeberta_xnli"

    if dataset_kind in {"ledgar", "hupd"}:
        print("Using standard AlignScore for English data.")
        return StandardAlignScoreRunner(), "standard_alignscore"

    raise ValueError(
        f"Unknown dataset kind: {dataset_kind}. "
        "Expected tender, ledgar, or hupd."
    )


def clear_gpu_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Evaluate all rows of one JSON or JSONL result file.

def analyze_result_file(result_path: Path, metric_runner, metric_backend: str, dataset_kind: str):
    df = load_json_or_jsonl(result_path)

    print("\nProcessing file:")
    print(result_path)
    print("Rows loaded:", len(df))
    print("Columns:", list(df.columns))

    required_columns = [
        "dataset",
        "experiment",
        "generator_model",
        "answer",
        "ground_truth",
        "contexts",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if len(df) == 0:
        raise ValueError("No rows found in result file.")

    dataset_from_json = get_single_or_joined_value(
        df=df,
        column="dataset",
        fallback="unknown_dataset",
    )

    df["method"] = df["experiment"].astype(str).map(
        infer_method_from_experiment_name
    )

    top_k_values = df["experiment"].astype(str).map(
        infer_top_k_from_experiment_name
    )

    for possible_top_k_column in ["top_k", "top_k_parents", "top_k_children"]:
        if possible_top_k_column in df.columns:
            fallback_values = pd.to_numeric(
                df[possible_top_k_column],
                errors="coerce",
            )
            top_k_values = top_k_values.fillna(fallback_values)

    df["top_k"] = pd.to_numeric(top_k_values, errors="coerce").astype("Int64")

    df["top_k"] = df["top_k"].astype("Int64")

    if df["top_k"].isna().any():
        missing_top_k_count = int(df["top_k"].isna().sum())
        raise ValueError(f"Could not infer top_k for {missing_top_k_count} rows.")

    method_from_json = get_single_or_joined_value(
        df=df,
        column="method",
        fallback="unknown_method",
    )

    print("Dataset:", dataset_from_json)
    print("Selected dataset kind:", dataset_kind)
    print("Method:", method_from_json)
    print("Metric backend:", metric_backend)

    print("\nGenerator models:")
    print(df["generator_model"].value_counts(dropna=False))

    print("\nTop-k values:")
    print(df["top_k"].value_counts(dropna=False).sort_index())

    df["answer"] = df["answer"].fillna("").astype(str)
    df["ground_truth"] = df["ground_truth"].fillna("").astype(str)

    df["retrieved_context_text"] = df["contexts"].apply(
    get_context_text_from_contexts_field)

    print("\nEmpty generated answers:", int((df["answer"].str.strip() == "").sum()))
    print("Empty ground truths:", int((df["ground_truth"].str.strip() == "").sum()))
    print("Empty contexts:", int((df["retrieved_context_text"].str.strip() == "").sum()))

    faithfulness_scores = []
    context_recall_scores = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"{metric_backend} for {result_path.name}",
    ):
        context_text = row["retrieved_context_text"]

        faithfulness_score = metric_runner.score_context_to_claim(
            context=context_text,
            claim=row["answer"],
        )

        context_recall_score = metric_runner.score_context_to_claim(
            context=context_text,
            claim=row["ground_truth"],
        )

        faithfulness_scores.append(faithfulness_score)
        context_recall_scores.append(context_recall_score)

    scores = df.reset_index(drop=True).copy()
    scores["faithfulness"] = faithfulness_scores
    scores["context_recall"] = context_recall_scores
    scores["metric_backend"] = metric_backend

    scores["test_case_id"] = scores.apply(infer_test_case_id, axis=1)

    if scores["test_case_id"].isna().any():
        missing_ids = int(scores["test_case_id"].isna().sum())
        raise ValueError(f"Could not infer test_case_id for {missing_ids} rows.")

    if dataset_kind == "tender":
        scores["test_case_id"] = pd.to_numeric(
            scores["test_case_id"],
            errors="coerce",
        ).astype("Int64")
    else:
        scores["test_case_id"] = scores["test_case_id"].astype(str).str.strip()

    metric_columns = [
        "faithfulness",
        "context_recall",
    ]

    per_question_group_columns = [
        "dataset",
        "method",
        "top_k",
        "generator_model",
        "test_case_id",
    ]

    metadata_columns = [
        "test_id",
        "block_id",
        "theme",
        "chapter",
        "metric_backend",
    ]

    metadata_columns = [
        column for column in metadata_columns
        if column in scores.columns
    ]

    per_question_metrics = (
        scores
        .groupby(per_question_group_columns, dropna=False)[metric_columns]
        .mean()
        .round(4)
        .reset_index()
    )

    per_question_counts = (
        scores
        .groupby(per_question_group_columns, dropna=False)
        .agg(
            n_rows=("experiment", "size"),
            n_empty_answers=(
                "answer",
                lambda values: int((values.fillna("").astype(str).str.strip() == "").sum()),
            ),
            n_empty_ground_truth=(
                "ground_truth",
                lambda values: int((values.fillna("").astype(str).str.strip() == "").sum()),
            ),
            n_empty_contexts=(
                "retrieved_context_text",
                lambda values: int((values.fillna("").astype(str).str.strip() == "").sum()),
            ),
        )
        .reset_index()
    )

    if metadata_columns:
        per_question_metadata = (
            scores
            .groupby(per_question_group_columns, dropna=False)[metadata_columns]
            .first()
            .reset_index()
        )
    else:
        per_question_metadata = scores[per_question_group_columns].drop_duplicates()

    per_question_summary = (
        per_question_counts
        .merge(
            per_question_metrics,
            on=per_question_group_columns,
            how="left",
        )
        .merge(
            per_question_metadata,
            on=per_question_group_columns,
            how="left",
        )
        .sort_values(
            [
                "dataset",
                "method",
                "generator_model",
                "top_k",
                "test_case_id",
            ]
        )
        .reset_index(drop=True)
    )

    setup_group_columns = [
        "dataset",
        "method",
        "top_k",
        "generator_model",
    ]

    setup_metrics = (
        scores
        .groupby(setup_group_columns, dropna=False)[metric_columns]
        .mean()
        .round(4)
        .reset_index()
    )

    setup_counts = (
        scores
        .groupby(setup_group_columns, dropna=False)
        .agg(
            n_rows=("experiment", "size"),
            n_questions=("test_case_id", "nunique"),
            n_empty_answers=(
                "answer",
                lambda values: int((values.fillna("").astype(str).str.strip() == "").sum()),
            ),
            n_empty_ground_truth=(
                "ground_truth",
                lambda values: int((values.fillna("").astype(str).str.strip() == "").sum()),
            ),
            n_empty_contexts=(
                "retrieved_context_text",
                lambda values: int((values.fillna("").astype(str).str.strip() == "").sum()),
            ),
            metric_backend=("metric_backend", "first"),
        )
        .reset_index()
    )

    setup_summary = (
        setup_counts
        .merge(
            setup_metrics,
            on=setup_group_columns,
            how="left",
        )
        .sort_values(
            [
                "dataset",
                "method",
                "generator_model",
                "top_k",
            ]
        )
        .reset_index(drop=True)
    )

    print("\nFinished result file:")
    print(result_path)
    print("Per-question rows:", len(per_question_summary))
    print("Setup-level rows:", len(setup_summary))

    display(per_question_summary)
    display(setup_summary)

    return {
        "dataset": dataset_from_json,
        "method": method_from_json,
        "per_question_summary": per_question_summary,
        "setup_summary": setup_summary,
    }


# Run the evaluation and combine the outputs across all result files.

dataset_kind = infer_dataset_kind_from_path(DATASET_DIR)
result_files = find_result_files(DATASET_DIR)
if EXPERIMENT_FILTER:
    file_prefix = f"{DATASET_NAME}_{EXPERIMENT_FILTER}_".casefold()
    result_files = [path for path in result_files if path.stem.casefold().startswith(file_prefix)]
    if not result_files:
        raise FileNotFoundError(
            f"No result files for experiment '{EXPERIMENT_FILTER}' in {DATASET_DIR / 'results'}"
        )
    print(f"Experiment filter: {EXPERIMENT_FILTER} ({len(result_files)} files)")

print("Selected dataset folder:")
print(DATASET_DIR)

print("\nDetected dataset kind:")
print(dataset_kind)

print("\nResult files found:")
for result_file in result_files:
    print("-", result_file)

metric_runner, metric_backend = load_metric_runner(dataset_kind)

created_outputs = []

for result_file in result_files:
    try:
        output = analyze_result_file(
            result_path=result_file,
            metric_runner=metric_runner,
            metric_backend=metric_backend,
            dataset_kind=dataset_kind,
        )
        created_outputs.append(output)

    except Exception as error:
        print("\nError while processing:")
        print(result_file)
        print(type(error).__name__, str(error))

clear_gpu_memory()

if not created_outputs:
    raise RuntimeError("No result files were processed successfully.")


all_per_question = pd.concat(
    [output["per_question_summary"] for output in created_outputs],
    ignore_index=True,
)

all_setup = pd.concat(
    [output["setup_summary"] for output in created_outputs],
    ignore_index=True,
)


per_question_group_columns = [
    "dataset",
    "method",
    "top_k",
    "generator_model",
    "test_case_id",
]

setup_group_columns = [
    "dataset",
    "method",
    "top_k",
    "generator_model",
]

metric_columns = [
    "faithfulness",
    "context_recall",
]

metadata_columns = [
    "test_id",
    "block_id",
    "theme",
    "chapter",
    "metric_backend",
]

metadata_columns = [
    column for column in metadata_columns
    if column in all_per_question.columns
]


combined_per_question_metrics = (
    all_per_question
    .groupby(per_question_group_columns, dropna=False)[metric_columns]
    .mean()
    .round(4)
    .reset_index()
)

combined_per_question_counts = (
    all_per_question
    .groupby(per_question_group_columns, dropna=False)
    .agg(
        n_rows=("n_rows", "sum"),
        n_empty_answers=("n_empty_answers", "sum"),
        n_empty_ground_truth=("n_empty_ground_truth", "sum"),
        n_empty_contexts=("n_empty_contexts", "sum"),
    )
    .reset_index()
)

if metadata_columns:
    combined_per_question_metadata = (
        all_per_question
        .groupby(per_question_group_columns, dropna=False)[metadata_columns]
        .first()
        .reset_index()
    )
else:
    combined_per_question_metadata = (
        all_per_question[per_question_group_columns]
        .drop_duplicates()
    )

combined_per_question = (
    combined_per_question_counts
    .merge(
        combined_per_question_metrics,
        on=per_question_group_columns,
        how="left",
    )
    .merge(
        combined_per_question_metadata,
        on=per_question_group_columns,
        how="left",
    )
    .sort_values(
        [
            "dataset",
            "method",
            "generator_model",
            "top_k",
            "test_case_id",
        ]
    )
    .reset_index(drop=True)
)


combined_setup_metrics = (
    all_setup
    .groupby(setup_group_columns, dropna=False)[metric_columns]
    .mean()
    .round(4)
    .reset_index()
)

combined_setup_counts = (
    all_setup
    .groupby(setup_group_columns, dropna=False)
    .agg(
        n_rows=("n_rows", "sum"),
        n_questions=("n_questions", "max"),
        n_empty_answers=("n_empty_answers", "sum"),
        n_empty_ground_truth=("n_empty_ground_truth", "sum"),
        n_empty_contexts=("n_empty_contexts", "sum"),
        metric_backend=("metric_backend", "first"),
    )
    .reset_index()
)

combined_setup = (
    combined_setup_counts
    .merge(
        combined_setup_metrics,
        on=setup_group_columns,
        how="left",
    )
    .sort_values(
        [
            "dataset",
            "method",
            "generator_model",
            "top_k",
        ]
    )
    .reset_index(drop=True)
)


output_dir = get_output_dir(DATASET_DIR)

for (dataset_name, method_name), per_question_part in combined_per_question.groupby(
    ["dataset", "method"],
    dropna=False,
):
    dataset_safe = safe_filename(dataset_name)
    method_safe = safe_filename(method_name)

    setup_part = combined_setup[
        (combined_setup["dataset"] == dataset_name) &
        (combined_setup["method"] == method_name)
    ].copy()

    per_question_csv_path = (
        output_dir /
        f"per_question_retrieval_metrics_{dataset_safe}_{method_safe}_{OUTPUT_SUFFIX}.csv"
    )

    summary_csv_path = (
        output_dir /
        f"summary_retrieval_metrics_{dataset_safe}_{method_safe}_{OUTPUT_SUFFIX}.csv"
    )

    per_question_part.to_csv(
        per_question_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    setup_part.to_csv(
        summary_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nSaved combined per-question CSV:")
    print(per_question_csv_path)

    print("\nSaved combined setup-level summary CSV:")
    print(summary_csv_path)


print("\nFinished retrieval-grounding analysis.")
print("Selected dataset:", DATASET_DIR)
print("Successful input files:", len(created_outputs))
print("Failed input files:", len(result_files) - len(created_outputs))

print("\nCombined setup summary:")
display(combined_setup)

