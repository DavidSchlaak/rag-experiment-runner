#!/usr/bin/env python
# coding: utf-8

# # Metric runner with per-question summaries
# 
# This runner writes one metric row per test case using `test_id` or `block_id`.



# ---
# Automatic reference-based metrics for longer answers:
# ROUGE-1/2/L precision/recall/F1,
# chunked BERTScore precision/recall/F1,
# chunked SimCSE precision/recall/F1,
# BGE-M3 ECS whole-answer similarity,
# chunked MATCHA precision/recall/F1
#
# Output:
# only summary metrics are saved.
#
# Summary logic:
# all metrics are averaged over all questions,
# grouped by dataset + method + top_k + generator_model.
#
# Filename:
# summary_metrics_{dataset_from_json}_{method_from_experiment}_{suffix}.xlsx
#
# Method inference:
# "dense_parent_child_top_5"  -> "dense_parent_child"
# "dense_parent_child_top_10" -> "dense_parent_child"
# "dense_parent_child_top_15" -> "dense_parent_child"
# "dense_parent_child_top_20" -> "dense_parent_child"
#
# All generators in the results file are included.
# ---


import argparse
import json
import re
import gc
import subprocess
import sys
from pathlib import Path


def parse_args() -> tuple[list[str], str | None]:
    parser = argparse.ArgumentParser(
        description="Calculate generation metrics from existing experiment result files."
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
        print(f"\n===== Generation metric analysis: {dataset} =====", flush=True)
        result = subprocess.run(
            [sys.executable, str(script_path), "--datasets", dataset]
            + (["--experiment", EXPERIMENT_FILTER] if EXPERIMENT_FILTER else []),
            check=False,
        )
        if result.returncode:
            failed.append((dataset, result.returncode))
    if failed:
        print(f"Generation metric analyses failed: {failed}", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(0)

# Preload Triton before PyTorch / TensorFlow-related dependencies.
import triton

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from rouge_score import rouge_scorer
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from matcha_metric import MATCHA


# ---
# Paths
# ---

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = SELECTED_DATASETS[0]
BASE_DIR = PROJECT_ROOT / f"{DATASET_NAME}_dataset"
#
# BERTScore language is inferred automatically from this folder name:
# tender -> lang="de"
# ledgar/hupd -> lang="en"


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


DATASET_KIND = infer_dataset_kind_from_path(BASE_DIR)
BERTSCORE_LANG = "de" if DATASET_KIND == "tender" else "en"

print("Selected dataset folder:", BASE_DIR)
print("Detected dataset kind:", DATASET_KIND)
print("BERTScore language:", BERTSCORE_LANG)

# Directory that contains the generated result files.
INPUT_RESULTS_DIR = BASE_DIR / "results"

# Keep evaluation outputs separate from the source experiment results.
OUTPUT_DIR = BASE_DIR / "results_test"
ANALYSIS_OUTPUT_ROOT = OUTPUT_DIR / "metric_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def find_result_files(input_dir: Path):
    """
    Finds all JSON and JSONL result files in the input directory.
    Only direct files are used, not recursive subfolders.
    This avoids re-processing generated output folders.
    """
    result_files = sorted(
        list(input_dir.glob("*.json")) +
        list(input_dir.glob("*.jsonl"))
    )

    result_files = [
        path for path in result_files
        if path.is_file()
    ]

    if not result_files:
        raise FileNotFoundError(
            f"No .json or .jsonl result files found in: {input_dir}"
        )

    return result_files


RESULT_FILES = find_result_files(INPUT_RESULTS_DIR)
if EXPERIMENT_FILTER:
    file_prefix = f"{DATASET_NAME}_{EXPERIMENT_FILTER}_".casefold()
    RESULT_FILES = [path for path in RESULT_FILES if path.stem.casefold().startswith(file_prefix)]
    if not RESULT_FILES:
        raise FileNotFoundError(
            f"No result files for experiment '{EXPERIMENT_FILTER}' in {INPUT_RESULTS_DIR}"
        )
    print(f"Experiment filter: {EXPERIMENT_FILTER} ({len(RESULT_FILES)} files)")

print("Result files found:")
for path in RESULT_FILES:
    print("-", path)

    
    


# ---
# Output settings
# ---

suffix = "all_generators"


# ---
# Helper: safe filenames
# ---

def make_safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name)).strip("_")


# ---
# Load generated system results JSON / JSONL
# ---

def load_json_or_jsonl(path: Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)

        if first == "[":
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]

    return pd.DataFrame(data)







def analyze_result_file(result_path: Path):
    df = load_json_or_jsonl(result_path)

    print("\nProcessing:")
    print(result_path)
    print("Rows loaded:", len(df))
    print("Columns:", list(df.columns))
    
    required = [
        "dataset",
        "experiment",
        "generator_model",
        "answer",
        "ground_truth"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


    if len(df) == 0:
        raise ValueError("No rows found in result file.")


    # ---
    # Dataset, method, and top-k inference from JSON
    # ---

    def infer_method_from_experiment_name(experiment_name: str) -> str:
        """
        Converts experiment names such as:
            dense_parent_child_top_5
            dense_parent_child_top_10
            dense_parent_child_top_15
            dense_parent_child_top_20

        into:
            dense_parent_child
        """
        experiment_name = str(experiment_name or "").strip()
        experiment_name = re.sub(r"_top_\d+$", "", experiment_name)
        return experiment_name.strip()


    def infer_top_k_from_experiment_name(experiment_name: str):
        """
        Extracts top-k from experiment names such as:
            dense_parent_child_top_5  -> 5
            dense_parent_child_top_10 -> 10
        """
        experiment_name = str(experiment_name or "").strip()
        match = re.search(r"_top_(\d+)$", experiment_name)

        if match:
            return int(match.group(1))

        return np.nan


    def get_single_or_joined_value_from_json(
        df: pd.DataFrame,
        column: str,
        fallback: str
    ) -> str:
        values = (
            df[column]
            .dropna()
            .astype(str)
            .str.strip()
        )

        values = sorted([v for v in values.unique() if v])

        if len(values) == 0:
            return fallback

        if len(values) == 1:
            return make_safe_filename(values[0])

        joined = "__".join(make_safe_filename(v) for v in values)

        if len(joined) > 180:
            return f"multiple_{column}"

        return joined


    df["method"] = df["experiment"].astype(str).map(infer_method_from_experiment_name)

    top_k_values = df["experiment"].astype(str).map(infer_top_k_from_experiment_name)

    # Fallback if the experiment name does not contain _top_{k}.
    for possible_top_k_col in ["top_k", "top_k_parents", "top_k_children"]:
        if possible_top_k_col in df.columns:
            fallback_values = pd.to_numeric(df[possible_top_k_col], errors="coerce")
            top_k_values = top_k_values.fillna(fallback_values)

    df["top_k"] = pd.to_numeric(top_k_values, errors="coerce").astype("Int64")

    if df["top_k"].isna().any():
        missing_top_k_count = int(df["top_k"].isna().sum())
        raise ValueError(
            f"Could not infer top_k for {missing_top_k_count} rows. "
            "Expected experiment names ending in _top_{k}, "
            "a top_k/top_k_parents/top_k_children column, "
            "or another supported top_k column."
        )

    dataset_from_json = get_single_or_joined_value_from_json(
        df=df,
        column="dataset",
        fallback="unknown_dataset"
    )

    method_from_json = get_single_or_joined_value_from_json(
        df=df,
        column="method",
        fallback="unknown_method"
    )

    print("Dataset from JSON:", dataset_from_json)
    print("Method from experiment column:", method_from_json)

    print("\nIncluded generator models:")
    print(df["generator_model"].value_counts(dropna=False))

    print("\nIncluded top-k values:")
    print(df["top_k"].value_counts(dropna=False).sort_index())


    # ---
    # Settings
    # ---

    SIMCSE_MODEL = "WENGSYX/Multilingual_SimCSE"
    BGE_M3_MODEL = "BAAI/bge-m3"
    MATCHA_MODEL = "Siran-Li/MATCHA"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    MAX_UNIT_WORDS = 80

    BATCH_SIZE = 16
    BGE_BATCH_SIZE = 1
    MATCHA_PAIR_BATCH_SIZE = 16


    # ---
    # Helper functions
    # ---

    def split_into_units(text: str, max_unit_words: int = MAX_UNIT_WORDS):
        """
        Splits long answers into sentence-preserving evaluation units.

        max_unit_words is a soft target, not a hard limit.
        Overlong single sentences are kept intact.
        """
        if text is None:
            return []

        text = str(text).strip()

        if not text or text.lower() == "nan":
            return []

        text = re.sub(r"\s+", " ", text)

        sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9])", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence_word_count = len(sentence.split())

            if sentence_word_count > max_unit_words:
                if current_chunk:
                    chunks.append(" ".join(current_chunk).strip())
                    current_chunk = []
                    current_word_count = 0

                chunks.append(sentence)
                continue

            if current_chunk and current_word_count + sentence_word_count > max_unit_words:
                chunks.append(" ".join(current_chunk).strip())
                current_chunk = [sentence]
                current_word_count = sentence_word_count
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_word_count

        if current_chunk:
            chunks.append(" ".join(current_chunk).strip())

        return chunks


    def harmonic_mean(p: float, r: float) -> float:
        if p + r == 0:
            return 0.0

        return 2 * p * r / (p + r)


    def to_numpy_scores(scores):
        """
        Converts MATCHA output to a flat numpy array.
        Handles tensor, numpy array, list, tuple, or scalar return values.
        """
        if torch.is_tensor(scores):
            return scores.detach().cpu().numpy().reshape(-1)

        if isinstance(scores, np.ndarray):
            return scores.reshape(-1)

        if isinstance(scores, (list, tuple)):
            converted = []

            for s in scores:
                if torch.is_tensor(s):
                    converted.append(float(s.detach().cpu().item()))
                else:
                    converted.append(float(s))

            return np.array(converted, dtype=float).reshape(-1)

        return np.array([float(scores)], dtype=float)


    def safe_text_series(series: pd.Series):
        return series.fillna("").astype(str).tolist()


    def release_cuda_memory():
        # Release models/temporary tensors that are no longer referenced.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    # ---
    # Prepare predictions and references
    # ---

    preds = safe_text_series(df["answer"])
    refs = safe_text_series(df["ground_truth"])

    empty_answer_count = sum(
        1 for x in preds
        if not str(x).strip() or str(x).strip().lower() == "nan"
    )

    empty_ref_count = sum(
        1 for x in refs
        if not str(x).strip() or str(x).strip().lower() == "nan"
    )

    print("\nEmpty generated answers:", empty_answer_count)
    print("Empty ground-truth answers:", empty_ref_count)


    # ---
    # ROUGE-1, ROUGE-2, ROUGE-L
    # ---

    rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    rouge_rows = []

    for pred, ref in zip(preds, refs):
        pred = str(pred).strip()
        ref = str(ref).strip()

        if not pred or not ref:
            rouge_rows.append({
                "rouge1_precision": 0.0,
                "rouge1_recall": 0.0,
                "rouge1_f1": 0.0,
                "rouge2_precision": 0.0,
                "rouge2_recall": 0.0,
                "rouge2_f1": 0.0,
                "rougeL_precision": 0.0,
                "rougeL_recall": 0.0,
                "rougeL_f1": 0.0,
            })
            continue

        scores = rouge.score(
            target=ref,
            prediction=pred
        )

        rouge_rows.append({
            "rouge1_precision": scores["rouge1"].precision,
            "rouge1_recall": scores["rouge1"].recall,
            "rouge1_f1": scores["rouge1"].fmeasure,
            "rouge2_precision": scores["rouge2"].precision,
            "rouge2_recall": scores["rouge2"].recall,
            "rouge2_f1": scores["rouge2"].fmeasure,
            "rougeL_precision": scores["rougeL"].precision,
            "rougeL_recall": scores["rougeL"].recall,
            "rougeL_f1": scores["rougeL"].fmeasure,
        })

    rouge_df = pd.DataFrame(rouge_rows)


    # ---
    # Chunked BERTScore with max-matching
    # ---

    bert_scorer = BERTScorer(
        lang=BERTSCORE_LANG,
        rescale_with_baseline=False,
        device=DEVICE
    )


    def chunked_bertscore_max_matching(pred_text: str, ref_text: str):
        pred_units = split_into_units(pred_text)
        ref_units = split_into_units(ref_text)

        if len(pred_units) == 0 or len(ref_units) == 0:
            return 0.0, 0.0, 0.0, len(pred_units), len(ref_units)

        pair_preds = []
        pair_refs = []

        for pred_unit in pred_units:
            for ref_unit in ref_units:
                pair_preds.append(pred_unit)
                pair_refs.append(ref_unit)

        _, _, pair_f1 = bert_scorer.score(
            pair_preds,
            pair_refs,
            batch_size=BATCH_SIZE,
            verbose=False
        )

        sim_matrix = pair_f1.cpu().numpy().reshape(len(pred_units), len(ref_units))

        precision = float(sim_matrix.max(axis=1).mean())
        recall = float(sim_matrix.max(axis=0).mean())
        f1 = harmonic_mean(precision, recall)

        return precision, recall, f1, len(pred_units), len(ref_units)


    bert_rows = []

    for pred, ref in zip(preds, refs):
        p, r, f1, n_pred_units, n_ref_units = chunked_bertscore_max_matching(pred, ref)

        bert_rows.append({
            "bertscore_precision": p,
            "bertscore_recall": r,
            "bertscore_f1": f1,
            "bertscore_pred_units": n_pred_units,
            "bertscore_ref_units": n_ref_units,
        })

    bert_df = pd.DataFrame(bert_rows)

    # BERTScore is finished. Do not keep RoBERTa on the GPU while
    # SimCSE and BGE-M3 are evaluated.
    del bert_scorer
    release_cuda_memory()


    # ---
    # Chunked SimCSE semantic similarity with max-matching
    # ---

    tokenizer = AutoTokenizer.from_pretrained(SIMCSE_MODEL)
    simcse_model = AutoModel.from_pretrained(SIMCSE_MODEL).to(DEVICE)
    simcse_model.eval()


    def mean_pooling(last_hidden_state, attention_mask):
        attention_mask = attention_mask.unsqueeze(-1)
        token_embeddings = last_hidden_state * attention_mask
        sentence_embeddings = token_embeddings.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
        return sentence_embeddings


    def encode_simcse_units(text_units, batch_size: int = BATCH_SIZE, max_length: int = 512):
        all_embeddings = []

        for start in range(0, len(text_units), batch_size):
            batch_texts = text_units[start:start + batch_size]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )

            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = simcse_model(**inputs)
                embeddings = mean_pooling(
                    outputs.last_hidden_state,
                    inputs["attention_mask"]
                )
                embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0).numpy()


    def chunked_simcse_max_matching(pred_text: str, ref_text: str):
        pred_units = split_into_units(pred_text)
        ref_units = split_into_units(ref_text)

        if len(pred_units) == 0 or len(ref_units) == 0:
            return 0.0, 0.0, 0.0, len(pred_units), len(ref_units)

        pred_emb = encode_simcse_units(pred_units)
        ref_emb = encode_simcse_units(ref_units)

        sim_matrix = pred_emb @ ref_emb.T

        precision = float(sim_matrix.max(axis=1).mean())
        recall = float(sim_matrix.max(axis=0).mean())
        f1 = harmonic_mean(precision, recall)

        return precision, recall, f1, len(pred_units), len(ref_units)


    simcse_rows = []

    for pred, ref in zip(preds, refs):
        p, r, f1, n_pred_units, n_ref_units = chunked_simcse_max_matching(pred, ref)

        simcse_rows.append({
            "simcse_precision": p,
            "simcse_recall": r,
            "simcse_f1": f1,
            "simcse_pred_units": n_pred_units,
            "simcse_ref_units": n_ref_units,
        })

    simcse_df = pd.DataFrame(simcse_rows)

    # SimCSE is finished. Free its model before loading BGE-M3.
    del simcse_model
    del tokenizer
    release_cuda_memory()


    # ---
    # ECS with BGE-M3, whole-answer similarity without splitting
    # ---

    bge_model = SentenceTransformer(BGE_M3_MODEL, device=DEVICE)

    preds_for_bge = [p if str(p).strip() else " " for p in preds]
    refs_for_bge = [r if str(r).strip() else " " for r in refs]

    answer_bge_emb = bge_model.encode(
        preds_for_bge,
        normalize_embeddings=True,
        batch_size=BGE_BATCH_SIZE,
        show_progress_bar=True
    )

    gold_bge_emb = bge_model.encode(
        refs_for_bge,
        normalize_embeddings=True,
        batch_size=BGE_BATCH_SIZE,
        show_progress_bar=True
    )

    ecs_bge_m3 = np.sum(answer_bge_emb * gold_bge_emb, axis=1)

    for i, (pred, ref) in enumerate(zip(preds, refs)):
        if not str(pred).strip() or not str(ref).strip():
            ecs_bge_m3[i] = 0.0

    ecs_df = pd.DataFrame({
        "ecs_bge_m3": ecs_bge_m3
    })

    # BGE-M3 is finished. Free it before MATCHA allocates its own model.
    del bge_model
    release_cuda_memory()


    # ---
    # Chunked MATCHA with max-matching
    # ---

    matcha_model = MATCHA.from_pretrained(MATCHA_MODEL)


    def score_matcha_pairs(pair_preds, pair_refs, batch_size: int = MATCHA_PAIR_BATCH_SIZE):
        if len(pair_preds) == 0:
            return np.array([], dtype=float)

        all_scores = []

        for start in range(0, len(pair_preds), batch_size):
            batch_preds = pair_preds[start:start + batch_size]
            batch_refs = pair_refs[start:start + batch_size]

            try:
                batch_scores = matcha_model.score(batch_preds, batch_refs)
                batch_scores = to_numpy_scores(batch_scores)

            except Exception:
                single_scores = []

                for p, r in zip(batch_preds, batch_refs):
                    s = matcha_model.score(p, r)
                    single_scores.extend(to_numpy_scores(s).tolist())

                batch_scores = np.array(single_scores, dtype=float)

            all_scores.append(batch_scores)

        return np.concatenate(all_scores, axis=0)


    def chunked_matcha_max_matching(pred_text: str, ref_text: str):
        pred_units = split_into_units(pred_text)
        ref_units = split_into_units(ref_text)

        if len(pred_units) == 0 or len(ref_units) == 0:
            return 0.0, 0.0, 0.0, len(pred_units), len(ref_units)

        pair_preds = []
        pair_refs = []

        for pred_unit in pred_units:
            for ref_unit in ref_units:
                pair_preds.append(pred_unit)
                pair_refs.append(ref_unit)

        pair_scores = score_matcha_pairs(
            pair_preds,
            pair_refs,
            batch_size=MATCHA_PAIR_BATCH_SIZE
        )

        sim_matrix = pair_scores.reshape(len(pred_units), len(ref_units))

        precision = float(sim_matrix.max(axis=1).mean())
        recall = float(sim_matrix.max(axis=0).mean())
        f1 = harmonic_mean(precision, recall)

        return precision, recall, f1, len(pred_units), len(ref_units)


    matcha_rows = []

    for pred, ref in zip(preds, refs):
        p, r, f1, n_pred_units, n_ref_units = chunked_matcha_max_matching(pred, ref)

        matcha_rows.append({
            "matcha_precision": p,
            "matcha_recall": r,
            "matcha_f1": f1,
            "matcha_pred_units": n_pred_units,
            "matcha_ref_units": n_ref_units,
        })

    matcha_df = pd.DataFrame(matcha_rows)


    # ---
    # Combine row-level scores internally
    # ---

    auto_scores = pd.concat(
        [
            df.reset_index(drop=True),
            rouge_df,
            bert_df,
            simcse_df,
            ecs_df,
            matcha_df
        ],
        axis=1
    )


    # ---
    # Combine row-level scores and build per-question summaries
    # ---

    metric_cols = [
        "rouge1_precision",
        "rouge1_recall",
        "rouge1_f1",
        "rouge2_precision",
        "rouge2_recall",
        "rouge2_f1",
        "rougeL_precision",
        "rougeL_recall",
        "rougeL_f1",
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
        "simcse_precision",
        "simcse_recall",
        "simcse_f1",
        "ecs_bge_m3",
        "matcha_precision",
        "matcha_recall",
        "matcha_f1"
    ]

    def extract_trailing_number(value):
        """
        Extracts a trailing numeric suffix from IDs such as:
            tender_01         -> 1
            tender_24        -> 24

        Returns pandas.NA if no trailing number exists.
        """
        if pd.isna(value):
            return pd.NA

        value = str(value).strip()
        match = re.search(r"(\d+)$", value)

        if not match:
            return pd.NA

        return int(match.group(1))


    def infer_test_case_id(row):
        """
        Infers the test-case identifier used for per-question aggregation.

        Tender:
            test_id = "tender_01"
            -> test_case_id = 1
            This corresponds to column "ID" in test_cases.csv.

        LEDGAR:
            test_id = "a-ledgar_train_007371-Amendments"
            -> test_case_id = "a-ledgar_train_007371-Amendments"
            This corresponds to column "block_id" in ledgar_test_cases.csv.

        HUPD:
            test_id or block_id = "hupd_g06q_14987114-background"
            -> test_case_id = "hupd_g06q_14987114-background"
            This corresponds to column "block_id" in hupd_test_cases.csv.
        """
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

        if "ledgar" in dataset_value:
            if not pd.isna(test_id_value) and str(test_id_value).strip():
                return str(test_id_value).strip()

            if not pd.isna(block_id_value) and str(block_id_value).strip():
                return str(block_id_value).strip()

            return pd.NA

        if "hupd" in dataset_value:
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


    auto_scores["test_case_id"] = auto_scores.apply(infer_test_case_id, axis=1)

    if auto_scores["test_case_id"].isna().any():
        missing_ids = int(auto_scores["test_case_id"].isna().sum())
        raise ValueError(
            f"Could not infer test_case_id for {missing_ids} rows. "
            "Expected a usable test_id or block_id field."
        )

    if dataset_from_json.lower().find("tender") >= 0:
        auto_scores["test_case_id"] = pd.to_numeric(
            auto_scores["test_case_id"],
            errors="coerce"
        ).astype("Int64")
    else:
        auto_scores["test_case_id"] = auto_scores["test_case_id"].astype(str).str.strip()

    print("\nInferred test-case IDs:")
    print(auto_scores["test_case_id"].value_counts(dropna=False).sort_index())

    per_question_group_cols = [
        "dataset",
        "method",
        "top_k",
        "generator_model",
        "test_case_id"
    ]

    optional_metadata_cols = [
        "test_id",
        "block_id",
        "project_name",
        "theme",
        "chapter",
        "question",
        "original_question",
        "retrieval_query",
    ]

    optional_metadata_cols = [
        col for col in optional_metadata_cols
        if col in auto_scores.columns
    ]

    per_question_metrics = (
        auto_scores
        .groupby(per_question_group_cols, dropna=False)[metric_cols]
        .mean()
        .round(4)
        .reset_index()
    )

    per_question_counts = (
        auto_scores
        .groupby(per_question_group_cols, dropna=False)
        .agg(
            n_rows=("experiment", "size"),
            n_empty_answers=("answer", lambda x: int((x.fillna("").astype(str).str.strip() == "").sum())),
            n_empty_ground_truth=("ground_truth", lambda x: int((x.fillna("").astype(str).str.strip() == "").sum()))
        )
        .reset_index()
    )

    if optional_metadata_cols:
        per_question_metadata = (
            auto_scores
            .groupby(per_question_group_cols, dropna=False)[optional_metadata_cols]
            .first()
            .reset_index()
        )
    else:
        per_question_metadata = auto_scores[per_question_group_cols].drop_duplicates()

    per_question_summary = (
        per_question_counts
        .merge(
            per_question_metrics,
            on=per_question_group_cols,
            how="left"
        )
        .merge(
            per_question_metadata,
            on=per_question_group_cols,
            how="left"
        )
    )

    per_question_sort_cols = [
        "dataset",
        "method",
        "generator_model",
        "top_k",
        "test_case_id"
    ]

    per_question_summary = per_question_summary.sort_values(
        per_question_sort_cols
    ).reset_index(drop=True)


    # ---
    # Build the original setup-level summary as before
    # ---

    group_cols = [
        "dataset",
        "method",
        "top_k",
        "generator_model"
    ]

    summary_metrics = (
        auto_scores
        .groupby(group_cols, dropna=False)[metric_cols]
        .mean()
        .round(4)
        .reset_index()
    )

    summary_counts = (
        auto_scores
        .groupby(group_cols, dropna=False)
        .agg(
            n_rows=("experiment", "size"),
            n_questions=("test_case_id", "nunique"),
            n_empty_answers=("answer", lambda x: int((x.fillna("").astype(str).str.strip() == "").sum())),
            n_empty_ground_truth=("ground_truth", lambda x: int((x.fillna("").astype(str).str.strip() == "").sum()))
        )
        .reset_index()
    )

    summary = (
        summary_counts
        .merge(
            summary_metrics,
            on=group_cols,
            how="left"
        )
    )

    summary = summary.sort_values(
        ["dataset", "method", "generator_model", "top_k"]
    ).reset_index(drop=True)


    # ---
    # Save per-question and setup-level summary metrics
    # ---

    result_output_dir = ANALYSIS_OUTPUT_ROOT / make_safe_filename(result_path.stem)
    result_output_dir.mkdir(parents=True, exist_ok=True)

    per_question_xlsx_path = (
        result_output_dir /
        f"per_question_metrics_{dataset_from_json}_{method_from_json}_{suffix}.xlsx"
    )

    per_question_csv_path = (
        result_output_dir /
        f"per_question_metrics_{dataset_from_json}_{method_from_json}_{suffix}.csv"
    )

    per_question_summary.to_excel(
        per_question_xlsx_path,
        index=False
    )

    per_question_summary.to_csv(
        per_question_csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    summary_path = (
        result_output_dir /
        f"summary_metrics_{dataset_from_json}_{method_from_json}_{suffix}.xlsx"
    )

    summary.to_excel(
        summary_path,
        index=False
    )

    print("\nPer-question metrics saved to:")
    print(per_question_xlsx_path)
    print(per_question_csv_path)

    print("\nSetup-level summary metrics saved to:")
    print(summary_path)

    print("\nPer-question rows saved:", len(per_question_summary))
    print("Per-question columns saved:", len(per_question_summary.columns))

    print("\nSetup-level rows saved:", len(summary))
    print("Setup-level columns saved:", len(summary.columns))

    print("\nPer-question summary:")
    print(per_question_summary.to_string(index=False))

    print("\nSetup-level summary:")
    print(summary.to_string(index=False))

    return {
        "per_question_xlsx": per_question_xlsx_path,
        "per_question_csv": per_question_csv_path,
        "summary_xlsx": summary_path,
    }
    
summary_paths = []

for result_file in RESULT_FILES:
    try:
        summary_path = analyze_result_file(result_file)
        summary_paths.append(summary_path)

    except Exception as error:
        print("\nERROR while processing:")
        print(result_file)
        print(type(error).__name__, str(error))

print("\nFinished batch analysis.")
print("Successful result files:", len(summary_paths))
print("Failed result files:", len(RESULT_FILES) - len(summary_paths))

print("\nCreated summary files:")
for path in summary_paths:
    print("-", path)
