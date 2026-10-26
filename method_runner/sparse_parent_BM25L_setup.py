#!/usr/bin/env python
# coding: utf-8

# --- BM25L parent-only retrieval ---
# Rank parent records with BM25L and generate from the retrieved parent text.

import os
import re
import gc
import json
import yaml
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from rank_bm25 import BM25L

from llm_clients_vllm import (
    build_generator_configs,
    setup_generator,
    generate_answer,
    release_generator,
)


# --- Shared runner utilities ---
try:
    from .runner_utils import ensure_dir, make_safe_filename, row_value, utc_now
except ImportError:
    from runner_utils import ensure_dir, make_safe_filename, row_value, utc_now

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Set EXPERIMENT_CONFIG to select a dataset config; defaults to HUPD.
CONFIG_PATH = Path(os.environ.get(
    "EXPERIMENT_CONFIG", PROJECT_ROOT / "hupd_dataset" / "hupd_config.yaml"
)).expanduser().resolve()

# The launcher sets this automatically for each worker.
# Example:
# RUN_ONLY_GENERATOR=Qwen/Qwen2.5-7B-Instruct python sparse_parent_setup.py
run_only_generator_env = os.environ.get("RUN_ONLY_GENERATOR")

RUN_ONLY_GENERATORS: Optional[List[str]] = (
    [run_only_generator_env]
    if run_only_generator_env
    else None
)

# This script implements the sparse parent-only experiment.
# BM25L is the sparse retriever used internally.
EXPERIMENT_BASE_NAME = "sparse_parent_L"




def log_answer(generator_name: str, test_id: str, top_k: int, answer: str) -> None:
    log_dir = RESULTS_DIR.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{make_safe_filename(generator_name)}.log"

    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"\n{'=' * 80}\n"
            f"test_id={test_id} | top_k={top_k}\n"
            f"{'=' * 80}\n"
            f"{answer}\n"
        )
        f.flush()



# ---
# SQLite checkpoint / queue helpers
# ---





def get_queue_db_path(generator_name: str) -> Path:
    safe_generator_name = make_safe_filename(generator_name)
    return RESULTS_DIR / f"{DATASET_NAME}_{EXPERIMENT_BASE_NAME}_{safe_generator_name}.sqlite"


def init_result_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path, timeout=60) as connection:
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA busy_timeout=60000;")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                job_id TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                experiment_name TEXT NOT NULL,
                generator_name TEXT NOT NULL,
                row_index INTEGER NOT NULL,
                test_id TEXT NOT NULL,
                top_k INTEGER NOT NULL,
                status TEXT NOT NULL,
                result_json TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )


def make_job_id(generator_name: str, row_index: int, top_k: int) -> str:
    return (
        f"{DATASET_NAME}__{EXPERIMENT_BASE_NAME}__{generator_name}"
        f"__row_{row_index:08d}__topk_{top_k}"
    )


def is_job_completed(db_path: Path, job_id: str) -> bool:
    with sqlite3.connect(db_path, timeout=60) as connection:
        row = connection.execute(
            "SELECT status FROM results WHERE job_id = ?;",
            (job_id,),
        ).fetchone()

    return row is not None and row[0] == "completed"


def claim_job(
    db_path: Path,
    job_id: str,
    dataset_name: str,
    experiment_name: str,
    generator_name: str,
    row_index: int,
    test_id: str,
    top_k: int,
) -> bool:
    now = utc_now()

    connection = sqlite3.connect(
        db_path,
        timeout=60,
        isolation_level=None,
    )

    try:
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA busy_timeout=60000;")
        connection.execute("BEGIN IMMEDIATE;")

        row = connection.execute(
            "SELECT status FROM results WHERE job_id = ?;",
            (job_id,),
        ).fetchone()

        if row is not None:
            status = str(row[0])
            if status in {"completed", "running"}:
                connection.execute("COMMIT;")
                return False

        connection.execute(
            """
            INSERT OR REPLACE INTO results (
                job_id,
                dataset_name,
                experiment_name,
                generator_name,
                row_index,
                test_id,
                top_k,
                status,
                result_json,
                error_message,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'running', NULL, NULL, ?, ?);
            """,
            (
                job_id,
                dataset_name,
                experiment_name,
                generator_name,
                int(row_index),
                str(test_id),
                int(top_k),
                now,
                now,
            ),
        )

        connection.execute("COMMIT;")
        return True

    except Exception:
        try:
            connection.execute("ROLLBACK;")
        except sqlite3.OperationalError:
            pass
        raise

    finally:
        connection.close()


def save_completed_job(
    db_path: Path,
    job_id: str,
    result: Dict[str, Any],
) -> None:
    now = utc_now()
    result_json = json.dumps(result, ensure_ascii=False)

    with sqlite3.connect(db_path, timeout=60) as connection:
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA busy_timeout=60000;")
        connection.execute(
            """
            INSERT OR REPLACE INTO results (
                job_id,
                dataset_name,
                experiment_name,
                generator_name,
                row_index,
                test_id,
                top_k,
                status,
                result_json,
                error_message,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'completed', ?, NULL, ?, ?);
            """,
            (
                job_id,
                result["dataset"],
                result["experiment"],
                result["generator_name"],
                int(result["row_index"]),
                str(result["test_id"]),
                int(result["top_k_parents"]),
                result_json,
                now,
                now,
            ),
        )


def save_failed_job(
    db_path: Path,
    job_id: str,
    error_message: str,
) -> None:
    now = utc_now()

    with sqlite3.connect(db_path, timeout=60) as connection:
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA busy_timeout=60000;")
        connection.execute(
            """
            UPDATE results
            SET
                status = 'failed',
                error_message = ?,
                updated_at = ?
            WHERE job_id = ?;
            """,
            (
                str(error_message),
                now,
                job_id,
            ),
        )


def export_completed_jobs(
    db_path: Path,
    output_path: Path,
) -> List[Dict[str, Any]]:
    with sqlite3.connect(db_path, timeout=60) as connection:
        rows = connection.execute(
            """
            SELECT result_json
            FROM results
            WHERE status = 'completed'
            ORDER BY row_index ASC, top_k ASC;
            """
        ).fetchall()

    results = [json.loads(row[0]) for row in rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Multiple workers can export the same final JSON at nearly the same time.
    # Each worker therefore needs its own temporary file name.
    tmp_path = output_path.with_name(
        f"{output_path.name}.pid_{os.getpid()}.tmp"
    )

    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())

    tmp_path.replace(output_path)

    return results


# ---
# Config loading and validation
# ---

def load_config(config_path: Path) -> Dict[str, Any]:
    print("Loading config from:", config_path)
    print("Exists:", config_path.exists())

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("YAML config did not load as a dictionary.")

    return config


def require_config_path(config: Dict[str, Any], key: str) -> Path:
    try:
        value = config["paths"][key]
    except KeyError as exc:
        raise KeyError(f"Missing config path: paths.{key}") from exc

    path = Path(value).expanduser()
    base_dir = Path(str(config.get("base_dir", "."))).expanduser()
    if not base_dir.is_absolute():
        base_dir = PROJECT_ROOT / base_dir
    return (path if path.is_absolute() else base_dir / path).resolve()




config = load_config(CONFIG_PATH)

DATASET_NAME = str(config.get("dataset", "dataset")).strip()

PARENTS_PATH = require_config_path(config, "parents")
RESULTS_DIR = require_config_path(config, "results_dir")
SYSTEM_PROMPT_PATH = require_config_path(config, "system_prompt")
TEST_CASES_PATH = require_config_path(config, "test_cases")

ensure_dir(RESULTS_DIR)

TOP_K_VARIANTS = list(config.get("retrieval", {}).get("top_k", [5, 10, 15, 20]))
GENERATOR_CONFIGS = build_generator_configs(config)

MAX_OUTPUT_TOKENS = int(config.get("generation", {}).get("max_output_tokens", 1500))
TEMPERATURE = float(config.get("generation", {}).get("temperature", 0.0))

# BM25L parameters. Defaults preserve the original sparse parent-only pipeline.
bm25l_config = config.get("retrieval", {}).get("bm25l", {})
BM25L_K1 = float(bm25l_config.get("k1", 1.5))
BM25L_B = float(bm25l_config.get("b", 0.75))
BM25L_DELTA = float(bm25l_config.get("delta", 0.5))
REQUIRE_POSITIVE_BM25_SCORE = bool(bm25l_config.get("require_positive_score", True))

if RUN_ONLY_GENERATORS is not None:
    allowed = set(RUN_ONLY_GENERATORS)
    GENERATOR_CONFIGS = [
        generator_config
        for generator_config in GENERATOR_CONFIGS
        if generator_config.get("name") in allowed
    ]

if not GENERATOR_CONFIGS:
    raise ValueError("No generators found in config or RUN_ONLY_GENERATORS filtered all generators out.")

configured_methods = set(config.get("retrieval", {}).get("methods", []))
if configured_methods and EXPERIMENT_BASE_NAME not in configured_methods:
    print(
        f"Warning: this script runs '{EXPERIMENT_BASE_NAME}', "
        f"but that method is not listed in retrieval.methods: {sorted(configured_methods)}"
    )

print("Dataset:", DATASET_NAME)
print("Experiment:", EXPERIMENT_BASE_NAME)
print("Top-k variants:", TOP_K_VARIANTS)
print("Generators:", [g.get("name") for g in GENERATOR_CONFIGS])
print("Results dir:", RESULTS_DIR)
print("BM25L k1:", BM25L_K1)
print("BM25L b:", BM25L_B)
print("BM25L delta:", BM25L_DELTA)
print("Require positive BM25 score:", REQUIRE_POSITIVE_BM25_SCORE)


# ---
# System prompt
# ---
# Metadata is intentionally NOT provided to the generator.
# The prompt body is dataset-specific and loaded from YAML path.

if not SYSTEM_PROMPT_PATH.exists():
    raise FileNotFoundError(f"System prompt file not found: {SYSTEM_PROMPT_PATH}")

SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


# ---
# Utility
# ---

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def normalize_regex_from_yaml(pattern: str) -> str:
    """
    YAML should normally contain only the regex itself, for example:
    '\\s*Write\\s+a\\s+concise...'

    This helper also tolerates accidental Python-style adjacent raw strings
    pasted into YAML, for example:
    r"..."
      r"..."
    and converts them into one regex string.
    """

    pattern = str(pattern or "").strip()

    fragments = re.findall(r"r?(['\"])(.*?)(?<!\\)\1", pattern, flags=re.DOTALL)
    if fragments:
        return "".join(fragment for _, fragment in fragments)

    return pattern


def clean_query_for_retrieval(question: str) -> str:
    text = str(question or "")

    for pattern_config in config.get("query_cleaning", {}).get("patterns", []):
        pattern_name = pattern_config.get("name", "unnamed_pattern")
        raw_pattern = pattern_config.get("regex", "")
        pattern = normalize_regex_from_yaml(raw_pattern)

        if not pattern:
            continue

        try:
            text = re.sub(
                pattern,
                "",
                text,
                flags=re.IGNORECASE | re.UNICODE,
            )
        except re.error as exc:
            raise ValueError(
                f"Invalid regex in query_cleaning pattern '{pattern_name}': {raw_pattern}"
            ) from exc

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# Backwards-compatible name from the original script.
def strip_max_words_instruction(question: str) -> str:
    return clean_query_for_retrieval(question)




def has_any_column(df: pd.DataFrame, *names: str) -> bool:
    return any(name in df.columns for name in names)


def get_chapter_title_from_row(row: pd.Series) -> str:
    chapter = row_value(row, "Chapter", "chapter", "label_name", default="").strip()

    if not chapter:
        raise ValueError(
            "Missing chapter title. Expected one of the columns: Chapter, chapter, label_name."
        )

    return chapter


def build_retrieval_query(chapter_title: str, example_question: str) -> str:
    """
    Build the retrieval query from structured test-case fields.

    Retrieval uses:
    - chapter title from the separate Chapter/chapter/label_name column
    - cleaned Example Question with configured generation-instruction patterns removed

    The generator still receives the full, original Example Question.
    """

    chapter_title = str(chapter_title or "").strip()
    cleaned_question = clean_query_for_retrieval(example_question)

    retrieval_query = f"{chapter_title}\n\n{cleaned_question}".strip()
    retrieval_query = re.sub(r"\n{3,}", "\n\n", retrieval_query)

    return retrieval_query


# ---
# Load artifacts
# ---

parents = load_jsonl(PARENTS_PATH)

parents = [
    parent
    for parent in parents
    if str(parent.get("text", "")).strip()
]

parent_lookup = {}
for parent in parents:
    for key in ["parent_id", "block_id"]:
        value = parent.get(key)
        if str(value or "").strip():
            parent_lookup[str(value).strip()] = parent

print("Parents loaded:", len(parents))
print("Parent lookup size:", len(parent_lookup))

if not parents:
    raise ValueError("No parent records with non-empty text found.")


# ---
# Sparse BM25L parent index
# ---

def tokenize_for_bm25(text: str) -> List[str]:
    """
    German-compatible tokenizer for BM25L.
    Keeps terms such as DSGVO, RBAC, LoRaWAN, API, OAuth2, etc.
    """

    text = str(text or "").lower()

    tokens = re.findall(
        r"[a-zäöüßA-ZÄÖÜ0-9][a-zäöüßA-ZÄÖÜ0-9_.-]*",
        text,
    )

    return tokens


def build_parent_retrieval_text(parent: Dict[str, Any]) -> str:
    """
    Text used for BM25L retrieval.

    Metadata from parents.jsonl is included here for retrieval matching only.
    Metadata is not passed to the generator context.
    """

    return f"""
Theme: {parent.get("theme", "")}
Chapter: {parent.get("chapter", "")}
Structure Path: {parent.get("structure_path", "")}

{parent.get("text", "")}
""".strip()


parent_retrieval_texts = [
    build_parent_retrieval_text(parent)
    for parent in parents
]

tokenized_parent_texts = [
    tokenize_for_bm25(text)
    for text in parent_retrieval_texts
]

bm25l_parent_index = BM25L(
    tokenized_parent_texts,
    k1=BM25L_K1,
    b=BM25L_B,
    delta=BM25L_DELTA,
)

print("BM25L indexed parents:", len(tokenized_parent_texts))


# ---
# Sparse parent retrieval with BM25L
# ---

def search_parents_bm25l(
    query: str,
    top_k_parents: int,
    require_positive_score: bool = REQUIRE_POSITIVE_BM25_SCORE,
) -> List[Tuple[Dict[str, Any], float]]:
    retrieval_query = strip_max_words_instruction(query)
    tokenized_query = tokenize_for_bm25(retrieval_query)

    if not tokenized_query:
        raise ValueError("Retrieval query contains no valid BM25L tokens.")

    scores = bm25l_parent_index.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    hits = []

    for idx in ranked_indices:
        score = float(scores[idx])

        if require_positive_score and score <= 0:
            continue

        hits.append((parents[idx], score))

        if len(hits) >= top_k_parents:
            break

    if not hits:
        raise ValueError(
            "BM25L found no parents with a positive score for this retrieval query."
        )

    return hits


def retrieve_top_parents(
    query: str,
    top_k_parents: int,
) -> Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], float]]]:
    parent_hits = search_parents_bm25l(
        query=query,
        top_k_parents=top_k_parents,
    )

    parents_context = [
        parent
        for parent, score in parent_hits
    ]

    return parents_context, parent_hits


# ---
# Context and prompt
# ---

def build_parent_only_context(
    parents_context: List[Dict[str, Any]],
) -> str:
    """
    Generator-facing context WITHOUT metadata.

    This intentionally does NOT include:
    - theme
    - chapter
    - structure_path
    - score
    - child chunks

    The generator receives only the parent text.
    """

    context = ""

    for i, parent in enumerate(parents_context, start=1):
        context += f"[PARENT-{i}]\n\n"
        context += str(parent.get("text", ""))
        context += "\n\n"

    return context.strip()


def build_prompt(chapter_title: str, example_question: str, context: str) -> str:
    return f"""{SYSTEM_PROMPT}

---

## Input / Eingabe

Retrieved context / Bereitgestellte Dokumente / Kontext:

{context}

Chapter / Zu schreibendes Kapitel:

{chapter_title}

Example Question / Nutzeranfrage / Anforderungen:

{example_question}

---

## Answer / Antwort
""".strip()


def get_max_model_len_from_generator_state(
    generator_state: Dict[str, Any]
) -> Optional[int]:
    """
    Return the configured vLLM max_model_len if it is available.

    The launcher/YAML value is attached to generator_state in
    run_full_experiment_for_generator. We intentionally prefer that value
    because it is stable and does not depend on vLLM internals.
    """

    for key in ["max_model_len", "model_max_length"]:
        value = generator_state.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass

    generator_config = generator_state.get("generator_config", {})
    if isinstance(generator_config, dict):
        for key in ["max_model_len", "model_max_length"]:
            value = generator_config.get(key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass

    return None


def get_tokenizer_from_generator_state(generator_state: Dict[str, Any]):
    """
    Return the vLLM tokenizer if possible.

    This is used only for prompt-size estimation before calling vLLM.
    If tokenization is unavailable, the code falls back to the original
    behavior and lets vLLM validate the prompt.
    """

    model = generator_state.get("model")

    if model is None:
        return None

    if hasattr(model, "get_tokenizer"):
        try:
            return model.get_tokenizer()
        except Exception:
            return None

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer

    return None


def count_prompt_tokens_for_vllm_chat(
    prompt: str,
    generator_state: Dict[str, Any],
) -> Optional[int]:
    """
    Estimate the same chat prompt token length that vLLM will validate.

    llm_clients_vllm.generate_answer() wraps the prompt as one user message.
    Therefore the estimate also uses one user message and the model chat template.
    """

    tokenizer = get_tokenizer_from_generator_state(generator_state)

    if tokenizer is None:
        return None

    messages = [
        {
            "role": "user",
            "content": str(prompt),
        }
    ]

    try:
        if hasattr(tokenizer, "apply_chat_template"):
            token_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        else:
            token_ids = tokenizer.encode(str(prompt))
    except Exception:
        try:
            token_ids = tokenizer.encode(str(prompt))
        except Exception:
            return None

    return int(len(token_ids))


def build_prompt_fitted_to_context_window(
    chapter_title: str,
    example_question: str,
    parents_context: List[Dict[str, Any]],
    generator_state: Dict[str, Any],
    max_output_tokens: int,
) -> Tuple[str, str, List[Dict[str, Any]], Optional[int], Optional[int], bool, int]:
    """
    Build a prompt and, if possible, trim trailing retrieved parents until
    prompt + reserved output fits into max_model_len.

    Highest-ranked parents are kept first.
    """

    max_model_len = get_max_model_len_from_generator_state(generator_state)
    selected_parents = list(parents_context)
    context_was_truncated = False

    if max_model_len is None:
        context = build_parent_only_context(selected_parents)
        prompt = build_prompt(
            chapter_title=chapter_title,
            example_question=example_question,
            context=context,
        )
        prompt_tokens = count_prompt_tokens_for_vllm_chat(prompt, generator_state)
        return (
            context,
            prompt,
            selected_parents,
            prompt_tokens,
            None,
            context_was_truncated,
            int(max_output_tokens),
        )

    max_model_len = int(max_model_len)
    requested_output_tokens = max(1, int(max_output_tokens))

    # Reserve the requested output budget, but never reserve the entire context.
    output_reserve = min(requested_output_tokens, max(1, max_model_len - 1))
    prompt_token_budget = max(1, max_model_len - output_reserve)

    while True:
        context = build_parent_only_context(selected_parents)
        prompt = build_prompt(
            chapter_title=chapter_title,
            example_question=example_question,
            context=context,
        )

        prompt_tokens = count_prompt_tokens_for_vllm_chat(prompt, generator_state)

        # If token counting failed, keep original behavior and let vLLM validate.
        if prompt_tokens is None:
            return (
                context,
                prompt,
                selected_parents,
                None,
                prompt_token_budget,
                context_was_truncated,
                requested_output_tokens,
            )

        available_output_tokens = max_model_len - prompt_tokens

        if prompt_tokens <= prompt_token_budget and available_output_tokens >= 1:
            effective_max_output_tokens = min(
                requested_output_tokens,
                available_output_tokens,
            )
            return (
                context,
                prompt,
                selected_parents,
                prompt_tokens,
                prompt_token_budget,
                context_was_truncated,
                int(effective_max_output_tokens),
            )

        if selected_parents:
            selected_parents = selected_parents[:-1]
            context_was_truncated = True
            continue

        # Even an empty context does not fit. Return it and allow the caller
        # to record the vLLM validation failure for this job.
        effective_max_output_tokens = max(0, available_output_tokens)
        return (
            context,
            prompt,
            selected_parents,
            prompt_tokens,
            prompt_token_budget,
            context_was_truncated,
            int(effective_max_output_tokens),
        )


def build_contexts_for_ragas(
    parents_context: List[Dict[str, Any]],
) -> List[str]:
    """
    RAGAS contexts match the generator-visible context units:
    parent texts only, without metadata.
    """

    return [
        str(parent.get("text", "")).strip()
        for parent in parents_context
        if str(parent.get("text", "")).strip()
    ]


# ---
# Full answer pipeline
# ---

def answer_query(
    chapter_title: str,
    example_question: str,
    top_k_parents: int,
    generator_state: Dict[str, Any],
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    temperature: float = TEMPERATURE,
) -> Dict[str, Any]:
    retrieval_query = build_retrieval_query(
        chapter_title=chapter_title,
        example_question=example_question,
    )

    parents_context, parent_hits = retrieve_top_parents(
        query=retrieval_query,
        top_k_parents=top_k_parents,
    )

    (
        context,
        prompt,
        fitted_parents_context,
        prompt_token_count,
        prompt_token_budget,
        context_was_truncated,
        effective_max_output_tokens,
    ) = build_prompt_fitted_to_context_window(
        chapter_title=chapter_title,
        example_question=example_question,
        parents_context=parents_context,
        generator_state=generator_state,
        max_output_tokens=max_output_tokens,
    )

    parents_context = fitted_parents_context

    if context_was_truncated:
        print(
            "[CONTEXT_TRUNCATED] "
            f"requested_top_k_parents={top_k_parents} | "
            f"used_top_k_parents={len(parents_context)} | "
            f"prompt_tokens={prompt_token_count} | "
            f"prompt_token_budget={prompt_token_budget}",
            flush=True,
        )

    answer = generate_answer(
        prompt=prompt,
        generator_state=generator_state,
        max_output_tokens=effective_max_output_tokens,
        temperature=temperature,
    )

    return {
        "chapter_title": chapter_title,
        "query": example_question,
        "retrieval_query": retrieval_query,
        "answer": answer,
        "top_k_parents": top_k_parents,
        "requested_top_k_parents": top_k_parents,
        "used_top_k_parents": len(parents_context),
        "context_was_truncated": context_was_truncated,
        "prompt_token_count": prompt_token_count,
        "prompt_token_budget": prompt_token_budget,
        "effective_max_output_tokens": effective_max_output_tokens,
        "experiment": f"{EXPERIMENT_BASE_NAME}_top_{top_k_parents}",
        "generator_name": generator_state.get("name"),
        "generator_model": generator_state.get("model_name"),
        "parent_hits": [
            {
                "rank": rank,
                "parent_id": parent.get("parent_id", ""),
                "block_id": parent.get("block_id", ""),
                "score": score,
                "theme": parent.get("theme", ""),
                "chapter": parent.get("chapter", ""),
                "structure_path": parent.get("structure_path", ""),
                "text": parent.get("text", ""),
            }
            for rank, (parent, score) in enumerate(parent_hits, start=1)
        ],
        "retrieved_parent_ids": [
            parent.get("parent_id", parent.get("block_id", ""))
            for parent in parents_context
        ],
        "parents_context": parents_context,
        # Generator-facing context.
        # This contains parent text only.
        # It contains no metadata and no child chunks.
        "context": context,
        "prompt": prompt,
        # RAGAS contexts also contain parent text only.
        "contexts_for_ragas": build_contexts_for_ragas(parents_context),
    }


# ---
# Optional parent retrieval debug table
# ---

def run_sparse_parent_only_bm25l_test_with_full_output(
    chapter_title: str,
    example_question: str,
    top_k_parents: int,
    generator_name: Optional[str] = None,
) -> Dict[str, Any]:
    if generator_name is None:
        generator_config = GENERATOR_CONFIGS[0]
    else:
        matches = [g for g in GENERATOR_CONFIGS if g.get("name") == generator_name]
        if not matches:
            raise ValueError(f"Generator not found in config: {generator_name}")
        generator_config = matches[0]

    generator_state = setup_generator(generator_config)
    generator_state["generator_config"] = dict(generator_config)

    try:
        result = answer_query(
            chapter_title=chapter_title,
            example_question=example_question,
            top_k_parents=top_k_parents,
            generator_state=generator_state,
        )
        
        log_answer(
            generator_name=generator_name,
            test_id=test_id,
            top_k=top_k,
            answer=result["answer"],
        )
    finally:
        release_generator(generator_state)

    rows = []

    for hit in result["parent_hits"]:
        rows.append({
            "Rank": hit["rank"],
            "Parent ID": hit["parent_id"],
            "Block ID": hit["block_id"],
            "BM25L Score": hit["score"],
            "Theme": hit["theme"],
            "Chapter": hit["chapter"],
            "Path": hit["structure_path"],
            "Text Preview": str(hit["text"])[:300],
        })

    retrieval_table = pd.DataFrame(rows)

    print("\n" + "=" * 120)
    print("CHAPTER TITLE")
    print("=" * 120)
    print(result["chapter_title"])

    print("\n" + "=" * 120)
    print("ORIGINAL GENERATOR QUERY")
    print("=" * 120)
    print(result["query"])

    print("\n" + "=" * 120)
    print("CLEANED RETRIEVAL QUERY")
    print("=" * 120)
    print(result["retrieval_query"])

    print("\n" + "=" * 120)
    print("TOP PARENT BM25L RETRIEVAL TABLE")
    print("=" * 120)
    try:
        display(retrieval_table)
    except NameError:
        print(retrieval_table)

    print("\n" + "=" * 120)
    print("CONTEXT PROVIDED TO THE LLM")
    print("Parent text only. No metadata. No child chunks.")
    print("=" * 120)
    print(result["context"])

    print("\n" + "=" * 120)
    print("FINAL ANSWER")
    print("=" * 120)
    print(result["answer"])

    result["retrieval_table"] = retrieval_table

    return result


# ---
# Full experiment runner from YAML test cases
# ---

def load_test_cases(test_cases_path: Path) -> pd.DataFrame:
    print("Loading test cases from:", test_cases_path)
    print("Exists:", test_cases_path.exists())

    if not test_cases_path.exists():
        raise FileNotFoundError(f"Test cases file not found: {test_cases_path}")

    suffix = test_cases_path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(test_cases_path)

    elif suffix == ".csv":
        dataset_name = str(DATASET_NAME).lower().strip()

        if dataset_name == "tender":
            # Tender CSV files are semicolon-separated.
            df = pd.read_csv(test_cases_path, sep=";")

        elif dataset_name == "ledgar":
            # LEDGAR CSV files are comma-separated.
            df = pd.read_csv(test_cases_path)

        else:
            # Fallback: let pandas infer the delimiter.
            df = pd.read_csv(test_cases_path, sep=None, engine="python")

    else:
        raise ValueError(
            f"Unsupported test case file type '{suffix}'. Use .csv, .xlsx, or .xls."
        )

    df.columns = df.columns.str.strip()

    print("Columns:", list(df.columns))

    required_columns = [
        "Example Question",
        "Gold Text",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required test case columns: {missing}"
        )

    if not has_any_column(df, "Chapter", "chapter", "label_name"):
        raise ValueError(
            "Missing required chapter column. Expected one of: "
            "'Chapter', 'chapter', or 'label_name'."
        )

    return df


def build_ragas_row(
    row: pd.Series,
    row_index: int,
    chapter_title: str,
    original_question: str,
    question_for_ragas: str,
    ground_truth: str,
    top_k: int,
    result: Dict[str, Any],
    generator_state: Dict[str, Any],
) -> Dict[str, Any]:
    test_id = row_value(row, "ID", "id", "block_id", "parent_id", default=str(row_index + 1))
    project_name = row_value(row, "Project Name", "project_name", default="")
    theme = row_value(row, "Theme", "theme", default="")
    chapter = chapter_title

    ragas_row = {
        "dataset": DATASET_NAME,
        "experiment": f"{EXPERIMENT_BASE_NAME}_top_{top_k}",
        "generator_name": generator_state.get("name"),
        "generator_model": generator_state.get("model_name"),
        "test_id": test_id,
        "project_name": project_name,
        "theme": theme,
        "chapter": chapter,
        "chapter_title": chapter_title,
        "row_index": int(row_index),
        # RAGAS fields
        "question": question_for_ragas,
        "answer": result["answer"],
        "contexts": result["contexts_for_ragas"],
        "ground_truth": ground_truth,
        # Audit fields
        "original_question": original_question,
        "retrieval_query": result["retrieval_query"],
        "top_k_parents": top_k,
        "requested_top_k_parents": result.get("requested_top_k_parents", top_k),
        "used_top_k_parents": result.get("used_top_k_parents", top_k),
        "context_was_truncated": result.get("context_was_truncated", False),
        "prompt_token_count": result.get("prompt_token_count"),
        "prompt_token_budget": result.get("prompt_token_budget"),
        "effective_max_output_tokens": result.get("effective_max_output_tokens"),
        "retrieved_parent_ids": result["retrieved_parent_ids"],
        "parent_hits": result["parent_hits"],
        # Generator-facing context/prompt.
        # This context contains no metadata and no child chunks.
        "generator_context": result["context"],
        "prompt_metadata": result["prompt"],
    }

    # Preserve additional dataset-specific test-case fields if present.
    for extra_col in ["length_tokens", "length_words_raw", "block_id", "parent_id"]:
        if extra_col in row.index and not pd.isna(row[extra_col]):
            value = row[extra_col]
            ragas_row[extra_col] = value.item() if hasattr(value, "item") else value

    return ragas_row


def is_nonfatal_job_error(exc: Exception) -> bool:
    """
    Decide whether a job-level error should be saved and skipped while the
    worker continues with the next job.
    """

    message = str(exc).lower()

    nonfatal_markers = [
        "maximum context length",
        "input_tokens",
        "prompt contains",
        "too many tokens",
        "vllmvalidationerror",
        "retrieval query contains no valid",
        "bm25l found no parents",
    ]

    return any(marker in message for marker in nonfatal_markers)


def run_full_experiment_for_generator(
    generator_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    generator_name = str(generator_config["name"])
    model_name = str(generator_config["model"])

    output_path = RESULTS_DIR / (
        f"{DATASET_NAME}_{EXPERIMENT_BASE_NAME}_{make_safe_filename(generator_name)}.json"
    )

    db_path = get_queue_db_path(generator_name)
    init_result_db(db_path)

    df = load_test_cases(TEST_CASES_PATH)

    pending_jobs = []

    for row_index, row in df.iterrows():
        chapter_title = get_chapter_title_from_row(row)
        original_question = row_value(row, "Example Question")
        ground_truth = row_value(row, "Gold Text")

        test_id = row_value(
            row,
            "ID",
            "id",
            "block_id",
            "parent_id",
            default=str(row_index + 1),
        )

        for top_k in TOP_K_VARIANTS:
            job_id = make_job_id(
                generator_name=generator_name,
                row_index=int(row_index),
                top_k=int(top_k),
            )

            if is_job_completed(db_path, job_id):
                print(f"[SKIP_COMPLETED] {job_id}", flush=True)
                continue

            pending_jobs.append(
                {
                    "job_id": job_id,
                    "row_index": int(row_index),
                    "row": row,
                    "chapter_title": chapter_title,
                    "original_question": original_question,
                    "ground_truth": ground_truth,
                    "test_id": test_id,
                    "top_k": int(top_k),
                }
            )

    print("=" * 100)
    print(f"[GENERATOR] {generator_name}")
    print(f"[MODEL] {model_name}")
    print(f"[DB] {db_path}")
    print(f"[OUTPUT] {output_path}")
    print(f"[PENDING_JOBS] {len(pending_jobs)}")
    print("=" * 100)

    if not pending_jobs:
        print(f"[GENERATOR_ALREADY_COMPLETE] {generator_name}", flush=True)
        return export_completed_jobs(db_path, output_path)

    generator_state = setup_generator(generator_config)
    generator_state["generator_config"] = dict(generator_config)

    try:
        for job in pending_jobs:
            row = job["row"]
            row_index = job["row_index"]
            top_k = job["top_k"]
            job_id = job["job_id"]
            chapter_title = job["chapter_title"]
            original_question = job["original_question"]
            ground_truth = job["ground_truth"]
            test_id = job["test_id"]

            question_for_ragas = build_retrieval_query(
                chapter_title=chapter_title,
                example_question=original_question,
            )

            claimed = claim_job(
                db_path=db_path,
                job_id=job_id,
                dataset_name=DATASET_NAME,
                experiment_name=f"{EXPERIMENT_BASE_NAME}_top_{top_k}",
                generator_name=generator_name,
                row_index=row_index,
                test_id=test_id,
                top_k=top_k,
            )

            if not claimed:
                print(f"[SKIP_CLAIMED_OR_COMPLETED] {job_id}", flush=True)
                continue

            print(
                f"Running dataset={DATASET_NAME} | "
                f"test_id={test_id} | "
                f"top_k_parents={top_k} | "
                f"generator={generator_name} | "
                f"model={model_name}",
                flush=True,
            )

            try:
                result = answer_query(
                    chapter_title=chapter_title,
                    example_question=original_question,
                    top_k_parents=top_k,
                    generator_state=generator_state,
                )

                
                log_answer(
                    generator_name=generator_name,
                    test_id=test_id,
                    top_k=top_k,
                    answer=result["answer"],
                )
            except Exception as exc:
                save_failed_job(
                    db_path=db_path,
                    job_id=job_id,
                    error_message=f"{type(exc).__name__}: {exc}",
                )

                print(
                    f"[FAILED] {job_id} | {type(exc).__name__}: {exc}",
                    flush=True,
                )

                if is_nonfatal_job_error(exc):
                    continue

                raise

            ragas_row = build_ragas_row(
                row=row,
                row_index=row_index,
                chapter_title=chapter_title,
                original_question=original_question,
                question_for_ragas=question_for_ragas,
                ground_truth=ground_truth,
                top_k=top_k,
                result=result,
                generator_state=generator_state,
            )

            ragas_row["job_id"] = job_id
            ragas_row["status"] = "completed"
            ragas_row["error"] = None

            save_completed_job(
                db_path=db_path,
                job_id=job_id,
                result=ragas_row,
            )

            print(f"[SAVED] {job_id}", flush=True)

    finally:
        release_generator(generator_state)

    return export_completed_jobs(db_path, output_path)


def run_all_generators_experiment() -> Dict[str, List[Dict[str, Any]]]:
    results_by_generator = {}

    for generator_config in GENERATOR_CONFIGS:
        generator_name = generator_config.get("name", "unknown_generator")

        results_by_generator[generator_name] = run_full_experiment_for_generator(
            generator_config=generator_config,
        )

    return results_by_generator


# ---
# Example single prompt check
# ---

single_chapter_title = "Rollen- und Nutzermanagement, Rechteverwaltung"

single_example_question = """Ausgangssituation: Der AG beabsichtigt, eine Urbane Datenplattform as a Service zu vergeben.

Anforderungen: Ein zentrales User- und Rollenmanagement ist zwingend notwendig. Verfasse ein prägnantes, aber vollständiges Kapitel mit maximal 150 Wörtern."""

# Uncomment to test one prompt with the first configured generator:
# debug_result = run_sparse_parent_only_bm25l_test_with_full_output(
#     chapter_title=single_chapter_title,
#     example_question=single_example_question,
#     top_k_parents=20,
#     generator_name="Qwen/Qwen2.5-7B-Instruct",
# )


# ---
# Run all configured generators × all test cases × all top-k variants
# ---

def main() -> None:
    experiment_results = run_all_generators_experiment()


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
