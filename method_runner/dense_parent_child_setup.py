# --- Dense parent-child retrieval ---
# Rank child chunks by dense similarity, expand them to parent records, and generate from parent text.

import os
import re
import gc
import json
import yaml
import getpass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import sqlite3
from datetime import datetime, timezone

import faiss
import numpy as np
import pandas as pd


from sentence_transformers import SentenceTransformer

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

# Optional: restrict the run during debugging.
# Example: RUN_ONLY_GENERATORS = ["gpt-5.4-mini"]
run_only_generator_env = os.environ.get("RUN_ONLY_GENERATOR")

RUN_ONLY_GENERATORS: Optional[List[str]] = (
    [run_only_generator_env]
    if run_only_generator_env
    else None
)

# This script implements the dense parent-child experiment.
# The experiment name follows the YAML method name.
EXPERIMENT_BASE_NAME = "dense_parent_child"





def get_queue_db_path(generator_name: str) -> Path:
    safe_generator_name = generator_name.replace("/", "_").replace(":", "_")
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
    return f"{DATASET_NAME}__{EXPERIMENT_BASE_NAME}__{generator_name}__row_{row_index:08d}__topk_{top_k}"


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
        connection.execute("ROLLBACK;")
        raise

    finally:
        connection.close()




def save_completed_job(db_path: Path, job_id: str, result: Dict[str, Any]) -> None:
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
                int(result["top_k_children"]),
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
                error_message,
                now,
                job_id,
            ),
        )

def export_completed_jobs(db_path: Path, output_path: Path) -> List[Dict[str, Any]]:
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

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("YAML config did not load as a dictionary.")

    return config


def require_config_path(config: Dict[str, Any], key: str) -> Path:
    try:
        value = config["paths"][key]
    except KeyError as exc:
        raise KeyError(f"Missing config path: paths.{key}") from exc

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path



config = load_config(CONFIG_PATH)

DATASET_NAME = str(config.get("dataset", "dataset")).strip()

PARENTS_PATH = require_config_path(config, "parents")
CHILDREN_PATH = require_config_path(config, "children")
CHILD_EMBEDDINGS_PATH = require_config_path(config, "child_embeddings")
RESULTS_DIR = require_config_path(config, "results_dir")
SYSTEM_PROMPT_PATH = require_config_path(config, "system_prompt")
TEST_CASES_PATH = require_config_path(config, "test_cases")

ensure_dir(RESULTS_DIR)

EMBEDDING_MODEL = config.get("embedding_model", "BAAI/bge-m3")
TOP_K_VARIANTS = list(config.get("retrieval", {}).get("top_k", [5, 10, 15, 20]))
GENERATOR_CONFIGS = build_generator_configs(config)


MAX_OUTPUT_TOKENS = int(config.get("generation", {}).get("max_output_tokens", 1500))
TEMPERATURE = float(config.get("generation", {}).get("temperature", 0.0))

if RUN_ONLY_GENERATORS is not None:
    allowed = set(RUN_ONLY_GENERATORS)
    GENERATOR_CONFIGS = [g for g in GENERATOR_CONFIGS if g.get("name") in allowed]

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
print("Embedding model:", EMBEDDING_MODEL)


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
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]




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

    # If the YAML value accidentally contains Python raw-string fragments,
    # join the inside of all quoted fragments.
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
                flags=re.IGNORECASE | re.UNICODE
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




def get_chapter_title(row: pd.Series) -> str:
    chapter = row_value(row, "Chapter", "chapter", "label_name", default="").strip()

    if not chapter:
        raise ValueError(
            "Missing chapter title. Expected one of the columns: Chapter, chapter, label_name."
        )

    return chapter


def build_retrieval_query(chapter_title: str, example_question: str) -> str:
    """
    Retrieval query follows the requested design:
    - Chapter title from the structured CSV column
    - plus cleaned Example Question
    - generation-only instruction sentences are removed only from the Example Question
    """

    cleaned_question = clean_query_for_retrieval(example_question)

    retrieval_query = f"""
{chapter_title}

{cleaned_question}
""".strip()

    retrieval_query = re.sub(r"\n{3,}", "\n\n", retrieval_query)

    return retrieval_query.strip()


def get_child_text(child: Dict[str, Any]) -> str:
    for key in ["child_text", "text", "content"]:
        value = child.get(key)
        if str(value or "").strip():
            return str(value).strip()
    return ""


def get_child_parent_key(child: Dict[str, Any]) -> Optional[str]:
    for key in ["parent_id", "parent_block_id", "block_id"]:
        value = child.get(key)
        if str(value or "").strip():
            return str(value).strip()
    return None


# ---
# Load artifacts
# ---

children = load_jsonl(CHILDREN_PATH)
parents = load_jsonl(PARENTS_PATH)

children = [
    c for c in children
    if get_child_text(c)
]

child_embeddings = np.load(CHILD_EMBEDDINGS_PATH).astype("float32")

parent_lookup = {}

for p in parents:
    for key in ["parent_id", "block_id"]:
        value = p.get(key)
        if str(value or "").strip():
            parent_lookup[str(value).strip()] = p

child_lookup = {
    str(c.get("child_id", idx)): c
    for idx, c in enumerate(children)
}

print("Children loaded:", len(children))
print("Parents loaded:", len(parents))
print("Parent lookup keys:", len(parent_lookup))
print("Child lookup size:", len(child_lookup))
print("Embedding matrix:", child_embeddings.shape)

if not children:
    raise ValueError("No child records with non-empty text found.")

if len(children) != child_embeddings.shape[0]:
    raise ValueError(
        f"Mismatch: {len(children)} children but {child_embeddings.shape[0]} embeddings."
    )

missing_parent_links = 0

for child in children:
    parent_key = get_child_parent_key(child)
    if parent_key not in parent_lookup:
        missing_parent_links += 1

print("Children without matching parent:", missing_parent_links)

if missing_parent_links == len(children):
    raise ValueError(
        "None of the children could be matched to parents. "
        "Check whether child['parent_id'] matches parent['parent_id'] or parent['block_id']."
    )


# ---
# Dense child retrieval
# ---

embed_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def build_faiss_index(embeddings: np.ndarray):
    embeddings = np.ascontiguousarray(embeddings.astype("float32"))

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


faiss_index = build_faiss_index(child_embeddings)

print("FAISS ntotal:", faiss_index.ntotal)


def embed_query(query: str) -> np.ndarray:
    q = embed_model.encode(
        query,
        normalize_embeddings=True
    ).astype("float32")

    return np.ascontiguousarray(q.reshape(1, -1))


def search_children(
    query: str,
    top_k_children: int
) -> List[Tuple[Dict[str, Any], float]]:

    retrieval_query = strip_max_words_instruction(query)

    q_emb = embed_query(retrieval_query)

    scores, indices = faiss_index.search(
        q_emb,
        min(top_k_children, len(children))
    )

    hits = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        hits.append(
            (
                children[int(idx)],
                float(score)
            )
        )

    return hits


def rank_unique_parents_from_children(
    child_hits: List[Tuple[Dict[str, Any], float]]
) -> Tuple[List[str], Dict[str, float]]:

    ranked_parent_ids = []
    parent_scores = {}

    for child, score in child_hits:
        parent_id = get_child_parent_key(child)

        if parent_id is None:
            continue

        if parent_id not in parent_lookup:
            continue

        if parent_id not in parent_scores:
            parent_scores[parent_id] = score
            ranked_parent_ids.append(parent_id)
        else:
            parent_scores[parent_id] = max(
                parent_scores[parent_id],
                score
            )

    return ranked_parent_ids, parent_scores


def retrieve_parents_from_top_children(
    query: str,
    top_k_children: int
) -> Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], float]], Dict[str, float]]:

    child_hits = search_children(
        query=query,
        top_k_children=top_k_children
    )

    ranked_parent_ids, parent_scores = rank_unique_parents_from_children(
        child_hits
    )

    parents_context = [
        parent_lookup[parent_id]
        for parent_id in ranked_parent_ids
        if parent_id in parent_lookup
    ]

    return parents_context, child_hits, parent_scores


# ---
# Context and prompt
# ---

def build_parent_only_context(
    parents_context: List[Dict[str, Any]]
) -> str:
    """
    Generator-facing context WITHOUT metadata.

    This intentionally does NOT include:
    - theme
    - chapter
    - structure_path
    - child chunks

    The generator receives only the expanded parent text.
    """

    context = ""

    for i, parent in enumerate(parents_context, start=1):
        context += f"[PARENT-{i}]\n\n"
        context += parent.get("text", "")
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
"""


def build_contexts_for_ragas(
    parents_context: List[Dict[str, Any]]
) -> List[str]:
    """
    Preserved from the original dense parent-child pipeline:
    RAGAS contexts include metadata for evaluation/audit purposes.
    This does not affect the generator prompt.
    """

    return [
        (
            f"Theme: {p.get('theme', '')}\n"
            f"Chapter: {p.get('chapter', '')}\n"
            f"Structure path: {p.get('structure_path', '')}\n\n"
            f"{p.get('text', '')}"
        ).strip()
        for p in parents_context
    ]


# ---
# Generator setup
# ---



# ---
# Full answer pipeline
# ---

def answer_query(
    chapter_title: str,
    example_question: str,
    top_k_children: int,
    generator_state: Dict[str, Any],
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    temperature: float = TEMPERATURE
) -> Dict[str, Any]:

    retrieval_query = build_retrieval_query(
        chapter_title=chapter_title,
        example_question=example_question
    )

    parents_context, child_hits, parent_scores = retrieve_parents_from_top_children(
        query=retrieval_query,
        top_k_children=top_k_children
    )

    context = build_parent_only_context(parents_context)

    prompt = build_prompt(
        chapter_title=chapter_title,
        example_question=example_question,
        context=context
    )

    answer = generate_answer(
        prompt=prompt,
        generator_state=generator_state,
        max_output_tokens=max_output_tokens,
        temperature=temperature
    )

    return {
        "chapter_title": chapter_title,
        "query": example_question,
        "retrieval_query": retrieval_query,
        "answer": answer,
        "top_k_children": top_k_children,
        "experiment": f"{EXPERIMENT_BASE_NAME}_top_{top_k_children}",
        "generator_name": generator_state["name"],
        "generator_model": generator_state["model_name"],
        "child_hits": [
            {
                "child_id": child.get("child_id", ""),
                "parent_id": get_child_parent_key(child) or "",
                "score": score,
                "theme": child.get("theme", ""),
                "chapter": child.get("chapter", ""),
                "structure_path": child.get("structure_path", ""),
                "child_text": get_child_text(child)
            }
            for child, score in child_hits
        ],
        "parent_scores": parent_scores,
        "retrieved_parent_ids": [
            p.get("parent_id", p.get("block_id", ""))
            for p in parents_context
        ],
        "parents_context": parents_context,

        # Generator-facing context.
        # This contains parent text only.
        # It contains no metadata and no child chunks.
        "context": context,
        "prompt": prompt,

        # Preserved original behavior for this pipeline:
        # RAGAS contexts include parent metadata.
        "contexts_for_ragas": build_contexts_for_ragas(parents_context)
    }


# ---
# Optional retrieval debug table
# ---

def run_dense_parent_child_test_with_full_output(
    chapter_title: str,
    example_question: str,
    top_k_children: int,
    generator_name: Optional[str] = None
) -> Dict[str, Any]:

    if generator_name is None:
        generator_config = GENERATOR_CONFIGS[0]
    else:
        matches = [g for g in GENERATOR_CONFIGS if g.get("name") == generator_name]
        if not matches:
            raise ValueError(f"Generator not found in config: {generator_name}")
        generator_config = matches[0]

    generator_state = setup_generator(generator_config)

    try:
        result = answer_query(
            chapter_title=chapter_title,
            example_question=example_question,
            top_k_children=top_k_children,
            generator_state=generator_state
        )
    finally:
        release_generator(generator_state)

    rows = []

    for rank, hit in enumerate(result["child_hits"], start=1):
        rows.append({
            "Rank": rank,
            "Child ID": hit["child_id"],
            "Parent ID": hit["parent_id"],
            "Score": hit["score"],
            "Theme": hit["theme"],
            "Chapter": hit["chapter"],
            "Path": hit["structure_path"],
            "Child Text Preview": str(hit["child_text"])[:300]
        })

    retrieval_table = pd.DataFrame(rows)

    print("\n" + "=" * 120)
    print("CHAPTER")
    print("=" * 120)
    print(result["chapter_title"])

    print("\n" + "=" * 120)
    print("ORIGINAL GENERATOR QUESTION")
    print("=" * 120)
    print(result["query"])

    print("\n" + "=" * 120)
    print("RETRIEVAL QUERY")
    print("=" * 120)
    print(result["retrieval_query"])

    print("\n" + "=" * 120)
    print("TOP CHILD DENSE RETRIEVAL TABLE")
    print("=" * 120)
    try:
        display(retrieval_table)
    except NameError:
        print(retrieval_table)

    print("\n" + "=" * 120)
    print("RETRIEVED PARENTS")
    print("=" * 120)
    print(result["retrieved_parent_ids"])

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
    DATASET_NAME = str(config.get("dataset", "dataset")).strip()
    dataset_name = str(DATASET_NAME).lower().strip()
    if not test_cases_path.exists():
        raise FileNotFoundError(f"Test cases file not found: {test_cases_path}")

    suffix = test_cases_path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(test_cases_path)
    elif suffix == ".csv":
        df = pd.read_csv(test_cases_path)#, sep=";")
    else:
        raise ValueError(
            f"Unsupported test case file type '{suffix}'. Use .csv, .xlsx, or .xls."
        )

    df.columns = df.columns.str.strip()

    print("Columns:", list(df.columns))

    required_columns = [
        "Example Question",
        "Gold Text"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required test case columns: {missing}"
        )

    # Ensure the standardized chapter field is present, allowing dataset-specific capitalization.
    if not any(col in df.columns for col in ["Chapter", "chapter", "label_name"]):
        raise ValueError(
            "Missing chapter column. Expected one of: Chapter, chapter, label_name."
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
    generator_state: Dict[str, Any]
) -> Dict[str, Any]:

    test_id = row_value(row, "ID", "id", "block_id", "parent_id", default=str(row_index + 1))
    project_name = row_value(row, "Project Name", "project_name", default="")
    theme = row_value(row, "Theme", "theme", default="")

    ragas_row = {
        "dataset": DATASET_NAME,
        "experiment": f"{EXPERIMENT_BASE_NAME}_top_{top_k}",
        "generator_name": generator_state["name"],
        "generator_model": generator_state["model_name"],
        "test_id": test_id,
        "project_name": project_name,
        "theme": theme,
        "chapter": chapter_title,
        "row_index": int(row_index),
        # RAGAS fields
        # This follows the new retrieval-query construction:
        # Chapter + cleaned Example Question.
        "question": question_for_ragas,
        "answer": result["answer"],
        "contexts": result["contexts_for_ragas"],
        "ground_truth": ground_truth,

        # Audit fields
        "original_question": original_question,
        "retrieval_query": result["retrieval_query"],
        "top_k_children": top_k,
        "retrieved_child_ids": [
            hit["child_id"]
            for hit in result["child_hits"]
        ],
        "retrieved_parent_ids": result["retrieved_parent_ids"],
        "child_hits": result["child_hits"],
        "parent_scores": result["parent_scores"],

        # Generator-facing context/prompt.
        # This context contains no metadata and no child chunks.
        "generator_context": result["context"],
        "prompt_metadata": result["prompt"]
    }

    # Preserve additional dataset-specific test-case fields if present.
    for extra_col in ["length_tokens", "length_words_raw", "block_id", "parent_id"]:
        if extra_col in row.index and not pd.isna(row[extra_col]):
            ragas_row[extra_col] = row[extra_col].item() if hasattr(row[extra_col], "item") else row[extra_col]

    return ragas_row


def run_full_experiment_for_generator(
    generator_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    generator_name = str(generator_config["name"])
    model_name = str(generator_config["model"])

    output_path = RESULTS_DIR / (
        f"{DATASET_NAME}_{EXPERIMENT_BASE_NAME}_{generator_name}.json"
    )

    db_path = get_queue_db_path(generator_name)
    init_result_db(db_path)

    df = load_test_cases(TEST_CASES_PATH)

    pending_jobs = []

    for row_index, row in df.iterrows():
        chapter_title = get_chapter_title(row)
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
                f"top_k_children={top_k} | "
                f"generator={generator_name} | "
                f"model={model_name}",
                flush=True,
            )

            try:
                result = answer_query(
                    chapter_title=chapter_title,
                    example_question=original_question,
                    top_k_children=top_k,
                    generator_state=generator_state,
                )
            except Exception as exc:
                save_failed_job(
                    db_path=db_path,
                    job_id=job_id,
                    error_message=str(exc),
                )
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

single_query = """Ausgangssituation: Der AG beabsichtigt, eine Urbane Datenplattform as a Service zu vergeben.

Anforderungen: Ein zentrales User- und Rollenmanagement ist zwingend notwendig. Verfasse ein prägnantes, aber vollständiges Kapitel mit maximal 150 Wörtern."""

# Uncomment to test one prompt with the first configured generator:
# debug_result = run_dense_parent_child_test_with_full_output(
#     chapter_title=single_chapter_title,
#     example_question=single_query,
#     top_k_children=20,
#     generator_name="gpt-5.4-mini"
# )


# ---
# Run all configured generators × all test cases × all top-k variants
# ---

# Uncomment when ready:
def main() -> None:
    experiment_results = run_all_generators_experiment()


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()






