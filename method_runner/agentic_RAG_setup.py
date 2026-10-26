"""
Fixed agentic RAG experiment runner.

Workflow
--------
1. Dense retrieval over the top 20 child embeddings using:
      chapter title + cleaned Example Question
2. Expansion of those children to unique parents.
3. Parent reranking by:
      a. exact normalized chapter-title match,
      b. semantic chapter-title similarity,
      c. original dense retrieval score.
4. Qwen 32B outline induction from the five best parents.
5. Qwen 7B conservative splitting of the cleaned Example Question into at
   most seven atomic requirements.
6. Dense top-3 child retrieval for each atomic requirement.
7. One Qwen 7B call that produces exactly one answer per requirement while
   seeing the full original Example Question and all requirement evidence.
8. Qwen 32B final generation from the outline, original Example Question,
   and the synthesized requirement answers.

The original Example Question is preserved unchanged for the two generation
steps. Dataset-configured query_cleaning patterns are applied only to the
retrieval-oriented steps.

Execution architecture
----------------------
The script follows the existing experiment architecture:
- YAML configuration
- FAISS child retrieval
- direct vLLM inference through llm_clients_vllm.py
- SQLite WAL job queue for resumability and parallel replicas
- atomic JSON export

Models are loaded in three sequential phases and released between phases:
- outline_generation
- requirement_processing (split + requirement-answer synthesis)
- final_generation
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import socket
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer

from llm_clients_vllm import (
    generate_answer,
    release_generator,
    setup_generator,
)


# ---
# Constants

# --- Shared runner utilities ---
try:
    from .runner_utils import ensure_dir, make_safe_filename, row_value, utc_now
except ImportError:
    from runner_utils import ensure_dir, make_safe_filename, row_value, utc_now

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Select the dataset config to use by default; the matching agentic workflow
# block is loaded from that dataset's multi_agent folder.
# DEFAULT_CONFIG_PATH = PROJECT_ROOT / "tender_dataset" / "tender_config.yaml"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "ledgar_dataset" / "ledgar_config.yaml"
# DEFAULT_CONFIG_PATH = PROJECT_ROOT / "hupd_dataset" / "hupd_config.yaml"

DEFAULT_EXPERIMENT_NAME = (
    "agentic_chapter_rerank_outline_requirement_answers"
)

STATUS_PENDING = "pending"
STATUS_OUTLINE_RUNNING = "outline_running"
STATUS_OUTLINE_COMPLETED = "outline_completed"
STATUS_AGENT_RUNNING = "agent_running"
STATUS_AGENT_COMPLETED = "agent_completed"
STATUS_FINAL_RUNNING = "final_running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


# ---
# CLI and time helpers
# ---


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed agentic RAG workflow."
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("CONFIG_PATH", str(DEFAULT_CONFIG_PATH)),
        help="Path to the dataset YAML config.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=2.0,
        help="Queue polling interval while replica workers finish a phase.",
    )
    parser.add_argument(
        "--stale-job-minutes",
        type=int,
        default=360,
        help=(
            "Recover running jobs older than this many minutes. "
            "Use a value longer than the slowest expected row."
        ),
    )
    return parser.parse_args()




def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value)


# ---
# General config and path helpers
# ---


def resolve_config_file(config_path: Path) -> Path:
    """Resolve a config path from the project root or the caller's directory."""
    path = Path(config_path).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidates = [PROJECT_ROOT / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def load_config(config_path: Path) -> Dict[str, Any]:
    config_path = resolve_config_file(config_path)
    print(f"[CONFIG] {config_path}", flush=True)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("YAML config did not load as a dictionary.")

    # Accept either a normal dataset config (and auto-load its companion
    # agentic block) or an explicit agentic_workflow_*_config_block.yaml.
    if "paths" not in config:
        block_config = config
        dataset_dir = config_path.parent.parent
        dataset_name = dataset_dir.name.removesuffix("_dataset")
        base_config_path = PROJECT_ROOT / f"{dataset_name}_dataset" / f"{dataset_name}_config.yaml"
        if not base_config_path.is_file():
            raise FileNotFoundError(
                "The supplied agentic workflow block has no matching dataset "
                f"config: {base_config_path}"
            )
        with base_config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        config["agentic_workflow"] = block_config.get("agentic_workflow")
        config_path = base_config_path
    elif "agentic_workflow" not in config:
        dataset_name = str(config.get("dataset", "")).strip()
        block_path = (
            PROJECT_ROOT
            / f"{dataset_name}_dataset"
            / "multi_agent"
            / f"agentic_workflow_{dataset_name}_config_block.yaml"
        )
        if not block_path.is_file():
            raise FileNotFoundError(
                "No agentic workflow config block found. Expected: "
                f"{block_path}"
            )
        with block_path.open("r", encoding="utf-8") as file:
            block_config = yaml.safe_load(file)
        config["agentic_workflow"] = block_config.get("agentic_workflow")

    if not isinstance(config.get("agentic_workflow"), dict):
        raise ValueError("The config must define an agentic_workflow mapping.")

    return config


def resolve_config_path(
    raw_value: str,
    config_path: Path,
    config: Dict[str, Any],
    *,
    must_exist: bool,
) -> Path:
    """Resolve configured relative paths from experiment_runner_final."""
    raw_path = Path(str(raw_value)).expanduser()
    result = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path
    result = result.resolve()

    if must_exist and not result.exists():
        raise FileNotFoundError(f"Configured path does not exist: {result}")

    return result



def require_standard_path(
    config: Dict[str, Any],
    config_path: Path,
    key: str,
    *,
    must_exist: bool = True,
) -> Path:
    try:
        raw_value = config["paths"][key]
    except KeyError as exc:
        raise KeyError(f"Missing config path: paths.{key}") from exc

    return resolve_config_path(
        raw_value,
        config_path,
        config,
        must_exist=must_exist,
    )


def require_agentic_prompt_path(
    config: Dict[str, Any],
    config_path: Path,
    key: str,
) -> Path:
    try:
        raw_value = config["agentic_workflow"]["prompts"][key]
    except KeyError as exc:
        raise KeyError(
            f"Missing agentic prompt path: agentic_workflow.prompts.{key}"
        ) from exc

    return resolve_config_path(
        raw_value,
        config_path,
        config,
        must_exist=True,
    )






# ---
# JSON and dataframe utilities
# ---


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]




def get_chapter_title(row: pd.Series) -> str:
    chapter = row_value(
        row,
        "Chapter",
        "chapter",
        "label_name",
        default="",
    ).strip()

    if not chapter:
        raise ValueError(
            "Missing chapter title. Expected one of: "
            "Chapter, chapter, label_name."
        )

    return chapter


def load_test_cases(test_cases_path: Path, dataset_name: str) -> pd.DataFrame:
    print(f"[TEST_CASES] {test_cases_path}", flush=True)

    suffix = test_cases_path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        dataframe = pd.read_excel(test_cases_path)
    elif suffix == ".csv":
        delimiter = ";" if dataset_name.strip().lower() == "tender" else ","
        dataframe = pd.read_csv(test_cases_path, sep=delimiter)
    else:
        raise ValueError(
            f"Unsupported test-case file type '{suffix}'. "
            "Use .csv, .xlsx, or .xls."
        )

    dataframe.columns = dataframe.columns.str.strip()

    missing = [
        column
        for column in ["Example Question", "Gold Text"]
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(f"Missing required test-case columns: {missing}")

    if not any(
        column in dataframe.columns
        for column in ["Chapter", "chapter", "label_name"]
    ):
        raise ValueError(
            "Missing chapter column. Expected one of: "
            "Chapter, chapter, label_name."
        )

    print(f"[TEST_CASE_COLUMNS] {list(dataframe.columns)}", flush=True)
    return dataframe


def extract_json_object(text: str) -> Dict[str, Any]:
    """Parse raw JSON or a JSON object wrapped in a Markdown fence."""
    raw = str(text or "").strip()
    raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"^```\s*", "", raw).strip()
    raw = re.sub(r"\s*```$", "", raw).strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError(
            "No JSON object found in model output:\n" + raw[:1500]
        )

    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Parsed model output is not a JSON object.")

    return parsed


# ---
# SQLite job queue
# ---


def init_result_db(db_path: Path) -> None:
    ensure_dir(db_path.parent)

    with sqlite3.connect(db_path, timeout=60) as connection:
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA busy_timeout=60000;")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agentic_jobs (
                job_id TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                experiment_name TEXT NOT NULL,
                final_generator_name TEXT NOT NULL,
                row_index INTEGER NOT NULL,
                test_id TEXT NOT NULL,
                status TEXT NOT NULL,
                worker_id TEXT,
                plan_json TEXT,
                result_json TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_agentic_jobs_status_row
            ON agentic_jobs(status, row_index);
            """
        )


def make_job_id(
    dataset_name: str,
    experiment_name: str,
    final_generator_name: str,
    row_index: int,
) -> str:
    return (
        f"{dataset_name}__{experiment_name}__{final_generator_name}__"
        f"row_{row_index:08d}"
    )


def initialize_jobs(
    db_path: Path,
    dataframe: pd.DataFrame,
    dataset_name: str,
    experiment_name: str,
    final_generator_name: str,
) -> None:
    now = utc_now()

    with sqlite3.connect(db_path, timeout=60) as connection:
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA busy_timeout=60000;")

        for row_index, row in dataframe.iterrows():
            test_id = row_value(
                row,
                "ID",
                "id",
                "block_id",
                "parent_id",
                default=str(row_index + 1),
            )
            job_id = make_job_id(
                dataset_name=dataset_name,
                experiment_name=experiment_name,
                final_generator_name=final_generator_name,
                row_index=int(row_index),
            )

            connection.execute(
                """
                INSERT OR IGNORE INTO agentic_jobs (
                    job_id,
                    dataset_name,
                    experiment_name,
                    final_generator_name,
                    row_index,
                    test_id,
                    status,
                    worker_id,
                    plan_json,
                    result_json,
                    error_message,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?, ?);
                """,
                (
                    job_id,
                    dataset_name,
                    experiment_name,
                    final_generator_name,
                    int(row_index),
                    str(test_id),
                    STATUS_PENDING,
                    now,
                    now,
                ),
            )


def recover_stale_running_jobs(
    db_path: Path,
    stale_job_minutes: int,
) -> None:
    """Recover only jobs old enough not to belong to an active replica."""
    cutoff = datetime.now(timezone.utc) - timedelta(
        minutes=int(stale_job_minutes)
    )

    recovery_map = {
        STATUS_OUTLINE_RUNNING: STATUS_PENDING,
        STATUS_AGENT_RUNNING: STATUS_OUTLINE_COMPLETED,
        STATUS_FINAL_RUNNING: STATUS_AGENT_COMPLETED,
    }

    with sqlite3.connect(db_path, timeout=60) as connection:
        rows = connection.execute(
            """
            SELECT job_id, status, updated_at
            FROM agentic_jobs
            WHERE status IN (?, ?, ?);
            """,
            (
                STATUS_OUTLINE_RUNNING,
                STATUS_AGENT_RUNNING,
                STATUS_FINAL_RUNNING,
            ),
        ).fetchall()

        for job_id, status, updated_at in rows:
            try:
                updated = parse_utc(str(updated_at))
            except ValueError:
                updated = datetime.min.replace(tzinfo=timezone.utc)

            if updated > cutoff:
                continue

            recovered_status = recovery_map[str(status)]
            connection.execute(
                """
                UPDATE agentic_jobs
                SET status = ?, worker_id = NULL,
                    error_message = NULL, updated_at = ?
                WHERE job_id = ?;
                """,
                (recovered_status, utc_now(), job_id),
            )
            print(
                f"[RECOVERED_STALE_JOB] {job_id} -> {recovered_status}",
                flush=True,
            )


def claim_next_job(
    db_path: Path,
    expected_status: str,
    running_status: str,
    worker_id: str,
) -> Optional[Dict[str, Any]]:
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
            """
            SELECT job_id, row_index, test_id, plan_json
            FROM agentic_jobs
            WHERE status = ?
            ORDER BY row_index ASC
            LIMIT 1;
            """,
            (expected_status,),
        ).fetchone()

        if row is None:
            connection.execute("COMMIT;")
            return None

        job_id, row_index, test_id, plan_json = row
        now = utc_now()

        updated = connection.execute(
            """
            UPDATE agentic_jobs
            SET status = ?, worker_id = ?, updated_at = ?
            WHERE job_id = ? AND status = ?;
            """,
            (
                running_status,
                worker_id,
                now,
                job_id,
                expected_status,
            ),
        ).rowcount

        if updated != 1:
            connection.execute("ROLLBACK;")
            return None

        connection.execute("COMMIT;")

        return {
            "job_id": str(job_id),
            "row_index": int(row_index),
            "test_id": str(test_id),
            "plan": json.loads(plan_json) if plan_json else {},
        }

    except Exception:
        connection.execute("ROLLBACK;")
        raise

    finally:
        connection.close()


def save_stage_plan(
    db_path: Path,
    job_id: str,
    completed_status: str,
    plan: Dict[str, Any],
) -> None:
    with sqlite3.connect(db_path, timeout=60) as connection:
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA busy_timeout=60000;")
        connection.execute(
            """
            UPDATE agentic_jobs
            SET status = ?, worker_id = NULL, plan_json = ?,
                error_message = NULL, updated_at = ?
            WHERE job_id = ?;
            """,
            (
                completed_status,
                json.dumps(plan, ensure_ascii=False),
                utc_now(),
                job_id,
            ),
        )


def save_completed_result(
    db_path: Path,
    job_id: str,
    plan: Dict[str, Any],
    result: Dict[str, Any],
) -> None:
    with sqlite3.connect(db_path, timeout=60) as connection:
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA busy_timeout=60000;")
        connection.execute(
            """
            UPDATE agentic_jobs
            SET status = ?, worker_id = NULL,
                plan_json = ?, result_json = ?,
                error_message = NULL, updated_at = ?
            WHERE job_id = ?;
            """,
            (
                STATUS_COMPLETED,
                json.dumps(plan, ensure_ascii=False),
                json.dumps(result, ensure_ascii=False),
                utc_now(),
                job_id,
            ),
        )


def save_failed_job(
    db_path: Path,
    job_id: str,
    error_message: str,
) -> None:
    with sqlite3.connect(db_path, timeout=60) as connection:
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA busy_timeout=60000;")
        connection.execute(
            """
            UPDATE agentic_jobs
            SET status = ?, worker_id = NULL,
                error_message = ?, updated_at = ?
            WHERE job_id = ?;
            """,
            (
                STATUS_FAILED,
                str(error_message),
                utc_now(),
                job_id,
            ),
        )


def get_status_counts(db_path: Path) -> Dict[str, int]:
    with sqlite3.connect(db_path, timeout=60) as connection:
        rows = connection.execute(
            """
            SELECT status, COUNT(*)
            FROM agentic_jobs
            GROUP BY status;
            """
        ).fetchall()

    return {str(status): int(count) for status, count in rows}


def assert_no_failed_jobs(db_path: Path) -> None:
    with sqlite3.connect(db_path, timeout=60) as connection:
        row = connection.execute(
            """
            SELECT job_id, error_message
            FROM agentic_jobs
            WHERE status = ?
            ORDER BY row_index ASC
            LIMIT 1;
            """,
            (STATUS_FAILED,),
        ).fetchone()

    if row is not None:
        raise RuntimeError(
            f"Agentic job failed: {row[0]}\n{row[1]}"
        )


def stage_is_complete(
    db_path: Path,
    incomplete_statuses: Sequence[str],
) -> bool:
    placeholders = ",".join("?" for _ in incomplete_statuses)
    query = (
        "SELECT COUNT(*) FROM agentic_jobs "
        f"WHERE status IN ({placeholders});"
    )

    with sqlite3.connect(db_path, timeout=60) as connection:
        count = connection.execute(
            query,
            tuple(incomplete_statuses),
        ).fetchone()[0]

    return int(count) == 0


def export_completed_jobs(
    db_path: Path,
    output_path: Path,
) -> List[Dict[str, Any]]:
    with sqlite3.connect(db_path, timeout=60) as connection:
        rows = connection.execute(
            """
            SELECT result_json
            FROM agentic_jobs
            WHERE status = ?
            ORDER BY row_index ASC;
            """,
            (STATUS_COMPLETED,),
        ).fetchall()

    results = [json.loads(row[0]) for row in rows]
    ensure_dir(output_path.parent)

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
# Agentic runtime
# ---


class AgenticRAGRuntime:
    def __init__(self, config_path: Path) -> None:
        self.config_path = resolve_config_file(config_path)
        self.config = load_config(self.config_path)

        self.dataset_name = str(
            self.config.get("dataset", "dataset")
        ).strip()

        agentic = self.config.get("agentic_workflow")
        if not isinstance(agentic, dict):
            raise ValueError(
                "Missing independent top-level 'agentic_workflow' config block."
            )
        if not bool(agentic.get("enabled", True)):
            raise ValueError("agentic_workflow.enabled is false.")

        self.agentic_config = agentic
        self.experiment_name = str(
            agentic.get("experiment_name", DEFAULT_EXPERIMENT_NAME)
        ).strip()

        self.parents_path = require_standard_path(
            self.config,
            self.config_path,
            "parents",
        )
        self.children_path = require_standard_path(
            self.config,
            self.config_path,
            "children",
        )
        self.child_embeddings_path = require_standard_path(
            self.config,
            self.config_path,
            "child_embeddings",
        )
        self.results_dir = require_standard_path(
            self.config,
            self.config_path,
            "results_dir",
            must_exist=False,
        )
        self.test_cases_path = require_standard_path(
            self.config,
            self.config_path,
            "test_cases",
        )
        ensure_dir(self.results_dir)

        self.outline_prompt = require_agentic_prompt_path(
            self.config,
            self.config_path,
            "outline_system_prompt",
        ).read_text(encoding="utf-8").strip()
        self.requirement_split_prompt = require_agentic_prompt_path(
            self.config,
            self.config_path,
            "requirement_split_system_prompt",
        ).read_text(encoding="utf-8").strip()
        self.requirement_answer_prompt = require_agentic_prompt_path(
            self.config,
            self.config_path,
            "requirement_answer_system_prompt",
        ).read_text(encoding="utf-8").strip()
        self.final_system_prompt = require_agentic_prompt_path(
            self.config,
            self.config_path,
            "final_system_prompt",
        ).read_text(encoding="utf-8").strip()

        retrieval = agentic.get("retrieval", {})
        self.initial_child_top_k = int(
            retrieval.get("initial_child_top_k", 20)
        )
        self.outline_parent_top_n = int(
            retrieval.get("outline_parent_top_n", 5)
        )
        self.requirement_child_top_k = int(
            retrieval.get("requirement_child_top_k", 3)
        )

        if self.initial_child_top_k <= 0:
            raise ValueError("initial_child_top_k must be positive.")
        if self.outline_parent_top_n <= 0:
            raise ValueError("outline_parent_top_n must be positive.")
        if self.requirement_child_top_k <= 0:
            raise ValueError("requirement_child_top_k must be positive.")

        models = agentic.get("models", {})
        self.outline_model_config = self._require_model_config(
            models,
            "outline_generation",
        )
        self.requirement_split_model_config = self._require_model_config(
            models,
            "requirement_splitting",
        )
        self.requirement_answer_model_config = self._require_model_config(
            models,
            "requirement_answer_synthesis",
        )
        self.final_model_config = self._require_model_config(
            models,
            "final_generation",
        )

        self._validate_shared_requirement_model()

        self.global_max_output_tokens = int(
            self.config.get("generation", {}).get(
                "max_output_tokens",
                5000,
            )
        )
        self.global_temperature = float(
            self.config.get("generation", {}).get("temperature", 0.0)
        )

        self.embedding_model_name = str(
            self.config.get("embedding_model", "BAAI/bge-m3")
        )
        self.embedding_device = str(
            agentic.get("embedding_device", "cpu")
        )

        self.dataframe = load_test_cases(self.test_cases_path, self.dataset_name)
        self.parents = load_jsonl(self.parents_path)
        self.children = [
            child
            for child in load_jsonl(self.children_path)
            if self.get_child_text(child)
        ]
        self.child_embeddings = np.load(
            self.child_embeddings_path
        ).astype("float32")

        if len(self.children) != self.child_embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(self.children)} children but "
                f"{self.child_embeddings.shape[0]} child embeddings."
            )

        self.parent_lookup: Dict[str, Dict[str, Any]] = {}
        for parent in self.parents:
            for key in ["parent_id", "block_id"]:
                value = parent.get(key)
                if str(value or "").strip():
                    self.parent_lookup[str(value).strip()] = parent

        if not self.parent_lookup:
            raise ValueError("No parent IDs found in the parent artifact.")

        linked_children = sum(
            1
            for child in self.children
            if self.get_child_parent_key(child) in self.parent_lookup
        )
        if linked_children == 0:
            raise ValueError(
                "None of the children could be matched to a parent."
            )

        print(
            f"[EMBEDDING_MODEL] {self.embedding_model_name} "
            f"device={self.embedding_device}",
            flush=True,
        )
        self.embed_model = SentenceTransformer(
            self.embedding_model_name,
            device=self.embedding_device,
        )
        self.faiss_index = self.build_faiss_index(self.child_embeddings)

        self.final_generator_name = str(
            self.final_model_config["name"]
        )
        safe_generator = make_safe_filename(self.final_generator_name)
        self.db_path = self.results_dir / (
            f"{self.dataset_name}_{self.experiment_name}_"
            f"{safe_generator}.sqlite"
        )
        self.output_path = self.results_dir / (
            f"{self.dataset_name}_{self.experiment_name}_"
            f"{safe_generator}.json"
        )

        print("=" * 100, flush=True)
        print(f"[DATASET] {self.dataset_name}", flush=True)
        print(f"[EXPERIMENT] {self.experiment_name}", flush=True)
        print(
            f"[FIXED_RETRIEVAL] initial_children={self.initial_child_top_k} "
            f"outline_parents={self.outline_parent_top_n} "
            f"children_per_requirement={self.requirement_child_top_k}",
            flush=True,
        )
        print(
            f"[MODELS] outline={self.outline_model_config['name']} | "
            f"requirements={self.requirement_split_model_config['name']} | "
            f"final={self.final_model_config['name']}",
            flush=True,
        )
        print(f"[DB] {self.db_path}", flush=True)
        print(f"[OUTPUT] {self.output_path}", flush=True)
        print("=" * 100, flush=True)

    @staticmethod
    def _require_model_config(
        models: Dict[str, Any],
        role: str,
    ) -> Dict[str, Any]:
        value = models.get(role)
        if not isinstance(value, dict):
            raise ValueError(
                f"Missing agentic_workflow.models.{role} config."
            )

        required = ["name", "provider", "model"]
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(
                f"agentic_workflow.models.{role} is missing: {missing}"
            )

        if str(value.get("provider")) != "vllm":
            raise ValueError(
                f"Agentic role '{role}' must use provider='vllm' "
                "with the attached llm_clients_vllm architecture."
            )

        return dict(value)

    def _validate_shared_requirement_model(self) -> None:
        split = self.requirement_split_model_config
        answer = self.requirement_answer_model_config

        runtime_keys = [
            "provider",
            "model",
            "tensor_parallel_size",
            "dtype",
            "quantization",
            "max_model_len",
        ]

        differences = [
            key
            for key in runtime_keys
            if split.get(key) != answer.get(key)
        ]

        if differences:
            raise ValueError(
                "Requirement splitting and requirement-answer synthesis are "
                "executed while one 7B model is loaded. Their runtime model "
                f"settings must match. Differences: {differences}"
            )

    # ---
    # Query cleaning and artifact helpers
    # ---

    @staticmethod
    def normalize_regex_from_yaml(pattern: str) -> str:
        pattern = str(pattern or "").strip()
        fragments = re.findall(
            r"r?(['\"])(.*?)(?<!\\)\1",
            pattern,
            flags=re.DOTALL,
        )
        if fragments:
            return "".join(fragment for _, fragment in fragments)
        return pattern

    def clean_query_for_retrieval(self, question: str) -> str:
        text = str(question or "")

        for pattern_config in self.config.get(
            "query_cleaning",
            {},
        ).get("patterns", []):
            pattern_name = pattern_config.get(
                "name",
                "unnamed_pattern",
            )
            raw_pattern = pattern_config.get("regex", "")
            pattern = self.normalize_regex_from_yaml(raw_pattern)

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
                    f"Invalid query_cleaning regex '{pattern_name}': "
                    f"{raw_pattern}"
                ) from exc

        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def build_retrieval_query(
        self,
        chapter_title: str,
        question: str,
    ) -> str:
        cleaned_question = self.clean_query_for_retrieval(question)
        query = f"{chapter_title}\n\n{cleaned_question}".strip()
        return re.sub(r"\n{3,}", "\n\n", query).strip()

    @staticmethod
    def get_child_text(child: Dict[str, Any]) -> str:
        for key in ["child_text", "text", "content"]:
            value = child.get(key)
            if str(value or "").strip():
                return str(value).strip()
        return ""

    @staticmethod
    def get_child_parent_key(
        child: Dict[str, Any],
    ) -> Optional[str]:
        for key in ["parent_id", "parent_block_id", "block_id"]:
            value = child.get(key)
            if str(value or "").strip():
                return str(value).strip()
        return None

    @staticmethod
    def get_parent_id(parent: Dict[str, Any]) -> str:
        for key in ["parent_id", "block_id"]:
            value = parent.get(key)
            if str(value or "").strip():
                return str(value).strip()
        return ""

    @staticmethod
    def get_parent_chapter(
        parent: Dict[str, Any],
        fallback_child: Optional[Dict[str, Any]],
    ) -> str:
        value = parent.get("chapter")
        if str(value or "").strip():
            return str(value).strip()

        if fallback_child is not None:
            value = fallback_child.get("chapter")
            if str(value or "").strip():
                return str(value).strip()

        return ""

    # ---
    # Dense retrieval
    # ---

    @staticmethod
    def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
        normalized = np.ascontiguousarray(
            embeddings.astype("float32")
        )
        index = faiss.IndexFlatIP(normalized.shape[1])
        index.add(normalized)
        return index

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.embed_model.encode(
            query,
            normalize_embeddings=True,
        ).astype("float32")
        return np.ascontiguousarray(embedding.reshape(1, -1))

    def search_children(
        self,
        query: str,
        top_k_children: int,
        *,
        query_is_already_cleaned: bool,
    ) -> List[Tuple[Dict[str, Any], float]]:
        retrieval_query = (
            str(query).strip()
            if query_is_already_cleaned
            else self.clean_query_for_retrieval(query)
        )

        query_embedding = self.embed_query(retrieval_query)
        scores, indices = self.faiss_index.search(
            query_embedding,
            min(int(top_k_children), len(self.children)),
        )

        hits: List[Tuple[Dict[str, Any], float]] = []
        for score, index in zip(scores[0], indices[0]):
            if int(index) == -1:
                continue
            hits.append((self.children[int(index)], float(score)))

        return hits

    def rank_unique_parents_from_children(
        self,
        child_hits: Sequence[Tuple[Dict[str, Any], float]],
    ) -> Tuple[List[str], Dict[str, float]]:
        ranked_parent_ids: List[str] = []
        parent_scores: Dict[str, float] = {}

        for child, score in child_hits:
            parent_id = self.get_child_parent_key(child)
            if parent_id is None or parent_id not in self.parent_lookup:
                continue

            if parent_id not in parent_scores:
                parent_scores[parent_id] = float(score)
                ranked_parent_ids.append(parent_id)
            else:
                parent_scores[parent_id] = max(
                    parent_scores[parent_id],
                    float(score),
                )

        return ranked_parent_ids, parent_scores

    # ---
    # Chapter-title parent reranking
    # ---

    @staticmethod
    def normalize_chapter_name(text: str) -> str:
        normalized = str(text or "").lower().strip()
        normalized = re.sub(
            r"^\s*(§\s*)?\d+(\.\d+)*[\).:\-\s]+",
            "",
            normalized,
        )
        normalized = re.sub(
            r"^\s*\([a-z0-9]+\)\s*",
            "",
            normalized,
        )
        normalized = re.sub(
            r"[^a-zA-Z0-9äöüÄÖÜß]+",
            " ",
            normalized,
        )
        return re.sub(r"\s+", " ", normalized).strip()

    def chapter_similarities(
        self,
        intended_chapter: str,
        candidate_chapters: Sequence[str],
    ) -> List[float]:
        if not candidate_chapters:
            return []

        embeddings = self.embed_model.encode(
            [intended_chapter, *candidate_chapters],
            normalize_embeddings=True,
        ).astype("float32")

        intended_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]
        scores = candidate_embeddings @ intended_embedding
        return [float(score) for score in scores]

    def select_outline_parents(
        self,
        chapter_title: str,
        child_hits: Sequence[Tuple[Dict[str, Any], float]],
    ) -> Tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        Dict[str, float],
    ]:
        ranked_parent_ids, parent_scores = (
            self.rank_unique_parents_from_children(child_hits)
        )

        first_child_for_parent: Dict[str, Dict[str, Any]] = {}
        for child, _score in child_hits:
            parent_id = self.get_child_parent_key(child)
            if parent_id and parent_id not in first_child_for_parent:
                first_child_for_parent[parent_id] = child

        intended_normalized = self.normalize_chapter_name(chapter_title)
        candidates: List[Dict[str, Any]] = []

        for original_rank, parent_id in enumerate(
            ranked_parent_ids,
            start=1,
        ):
            parent = self.parent_lookup[parent_id]
            candidate_chapter = self.get_parent_chapter(
                parent,
                first_child_for_parent.get(parent_id),
            )
            candidate_normalized = self.normalize_chapter_name(
                candidate_chapter
            )

            candidates.append(
                {
                    "parent_id": parent_id,
                    "parent": parent,
                    "candidate_chapter": candidate_chapter,
                    "candidate_chapter_normalized": candidate_normalized,
                    "exact_chapter_match": bool(
                        intended_normalized
                        and candidate_normalized
                        and intended_normalized == candidate_normalized
                    ),
                    "parent_retrieval_score": float(
                        parent_scores[parent_id]
                    ),
                    "original_parent_rank": int(original_rank),
                }
            )

        similarities = self.chapter_similarities(
            intended_chapter=chapter_title,
            candidate_chapters=[
                candidate["candidate_chapter"]
                for candidate in candidates
            ],
        )

        for candidate, similarity in zip(candidates, similarities):
            candidate["chapter_similarity"] = float(similarity)

        ranked_candidates = sorted(
            candidates,
            key=lambda candidate: (
                candidate["exact_chapter_match"],
                candidate["chapter_similarity"],
                candidate["parent_retrieval_score"],
                -candidate["original_parent_rank"],
            ),
            reverse=True,
        )

        selected = ranked_candidates[: self.outline_parent_top_n]
        selected_parents = [candidate["parent"] for candidate in selected]

        if not selected_parents:
            raise ValueError(
                "Initial retrieval produced no parent suitable for outline "
                "generation."
            )

        return selected_parents, ranked_candidates, parent_scores

    # ---
    # Prompt construction
    # ---

    def build_outline_prompt(
        self,
        chapter_title: str,
        selected_parents: Sequence[Dict[str, Any]],
    ) -> str:
        parent_blocks = []
        for index, parent in enumerate(selected_parents, start=1):
            parent_blocks.append(
                f"[PARENT-{index}]\n{str(parent.get('text', '')).strip()}"
            )

        return f"""{self.outline_prompt}

---

## Intended text-unit title

{chapter_title}

## Precedent parent text units

{chr(10).join(parent_blocks)}

---

Return only valid JSON.
"""

    def build_requirement_split_prompt(
        self,
        cleaned_question: str,
    ) -> str:
        return f"""{self.requirement_split_prompt}

---

## Cleaned Example Question

{cleaned_question}

---

Return only valid JSON.
"""

    @staticmethod
    def build_atomic_requirements_block(
        requirement_split: Dict[str, Any],
    ) -> str:
        return "\n".join(
            f"{item['id']}: {item['text']}"
            for item in requirement_split["content_requirements"]
        )

    @staticmethod
    def build_requirement_evidence_blocks(
        evidence_by_requirement: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        blocks: List[str] = []

        for requirement_id, hits in evidence_by_requirement.items():
            requirement_text = (
                hits[0]["requirement_text"] if hits else ""
            )
            lines = [
                f"[REQUIREMENT {requirement_id}]",
                f"Requirement text:\n{requirement_text}",
                "Retrieved child evidence:",
            ]

            for index, hit in enumerate(hits, start=1):
                lines.append(
                    f"[CHILD-{requirement_id}-{index}]\n"
                    f"{hit['child_text']}"
                )

            blocks.append("\n\n".join(lines))

        return "\n\n".join(blocks)

    def build_requirement_answer_prompt(
        self,
        chapter_title: str,
        original_question: str,
        requirement_split: Dict[str, Any],
        evidence_by_requirement: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        return f"""{self.requirement_answer_prompt}

---

## Intended text-unit title

{chapter_title}

## Full original Example Question

{original_question}

## All atomic requirements

{self.build_atomic_requirements_block(requirement_split)}

## Three retrieved children per atomic requirement

{self.build_requirement_evidence_blocks(evidence_by_requirement)}

---

## Task

Return exactly one synthesized answer for every atomic requirement.
The full original Example Question contains all applicable generation
instructions, including any maximum word limit. Treat those instructions as
binding across the combined requirement-level answers.

Return only valid JSON.
"""

    @staticmethod
    def build_requirement_answers_block(
        requirement_answers: Dict[str, Any],
    ) -> str:
        blocks = []
        for item in requirement_answers["requirement_answers"]:
            blocks.append(
                f"[REQUIREMENT {item['requirement_id']}]\n"
                f"Requirement:\n{item['requirement_text']}\n\n"
                f"Synthesized answer:\n{item['answer']}"
            )
        return "\n\n".join(blocks)

    def build_final_prompt(
        self,
        chapter_title: str,
        original_question: str,
        outline: Dict[str, Any],
        requirement_answers: Dict[str, Any],
    ) -> str:
        return f"""{self.final_system_prompt}

---

## Intended text-unit title

{chapter_title}

## Full original Example Question

{original_question}

## Reusable outline induced from precedent parent text units

{json.dumps(outline, ensure_ascii=False, indent=2)}

## One synthesized answer per atomic requirement

{self.build_requirement_answers_block(requirement_answers)}

---

## Final task

Write the final text unit. The full original Example Question contains all
applicable generation instructions, including any maximum word limit. Follow
those instructions exactly. Use the outline as structural guidance and
integrate the requirement-level answers into one coherent text.

Produce only the final answer.
"""

    # ---
    # Validation of agent outputs
    # ---

    def validate_outline(
        self,
        parsed: Dict[str, Any],
        raw_output: str,
    ) -> Dict[str, Any]:
        outline = parsed.get("outline")
        if not isinstance(outline, list) or not outline:
            raise ValueError(
                "Outline model must return a non-empty 'outline' list."
            )

        for section in outline:
            if not isinstance(section, dict):
                raise ValueError("Every outline section must be an object.")
            if not str(section.get("section_title", "")).strip():
                raise ValueError("Every outline section needs section_title.")

        result = dict(parsed)
        result["raw_outline_output"] = raw_output
        return result

    def validate_requirement_split(
        self,
        parsed: Dict[str, Any],
        raw_output: str,
    ) -> Dict[str, Any]:
        requirements = parsed.get("content_requirements")
        if not isinstance(requirements, list) or not requirements:
            raise ValueError(
                "Requirement splitter must return a non-empty "
                "'content_requirements' list."
            )

        maximum = int(
            self.requirement_split_model_config.get(
                "max_atomic_requirements",
                7,
            )
        )
        if len(requirements) > maximum:
            raise ValueError(
                f"Requirement splitter returned {len(requirements)} "
                f"requirements; maximum is {maximum}. No repair is applied."
            )

        cleaned: List[Dict[str, str]] = []
        for index, requirement in enumerate(requirements, start=1):
            if not isinstance(requirement, dict):
                raise ValueError("Each requirement must be a JSON object.")

            text = str(requirement.get("text", "")).strip()
            if not text:
                raise ValueError("Atomic requirement text must not be empty.")

            cleaned.append({"id": f"R{index}", "text": text})

        return {
            "content_requirements": cleaned,
            "raw_requirement_split_output": raw_output,
        }

    @staticmethod
    def validate_requirement_answers(
        parsed: Dict[str, Any],
        raw_output: str,
        requirement_split: Dict[str, Any],
    ) -> Dict[str, Any]:
        answers = parsed.get("requirement_answers")
        if not isinstance(answers, list):
            raise ValueError(
                "Requirement-answer model must return "
                "'requirement_answers' as a list."
            )

        expected = {
            item["id"]: item["text"]
            for item in requirement_split["content_requirements"]
        }
        received: Dict[str, Dict[str, Any]] = {}

        for item in answers:
            if not isinstance(item, dict):
                raise ValueError(
                    "Every requirement answer must be a JSON object."
                )

            requirement_id = str(
                item.get("requirement_id", "")
            ).strip()
            answer = str(item.get("answer", "")).strip()

            if requirement_id not in expected:
                raise ValueError(
                    f"Unexpected requirement answer ID: {requirement_id}"
                )
            if requirement_id in received:
                raise ValueError(
                    f"Duplicate requirement answer ID: {requirement_id}"
                )
            if not answer:
                raise ValueError(
                    f"Empty answer for requirement {requirement_id}."
                )

            received[requirement_id] = {
                "requirement_id": requirement_id,
                "requirement_text": expected[requirement_id],
                "answer": answer,
            }

        if set(received) != set(expected):
            missing = sorted(set(expected) - set(received))
            raise ValueError(
                "Requirement-answer model did not return exactly one answer "
                f"per requirement. Missing: {missing}"
            )

        ordered = [
            received[item["id"]]
            for item in requirement_split["content_requirements"]
        ]

        return {
            "requirement_answers": ordered,
            "raw_requirement_answer_output": raw_output,
        }

    # ---
    # Workflow stage implementations
    # ---

    def run_outline_for_row(
        self,
        row_index: int,
        outline_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        row = self.dataframe.iloc[row_index]
        chapter_title = get_chapter_title(row)
        original_question = row_value(row, "Example Question")
        ground_truth = row_value(row, "Gold Text")
        cleaned_question = self.clean_query_for_retrieval(
            original_question
        )
        retrieval_query = self.build_retrieval_query(
            chapter_title,
            original_question,
        )

        initial_child_hits = self.search_children(
            retrieval_query,
            self.initial_child_top_k,
            query_is_already_cleaned=True,
        )

        selected_parents, ranked_candidates, parent_scores = (
            self.select_outline_parents(
                chapter_title,
                initial_child_hits,
            )
        )

        outline_prompt = self.build_outline_prompt(
            chapter_title,
            selected_parents,
        )
        raw_outline = generate_answer(
            prompt=outline_prompt,
            generator_state=outline_state,
            max_output_tokens=int(
                self.outline_model_config.get(
                    "max_output_tokens",
                    1800,
                )
            ),
            temperature=float(
                self.outline_model_config.get("temperature", 0.0)
            ),
        )
        outline = self.validate_outline(
            extract_json_object(raw_outline),
            raw_outline,
        )

        initial_hits_json = [
            {
                "child_id": child.get("child_id", ""),
                "parent_id": self.get_child_parent_key(child) or "",
                "score": float(score),
                "theme": child.get("theme", ""),
                "chapter": child.get("chapter", ""),
                "structure_path": child.get("structure_path", ""),
                "child_text": self.get_child_text(child),
            }
            for child, score in initial_child_hits
        ]

        selected_parent_ids = [
            self.get_parent_id(parent)
            for parent in selected_parents
        ]

        return {
            "dataset": self.dataset_name,
            "experiment": self.experiment_name,
            "row_index": int(row_index),
            "test_id": row_value(
                row,
                "ID",
                "id",
                "block_id",
                "parent_id",
                default=str(row_index + 1),
            ),
            "project_name": row_value(
                row,
                "Project Name",
                "project_name",
                default="",
            ),
            "theme": row_value(row, "Theme", "theme", default=""),
            "chapter": chapter_title,
            "original_question": original_question,
            "cleaned_question": cleaned_question,
            "question": retrieval_query,
            "ground_truth": ground_truth,
            "retrieval_query": retrieval_query,
            "initial_top_k_children": self.initial_child_top_k,
            "outline_top_n_parents": self.outline_parent_top_n,
            "requirement_top_k_children": self.requirement_child_top_k,
            "initial_child_hits": initial_hits_json,
            "parent_scores": parent_scores,
            "ranked_parent_candidates": [
                {
                    key: value
                    for key, value in candidate.items()
                    if key != "parent"
                }
                for candidate in ranked_candidates
            ],
            "selected_outline_parent_ids": selected_parent_ids,
            "selected_outline_parents": selected_parents,
            "outline_prompt": outline_prompt,
            "outline": outline,
            "outline_model_name": outline_state["name"],
            "outline_model": outline_state["model_name"],
        }

    def retrieve_requirement_evidence(
        self,
        chapter_title: str,
        requirement_split: Dict[str, Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        evidence: Dict[str, List[Dict[str, Any]]] = {}

        for requirement in requirement_split["content_requirements"]:
            requirement_id = requirement["id"]
            requirement_text = requirement["text"]
            query = f"{chapter_title}\n\n{requirement_text}".strip()

            hits = self.search_children(
                query,
                self.requirement_child_top_k,
                query_is_already_cleaned=True,
            )

            evidence[requirement_id] = [
                {
                    "requirement_id": requirement_id,
                    "requirement_text": requirement_text,
                    "child_id": child.get("child_id", ""),
                    "parent_id": self.get_child_parent_key(child) or "",
                    "score": float(score),
                    "theme": child.get("theme", ""),
                    "chapter": child.get("chapter", ""),
                    "structure_path": child.get("structure_path", ""),
                    "child_text": self.get_child_text(child),
                }
                for child, score in hits
            ]

        return evidence

    def run_requirement_processing(
        self,
        plan: Dict[str, Any],
        requirement_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        split_prompt = self.build_requirement_split_prompt(
            plan["cleaned_question"]
        )
        raw_split = generate_answer(
            prompt=split_prompt,
            generator_state=requirement_state,
            max_output_tokens=int(
                self.requirement_split_model_config.get(
                    "max_output_tokens",
                    900,
                )
            ),
            temperature=float(
                self.requirement_split_model_config.get(
                    "temperature",
                    0.0,
                )
            ),
        )
        requirement_split = self.validate_requirement_split(
            extract_json_object(raw_split),
            raw_split,
        )

        evidence = self.retrieve_requirement_evidence(
            plan["chapter"],
            requirement_split,
        )

        answer_prompt = self.build_requirement_answer_prompt(
            chapter_title=plan["chapter"],
            original_question=plan["original_question"],
            requirement_split=requirement_split,
            evidence_by_requirement=evidence,
        )
        synthesis_limit = min(
            int(
                self.requirement_answer_model_config.get(
                    "max_output_tokens",
                    self.global_max_output_tokens,
                )
            ),
            self.global_max_output_tokens,
        )
        
        raw_answers = generate_answer(
            prompt=answer_prompt,
            generator_state=requirement_state,
            max_output_tokens=synthesis_limit,
            temperature=float(
                self.requirement_answer_model_config.get(
                    "temperature",
                    0.0,
                )
            ),
        )
        
        try:
            requirement_answers = self.validate_requirement_answers(
                extract_json_object(raw_answers),
                raw_answers,
                requirement_split,
            )
        
        except (json.JSONDecodeError, ValueError):
            retry_prompt = answer_prompt + """
        
        IMPORTANT FORMAT CORRECTION:
        Regenerate the complete response from scratch.
        The previous response could not be parsed as valid JSON.
        Return strictly valid JSON parseable by Python json.loads().
        Escape all double quotation marks inside string values as \\".
        Ensure all required commas, brackets, and braces are present.
        Return no Markdown and no text outside the JSON object.
        """
        
            raw_answers = generate_answer(
                prompt=retry_prompt,
                generator_state=requirement_state,
                max_output_tokens=synthesis_limit,
                temperature=float(
                    self.requirement_answer_model_config.get(
                        "temperature",
                        0.0,
                    )
                ),
            )
        
            requirement_answers = self.validate_requirement_answers(
                extract_json_object(raw_answers),
                raw_answers,
                requirement_split,
            )
    

        updated = dict(plan)
        updated.update(
            {
                "requirement_split_prompt": split_prompt,
                "requirement_split": requirement_split,
                "requirement_child_evidence": evidence,
                "requirement_answer_prompt": answer_prompt,
                "requirement_answer_synthesis": requirement_answers,
                "requirement_split_model_name": requirement_state["name"],
                "requirement_split_model": requirement_state["model_name"],
                "requirement_answer_model_name": (
                    self.requirement_answer_model_config["name"]
                ),
                "requirement_answer_model": (
                    self.requirement_answer_model_config["model"]
                ),
            }
        )
        return updated

    def build_contexts_for_ragas(
        self,
        plan: Dict[str, Any],
    ) -> List[str]:
        """
        Store the raw retrieved sources that influenced the pipeline.

        The selected parents influenced the generated outline, and the
        requirement children supported the intermediate requirement answers.
        Generated intermediate texts are kept separately for audit and are not
        treated as retrieved source contexts.
        """
        contexts: List[str] = []

        for parent in plan.get("selected_outline_parents", []):
            contexts.append(
                (
                    f"Source role: outline precedent\n"
                    f"Theme: {parent.get('theme', '')}\n"
                    f"Chapter: {parent.get('chapter', '')}\n"
                    f"Structure path: {parent.get('structure_path', '')}\n\n"
                    f"{parent.get('text', '')}"
                ).strip()
            )

        seen_children: set[str] = set()
        for hits in plan.get("requirement_child_evidence", {}).values():
            for hit in hits:
                child_key = str(
                    hit.get("child_id")
                    or f"{hit.get('parent_id')}::{hit.get('child_text')}"
                )
                if child_key in seen_children:
                    continue
                seen_children.add(child_key)
                contexts.append(
                    (
                        f"Source role: requirement evidence\n"
                        f"Requirement: {hit.get('requirement_text', '')}\n"
                        f"Theme: {hit.get('theme', '')}\n"
                        f"Chapter: {hit.get('chapter', '')}\n"
                        f"Structure path: {hit.get('structure_path', '')}\n\n"
                        f"{hit.get('child_text', '')}"
                    ).strip()
                )

        return contexts

    def run_final_generation(
        self,
        plan: Dict[str, Any],
        final_state: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        final_prompt = self.build_final_prompt(
            chapter_title=plan["chapter"],
            original_question=plan["original_question"],
            outline=plan["outline"],
            requirement_answers=plan["requirement_answer_synthesis"],
        )

        final_limit = min(
            int(
                self.final_model_config.get(
                    "max_output_tokens",
                    self.global_max_output_tokens,
                )
            ),
            self.global_max_output_tokens,
        )

        answer = generate_answer(
            prompt=final_prompt,
            generator_state=final_state,
            max_output_tokens=final_limit,
            temperature=float(
                self.final_model_config.get(
                    "temperature",
                    self.global_temperature,
                )
            ),
        )

        updated_plan = dict(plan)
        updated_plan["final_prompt"] = final_prompt

        result = {
            "dataset": self.dataset_name,
            "experiment": self.experiment_name,
            "generator_name": final_state["name"],
            "generator_model": final_state["model_name"],
            "test_id": plan["test_id"],
            "project_name": plan.get("project_name", ""),
            "theme": plan.get("theme", ""),
            "chapter": plan["chapter"],
            "row_index": int(plan["row_index"]),
            # RAGAS-compatible fields.
            "question": plan["question"],
            "answer": answer,
            "contexts": self.build_contexts_for_ragas(plan),
            "ground_truth": plan["ground_truth"],
            # Fixed workflow fields.
            "top_k_children": self.initial_child_top_k,
            "initial_top_k_children": self.initial_child_top_k,
            "outline_top_n_parents": self.outline_parent_top_n,
            "requirement_top_k_children": self.requirement_child_top_k,
            # Full audit trail.
            "original_question": plan["original_question"],
            "cleaned_question": plan["cleaned_question"],
            "retrieval_query": plan["retrieval_query"],
            "initial_child_hits": plan["initial_child_hits"],
            "parent_scores": plan["parent_scores"],
            "ranked_parent_candidates": plan[
                "ranked_parent_candidates"
            ],
            "selected_outline_parent_ids": plan[
                "selected_outline_parent_ids"
            ],
            "selected_outline_parents": plan[
                "selected_outline_parents"
            ],
            "outline": plan["outline"],
            "requirement_split": plan["requirement_split"],
            "requirement_child_evidence": plan[
                "requirement_child_evidence"
            ],
            "requirement_answer_synthesis": plan[
                "requirement_answer_synthesis"
            ],
            "outline_model_name": plan["outline_model_name"],
            "outline_model": plan["outline_model"],
            "requirement_split_model_name": plan[
                "requirement_split_model_name"
            ],
            "requirement_split_model": plan[
                "requirement_split_model"
            ],
            "requirement_answer_model_name": plan[
                "requirement_answer_model_name"
            ],
            "requirement_answer_model": plan[
                "requirement_answer_model"
            ],
            "final_max_output_tokens": final_limit,
            "outline_prompt": plan["outline_prompt"],
            "requirement_split_prompt": plan[
                "requirement_split_prompt"
            ],
            "requirement_answer_prompt": plan[
                "requirement_answer_prompt"
            ],
            "agentic_final_prompt": final_prompt,
            "job_id": make_job_id(
                dataset_name=self.dataset_name,
                experiment_name=self.experiment_name,
                final_generator_name=self.final_generator_name,
                row_index=int(plan["row_index"]),
            ),
            "status": STATUS_COMPLETED,
            "error": None,
        }

        row = self.dataframe.iloc[int(plan["row_index"])]
        for extra_column in [
            "length_tokens",
            "length_words_raw",
            "block_id",
            "parent_id",
        ]:
            if extra_column in row.index and not pd.isna(row[extra_column]):
                value = row[extra_column]
                result[extra_column] = (
                    value.item() if hasattr(value, "item") else value
                )

        return updated_plan, result


# ---
# Phase runner
# ---


def run_phase(
    *,
    runtime: AgenticRAGRuntime,
    db_path: Path,
    worker_id: str,
    expected_status: str,
    running_status: str,
    completed_status: str,
    model_config: Dict[str, Any],
    process_job: Any,
    incomplete_statuses: Sequence[str],
    poll_seconds: float,
) -> None:
    """Load one model once, process/claim rows, release, then cross barrier."""
    assert_no_failed_jobs(db_path)

    if stage_is_complete(db_path, incomplete_statuses):
        print(
            f"[PHASE_ALREADY_COMPLETE] {completed_status}",
            flush=True,
        )
        return

    generator_state = setup_generator(model_config)

    try:
        while True:
            assert_no_failed_jobs(db_path)

            job = claim_next_job(
                db_path=db_path,
                expected_status=expected_status,
                running_status=running_status,
                worker_id=worker_id,
            )

            if job is not None:
                print(
                    f"[CLAIM] phase={running_status} "
                    f"job={job['job_id']} worker={worker_id}",
                    flush=True,
                )

                try:
                    plan = process_job(job, generator_state)
                    save_stage_plan(
                        db_path=db_path,
                        job_id=job["job_id"],
                        completed_status=completed_status,
                        plan=plan,
                    )
                    print(
                        f"[SAVED_STAGE] {job['job_id']} -> "
                        f"{completed_status}",
                        flush=True,
                    )
                except Exception as exc:
                    save_failed_job(
                        db_path=db_path,
                        job_id=job["job_id"],
                        error_message=str(exc),
                    )
                    raise

                continue

            if stage_is_complete(db_path, incomplete_statuses):
                break

            counts = get_status_counts(db_path)
            print(
                f"[PHASE_WAIT] target={completed_status} "
                f"counts={counts}",
                flush=True,
            )
            time.sleep(float(poll_seconds))

    finally:
        release_generator(generator_state)


def run_final_phase(
    *,
    runtime: AgenticRAGRuntime,
    db_path: Path,
    worker_id: str,
    poll_seconds: float,
) -> None:
    assert_no_failed_jobs(db_path)

    if stage_is_complete(
        db_path,
        [STATUS_AGENT_COMPLETED, STATUS_FINAL_RUNNING],
    ):
        print("[PHASE_ALREADY_COMPLETE] final", flush=True)
        return

    final_state = setup_generator(runtime.final_model_config)

    try:
        while True:
            assert_no_failed_jobs(db_path)

            job = claim_next_job(
                db_path=db_path,
                expected_status=STATUS_AGENT_COMPLETED,
                running_status=STATUS_FINAL_RUNNING,
                worker_id=worker_id,
            )

            if job is not None:
                print(
                    f"[CLAIM] phase=final job={job['job_id']} "
                    f"worker={worker_id}",
                    flush=True,
                )

                try:
                    updated_plan, result = runtime.run_final_generation(
                        job["plan"],
                        final_state,
                    )
                    save_completed_result(
                        db_path=db_path,
                        job_id=job["job_id"],
                        plan=updated_plan,
                        result=result,
                    )
                    print(
                        f"[SAVED_FINAL] {job['job_id']}",
                        flush=True,
                    )
                except Exception as exc:
                    save_failed_job(
                        db_path=db_path,
                        job_id=job["job_id"],
                        error_message=str(exc),
                    )
                    raise

                continue

            if stage_is_complete(
                db_path,
                [STATUS_AGENT_COMPLETED, STATUS_FINAL_RUNNING],
            ):
                break

            counts = get_status_counts(db_path)
            print(f"[FINAL_WAIT] counts={counts}", flush=True)
            time.sleep(float(poll_seconds))

    finally:
        release_generator(final_state)


# ---
# Main
# ---


def main() -> None:
    args = parse_args()
    runtime = AgenticRAGRuntime(Path(args.config))

    init_result_db(runtime.db_path)
    initialize_jobs(
        db_path=runtime.db_path,
        dataframe=runtime.dataframe,
        dataset_name=runtime.dataset_name,
        experiment_name=runtime.experiment_name,
        final_generator_name=runtime.final_generator_name,
    )
    recover_stale_running_jobs(
        runtime.db_path,
        stale_job_minutes=args.stale_job_minutes,
    )

    worker_id = (
        f"host_{socket.gethostname()}__pid_{os.getpid()}__"
        f"gpus_{os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}"
    )

    # Phase 1: top-20 retrieval, parent reranking, Qwen 32B outline.
    run_phase(
        runtime=runtime,
        db_path=runtime.db_path,
        worker_id=worker_id,
        expected_status=STATUS_PENDING,
        running_status=STATUS_OUTLINE_RUNNING,
        completed_status=STATUS_OUTLINE_COMPLETED,
        model_config=runtime.outline_model_config,
        process_job=lambda job, state: runtime.run_outline_for_row(
            job["row_index"],
            state,
        ),
        incomplete_statuses=[
            STATUS_PENDING,
            STATUS_OUTLINE_RUNNING,
        ],
        poll_seconds=args.poll_seconds,
    )

    # Phase 2: Qwen 7B requirement split, top-3 retrieval per requirement,
    # and one global requirement-answer synthesis call per test case.
    run_phase(
        runtime=runtime,
        db_path=runtime.db_path,
        worker_id=worker_id,
        expected_status=STATUS_OUTLINE_COMPLETED,
        running_status=STATUS_AGENT_RUNNING,
        completed_status=STATUS_AGENT_COMPLETED,
        model_config=runtime.requirement_split_model_config,
        process_job=lambda job, state: runtime.run_requirement_processing(
            job["plan"],
            state,
        ),
        incomplete_statuses=[
            STATUS_OUTLINE_COMPLETED,
            STATUS_AGENT_RUNNING,
        ],
        poll_seconds=args.poll_seconds,
    )

    # Phase 3: Qwen 32B final chapter generation.
    run_final_phase(
        runtime=runtime,
        db_path=runtime.db_path,
        worker_id=worker_id,
        poll_seconds=args.poll_seconds,
    )

    results = export_completed_jobs(
        runtime.db_path,
        runtime.output_path,
    )
    counts = get_status_counts(runtime.db_path)

    print("=" * 100, flush=True)
    print(f"[ALL_DONE] completed={len(results)}", flush=True)
    print(f"[STATUS_COUNTS] {counts}", flush=True)
    print(f"[OUTPUT] {runtime.output_path}", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
