"""Prepares LEDGAR and HUPD source datasets and baseline test cases.

Generates synthetic queries with the dedicated scripts in
dataset_preparation/synthetic_query_generation/.
"""

# --- Dataset preparation ---
#
# LEDGAR and HUPD retain separate split, sampling, cleaning, and validation rules.

import os
import re
import json
import math
import time
import shutil
from pathlib import Path
from functools import partial
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    from IPython.display import display
except ImportError:
    display = print



def find_project_root(start=None):
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if all((candidate / name).is_dir() for name in (
            "tender_dataset", "ledgar_dataset", "hupd_dataset"
        )):
            return candidate
    raise FileNotFoundError(
        "Could not locate experiment_runner_final above the script directory."
    )

PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)

DATA_ROOT = PROJECT_ROOT
WORK_ROOT = PROJECT_ROOT
CACHE_ROOT = PROJECT_ROOT / ".cache"
SYNTHETIC_QUERY_DIR = PROJECT_ROOT / "dataset_preparation" / "synthetic_query_generation"
INTERIM_PREPARATION_DIR = PROJECT_ROOT / "dataset_preparation" / "interim_dataset_preparation"
LEDGAR_INTERIM_DIR = INTERIM_PREPARATION_DIR
DATA_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
INTERIM_PREPARATION_DIR.mkdir(parents=True, exist_ok=True)

_BGE_TOKENIZER = None

def count_bge_tokens(text: str) -> int:
    """Count tokens as the RAG artifact pipeline does."""
    global _BGE_TOKENIZER
    if _BGE_TOKENIZER is None:
        from transformers import AutoTokenizer
        _BGE_TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    return len(_BGE_TOKENIZER.encode(str(text or ""), add_special_tokens=False))

def load_parent_jsonl_as_blocks(path: Path) -> list[dict]:
    """Read RAG parent JSONL and expose the legacy field names used by prep steps."""
    if not path.exists():
        raise FileNotFoundError(f"Parent JSONL not found: {path}")
    blocks = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record["block_id"] = record.pop("parent_id", record.get("block_id"))
            record["length_tokens"] = record.pop(
                "length_tokens_bge_m3", record.get("length_tokens")
            )
            blocks.append(record)
    return blocks

def save_parent_jsonl_from_blocks(path: Path, blocks: list[dict]) -> None:
    """Writes parent records in the exact field schema consumed by the RAG pipeline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for block in blocks:
            text = str(block.get("text", "")).strip()
            record = {
                "parent_id": str(block.get("block_id", block.get("parent_id", ""))).strip(),
                "theme": block.get("theme"),
                "chapter": block.get("chapter"),
                "text": text,
                "length_tokens_bge_m3": count_bge_tokens(text),
                "length_words_raw": block.get("length_words_raw"),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_blocks_file(path: Path) -> list[dict]:
    """Read a JSON/JSONL parent source, mapping the RAG schema for prep logic."""
    if path.suffix.lower() == ".jsonl":
        return load_parent_jsonl_as_blocks(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "blocks" in data:
        return data["blocks"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported blocks file structure: {path}")

def save_blocks_file(path: Path, blocks: list[dict]) -> None:
    """Save parent blocks as JSONL for RAG parents, retaining other JSON outputs."""
    if path.suffix.lower() == ".jsonl":
        save_parent_jsonl_from_blocks(path, blocks)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"blocks": blocks}, handle, ensure_ascii=False, indent=2)

# --- 2. Shared utilities ---
#
#  word counting, user-query construction, HUPD split allocation and the API request. Dataset-specific prompts and validators remain in their generation stages.

def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", str(text)))


def allocate_test_counts_by_stratum(
    df: pd.DataFrame,
    stratum_cols: list[str],
    test_frac: float
) -> pd.DataFrame:
    strata = (
        df
        .groupby(stratum_cols)
        .size()
        .reset_index(name="n")
    )

    target_total = int(round(len(df) * test_frac))

    strata["expected"] = strata["n"] * test_frac
    strata["sample_n"] = np.floor(strata["expected"]).astype(int)
    strata["max_sample_n"] = strata["n"].apply(lambda n: max(0, n - 1))

    strata["sample_n"] = strata[["sample_n", "max_sample_n"]].min(axis=1)

    current_total = int(strata["sample_n"].sum())
    remaining = target_total - current_total

    strata["remainder"] = strata["expected"] - np.floor(strata["expected"])

    if remaining > 0:
        candidates = strata[
            strata["sample_n"] < strata["max_sample_n"]
        ].copy()

        candidates = candidates.sort_values(
            ["remainder", "n"],
            ascending=[False, False]
        )

        for idx in candidates.index:
            if remaining <= 0:
                break

            strata.loc[idx, "sample_n"] += 1
            remaining -= 1

    elif remaining < 0:
        candidates = strata[
            strata["sample_n"] > 0
        ].copy()

        candidates = candidates.sort_values(
            ["remainder", "n"],
            ascending=[True, True]
        )

        for idx in candidates.index:
            if remaining >= 0:
                break

            strata.loc[idx, "sample_n"] -= 1
            remaining += 1

    return strata

def stratified_train_test_split(
    df: pd.DataFrame,
    stratum_cols: list[str],
    test_frac: float,
    random_state: int
):
    alloc = allocate_test_counts_by_stratum(
        df=df,
        stratum_cols=stratum_cols,
        test_frac=test_frac
    )

    test_indices = []

    for _, row in alloc.iterrows():
        mask = pd.Series(True, index=df.index)

        for col in stratum_cols:
            mask &= df[col].eq(row[col])

        stratum_df = df[mask]
        sample_n = int(row["sample_n"])

        if sample_n <= 0:
            continue

        sampled_idx = (
            stratum_df
            .sample(
                n=sample_n,
                random_state=random_state
            )
            .index
            .tolist()
        )

        test_indices.extend(sampled_idx)

    test_df = df.loc[test_indices].copy()
    train_df = df.drop(index=test_indices).copy()

    train_df = train_df.sort_values(
        ["theme", "chapter", "block_id"]
    ).reset_index(drop=True)

    test_df = test_df.sort_values(
        ["theme", "chapter", "block_id"]
    ).reset_index(drop=True)

    return train_df, test_df, alloc



# --- 3. LEDGAR ---

# --- Load and filter LEDGAR clauses ---
#
# Combine the original LEDGAR splits, tokenize with BGE-M3 and retain clauses within the 300–4000 token limits.

def prepare_ledgar():
    """Load and filter LEDGAR clauses."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Configuration

    OUT_DIR = LEDGAR_INTERIM_DIR
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    PARENTS_JSON = OUT_DIR / "ledgar_clause_parents_min300.json"
    INVENTORY_CSV = OUT_DIR / "ledgar_clause_inventory_min300.csv"

    MIN_TOKEN_LIMIT = 300

    # Keep this to avoid extremely long provisions.
    # Set to None if you want no upper bound.
    MAX_TOKEN_LIMIT = 4000

    TOKENIZER_NAME = "BAAI/bge-m3"
    INCLUDE_LABEL_IN_TEXT = True

    # Load LEDGAR

    try:
        ds = load_dataset("coastalcph/lex_glue", "ledgar")
    except Exception:
        ds = load_dataset("lex_glue", "ledgar")

    print(ds)
    print(ds["train"].features)

    # Tokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    def bge_m3_token_counts(texts, batch_size=256):
        counts = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing"):
            batch = texts[i:i + batch_size]
            encoded = tokenizer(
                batch,
                add_special_tokens=True,
                truncation=False,
                padding=False,
            )
            counts.extend([len(ids) for ids in encoded["input_ids"]])

        return counts

    def safe_label(label: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", str(label)).strip("_")

    # Convert splits to dataframe

    rows = []
    label_feature = ds["train"].features["label"]

    for split_name, split_ds in ds.items():
        df_split = pd.DataFrame(split_ds)

        if hasattr(label_feature, "names"):
            df_split["label_name"] = df_split["label"].apply(
                lambda x: label_feature.names[int(x)]
            )
        else:
            df_split["label_name"] = df_split["label"].astype(str)

        df_split["split"] = split_name
        df_split["row_id"] = range(len(df_split))

        rows.append(df_split)

    df = pd.concat(rows, ignore_index=True)

    df["text"] = df["text"].fillna("").astype(str)
    df["length_words_raw"] = df["text"].apply(word_count)
    df["length_chars_raw"] = df["text"].apply(len)

    print("Total LEDGAR rows:", len(df))
    display(df.head())

    # Build final text first, then count tokens

    def make_final_text(row):
        label = str(row["label_name"])
        provision_text = str(row["text"]).strip()

        if INCLUDE_LABEL_IN_TEXT:
            return f"{label}\n\n{provision_text}"
        return provision_text

    df["final_text"] = df.apply(make_final_text, axis=1)
    df["length_tokens"] = bge_m3_token_counts(df["final_text"].tolist())

    # Filter: keep only long clauses

    if MAX_TOKEN_LIMIT is None:
        filtered_df = df[df["length_tokens"] >= MIN_TOKEN_LIMIT].copy()
        filter_description = f">= {MIN_TOKEN_LIMIT}"
    else:
        filtered_df = df[
            (df["length_tokens"] >= MIN_TOKEN_LIMIT)
            & (df["length_tokens"] <= MAX_TOKEN_LIMIT)
        ].copy()
        filter_description = f">= {MIN_TOKEN_LIMIT} and <= {MAX_TOKEN_LIMIT}"

    removed_short_df = df[df["length_tokens"] < MIN_TOKEN_LIMIT].copy()

    if MAX_TOKEN_LIMIT is not None:
        removed_long_df = df[df["length_tokens"] > MAX_TOKEN_LIMIT].copy()
    else:
        removed_long_df = pd.DataFrame()

    print("\n=== Filtering summary ===")
    print(f"Token filter: {filter_description}")
    print("Rows before:", len(df))
    print("Rows kept:", len(filtered_df))
    print("Rows removed because under 300:", len(removed_short_df))

    if MAX_TOKEN_LIMIT is not None:
        print(f"Rows removed because over {MAX_TOKEN_LIMIT}:", len(removed_long_df))

    print("Share kept:", round(len(filtered_df) / len(df), 4))

    # Build parent JSON blocks

    blocks = []

    for _, row in filtered_df.iterrows():
        split = row["split"]
        row_id = int(row["row_id"])
        label = str(row["label_name"])

        tender_id = f"ledgar_{split}_{row_id:06d}"
        label_safe = safe_label(label)
        block_id = f"a-{tender_id}-{label_safe}"

        blocks.append({
            "tender_id": tender_id,
            "theme": label,
            "chapter": label,
            "block_id": block_id,
            "text": row["final_text"],
            "length_tokens": int(row["length_tokens"]),
        })

    blocks_df = pd.DataFrame(blocks)

    print("\n=== Final parent blocks ===")
    print("Blocks saved:", len(blocks_df))

    if len(blocks_df) == 0:
        raise ValueError("No blocks left after filtering. Lower MIN_TOKEN_LIMIT or remove MAX_TOKEN_LIMIT.")

    # Overview statistics

    overall_stats = blocks_df[["length_tokens"]].describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    ).T

    label_stats = (
        blocks_df.groupby("chapter")
        .agg(
            n_blocks=("block_id", "count"),
            mean_tokens=("length_tokens", "mean"),
            median_tokens=("length_tokens", "median"),
            p90_tokens=("length_tokens", lambda x: x.quantile(0.90)),
            p95_tokens=("length_tokens", lambda x: x.quantile(0.95)),
            max_tokens=("length_tokens", "max"),
        )
        .reset_index()
        .sort_values("n_blocks", ascending=False)
    )

    split_stats = (
        blocks_df.assign(split=blocks_df["tender_id"].str.extract(r"ledgar_([^_]+)_")[0])
        .groupby("split")
        .agg(
            n_blocks=("block_id", "count"),
            mean_tokens=("length_tokens", "mean"),
            median_tokens=("length_tokens", "median"),
            max_tokens=("length_tokens", "max"),
        )
        .reset_index()
    )

    print("\n=== Overall token statistics after filtering ===")
    display(overall_stats)

    print("\n=== Blocks by split ===")
    display(split_stats)

    print("\n=== Most frequent long-clause labels ===")
    display(label_stats.head(30))

    # Token buckets

    bins = [300, 500, 750, 1000, 1500, 2000, 3000, 4000, float("inf")]
    labels = [
        "300-500",
        "501-750",
        "751-1000",
        "1001-1500",
        "1501-2000",
        "2001-3000",
        "3001-4000",
        ">4000",
    ]

    blocks_df["token_bucket"] = pd.cut(
        blocks_df["length_tokens"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    bucket_stats = (
        blocks_df["token_bucket"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    bucket_stats.columns = ["token_bucket", "n_blocks"]
    bucket_stats["share"] = bucket_stats["n_blocks"] / len(blocks_df)

    print("\n=== Token buckets for kept long clauses ===")
    display(bucket_stats)

    # Save files

    with PARENTS_JSON.open("w", encoding="utf-8") as f:
        json.dump(
            {"blocks": blocks},
            f,
            ensure_ascii=False,
            indent=2,
        )

    blocks_df.to_csv(INVENTORY_CSV, index=False)

    print("\nSaved files:")
    print("-", PARENTS_JSON)
    print("-", INVENTORY_CSV)

    # Preview

    preview = blocks_df[
        ["tender_id", "theme", "chapter", "block_id", "length_tokens", "text"]
    ].copy()

    preview["text"] = preview["text"].str[:500]

    print("\n=== Preview ===")
    display(preview.head(10))

prepare_ledgar()

# --- Assign LEDGAR themes ---
#
# Keep clause labels as chapters and apply the original broad-theme rules.

def map_ledgar_themes():
    """Assign LEDGAR themes."""
    IN_JSON = LEDGAR_INTERIM_DIR / "ledgar_clause_parents_min300.json"
    OUT_JSON = LEDGAR_INTERIM_DIR / "ledgar_clause_parents_min300_themes.json"
    OUT_XLSX = LEDGAR_INTERIM_DIR / "ledgar_theme_overview_min300.xlsx"
    OUT_CSV = LEDGAR_INTERIM_DIR / "ledgar_clause_inventory_min300_themes.csv"

    if not IN_JSON.exists():
        raise FileNotFoundError(f"Input file not found: {IN_JSON}")

    # Load blocks

    with IN_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    blocks = data.get("blocks", [])

    if not blocks:
        raise ValueError("No blocks found in input JSON.")

    df = pd.DataFrame(blocks)

    required_cols = ["tender_id", "theme", "chapter", "block_id", "text", "length_tokens"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["length_tokens"] = pd.to_numeric(df["length_tokens"], errors="coerce")

    # In the previous file, theme == chapter == LEDGAR label.
    df["ledgar_label"] = df["chapter"].astype(str)

    # Infer broad contract themes from LEDGAR label names
    THEME_RULES = {
        "financing_credit_and_payment_terms": [
            "erisa",
            "payments",
            "payment",
            "interests",
            "interest",
            "loans",
            "borrowings",
            "credit",
            "debt",
            "collateral",
            "security",
            "liens",
            "insurances",
            "taxes",
            "financial statements",
            "participations",
            "expenses",
        ],

        "corporate_transaction_and_securities_terms": [
            "capitalization",
            "closings",
            "closing",
            "authorizations",
            "no conflicts",
            "securities",
            "shares",
            "stock",
            "registration",
            "subsidiaries",
            "purchase",
            "exchange",
            "merger",
            "disclosures",
            "further assurances",
        ],

        "employment_compensation_and_equity_terms": [
            "base salary",
            "salary",
            "bonus",
            "compensation",
            "employment",
            "employee",
            "benefits",
            "vesting",
            "equity",
            "options",
            "awards",
            "termination of employment",
            "severance",
        ],

        "dispute_resolution_and_jurisdiction_terms": [
            "arbitration",
            "litigations",
            "litigation",
            "governing laws",
            "governing law",
            "submission to jurisdiction",
            "consent to jurisdiction",
            "waiver of jury trials",
            "jury",
            "remedies",
        ],

        "confidentiality_ip_and_general_boilerplate": [
            "confidentiality",
            "intellectual property",
            "publicity",
            "notices",
            "counterparts",
            "successors",
            "assignments",
            "amendments",
            "waivers",
            "waiver",
            "survival",
            "general",
            "releases",
            "indemnifications",
            "indemnification",
            "indemnity",
        ],
    }

    SELECTED_THEMES = [
        "financing_credit_and_payment_terms",
        "corporate_transaction_and_securities_terms",
        "employment_compensation_and_equity_terms",
        "dispute_resolution_and_jurisdiction_terms",
        "confidentiality_ip_and_general_boilerplate",
    ]

    def normalize_label(label: str) -> str:
        label = str(label).lower().strip()
        label = re.sub(r"[^a-z0-9]+", " ", label)
        label = re.sub(r"\s+", " ", label)
        return label.strip()

    def infer_broad_theme(label: str) -> str:
        norm = normalize_label(label)

        for theme, keywords in THEME_RULES.items():
            for kw in keywords:
                kw_norm = normalize_label(kw)

                # Exact or near-exact label match.
                if norm == kw_norm:
                    return theme

                # Controlled containment for multi-word terms.
                if kw_norm in norm or norm in kw_norm:
                    return theme

        return "other_or_unmapped"

    df["broad_theme"] = df["ledgar_label"].apply(infer_broad_theme)

    # Show mapping quality before filtering

    print("=== Original long-clause dataset ===")
    print("Blocks:", len(df))
    print("Total BGE-M3 tokens over all long clauses:", int(df["length_tokens"].sum()))
    print("Unique LEDGAR labels:", df["ledgar_label"].nunique())

    theme_mapping_overview = (
        df.groupby("broad_theme")
        .agg(
            n_blocks=("block_id", "count"),
            total_tokens=("length_tokens", "sum"),
            mean_tokens=("length_tokens", "mean"),
            median_tokens=("length_tokens", "median"),
            n_unique_ledgar_labels=("ledgar_label", "nunique"),
        )
        .reset_index()
        .sort_values("n_blocks", ascending=False)
    )

    print("\n=== Broad theme mapping overview before filtering ===")
    display(theme_mapping_overview)

    label_mapping_overview = (
        df.groupby(["broad_theme", "ledgar_label"])
        .agg(
            n_blocks=("block_id", "count"),
            total_tokens=("length_tokens", "sum"),
            mean_tokens=("length_tokens", "mean"),
            median_tokens=("length_tokens", "median"),
            max_tokens=("length_tokens", "max"),
        )
        .reset_index()
        .sort_values(["broad_theme", "n_blocks"], ascending=[True, False])
    )

    print("\n=== Label-to-theme mapping preview ===")
    display(label_mapping_overview.head(80))

    unmapped = (
        df[df["broad_theme"] == "other_or_unmapped"]
        .groupby("ledgar_label")
        .agg(
            n_blocks=("block_id", "count"),
            total_tokens=("length_tokens", "sum"),
            mean_tokens=("length_tokens", "mean"),
        )
        .reset_index()
        .sort_values("n_blocks", ascending=False)
    )

    print("\n=== Most frequent unmapped labels ===")
    display(unmapped.head(40))

    # Filter to selected 3-5 broad themes

    filtered_df = df[df["broad_theme"].isin(SELECTED_THEMES)].copy()

    print("\n=== Filtering to selected broad themes ===")
    print("Selected broad themes:", len(SELECTED_THEMES))
    print("Blocks before:", len(df))
    print("Blocks kept:", len(filtered_df))
    print("Blocks removed:", len(df) - len(filtered_df))
    print("Share kept:", round(len(filtered_df) / len(df), 4))
    print("Total BGE-M3 tokens in kept clauses:", int(filtered_df["length_tokens"].sum()))

    # Replace theme with broad theme, keep chapter as original LEDGAR label

    filtered_df["theme"] = filtered_df["broad_theme"]
    filtered_df["chapter"] = filtered_df["ledgar_label"]

    # Keep only schema-compatible fields.
    final_cols = [
        "tender_id",
        "theme",
        "chapter",
        "block_id",
        "text",
        "length_tokens",
    ]

    final_df = filtered_df[final_cols].copy()

    # Statistics

    theme_stats = (
        final_df.groupby("theme")
        .agg(
            n_blocks=("block_id", "count"),
            total_tokens=("length_tokens", "sum"),
            mean_tokens=("length_tokens", "mean"),
            median_tokens=("length_tokens", "median"),
            p90_tokens=("length_tokens", lambda x: x.quantile(0.90)),
            p95_tokens=("length_tokens", lambda x: x.quantile(0.95)),
            max_tokens=("length_tokens", "max"),
            n_chapter_labels=("chapter", "nunique"),
        )
        .reset_index()
        .sort_values("n_blocks", ascending=False)
    )

    chapter_stats = (
        final_df.groupby(["theme", "chapter"])
        .agg(
            n_blocks=("block_id", "count"),
            total_tokens=("length_tokens", "sum"),
            mean_tokens=("length_tokens", "mean"),
            median_tokens=("length_tokens", "median"),
            max_tokens=("length_tokens", "max"),
        )
        .reset_index()
        .sort_values(["theme", "n_blocks"], ascending=[True, False])
    )

    overall_summary = pd.DataFrame({
        "metric": [
            "blocks_before_theme_filter",
            "blocks_after_theme_filter",
            "unique_original_ledgar_labels_before",
            "unique_chapter_labels_after",
            "n_broad_themes",
            "total_tokens_before_theme_filter",
            "total_tokens_after_theme_filter",
            "mean_tokens_after_theme_filter",
            "median_tokens_after_theme_filter",
            "max_tokens_after_theme_filter",
        ],
        "value": [
            len(df),
            len(final_df),
            df["ledgar_label"].nunique(),
            final_df["chapter"].nunique(),
            final_df["theme"].nunique(),
            int(df["length_tokens"].sum()),
            int(final_df["length_tokens"].sum()),
            float(final_df["length_tokens"].mean()),
            float(final_df["length_tokens"].median()),
            int(final_df["length_tokens"].max()),
        ]
    })

    print("\n=== Final broad-theme statistics ===")
    display(theme_stats)

    print("\n=== Overall summary ===")
    display(overall_summary)

    print("\n=== Chapter labels within broad themes ===")
    display(chapter_stats.head(80))

    # Save updated JSON and overview files

    final_blocks = final_df.to_dict(orient="records")

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(
            {"blocks": final_blocks},
            f,
            ensure_ascii=False,
            indent=2,
        )

    final_df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        overall_summary.to_excel(writer, sheet_name="overall_summary", index=False)
        theme_stats.to_excel(writer, sheet_name="theme_stats", index=False)
        chapter_stats.to_excel(writer, sheet_name="chapter_stats", index=False)
        label_mapping_overview.to_excel(writer, sheet_name="label_mapping_all", index=False)
        unmapped.to_excel(writer, sheet_name="unmapped_labels", index=False)
        final_df.to_excel(writer, sheet_name="final_blocks", index=False)

    print("\nSaved files:")
    print("-", OUT_JSON)
    print("-", OUT_CSV)
    print("-", OUT_XLSX)

    # Preview final parent JSON blocks

    preview = final_df[
        ["tender_id", "theme", "chapter", "block_id", "length_tokens", "text"]
    ].copy()

    preview["text"] = preview["text"].str[:400]

    print("\n=== Final preview ===")
    display(preview.head(15))

map_ledgar_themes()

# --- Split LEDGAR by theme and chapter ---
#
# Use the original LEDGAR stratified 10% test selection with seed 42.

def split_ledgar():
    """Split LEDGAR by theme and chapter."""
    IN_JSON = LEDGAR_INTERIM_DIR / "ledgar_clause_parents_min300_themes.json"

    OUT_DIR = IN_JSON.parent

    TRAIN_JSON = OUT_DIR / "ledgar_clause_parents_min300_themes_train90_no_tender.json"
    TEST_JSON = OUT_DIR / "ledgar_clause_test_gold_min300_themes_10pct_no_tender.json"

    SPLIT_INVENTORY_CSV = OUT_DIR / "ledgar_clause_split_inventory_no_tender.csv"
    SPLIT_OVERVIEW_XLSX = OUT_DIR / "ledgar_clause_split_overview_no_tender.xlsx"

    TEST_FRAC = 0.10
    RANDOM_STATE = 42

    STRATIFY_COLS = ["theme", "chapter"]

    DROP_TENDER_ID = True

    # Load existing themed JSON

    if not IN_JSON.exists():
        raise FileNotFoundError(f"Input JSON not found: {IN_JSON}")

    with IN_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    blocks = data.get("blocks", [])

    if not blocks:
        raise ValueError("Input JSON contains no blocks.")

    df = pd.DataFrame(blocks)

    required_cols = ["theme", "chapter", "block_id", "text", "length_tokens"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["length_tokens"] = pd.to_numeric(df["length_tokens"], errors="coerce")

    if df["length_tokens"].isna().any():
        raise ValueError("Some length_tokens values could not be converted to numbers.")

    if df["block_id"].duplicated().any():
        duplicated = df[df["block_id"].duplicated(keep=False)]["block_id"].head(20).tolist()
        raise ValueError(f"Duplicate block_id values found, e.g.: {duplicated}")

    print("=== Loaded dataset ===")
    print("Input file:", IN_JSON)
    print("Blocks:", len(df))
    print("Unique block_id:", df["block_id"].nunique())
    print("Unique themes:", df["theme"].nunique())
    print("Unique chapters:", df["chapter"].nunique())
    print("Total BGE-M3 tokens:", int(df["length_tokens"].sum()))

    if "tender_id" in df.columns:
        print("Unique tender_id:", df["tender_id"].nunique())
        print("Note: tender_id is artificial and will be removed from output JSONs.")

    # Theme and chapter overview before split

    theme_stats_full = (
        df.groupby("theme")
        .agg(
            n_blocks=("block_id", "count"),
            total_tokens=("length_tokens", "sum"),
            mean_tokens=("length_tokens", "mean"),
            median_tokens=("length_tokens", "median"),
            n_chapters=("chapter", "nunique"),
        )
        .reset_index()
        .sort_values("n_blocks", ascending=False)
    )

    chapter_stats_full = (
        df.groupby(["theme", "chapter"])
        .agg(
            n_blocks=("block_id", "count"),
            total_tokens=("length_tokens", "sum"),
            mean_tokens=("length_tokens", "mean"),
            median_tokens=("length_tokens", "median"),
            max_tokens=("length_tokens", "max"),
        )
        .reset_index()
        .sort_values(["theme", "n_blocks"], ascending=[True, False])
    )

    print("\n=== Theme overview before split ===")
    display(theme_stats_full)

    print("\n=== Chapter overview before split ===")
    display(chapter_stats_full.head(80))

    # Exact proportional stratified sampling

    def compute_stratified_sample_sizes(data, strat_cols, frac):
        counts = (
            data.groupby(strat_cols)
            .size()
            .reset_index(name="n")
        )

        counts["expected"] = counts["n"] * frac
        counts["sample_n"] = np.floor(counts["expected"]).astype(int)
        counts["remainder"] = counts["expected"] - counts["sample_n"]

        target_total = int(round(len(data) * frac))
        current_total = int(counts["sample_n"].sum())
        missing = target_total - current_total

        if missing > 0:
            eligible = counts[counts["sample_n"] < counts["n"]].copy()
            eligible = eligible.sort_values(
                ["remainder", "n"],
                ascending=[False, False],
            )
            add_indices = eligible.index.tolist()[:missing]
            counts.loc[add_indices, "sample_n"] += 1

        elif missing < 0:
            removable = counts[counts["sample_n"] > 0].copy()
            removable = removable.sort_values(
                ["remainder", "n"],
                ascending=[True, True],
            )
            remove_indices = removable.index.tolist()[:abs(missing)]
            counts.loc[remove_indices, "sample_n"] -= 1

        counts["sample_n"] = counts["sample_n"].clip(lower=0, upper=counts["n"])

        return counts

    def stratified_sample_exact(data, strat_cols, frac, random_state=42):
        allocation_df = compute_stratified_sample_sizes(
            data=data,
            strat_cols=strat_cols,
            frac=frac,
        )

        rng = np.random.default_rng(random_state)

        sampled_parts = []

        for _, alloc in allocation_df.iterrows():
            sample_n = int(alloc["sample_n"])

            if sample_n <= 0:
                continue

            mask = pd.Series(True, index=data.index)

            for col in strat_cols:
                mask &= data[col] == alloc[col]

            group = data[mask]

            seed = int(rng.integers(0, 2**32 - 1))

            sampled = group.sample(
                n=min(sample_n, len(group)),
                random_state=seed,
                replace=False,
            )

            sampled_parts.append(sampled)

        if sampled_parts:
            test_df = pd.concat(sampled_parts, axis=0).copy()
        else:
            test_df = data.iloc[0:0].copy()

        train_df = data.drop(index=test_df.index).copy()

        test_df = test_df.reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)

        return train_df, test_df, allocation_df

    train_df, test_df, allocation_df = stratified_sample_exact(
        data=df,
        strat_cols=STRATIFY_COLS,
        frac=TEST_FRAC,
        random_state=RANDOM_STATE,
    )

    # Distribution checks

    def distribution_table(full, train, test, group_cols):
        full_counts = full.groupby(group_cols).size().reset_index(name="full_n")
        train_counts = train.groupby(group_cols).size().reset_index(name="train_n")
        test_counts = test.groupby(group_cols).size().reset_index(name="test_n")

        out = full_counts.merge(train_counts, on=group_cols, how="left")
        out = out.merge(test_counts, on=group_cols, how="left")

        out["train_n"] = out["train_n"].fillna(0).astype(int)
        out["test_n"] = out["test_n"].fillna(0).astype(int)

        out["full_share"] = out["full_n"] / out["full_n"].sum()
        out["test_share"] = out["test_n"] / max(1, out["test_n"].sum())
        out["share_diff"] = out["test_share"] - out["full_share"]

        return out.sort_values("full_n", ascending=False)

    theme_distribution = distribution_table(df, train_df, test_df, ["theme"])
    chapter_distribution = distribution_table(df, train_df, test_df, ["chapter"])
    theme_chapter_distribution = distribution_table(df, train_df, test_df, ["theme", "chapter"])

    overall_summary = pd.DataFrame({
        "metric": [
            "all_blocks",
            "all_tokens",
            "train_blocks",
            "train_tokens",
            "test_blocks",
            "test_tokens",
            "test_fraction_blocks",
            "test_fraction_tokens",
            "n_themes",
            "n_chapters",
            "mean_tokens_all",
            "median_tokens_all",
            "max_tokens_all",
        ],
        "value": [
            len(df),
            int(df["length_tokens"].sum()),
            len(train_df),
            int(train_df["length_tokens"].sum()),
            len(test_df),
            int(test_df["length_tokens"].sum()),
            len(test_df) / len(df),
            test_df["length_tokens"].sum() / df["length_tokens"].sum(),
            df["theme"].nunique(),
            df["chapter"].nunique(),
            float(df["length_tokens"].mean()),
            float(df["length_tokens"].median()),
            int(df["length_tokens"].max()),
        ]
    })

    print("\n=== Split summary ===")
    display(overall_summary)

    print("\n=== Theme distribution: full vs test ===")
    display(theme_distribution)

    print("\n=== Top chapter distribution: full vs test ===")
    display(chapter_distribution.head(50))

    print("\n=== Theme × chapter allocation preview ===")
    display(theme_chapter_distribution.head(80))

    # Drop tender_id and save train/test JSONs

    schema_cols = ["theme", "chapter", "block_id", "text", "length_tokens"]

    if not DROP_TENDER_ID and "tender_id" in df.columns:
        schema_cols = ["tender_id"] + schema_cols

    train_blocks = train_df[schema_cols].to_dict(orient="records")
    test_blocks = test_df[schema_cols].to_dict(orient="records")

    with TRAIN_JSON.open("w", encoding="utf-8") as f:
        json.dump(
            {"blocks": train_blocks},
            f,
            ensure_ascii=False,
            indent=2,
        )

    with TEST_JSON.open("w", encoding="utf-8") as f:
        json.dump(
            {"blocks": test_blocks},
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Save inventory and overview

    inventory_cols = schema_cols.copy()

    inventory_df = pd.concat(
        [
            train_df.assign(dataset_split="train_index"),
            test_df.assign(dataset_split="test_gold"),
        ],
        axis=0,
        ignore_index=True,
    )

    inventory_save_cols = ["dataset_split"] + inventory_cols

    inventory_df[inventory_save_cols].to_csv(
        SPLIT_INVENTORY_CSV,
        index=False,
        encoding="utf-8",
    )

    with pd.ExcelWriter(SPLIT_OVERVIEW_XLSX, engine="openpyxl") as writer:
        overall_summary.to_excel(writer, sheet_name="overall_summary", index=False)
        theme_stats_full.to_excel(writer, sheet_name="theme_stats_full", index=False)
        chapter_stats_full.to_excel(writer, sheet_name="chapter_stats_full", index=False)
        theme_distribution.to_excel(writer, sheet_name="theme_distribution", index=False)
        chapter_distribution.to_excel(writer, sheet_name="chapter_distribution", index=False)
        theme_chapter_distribution.to_excel(writer, sheet_name="theme_chapter_distribution", index=False)
        allocation_df.to_excel(writer, sheet_name="sample_allocation", index=False)

        inventory_preview = inventory_df[inventory_save_cols].copy()
        inventory_preview["text_preview"] = inventory_preview["text"].str[:500]
        inventory_preview = inventory_preview.drop(columns=["text"])
        inventory_preview.to_excel(writer, sheet_name="inventory_preview", index=False)

    print("\nSaved files:")
    print("-", TRAIN_JSON)
    print("-", TEST_JSON)
    print("-", SPLIT_INVENTORY_CSV)
    print("-", SPLIT_OVERVIEW_XLSX)

    # Preview final test set

    print("\n=== Test set preview ===")

    preview = test_df[["theme", "chapter", "block_id", "length_tokens", "text"]].copy()
    preview["text"] = preview["text"].str[:400]

    display(preview.head(20))

split_ledgar()

# --- Export LEDGAR parents and test cases ---
#
# Write the final training parent JSON and test CSV, including raw word lengths.

def export_ledgar():
    """Export LEDGAR parents and test cases."""
    TRAIN_SRC = LEDGAR_INTERIM_DIR / "ledgar_clause_parents_min300_themes_train90_no_tender.json"
    TEST_SRC = LEDGAR_INTERIM_DIR / "ledgar_clause_test_gold_min300_themes_10pct_no_tender.json"

    if not TRAIN_SRC.exists():
        raise FileNotFoundError(f"Train file not found: {TRAIN_SRC}")

    if not TEST_SRC.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_SRC}")

    # Destination paths

    DATASET_DIR = (DATA_ROOT / "ledgar_dataset")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    ARTIFACT_DIR = DATASET_DIR / "rag_artifacts"
    TRAIN_DST = ARTIFACT_DIR / "ledgar_parents.jsonl"
    TEST_DST = DATASET_DIR / "ledgar_test_cases.csv"

    # Helpers

    def strip_chapter_heading(text: str, chapter: str) -> str:
        """
    Removes the added leading chapter label if the text starts with:
      chapter + blank line + raw clause
    """
        text = str(text).strip()
        chapter = str(chapter).strip()

        if not chapter:
            return text

        # Normalize line endings
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")

        # Most common format: "Chapter\n\nClause..."
        prefix_1 = chapter + "\n\n"
        prefix_2 = chapter + "\n"

        if normalized.startswith(prefix_1):
            return normalized[len(prefix_1):].strip()

        if normalized.startswith(prefix_2):
            return normalized[len(prefix_2):].strip()

        return normalized.strip()

    def add_length_words_raw(df: pd.DataFrame) -> pd.DataFrame:
        """
    Adds or reconstructs length_words_raw.
    If it already exists, it is preserved.
    Otherwise it is computed from raw clause text after stripping the chapter heading.
    """
        df = df.copy()

        if "length_words_raw" not in df.columns:
            df["_raw_clause_text"] = df.apply(
                lambda row: strip_chapter_heading(row["text"], row["chapter"]),
                axis=1,
            )
            df["length_words_raw"] = df["_raw_clause_text"].apply(word_count)
            df = df.drop(columns=["_raw_clause_text"])
        else:
            df["length_words_raw"] = pd.to_numeric(
                df["length_words_raw"],
                errors="coerce",
            )

            missing_mask = df["length_words_raw"].isna()

            if missing_mask.any():
                df.loc[missing_mask, "length_words_raw"] = df.loc[missing_mask].apply(
                    lambda row: word_count(strip_chapter_heading(row["text"], row["chapter"])),
                    axis=1,
                )

            df["length_words_raw"] = df["length_words_raw"].astype(int)

        return df

    # Load train and test JSONs

    with TRAIN_SRC.open("r", encoding="utf-8") as f:
        train_data = json.load(f)

    with TEST_SRC.open("r", encoding="utf-8") as f:
        test_data = json.load(f)

    train_blocks = train_data.get("blocks", [])
    test_blocks = test_data.get("blocks", [])

    if not train_blocks:
        raise ValueError("Train JSON contains no blocks.")

    if not test_blocks:
        raise ValueError("Test JSON contains no blocks.")

    train_df = pd.DataFrame(train_blocks)
    test_df = pd.DataFrame(test_blocks)

    required_cols = ["chapter", "theme", "text", "length_tokens", "block_id"]

    for name, df in [("train", train_df), ("test", test_df)]:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {name} JSON: {missing}")

    # Add length_words_raw to train and test

    train_df = add_length_words_raw(train_df)
    test_df = add_length_words_raw(test_df)

    # Save train parents directly in the RAG artifact JSONL schema.

    train_cols = [
        "theme",
        "chapter",
        "block_id",
        "text",
        "length_tokens",
        "length_words_raw",
    ]

    train_df = train_df[train_cols].copy()

    save_parent_jsonl_from_blocks(TRAIN_DST, train_df.to_dict(orient="records"))

    # Convert test JSON to required CSV format

    test_df["Example Question"] = test_df.apply(
        lambda row: (
            f"Generate a {row['chapter']} clause for a contract provision "
            f"in the broader theme '{row['theme']}'."
        ),
        axis=1,
    )

    test_df["Gold Text"] = test_df["text"]

    test_csv_df = test_df[
        [
            "chapter",
            "theme",
            "Example Question",
            "Gold Text",
            "length_tokens",
            "length_words_raw",
            "block_id",
        ]
    ].copy()

    test_csv_df.to_csv(TEST_DST, index=False, encoding="utf-8")

    # Verification

    print("Saved LEDGAR dataset inside the project:")
    print(DATASET_DIR)

    print("\nFiles:")
    print("-", TRAIN_DST)
    print("-", TEST_DST)

    print("\nTrain parent blocks:", len(train_df))
    print("Test cases:", len(test_csv_df))

    print("\nTotal BGE-M3 tokens in test CSV:", int(test_csv_df["length_tokens"].sum()))
    print("Total raw words in test CSV:", int(test_csv_df["length_words_raw"].sum()))

    print("\nAverage BGE-M3 tokens per test case:", round(test_csv_df["length_tokens"].mean(), 2))
    print("Average raw words per test case:", round(test_csv_df["length_words_raw"].mean(), 2))

    print("\nTest CSV columns:")
    print(list(test_csv_df.columns))

    print("\nTest CSV preview:")
    display(test_csv_df.head(10))

export_ledgar()

# --- Inspect LEDGAR lengths ---
#
# Display and export the original LEDGAR length summaries.

def summarize_ledgar():
    """Inspect LEDGAR lengths."""
    BASE_DIR = (DATA_ROOT / "ledgar_dataset")

    TRAIN_JSONL = BASE_DIR / "rag_artifacts" / "ledgar_parents.jsonl"
    TEST_CSV = BASE_DIR / "ledgar_test_cases.csv"

    if not TRAIN_JSONL.exists():
        raise FileNotFoundError(f"Train file not found: {TRAIN_JSONL}")

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_CSV}")

    # Helpers

    def ensure_numeric_length_tokens(df: pd.DataFrame, name: str) -> pd.DataFrame:
        df = df.copy()

        if "length_tokens" not in df.columns:
            raise ValueError(f"Missing required column in {name}: length_tokens")

        df["length_tokens"] = pd.to_numeric(df["length_tokens"], errors="coerce")

        if df["length_tokens"].isna().any():
            n_missing = int(df["length_tokens"].isna().sum())
            raise ValueError(
                f"{name} contains {n_missing} missing or invalid length_tokens values."
            )

        return df

    def validate_required_columns(df: pd.DataFrame, required_cols, name: str):
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns in {name}: {missing}")

    def build_chapter_stats(df: pd.DataFrame, unit_name: str) -> pd.DataFrame:
        """
    Per-chapter statistics.
    Each row is one chapter.
    mean_tokens_per_unit is the mean parent/test-case length inside that chapter.
    """

        chapter_stats = (
            df.groupby("chapter")
            .agg(
                n_units=("block_id", "count"),
                total_tokens=("length_tokens", "sum"),
                median_tokens_per_unit=("length_tokens", "median"),
                mean_tokens_per_unit=("length_tokens", "mean"),
                sd_tokens_per_unit_within_chapter=("length_tokens", "std"),
                min_tokens_per_unit=("length_tokens", "min"),
                max_tokens_per_unit=("length_tokens", "max"),
            )
            .reset_index()
            .sort_values("n_units", ascending=False)
        )

        chapter_stats["unit"] = unit_name

        # If a chapter has only one unit, std is NaN. Set this within-chapter SD to 0.
        chapter_stats["sd_tokens_per_unit_within_chapter"] = (
            chapter_stats["sd_tokens_per_unit_within_chapter"].fillna(0.0)
        )

        return chapter_stats

    def print_dataset_summary(df: pd.DataFrame, title: str, unit_name: str):
        """
    Prints total statistics over all units and chapter-level statistics.

    Important:
    - Total tokens are calculated over all units in the set.
    - Average tokens per chapter is calculated as the mean of chapter-level means.
    - SD tokens per chapter is calculated as the SD of chapter-level means.
    """

        chapter_stats = build_chapter_stats(df, unit_name=unit_name)

        total_units = len(df)
        total_tokens = int(df["length_tokens"].sum())
        median_tokens_per_unit = float(df["length_tokens"].median())

        # These are calculated over chapter-level mean token lengths.
        mean_tokens_per_chapter_mean = float(chapter_stats["mean_tokens_per_unit"].mean())
        sd_tokens_per_chapter_mean = float(chapter_stats["mean_tokens_per_unit"].std(ddof=1))

        print("=" * 70)
        print(title)
        print("=" * 70)

        if title.lower().startswith("knowledge"):
            print("Train parent blocks:", total_units)
            print("Total BGE-M3 tokens train set:", total_tokens)
            print("Median tokens per parent block:", round(median_tokens_per_unit, 2))
            print(
                "Average tokens per parent block / chapter:",
                round(mean_tokens_per_chapter_mean, 2),
            )
            print(
                "SD tokens per parent block / chapter:",
                round(sd_tokens_per_chapter_mean, 2),
            )
        else:
            print("Test cases:", total_units)
            print("Total BGE-M3 tokens in test CSV:", total_tokens)
            print("Median tokens per test case:", round(median_tokens_per_unit, 2))
            print(
                "Average tokens per test case / parent block / chapter:",
                round(mean_tokens_per_chapter_mean, 2),
            )
            print(
                "SD tokens per test case / parent block / chapter:",
                round(sd_tokens_per_chapter_mean, 2),
            )

        print("Unique chapters:", int(df["chapter"].nunique()))

        print("\nPer-chapter statistics:")
        display(chapter_stats)

        return chapter_stats

    # Load train JSON

    train_blocks = load_parent_jsonl_as_blocks(TRAIN_JSONL)

    if not train_blocks:
        raise ValueError("Train JSONL contains no parent records.")

    train_df = pd.DataFrame(train_blocks)

    required_train_cols = [
        "chapter",
        "block_id",
        "text",
        "length_tokens",
    ]

    validate_required_columns(train_df, required_train_cols, "train JSONL")
    train_df = ensure_numeric_length_tokens(train_df, "train JSONL")

    # Load test CSV

    test_df = pd.read_csv(TEST_CSV)

    required_test_cols = [
        "chapter",
        "Example Question",
        "Gold Text",
        "length_tokens",
        "block_id",
    ]

    validate_required_columns(test_df, required_test_cols, "test CSV")
    test_df = ensure_numeric_length_tokens(test_df, "test CSV")

    # Print statistics

    train_chapter_stats = print_dataset_summary(
        train_df,
        title="Knowledge base",
        unit_name="parent block",
    )

    test_chapter_stats = print_dataset_summary(
        test_df,
        title="Test set",
        unit_name="test case / held-out parent block",
    )

summarize_ledgar()

# --- 4. HUPD ---

# --- Load and select HUPD blocks ---
#
# Load the original HUPD sample/date window and themes, select blocks and perform the initial split.

def prepare_hupd():
    """Load and select HUPD blocks."""
    import gc
    from collections import Counter

    # Cache safety configuration

    MAX_CACHE_GB = 2
    MAX_CACHE_BYTES = MAX_CACHE_GB * 1024 ** 3

    CACHE_DIR = (CACHE_ROOT / "hupd_hf_cache_max_2gb")
    RESET_CACHE_BEFORE_RUN = True

    if RESET_CACHE_BEFORE_RUN and CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(CACHE_DIR / "hf_home")
    os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "datasets")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    from datasets import load_dataset, DownloadConfig
    from transformers import AutoTokenizer

    def folder_size_bytes(path: Path) -> int:
        if not path.exists():
            return 0

        total = 0

        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass

        return total

    def print_cache_size(stage: str) -> int:
        size_bytes = folder_size_bytes(CACHE_DIR)
        size_gb = size_bytes / 1024 ** 3
        print(f"[Cache check: {stage}] {size_gb:.3f} GB used in {CACHE_DIR}")
        return size_bytes

    def enforce_cache_limit(stage: str):
        size_bytes = print_cache_size(stage)

        if size_bytes > MAX_CACHE_BYTES:
            size_gb = size_bytes / 1024 ** 3
            raise RuntimeError(
                f"Local Hugging Face cache exceeded {MAX_CACHE_GB} GB after stage '{stage}'. "
                f"Current size: {size_gb:.3f} GB."
            )

    def cleanup_hf_dataset_cache():
        """
    Delete Hugging Face dataset cache files after we have extracted our own records.
    This removes downloaded HUPD source files and generated Arrow files.
    The tokenizer cache is kept.
    """
        datasets_dir = CACHE_DIR / "datasets"

        if datasets_dir.exists():
            shutil.rmtree(datasets_dir)
            print("Removed Hugging Face datasets cache:", datasets_dir)

        datasets_dir.mkdir(parents=True, exist_ok=True)

        # Optional: remove cached custom dataset module after loading.
        modules_dir = CACHE_DIR / "hf_home" / "modules" / "datasets_modules"

        if modules_dir.exists():
            shutil.rmtree(modules_dir)
            print("Removed Hugging Face dataset module cache:", modules_dir)

    enforce_cache_limit("before loading")

    # Drive and output paths

    BASE_DIR = (DATA_ROOT / "hupd_dataset")
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_TRAIN_PARENTS_JSONL = BASE_DIR / "rag_artifacts" / "hupd_parents.jsonl"
    OUTPUT_TRAIN_PARENTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEST_CASES_CSV = BASE_DIR / "hupd_test_cases.csv"
    OUTPUT_FULL_BLOCKS_JSON = INTERIM_PREPARATION_DIR / "hupd_all_selected_blocks_before_split.json"
    OUTPUT_INVENTORY_XLSX = BASE_DIR / "hupd_selection_inventory.xlsx"
    OUTPUT_THEME_CHAPTER_STATS_CSV = BASE_DIR / "hupd_theme_chapter_stats.csv"

    # Dataset configuration

    LOAD_NAME = "sample"

    if LOAD_NAME != "sample":
        raise ValueError(
            "This script is capped to LOAD_NAME = 'sample'. "
            "Do not use the full HUPD dataset here."
        )

    HUPD_METADATA_URL = (
        "https://huggingface.co/datasets/HUPD/hupd/resolve/main/"
        "hupd_metadata_2022-02-22.feather"
    )

    # Loader-level CPC filter.
    # This restricts HUPD before our own 5-theme filtering.
    LOADER_CPC_LABEL = "G06"

    # HUPD sample is January 2016.
    # End date is treated as exclusive by the HUPD loader.
    TRAIN_FILING_START_DATE = "2016-01-01"
    TRAIN_FILING_END_DATE = "2016-01-15"

    # Dummy validation dates required by the HUPD loading script.
    # They are not used for our benchmark split.
    DUMMY_VAL_FILING_START_DATE = "2016-01-15"
    DUMMY_VAL_FILING_END_DATE = "2016-01-16"

    EMBEDDING_MODEL = "BAAI/bge-m3"

    MIN_TOKENS = 300
    MAX_TOKENS = 4000

    TEST_FRAC = 0.10
    RANDOM_STATE = 42

    # Fixed 5 similar G06 themes

    TARGET_CPC_THEMES = {
        "G06F": {
            "theme": "digital_data_processing",
            "theme_name": "Digital data processing",
        },
        "G06Q": {
            "theme": "business_and_administrative_data_processing",
            "theme_name": "Business and administrative data processing",
        },
        "G06K": {
            "theme": "data_recognition_and_handling",
            "theme_name": "Data recognition and handling",
        },
        "G06T": {
            "theme": "image_data_processing",
            "theme_name": "Image data processing",
        },
        "G06N": {
            "theme": "computational_models_and_machine_learning",
            "theme_name": "Computational models and machine learning",
        },
    }

    TARGET_CPC_CODES = list(TARGET_CPC_THEMES.keys())

    # Recurrent chapter configuration

    CHAPTER_FIELD_ALIASES = {
        "Abstract": ["abstract"],
        "Claims": ["claims"],
        "Background": ["background"],
        "Summary": ["summary"],
        "Full Description": ["description", "full_description"],
    }

    MIN_CHAPTER_TOTAL_BLOCKS = 5
    MIN_CHAPTER_THEME_PRESENCE = 2
    MIN_BLOCKS_PER_THEME_CHAPTER = 1

    # Tokenizer

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    global _BGE_TOKENIZER
    _BGE_TOKENIZER = tokenizer

    enforce_cache_limit("after tokenizer loading")

    def count_bge_tokens(text: str) -> int:
        return len(
            tokenizer.encode(
                str(text or ""),
                add_special_tokens=False
            )
        )

    def count_words_raw(text: str) -> int:
        return len(
            re.findall(
                r"\b\w+\b",
                str(text or "")
            )
        )

    # Helpers

    def normalize_text(x) -> str:
        if x is None:
            return ""

        if isinstance(x, (list, tuple, set, np.ndarray)):
            x = "\n".join(str(v) for v in x if v is not None)

        text = str(x)
        text = text.replace("\x00", " ")
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def get_first_existing_field(row: dict, aliases: list[str]) -> str:
        for field in aliases:
            if field in row:
                value = normalize_text(row.get(field, ""))

                if value:
                    return value

        return ""

    def normalize_cpc_label(value) -> str:
        if value is None:
            return ""

        label = str(value).strip().upper()
        label = re.sub(r"[^A-Z0-9]", "", label)

        if len(label) >= 4:
            return label[:4]

        return label

    def get_target_cpc_code(row: dict) -> str | None:
        for key in ["cpc_label", "main_cpc_label", "main_cpc"]:
            value = row.get(key, None)

            if value not in [None, ""]:
                code = normalize_cpc_label(value)

                if code in TARGET_CPC_THEMES:
                    return code

        return None

    def slugify(text: str, max_len: int = 90) -> str:
        text = str(text or "").lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")

        if not text:
            text = "unknown"

        return text[:max_len]

    def get_application_identifier(row: dict, row_idx: int) -> str:
        for key in [
            "patent_number",
            "application_number",
            "publication_number",
            "application_id",
        ]:
            value = row.get(key, None)

            if value not in [None, ""]:
                return str(value)

        return f"row_{row_idx}"

    def make_block_id(theme_code: str, application_id: str, chapter: str) -> str:
        return (
            "hupd_"
            + slugify(theme_code, max_len=10)
            + "_"
            + slugify(application_id, max_len=80)
            + "-"
            + slugify(chapter, max_len=60)
        )

    def build_parent_text(chapter: str, raw_text: str) -> str:
        return f"{chapter}\n\n{raw_text}".strip()

    # Load HUPD sample

    def load_hupd_iterator():
        """
    Prefer streaming mode to avoid generating large Arrow cache files.
    Fallback to non-streaming only if streaming fails.
    """
        download_config = DownloadConfig(
            cache_dir=str(CACHE_DIR / "datasets"),
            max_retries=2
        )

        common_kwargs = dict(
            path="HUPD/hupd",
            name=LOAD_NAME,
            data_files=HUPD_METADATA_URL,

            cpc_label=LOADER_CPC_LABEL,

            train_filing_start_date=TRAIN_FILING_START_DATE,
            train_filing_end_date=TRAIN_FILING_END_DATE,

            val_filing_start_date=DUMMY_VAL_FILING_START_DATE,
            val_filing_end_date=DUMMY_VAL_FILING_END_DATE,

            cache_dir=str(CACHE_DIR / "datasets"),
            download_config=download_config,
            split="train",
        )

        try:
            ds = load_dataset(
                **common_kwargs,
                streaming=True,
                trust_remote_code=True,
            )
            print("Loaded HUPD in streaming mode.")
            return ds, True

        except Exception as e:
            print("Streaming load failed. Falling back to non-streaming.")
            print("Streaming error:", repr(e))

            try:
                ds = load_dataset(
                    **common_kwargs,
                    keep_in_memory=True,
                    trust_remote_code=True,
                )
            except TypeError:
                ds = load_dataset(
                    **common_kwargs,
                    keep_in_memory=True,
                )

            print("Loaded HUPD in non-streaming fallback mode.")
            return ds, False

    hupd_iter, streaming_mode = load_hupd_iterator()

    print_cache_size("after HUPD iterator creation")

    # Convert fixed HUPD fields into candidate blocks

    records = []

    for row_idx, row in enumerate(tqdm(hupd_iter, desc="Processing HUPD rows")):
        row = dict(row)

        theme_code = get_target_cpc_code(row)

        if theme_code is None:
            continue

        theme = TARGET_CPC_THEMES[theme_code]["theme"]
        theme_name = TARGET_CPC_THEMES[theme_code]["theme_name"]

        application_id = get_application_identifier(
            row=row,
            row_idx=row_idx
        )

        title = normalize_text(row.get("title", ""))

        for chapter, aliases in CHAPTER_FIELD_ALIASES.items():
            raw_text = get_first_existing_field(row, aliases)

            if not raw_text:
                continue

            parent_text = build_parent_text(chapter, raw_text)
            length_tokens = count_bge_tokens(parent_text)
            length_words_raw = count_words_raw(raw_text)

            records.append({
                "theme": theme,
                "chapter": chapter,
                "block_id": make_block_id(
                    theme_code=theme_code,
                    application_id=application_id,
                    chapter=chapter
                ),
                "text": parent_text,
                "length_tokens": int(length_tokens),
                "length_words_raw": int(length_words_raw),

                # Inspection-only fields
                "theme_code": theme_code,
                "theme_name": theme_name,
                "application_id": application_id,
                "title": title,
            })

    print("\nRaw candidate records extracted:", len(records))

    # Delete HUPD iterator and all HUPD dataset cache after extracting our own records.
    del hupd_iter
    gc.collect()

    cleanup_hf_dataset_cache()
    enforce_cache_limit("after processing HUPD rows and cleanup")

    df_all = pd.DataFrame(records)

    if len(df_all) == 0:
        raise ValueError(
            "No HUPD blocks were extracted for the fixed five G06 themes."
        )

    print("\nAll extracted fixed-field blocks from the 5 target themes:", len(df_all))

    print("\nAll block token stats:")
    print(
        df_all["length_tokens"]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    )

    # Token filter

    df_len = df_all[
        (df_all["length_tokens"] >= MIN_TOKENS)
        & (df_all["length_tokens"] <= MAX_TOKENS)
    ].copy()

    print("\nBlocks after token filter:", len(df_len))
    print("Removed below 300:", int((df_all["length_tokens"] < MIN_TOKENS).sum()))
    print("Removed above 4000:", int((df_all["length_tokens"] > MAX_TOKENS).sum()))

    if len(df_len) == 0:
        raise ValueError("No blocks left after token filtering.")

    # Theme and chapter statistics

    theme_stats = (
        df_len
        .groupby(["theme_code", "theme", "theme_name"])
        .agg(
            n_blocks=("block_id", "count"),
            n_applications=("application_id", "nunique"),
            n_chapters=("chapter", "nunique"),
            mean_tokens=("length_tokens", "mean"),
            median_tokens=("length_tokens", "median"),
            min_tokens=("length_tokens", "min"),
            max_tokens=("length_tokens", "max"),
        )
        .reset_index()
        .sort_values(["theme_code"])
    )

    chapter_stats_all = (
        df_len
        .groupby("chapter")
        .agg(
            n_blocks=("block_id", "count"),
            n_themes=("theme", "nunique"),
            n_applications=("application_id", "nunique"),
            mean_tokens=("length_tokens", "mean"),
            median_tokens=("length_tokens", "median"),
            min_tokens=("length_tokens", "min"),
            max_tokens=("length_tokens", "max"),
        )
        .reset_index()
        .sort_values(["n_themes", "n_blocks"], ascending=False)
    )

    theme_chapter_stats = (
        df_len
        .groupby(["theme_code", "theme", "theme_name", "chapter"])
        .agg(
            n_blocks=("block_id", "count"),
            n_applications=("application_id", "nunique"),
            mean_tokens=("length_tokens", "mean"),
            median_tokens=("length_tokens", "median"),
            min_tokens=("length_tokens", "min"),
            max_tokens=("length_tokens", "max"),
        )
        .reset_index()
        .sort_values(["theme_code", "chapter"])
    )

    print("\nTheme stats after token filter:")
    print(theme_stats.to_string(index=False))

    print("\nChapter stats after token filter:")
    print(chapter_stats_all.to_string(index=False))

    print("\nTheme/chapter stats after token filter:")
    print(theme_chapter_stats.to_string(index=False))

    # Select only recurrent chapters

    valid_theme_chapter_pairs = theme_chapter_stats[
        theme_chapter_stats["n_blocks"] >= MIN_BLOCKS_PER_THEME_CHAPTER
    ][["theme", "chapter"]].copy()

    df_pair_filtered = df_len.merge(
        valid_theme_chapter_pairs,
        on=["theme", "chapter"],
        how="inner"
    )

    chapter_recurrence = (
        df_pair_filtered
        .groupby("chapter")
        .agg(
            n_blocks=("block_id", "count"),
            n_themes=("theme", "nunique"),
            n_applications=("application_id", "nunique"),
            mean_tokens=("length_tokens", "mean"),
            median_tokens=("length_tokens", "median"),
            min_tokens=("length_tokens", "min"),
            max_tokens=("length_tokens", "max"),
        )
        .reset_index()
        .sort_values(["n_themes", "n_blocks"], ascending=False)
    )

    selected_chapters_df = chapter_recurrence[
        (chapter_recurrence["n_blocks"] >= MIN_CHAPTER_TOTAL_BLOCKS)
        & (chapter_recurrence["n_themes"] >= MIN_CHAPTER_THEME_PRESENCE)
    ].copy()

    if len(selected_chapters_df) == 0:
        selected_chapters_df = chapter_recurrence[
            chapter_recurrence["n_themes"] >= MIN_CHAPTER_THEME_PRESENCE
        ].copy()

    if len(selected_chapters_df) == 0:
        raise ValueError(
            "No recurrent chapters found. "
            "Lower MIN_CHAPTER_THEME_PRESENCE or inspect the inventory."
        )

    selected_chapters = selected_chapters_df["chapter"].tolist()

    print("\nSelected recurrent chapters:")
    print(selected_chapters)

    df_final = df_pair_filtered[
        df_pair_filtered["chapter"].isin(selected_chapters)
    ].copy()

    # Final selected full corpus before split

    df_final = df_final.sort_values(
        ["theme_code", "chapter", "block_id"]
    ).reset_index(drop=True)

    duplicate_mask = df_final["block_id"].duplicated(keep=False)

    if duplicate_mask.any():
        counts = Counter()
        new_ids = []

        for block_id in df_final["block_id"]:
            counts[block_id] += 1

            if counts[block_id] == 1:
                new_ids.append(block_id)
            else:
                new_ids.append(f"{block_id}_{counts[block_id]}")

        df_final["block_id"] = new_ids

    assert (df_final["length_tokens"] >= MIN_TOKENS).all()
    assert (df_final["length_tokens"] <= MAX_TOKENS).all()
    assert df_final["block_id"].is_unique

    print("\nFull selected corpus blocks before 90/10 split:", len(df_final))
    print("Themes:", sorted(df_final["theme"].unique()))
    print("Chapters:", sorted(df_final["chapter"].unique()))

    missing_theme_codes = sorted(set(TARGET_CPC_CODES) - set(df_final["theme_code"].unique()))
    print("Missing target CPC themes after filtering:", missing_theme_codes)

    print("\nFull selected corpus token stats:")
    print(
        df_final["length_tokens"]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    )

    full_theme_chapter_counts = (
        df_final
        .groupby(["theme_code", "theme", "theme_name", "chapter"])
        .size()
        .reset_index(name="n_blocks")
        .sort_values(["theme_code", "chapter"])
    )

    print("\nFull selected theme/chapter counts:")
    print(full_theme_chapter_counts.to_string(index=False))

    # Stratified 10% test split like LEDGAR

    train_df, test_df, split_alloc = stratified_train_test_split(
        df=df_final,
        stratum_cols=["theme", "chapter"],
        test_frac=TEST_FRAC,
        random_state=RANDOM_STATE
    )

    print("\nSplit summary:")
    print("Full selected corpus:", len(df_final))
    print("Train corpus:", len(train_df))
    print("Test cases:", len(test_df))
    print("Actual test share:", round(len(test_df) / len(df_final), 4))

    print("\nTest counts by theme/chapter:")
    test_counts = (
        test_df
        .groupby(["theme", "chapter"])
        .size()
        .reset_index(name="n_test")
        .sort_values(["theme", "chapter"])
    )

    print(test_counts.to_string(index=False))

    # Save LEDGAR-compatible outputs

    train_blocks = train_df[
        [
            "theme",
            "chapter",
            "block_id",
            "text",
            "length_tokens",
            "length_words_raw",
        ]
    ].to_dict(orient="records")

    save_parent_jsonl_from_blocks(OUTPUT_TRAIN_PARENTS_JSONL, train_blocks)

    full_blocks = df_final[
        [
            "theme",
            "chapter",
            "block_id",
            "text",
            "length_tokens",
            "length_words_raw",
        ]
    ].to_dict(orient="records")

    full_json = {
        "blocks": full_blocks
    }

    with OUTPUT_FULL_BLOCKS_JSON.open("w", encoding="utf-8") as f:
        json.dump(
            full_json,
            f,
            ensure_ascii=False,
            indent=2
        )

    test_cases = pd.DataFrame({
        "chapter": test_df["chapter"],
        "theme": test_df["theme"],
        "Example Question": "",
        "Gold Text": test_df["text"],
        "length_tokens": test_df["length_tokens"],
        "length_words_raw": test_df["length_words_raw"],
        "block_id": test_df["block_id"],
    })

    test_cases.to_csv(
        OUTPUT_TEST_CASES_CSV,
        index=False,
        encoding="utf-8"
    )

    theme_chapter_stats.to_csv(
        OUTPUT_THEME_CHAPTER_STATS_CSV,
        index=False,
        encoding="utf-8"
    )

    # Save inspection Excel

    def preview_df(df: pd.DataFrame, n_chars: int = 1500) -> pd.DataFrame:
        out = df.copy()

        if "text" in out.columns:
            out["text_preview"] = out["text"].str[:n_chars]
            out = out.drop(columns=["text"])

        return out

    train_preview = preview_df(train_df)
    test_preview = preview_df(test_df)
    final_preview = preview_df(df_final)
    len_preview = preview_df(df_len)
    all_preview = preview_df(df_all)

    with pd.ExcelWriter(OUTPUT_INVENTORY_XLSX, engine="openpyxl") as writer:
        theme_stats.to_excel(writer, sheet_name="theme_stats", index=False)
        chapter_stats_all.to_excel(writer, sheet_name="chapter_stats_all", index=False)
        theme_chapter_stats.to_excel(writer, sheet_name="theme_chapter_stats", index=False)
        chapter_recurrence.to_excel(writer, sheet_name="chapter_recurrence", index=False)
        selected_chapters_df.to_excel(writer, sheet_name="selected_chapters", index=False)
        full_theme_chapter_counts.to_excel(writer, sheet_name="full_counts", index=False)
        split_alloc.to_excel(writer, sheet_name="split_allocation", index=False)
        test_counts.to_excel(writer, sheet_name="test_counts", index=False)
        train_preview.to_excel(writer, sheet_name="train_preview", index=False)
        test_preview.to_excel(writer, sheet_name="test_preview", index=False)
        final_preview.to_excel(writer, sheet_name="full_selected_preview", index=False)
        len_preview.head(5000).to_excel(writer, sheet_name="token_filtered_preview", index=False)
        all_preview.head(5000).to_excel(writer, sheet_name="all_blocks_preview", index=False)

    enforce_cache_limit("after saving outputs")

    # Final schema validation

    check = []
    with OUTPUT_TRAIN_PARENTS_JSONL.open("r", encoding="utf-8") as f:
        check = [json.loads(line) for line in f if line.strip()]

    assert len(check) == len(train_df)

    required_keys = {
        "parent_id",
        "theme",
        "chapter",
        "text",
        "length_tokens_bge_m3",
        "length_words_raw",
    }

    for i, block in enumerate(check[:10]):
        missing = required_keys - set(block.keys())

        if missing:
            raise ValueError(f"Block {i} missing keys: {missing}")

        extra = set(block.keys()) - required_keys

        if extra:
            raise ValueError(f"Block {i} has extra keys: {extra}")

    print("\nSaved:")
    print("Train parents JSONL:", OUTPUT_TRAIN_PARENTS_JSONL)
    print("Test cases CSV:", OUTPUT_TEST_CASES_CSV)
    print("Full selected blocks before split:", OUTPUT_FULL_BLOCKS_JSON)
    print("Inventory:", OUTPUT_INVENTORY_XLSX)
    print("Theme/chapter stats:", OUTPUT_THEME_CHAPTER_STATS_CSV)

    print("\nFinal train JSONL schema example:")
    print(json.dumps(check[0], ensure_ascii=False, indent=2)[:2000])

prepare_hupd()

# --- Keep HUPD Background, Summary and Claims ---
#
# Keep only Background, Summary and Claims, then recompute the original HUPD split and outputs.

def filter_hupd_chapters():
    """Keep HUPD Background, Summary and Claims."""
    from collections import Counter

    # Configuration

    BASE_DIR = (DATA_ROOT / "hupd_dataset")

    FULL_JSON_PATH = INTERIM_PREPARATION_DIR / "hupd_all_selected_blocks_before_split.json"
    TRAIN_JSON_PATH = BASE_DIR / "rag_artifacts" / "hupd_parents.jsonl"
    TEST_CSV_PATH = BASE_DIR / "hupd_test_cases.csv"
    THEME_CHAPTER_STATS_CSV = BASE_DIR / "hupd_theme_chapter_stats.csv"
    INVENTORY_XLSX_PATH = BASE_DIR / "hupd_selection_inventory.xlsx"

    KEEP_CHAPTERS = ["Background", "Summary", "Claims"]
    DROP_CHAPTERS = ["Abstract", "Full Description"]

    TEST_FRAC = 0.10
    RANDOM_STATE = 42

    REQUIRED_BLOCK_KEYS = [
        "theme",
        "chapter",
        "block_id",
        "text",
        "length_tokens",
        "length_words_raw",
    ]

    # Helpers

    def backup_once(path: Path):
        """
    Create a one-time backup before overwriting a file.
    """
        if not path.exists():
            return

        backup_path = path.with_name(path.name + ".bak_before_chapter_filter")

        if not backup_path.exists():
            shutil.copy2(path, backup_path)
            print(f"Backup created: {backup_path}")
        else:
            print(f"Backup already exists: {backup_path}")

    def load_blocks_json(path: Path) -> list[dict]:
        """
    Load JSON in either:
    1. {"blocks": [...]}
    2. [...]
    """
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

        blocks = load_blocks_file(path)

        if not isinstance(blocks, list):
            raise ValueError(f"'blocks' must be a list in: {path}")

        return blocks

    def save_blocks_json(path: Path, blocks: list[dict]):
        """
    Save LEDGAR-compatible parent JSON structure.
    """
        save_blocks_file(path, blocks)

    def normalize_block(block: dict) -> dict:
        """
    Keep only the required LEDGAR-compatible fields.
    """
        return {
            key: block.get(key)
            for key in REQUIRED_BLOCK_KEYS
        }

    # Backup existing files

    for path in [
        FULL_JSON_PATH,
        TRAIN_JSON_PATH,
        TEST_CSV_PATH,
        THEME_CHAPTER_STATS_CSV,
        INVENTORY_XLSX_PATH,
    ]:
        backup_once(path)

    # Load full selected corpus and filter chapters

    all_blocks = load_blocks_json(FULL_JSON_PATH)

    df_full = pd.DataFrame(all_blocks)

    if "chapter" not in df_full.columns:
        raise ValueError("The full JSON does not contain a 'chapter' column.")

    before_counts = df_full["chapter"].value_counts().sort_index()

    df_filtered = df_full[
        df_full["chapter"].isin(KEEP_CHAPTERS)
    ].copy()

    after_counts = df_filtered["chapter"].value_counts().sort_index()

    removed_counts = df_full[
        df_full["chapter"].isin(DROP_CHAPTERS)
    ]["chapter"].value_counts().sort_index()

    if len(df_filtered) == 0:
        raise ValueError("Filtering removed all blocks. Check chapter names.")

    if df_filtered["chapter"].isin(DROP_CHAPTERS).any():
        raise AssertionError("Dropped chapters are still present after filtering.")

    if not set(KEEP_CHAPTERS).issubset(set(df_filtered["chapter"].unique())):
        missing = sorted(set(KEEP_CHAPTERS) - set(df_filtered["chapter"].unique()))
        raise ValueError(f"Missing expected chapters after filtering: {missing}")

    if df_filtered["block_id"].duplicated().any():
        duplicated = df_filtered[df_filtered["block_id"].duplicated(keep=False)]["block_id"].tolist()[:20]
        raise ValueError(f"Duplicate block_id values found, examples: {duplicated}")

    # Recreate 90/10 split from filtered full corpus

    train_df, test_df, split_alloc = stratified_train_test_split(
        df=df_filtered,
        stratum_cols=["theme", "chapter"],
        test_frac=TEST_FRAC,
        random_state=RANDOM_STATE
    )

    train_blocks = [
        normalize_block(row)
        for row in train_df.to_dict(orient="records")
    ]

    full_blocks_filtered = [
        normalize_block(row)
        for row in df_filtered
        .sort_values(["theme", "chapter", "block_id"])
        .to_dict(orient="records")
    ]

    test_cases = pd.DataFrame({
        "chapter": test_df["chapter"],
        "theme": test_df["theme"],
        "Example Question": "",
        "Gold Text": test_df["text"],
        "length_tokens": test_df["length_tokens"],
        "length_words_raw": test_df["length_words_raw"],
        "block_id": test_df["block_id"],
    })

    # Overwrite files

    save_blocks_json(
        FULL_JSON_PATH,
        full_blocks_filtered
    )

    save_blocks_json(
        TRAIN_JSON_PATH,
        train_blocks
    )

    test_cases.to_csv(
        TEST_CSV_PATH,
        index=False,
        encoding="utf-8"
    )

    theme_chapter_stats = (
        df_filtered
        .groupby(["theme", "chapter"])
        .agg(
            n_blocks=("block_id", "count"),
            mean_tokens=("length_tokens", "mean"),
            median_tokens=("length_tokens", "median"),
            min_tokens=("length_tokens", "min"),
            max_tokens=("length_tokens", "max"),
        )
        .reset_index()
        .sort_values(["chapter", "theme"])
    )

    theme_chapter_stats.to_csv(
        THEME_CHAPTER_STATS_CSV,
        index=False,
        encoding="utf-8"
    )

    full_counts = (
        df_filtered
        .groupby(["theme", "chapter"])
        .size()
        .reset_index(name="n_blocks")
        .sort_values(["chapter", "theme"])
    )

    train_counts = (
        train_df
        .groupby(["theme", "chapter"])
        .size()
        .reset_index(name="n_train")
        .sort_values(["chapter", "theme"])
    )

    test_counts = (
        test_df
        .groupby(["theme", "chapter"])
        .size()
        .reset_index(name="n_test")
        .sort_values(["chapter", "theme"])
    )

    summary_df = pd.DataFrame({
        "metric": [
            "full_blocks_before_filter",
            "full_blocks_after_filter",
            "train_blocks_after_filter",
            "test_cases_after_filter",
            "actual_test_share",
        ],
        "value": [
            len(df_full),
            len(df_filtered),
            len(train_df),
            len(test_df),
            round(len(test_df) / len(df_filtered), 4),
        ],
    })

    with pd.ExcelWriter(INVENTORY_XLSX_PATH, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        before_counts.rename("n_before").reset_index().rename(columns={"index": "chapter"}).to_excel(
            writer,
            sheet_name="before_counts",
            index=False
        )
        after_counts.rename("n_after").reset_index().rename(columns={"index": "chapter"}).to_excel(
            writer,
            sheet_name="after_counts",
            index=False
        )
        removed_counts.rename("n_removed").reset_index().rename(columns={"index": "chapter"}).to_excel(
            writer,
            sheet_name="removed_counts",
            index=False
        )
        full_counts.to_excel(writer, sheet_name="full_counts", index=False)
        train_counts.to_excel(writer, sheet_name="train_counts", index=False)
        test_counts.to_excel(writer, sheet_name="test_counts", index=False)
        theme_chapter_stats.to_excel(writer, sheet_name="theme_chapter_stats", index=False)
        split_alloc.to_excel(writer, sheet_name="split_allocation", index=False)

    # Final validation

    check_train = load_blocks_json(TRAIN_JSON_PATH)
    check_full = load_blocks_json(FULL_JSON_PATH)
    check_test = pd.read_csv(TEST_CSV_PATH)

    for name, blocks in [
        ("train", check_train),
        ("full", check_full),
    ]:
        chapters = set([b.get("chapter") for b in blocks])

        if any(ch in chapters for ch in DROP_CHAPTERS):
            raise AssertionError(f"{name} JSON still contains dropped chapters: {DROP_CHAPTERS}")

        if not chapters.issubset(set(KEEP_CHAPTERS)):
            raise AssertionError(f"{name} JSON contains unexpected chapters: {chapters}")

    if check_test["chapter"].isin(DROP_CHAPTERS).any():
        raise AssertionError("Test CSV still contains dropped chapters.")

    print("\nOverwrite complete.")
    print("\nBefore filtering:")
    print(before_counts)

    print("\nRemoved:")
    print(removed_counts)

    print("\nAfter filtering:")
    print(after_counts)

    print("\nSplit summary:")
    print(summary_df.to_string(index=False))

    print("\nOverwritten files:")
    print(FULL_JSON_PATH)
    print(TRAIN_JSON_PATH)
    print(TEST_CSV_PATH)
    print(THEME_CHAPTER_STATS_CSV)
    print(INVENTORY_XLSX_PATH)

    print("\nNote: Existing RAG embeddings or child chunks are now stale and should be regenerated from the filtered HUPD parent JSONL.")

filter_hupd_chapters()

# --- Remove HUPD markers from parent text ---
#
# Clean the parent JSON using the original marker substitutions.

def clean_hupd_parents():
    """Remove HUPD markers from parent text."""
    path = DATA_ROOT / "hupd_dataset/rag_artifacts/hupd_parents.jsonl"
    blocks = load_parent_jsonl_as_blocks(path)

    pattern = re.compile(r"<SOH>\s*.*?\s*<EOH>", flags=re.DOTALL)

    changed = 0

    for block in blocks:
        text = str(block.get("text", ""))
        cleaned = pattern.sub("", text)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

        if cleaned != text:
            changed += 1
            block["text"] = cleaned

    save_parent_jsonl_from_blocks(path, blocks)

    print(f"Overwritten: {path}")
    print(f"Texts changed: {changed}")

clean_hupd_parents()

# --- Check HUPD split distributions ---
#
# Check chapter/theme distributions and train–test block separation.

def check_hupd_distributions():
    """Check HUPD split distributions."""
    BASE_DIR = (DATA_ROOT / "hupd_dataset")

    FULL_JSON_PATH = INTERIM_PREPARATION_DIR / "hupd_all_selected_blocks_before_split.json"
    TRAIN_JSON_PATH = BASE_DIR / "rag_artifacts" / "hupd_parents.jsonl"
    TEST_CSV_PATH = BASE_DIR / "hupd_test_cases.csv"

    TARGET_TEST_FRAC = 0.10

    ID_COL = "block_id"
    THEME_COL = "theme"
    CHAPTER_COL = "chapter"

    # Helpers

    def load_blocks_json(path: Path) -> list[dict]:
        """
    Load JSON in either:
    1. {"blocks": [...]}
    2. [...]
    """
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        blocks = load_blocks_file(path)

        if not isinstance(blocks, list):
            raise ValueError(f"'blocks' must be a list in: {path}")

        return blocks

    def validate_required_columns(df: pd.DataFrame, name: str, required_cols: list[str]):
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            raise ValueError(
                f"{name} is missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

    def build_distribution_check(
        full_df: pd.DataFrame,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        group_cols: list[str],
        label: str
    ) -> pd.DataFrame:
        full_dist = (
            full_df
            .groupby(group_cols)
            .size()
            .reset_index(name="n_full")
        )

        train_dist = (
            train_df
            .groupby(group_cols)
            .size()
            .reset_index(name="n_train")
        )

        test_dist = (
            test_df
            .groupby(group_cols)
            .size()
            .reset_index(name="n_test")
        )

        out = (
            full_dist
            .merge(train_dist, on=group_cols, how="left")
            .merge(test_dist, on=group_cols, how="left")
        )

        out["n_train"] = out["n_train"].fillna(0).astype(int)
        out["n_test"] = out["n_test"].fillna(0).astype(int)

        out["n_train_plus_test"] = out["n_train"] + out["n_test"]
        out["count_balance_ok"] = out["n_train_plus_test"].eq(out["n_full"])

        out["test_share_within_group"] = out["n_test"] / out["n_full"]
        out["target_test_share"] = TARGET_TEST_FRAC
        out["absolute_deviation_from_target"] = (
            out["test_share_within_group"] - TARGET_TEST_FRAC
        ).abs()

        out["full_share_of_total"] = out["n_full"] / len(full_df)
        out["train_share_of_total"] = out["n_train"] / len(train_df)
        out["test_share_of_total"] = out["n_test"] / len(test_df)

        out["test_distribution_minus_full_distribution"] = (
            out["test_share_of_total"] - out["full_share_of_total"]
        )

        out["abs_test_distribution_deviation"] = (
            out["test_distribution_minus_full_distribution"].abs()
        )

        out.insert(0, "level", label)

        return out.sort_values(group_cols).reset_index(drop=True)

    # Load saved files

    full_blocks = load_blocks_json(FULL_JSON_PATH)
    train_blocks = load_blocks_json(TRAIN_JSON_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    full_df = pd.DataFrame(full_blocks)
    train_df = pd.DataFrame(train_blocks)

    validate_required_columns(
        full_df,
        "Full selected blocks JSON",
        [ID_COL, THEME_COL, CHAPTER_COL]
    )

    validate_required_columns(
        train_df,
        "Train parents JSON",
        [ID_COL, THEME_COL, CHAPTER_COL]
    )

    validate_required_columns(
        test_df,
        "Test CSV",
        [ID_COL, THEME_COL, CHAPTER_COL]
    )

    full_df = full_df[[ID_COL, THEME_COL, CHAPTER_COL]].copy()
    train_df = train_df[[ID_COL, THEME_COL, CHAPTER_COL]].copy()
    test_df = test_df[[ID_COL, THEME_COL, CHAPTER_COL]].copy()

    # Integrity checks

    full_ids = set(full_df[ID_COL])
    train_ids = set(train_df[ID_COL])
    test_ids = set(test_df[ID_COL])

    overlap_train_test = train_ids.intersection(test_ids)
    missing_from_full = train_ids.union(test_ids) - full_ids
    missing_from_split = full_ids - train_ids.union(test_ids)

    integrity_df = pd.DataFrame([
        {"check": "n_full", "value": len(full_df)},
        {"check": "n_train", "value": len(train_df)},
        {"check": "n_test", "value": len(test_df)},
        {"check": "actual_test_share", "value": round(len(test_df) / len(full_df), 6)},
        {"check": "train_test_overlap_count", "value": len(overlap_train_test)},
        {"check": "ids_in_train_or_test_but_not_full", "value": len(missing_from_full)},
        {"check": "ids_in_full_but_not_train_or_test", "value": len(missing_from_split)},
    ])

    if full_df[ID_COL].duplicated().any():
        dupes = full_df[full_df[ID_COL].duplicated(keep=False)][ID_COL].head(20).tolist()
        raise ValueError(f"Duplicate block_id in full JSON, examples: {dupes}")

    if train_df[ID_COL].duplicated().any():
        dupes = train_df[train_df[ID_COL].duplicated(keep=False)][ID_COL].head(20).tolist()
        raise ValueError(f"Duplicate block_id in train JSON, examples: {dupes}")

    if test_df[ID_COL].duplicated().any():
        dupes = test_df[test_df[ID_COL].duplicated(keep=False)][ID_COL].head(20).tolist()
        raise ValueError(f"Duplicate block_id in test CSV, examples: {dupes}")

    # Distribution checks

    theme_chapter_dist = build_distribution_check(
        full_df=full_df,
        train_df=train_df,
        test_df=test_df,
        group_cols=[THEME_COL, CHAPTER_COL],
        label="theme_x_chapter"
    )

    chapter_dist = build_distribution_check(
        full_df=full_df,
        train_df=train_df,
        test_df=test_df,
        group_cols=[CHAPTER_COL],
        label="chapter"
    )

    theme_dist = build_distribution_check(
        full_df=full_df,
        train_df=train_df,
        test_df=test_df,
        group_cols=[THEME_COL],
        label="theme"
    )

    missing_represented_strata = theme_chapter_dist[
        (theme_chapter_dist["n_full"] >= 10)
        & (theme_chapter_dist["n_test"] == 0)
    ].copy()

    largest_deviations = (
        theme_chapter_dist
        .sort_values("abs_test_distribution_deviation", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )

    # Print results

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 80)

    print("\n==============================")
    print("SPLIT INTEGRITY")
    print("==============================")
    print(integrity_df.to_string(index=False))

    print("\n==============================")
    print("THEME × CHAPTER DISTRIBUTION")
    print("==============================")
    print(theme_chapter_dist.to_string(index=False))

    print("\n==============================")
    print("CHAPTER DISTRIBUTION")
    print("==============================")
    print(chapter_dist.to_string(index=False))

    print("\n==============================")
    print("THEME DISTRIBUTION")
    print("==============================")
    print(theme_dist.to_string(index=False))

    print("\n==============================")
    print("LARGE THEME × CHAPTER STRATA MISSING FROM TEST")
    print("==============================")
    if len(missing_represented_strata) == 0:
        print("None. Every theme × chapter stratum with at least 10 full examples is represented in the test set.")
    else:
        print(missing_represented_strata.to_string(index=False))

    print("\n==============================")
    print("LARGEST TEST DISTRIBUTION DEVIATIONS")
    print("==============================")
    print(largest_deviations.to_string(index=False))

    print("\n==============================")
    print("SUMMARY")
    print("==============================")
    print("Full blocks:", len(full_df))
    print("Train blocks:", len(train_df))
    print("Test cases:", len(test_df))
    print("Actual test share:", round(len(test_df) / len(full_df), 6))
    print("Train/test overlap:", len(overlap_train_test))
    print("Full IDs missing from train/test:", len(missing_from_split))
    print("Train/test IDs missing from full:", len(missing_from_full))

check_hupd_distributions()

# --- Clean HUPD test gold text ---
#
# Clean the test CSV gold text and preserve the original backup behavior.

def clean_hupd_gold_text():
    """Clean HUPD test gold text."""
    BASE_DIR = (DATA_ROOT / "hupd_dataset")
    TEST_CSV_PATH = BASE_DIR / "hupd_test_cases.csv"

    BACKUP_PATH = TEST_CSV_PATH.with_name(
        TEST_CSV_PATH.name + ".bak_before_gold_text_heading_cleanup"
    )

    GOLD_COL = "Gold Text"
    QUESTION_COL = "Example Question"
    CHAPTER_COL = "chapter"

    # Cleaning function

    def remove_hupd_headings_from_gold_text(text, chapter=None) -> str:
        """
    Remove HUPD section headings from Gold Text.

    Removes:
    - leading chapter wrapper inserted by preprocessing:
        Background\n\n
        Summary\n\n
        Claims\n\n

    - HUPD heading markers:
        <SOH> BACKGROUND <EOH>
        <SOH> BACKGROUND ART <EOH>
        <SOH> SUMMARY <EOH>
        <SOH> CLAIMS <EOH>

    - common plain heading prefixes at the beginning:
        BACKGROUND:
        BACKGROUND ART
        SUMMARY -
        CLAIMS.
    """
        if pd.isna(text):
            return ""

        s = str(text)

        # Remove leading wrapper from parent text, e.g. "Background\n\n..."
        if chapter is not None and not pd.isna(chapter):
            chapter_str = str(chapter).strip()

            if chapter_str:
                s = re.sub(
                    rf"^\s*{re.escape(chapter_str)}\s*(?:\r?\n)+",
                    "",
                    s,
                    flags=re.IGNORECASE
                )

        # Remove all HUPD <SOH> ... <EOH> section heading markers.
        s = re.sub(
            r"\s*<SOH>\s*[^<>]{0,200}?\s*<EOH>\s*",
            " ",
            s,
            flags=re.IGNORECASE
        )

        # Remove common plain heading prefixes at the beginning.
        s = re.sub(
            r"^\s*(BACKGROUND ART|BACKGROUND|SUMMARY OF THE INVENTION|SUMMARY|CLAIMS?)\s*[:.\-–—]?\s*",
            "",
            s,
            flags=re.IGNORECASE
        )

        # Clean whitespace but preserve paragraph breaks where possible.
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n[ \t]+", "\n", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        s = re.sub(r" {2,}", " ", s)
        s = s.strip()

        return s

    # Load CSV

    if not TEST_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing file: {TEST_CSV_PATH}")

    df = pd.read_csv(TEST_CSV_PATH)

    required_cols = [GOLD_COL, CHAPTER_COL]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available columns: {list(df.columns)}"
        )

    # Backup once

    if not BACKUP_PATH.exists():
        shutil.copy2(TEST_CSV_PATH, BACKUP_PATH)
        print(f"Backup created: {BACKUP_PATH}")
    else:
        print(f"Backup already exists: {BACKUP_PATH}")

    # Clean only Gold Text

    before_gold = df[GOLD_COL].fillna("").astype(str).copy()

    if QUESTION_COL in df.columns:
        before_questions = df[QUESTION_COL].copy()
    else:
        before_questions = None

    df[GOLD_COL] = df.apply(
        lambda row: remove_hupd_headings_from_gold_text(
            text=row[GOLD_COL],
            chapter=row[CHAPTER_COL]
        ),
        axis=1
    )

    after_gold = df[GOLD_COL].fillna("").astype(str)

    changed_mask = before_gold.ne(after_gold)
    n_changed = int(changed_mask.sum())

    # Validation

    remaining_marker_mask = (
        df[GOLD_COL]
        .fillna("")
        .astype(str)
        .str.contains(r"<SOH>|<EOH>", case=False, regex=True)
    )

    n_remaining_markers = int(remaining_marker_mask.sum())

    if n_remaining_markers > 0:
        examples = df.loc[
            remaining_marker_mask,
            [CHAPTER_COL, GOLD_COL]
        ].head(10)

        raise AssertionError(
            f"{n_remaining_markers} rows still contain <SOH> or <EOH> in Gold Text. Examples:\n"
            + examples.to_string(index=False)
        )

    # Check that Example Question was not modified.
    if QUESTION_COL in df.columns and before_questions is not None:
        if not df[QUESTION_COL].equals(before_questions):
            raise AssertionError("Example Question column was modified, but it should remain unchanged.")

    # Optional check: Gold Text must not be empty after cleaning.
    empty_gold_mask = df[GOLD_COL].fillna("").astype(str).str.strip().eq("")
    n_empty_gold = int(empty_gold_mask.sum())

    if n_empty_gold > 0:
        examples = df.loc[
            empty_gold_mask,
            [CHAPTER_COL, "theme", "block_id", GOLD_COL]
        ].head(10)

        raise AssertionError(
            f"{n_empty_gold} rows have empty Gold Text after cleaning. Examples:\n"
            + examples.to_string(index=False)
        )

    # Overwrite only test CSV

    df.to_csv(
        TEST_CSV_PATH,
        index=False,
        encoding="utf-8"
    )

    # Print summary and changed examples

    print("\nOverwrite complete.")
    print("File overwritten:", TEST_CSV_PATH)
    print("Rows:", len(df))
    print("Rows changed in Gold Text:", n_changed)
    print("Remaining <SOH>/<EOH> in Gold Text:", n_remaining_markers)
    print("Empty Gold Text rows after cleaning:", n_empty_gold)

    if n_changed > 0:
        preview = pd.DataFrame({
            "chapter": df.loc[changed_mask, CHAPTER_COL].head(10).tolist(),
            "before": before_gold[changed_mask].head(10).tolist(),
            "after": after_gold[changed_mask].head(10).tolist(),
        })

        preview["before"] = preview["before"].str[:500]
        preview["after"] = preview["after"].str[:500]

        print("\nChanged examples:")
        print(preview.to_string(index=False))
    else:
        print("\nNo headings found in Gold Text column.")

clean_hupd_gold_text()

# --- Inspect HUPD lengths ---
#
# Display the original parent/test token statistics.

def summarize_hupd():
    """Inspect HUPD lengths."""
    BASE_DIR = (DATA_ROOT / "hupd_dataset")

    TRAIN_JSONL = BASE_DIR / "rag_artifacts" / "hupd_parents.jsonl"
    TEST_CSV = BASE_DIR / "hupd_test_cases.csv"

    if not TRAIN_JSONL.exists():
        raise FileNotFoundError(f"Train JSONL not found: {TRAIN_JSONL}")

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")

    # Load HUPD knowledge base

    train_blocks = load_parent_jsonl_as_blocks(TRAIN_JSONL)

    train_df = pd.DataFrame(train_blocks)

    if train_df.empty:
        raise ValueError("hupd_parents.jsonl contains no records.")

    required_train_cols = ["chapter", "block_id", "text", "length_tokens"]
    missing_train = [c for c in required_train_cols if c not in train_df.columns]

    if missing_train:
        raise ValueError(f"Missing required train columns: {missing_train}")

    train_df["length_tokens"] = pd.to_numeric(
        train_df["length_tokens"],
        errors="coerce"
    )

    if train_df["length_tokens"].isna().any():
        n_missing = int(train_df["length_tokens"].isna().sum())
        raise ValueError(f"Train set contains {n_missing} invalid length_tokens values.")

    # Load HUPD test set

    test_df = pd.read_csv(TEST_CSV)

    if test_df.empty:
        raise ValueError("hupd_test_cases.csv contains no records.")

    required_test_cols = ["chapter", "Gold Text", "length_tokens", "block_id"]
    missing_test = [c for c in required_test_cols if c not in test_df.columns]

    if missing_test:
        raise ValueError(f"Missing required test columns: {missing_test}")

    test_df["length_tokens"] = pd.to_numeric(
        test_df["length_tokens"],
        errors="coerce"
    )

    if test_df["length_tokens"].isna().any():
        n_missing = int(test_df["length_tokens"].isna().sum())
        raise ValueError(f"Test set contains {n_missing} invalid length_tokens values.")

    # Statistics helper

    def token_stats(df: pd.DataFrame) -> dict:
        tokens = df["length_tokens"]

        return {
            "n": int(len(df)),
            "total_tokens": int(tokens.sum()),
            "median_tokens": float(tokens.median()),
            "average_tokens": float(tokens.mean()),
            "sd_tokens": float(tokens.std(ddof=1)) if len(tokens) > 1 else 0.0,
        }

    train_stats = token_stats(train_df)
    test_stats = token_stats(test_df)

    # Print raw statistics

    print("=" * 80)
    print("Knowledge base")
    print("=" * 80)
    print("Train parent blocks:", train_stats["n"])
    print("Total tokens:", train_stats["total_tokens"])
    print("Median tokens per chapter:", round(train_stats["median_tokens"], 2))
    print("Average tokens per chapter:", round(train_stats["average_tokens"], 2))
    print("SD tokens per chapter:", round(train_stats["sd_tokens"], 2))

    print("\n" + "=" * 80)
    print("Test set")
    print("=" * 80)
    print("Test cases:", test_stats["n"])
    print("Total tokens:", test_stats["total_tokens"])
    print("Median tokens per test case:", round(test_stats["median_tokens"], 2))
    print("Average tokens per test case:", round(test_stats["average_tokens"], 2))
    print("SD tokens per test case:", round(test_stats["sd_tokens"], 2))

    # Summary table for inspection

    summary_table = pd.DataFrame([
        {
            "Dataset split": "Knowledge base",
            "Statistic": r"\# Chapters",
            "Value": f"{train_stats['n']:,}",
        },
        {
            "Dataset split": "Knowledge base",
            "Statistic": "Total tokens",
            "Value": f"{train_stats['total_tokens']:,}",
        },
        {
            "Dataset split": "Knowledge base",
            "Statistic": "Median tokens per chapter",
            "Value": f"{train_stats['median_tokens']:.2f}",
        },
        {
            "Dataset split": "Knowledge base",
            "Statistic": "Average tokens per chapter",
            "Value": f"{train_stats['average_tokens']:.2f}",
        },
        {
            "Dataset split": "Knowledge base",
            "Statistic": "SD tokens per chapter",
            "Value": f"{train_stats['sd_tokens']:.2f}",
        },
        {
            "Dataset split": "Test set",
            "Statistic": r"\# Test cases",
            "Value": f"{test_stats['n']:,}",
        },
        {
            "Dataset split": "Test set",
            "Statistic": "Total tokens",
            "Value": f"{test_stats['total_tokens']:,}",
        },
        {
            "Dataset split": "Test set",
            "Statistic": "Median tokens per test case",
            "Value": f"{test_stats['median_tokens']:.2f}",
        },
        {
            "Dataset split": "Test set",
            "Statistic": "Average tokens per test case",
            "Value": f"{test_stats['average_tokens']:.2f}",
        },
        {
            "Dataset split": "Test set",
            "Statistic": "SD tokens per test case",
            "Value": f"{test_stats['sd_tokens']:.2f}",
        },
    ])

    print("\n" + "=" * 80)
    print("Summary table")
    print("=" * 80)
    display(summary_table)


summarize_hupd()

