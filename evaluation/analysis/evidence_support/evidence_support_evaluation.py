"""Evaluate retrieval effectiveness for Tender, LEDGAR, and HUPD.

Run from any working directory with `python evidence_support_evaluation.py`.
Select a dataset with `--dataset tender` (or ledgar/hupd), or select several
with `--datasets tender,ledgar`. Omit the option to evaluate all three.

Stages:
    1. Shared normalization, setup aggregation, and tables.
    2. Tender overall evaluation and the fixed hard-case comparison.
    3. LEDGAR evaluation from per-question CSV files.
    4. HUPD evaluation from per-question CSV files.
    5. Run all stages from the local project data folders.

The root contains tender_dataset, ledgar_dataset, and hupd_dataset. Each stage
reads its results/retrieval_analysis folder. Best-system selections and
stored generation scores are fixed inputs copied from the documented experiment results.
Script runs print complete tables.
"""

import argparse
from io import StringIO
from pathlib import Path
import re
import subprocess
import sys
import warnings


PROJECT_ROOT = Path(__file__).resolve().parents[3]


DATASETS = ("tender", "ledgar", "hupd")


def normalize_datasets(datasets):
    """Validate dataset names."""
    if isinstance(datasets, str):
        datasets = datasets.split(",")
    selected = tuple(name.strip().lower() for name in datasets)
    if (
        not selected
        or len(selected) != len(set(selected))
        or set(selected) - set(DATASETS)
    ):
        raise ValueError("Datasets must contain unique names from tender,ledgar,hupd.")
    return selected


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset", "--datasets",
        dest="datasets",
        default=DATASETS,
        metavar="NAME[,NAME...]",
        help="Evaluate only these datasets: tender, ledgar, hupd (default: all three).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root containing the three local dataset folders.",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install dependencies once before running.",
    )
    args = parser.parse_args(argv)
    try:
        args.datasets = normalize_datasets(args.datasets)
    except ValueError as exc:
        parser.error(str(exc))
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.install:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "numpy",
                "pandas",
                "openpyxl",
                "ipython",
                "jinja2",
            ],
            check=True,
        )

import numpy as np
import pandas as pd

from pandas.io.formats.style import Styler

try:
    from IPython import get_ipython
    from IPython.display import display as _rich_display
except ImportError:
    get_ipython = None
    _rich_display = None


def display(*objects):
    """Print complete tables in scripts and use rich tables in interactive IPython sessions."""
    if (
        __name__ != "__main__"
        and get_ipython is not None
        and getattr(get_ipython(), "kernel", None) is not None
    ):
        _rich_display(*objects)
        return

    for obj in objects:
        if isinstance(obj, Styler):
            with pd.option_context(
                "styler.render.max_elements", max(1, obj.data.size)
            ):
                print(
                    obj.to_string(
                        delimiter=" | ",
                        max_rows=len(obj.data.index),
                        max_columns=len(obj.data.columns),
                    )
                )
        elif isinstance(obj, pd.DataFrame):
            print(
                obj.to_string(
                    max_rows=None,
                    max_cols=None,
                    max_colwidth=None,
                    line_width=None,
                )
            )
        elif isinstance(obj, pd.Series):
            with pd.option_context("display.max_colwidth", None):
                print(obj.to_string(max_rows=None))
        else:
            print(obj)


# 1. Shared LEDGAR/HUPD helpers; Tender keeps its original normalization rules.

RETRIEVAL_COLUMNS = (
    "dataset",
    "method",
    "top_k",
    "generator_model",
    "test_case_id",
    "faithfulness",
    "context_recall",
)


def print_table(title, df):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    if df.empty:
        print("No data available.")
    else:
        display(df)


def natural_sort_key(value):
    value = str(value)
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split("(\\d+)", value)
    ]


def normalize_column_name(value):
    return str(value).strip()


def normalize_text(value):
    value = "" if pd.isna(value) else str(value)
    value = value.lower().strip()
    return re.sub("[^a-z0-9]+", "", value)


def normalize_generator(value):
    value = "" if pd.isna(value) else str(value).strip()
    value = value.replace("\\\\", "/")
    value = value.split("/")[-1]
    return normalize_text(value)


def normalize_tender_generator(value):
    """Return the canonical Hugging Face model ID for Tender generators."""
    if pd.isna(value):
        return ""
    compact = re.sub("[^a-z0-9]+", "", str(value).casefold())
    aliases = [
        ("llama318binstruct", "meta-llama/Llama-3.1-8B-Instruct"),
        ("llama3170binstruct", "meta-llama/Llama-3.1-70B-Instruct"),
        ("qwen257binstruct", "Qwen/Qwen2.5-7B-Instruct"),
        ("qwen2532binstruct", "Qwen/Qwen2.5-32B-Instruct"),
        ("qwen2572binstruct", "Qwen/Qwen2.5-72B-Instruct"),
        ("qwen3235ba22binstruct", "Qwen/Qwen3-235B-A22B-Instruct-2507"),
        ("qwen3235binstruct", "Qwen/Qwen3-235B-A22B-Instruct-2507"),
        ("qwen31235binstruct", "Qwen/Qwen3-235B-A22B-Instruct-2507"),
    ]
    for token, canonical_name in aliases:
        if token in compact:
            return canonical_name
    return compact



def display_generator(value):
    value = "" if pd.isna(value) else str(value).strip()
    value = value.replace("\\\\", "/")
    return value.split("/")[-1]


def parse_numeric_series(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})
        .pipe(pd.to_numeric, errors="coerce")
    )


def canonicalize_question_ids(series):
    out = series.astype("string").str.strip()
    out = out.mask(out.str.lower().isin(["", "nan", "none", "null", "<na>"]))
    numeric_like = out.str.fullmatch("[+-]?\\d+(?:\\\\.0+)?", na=False)
    out.loc[numeric_like] = out.loc[numeric_like].map(
        lambda value: str(int(float(value)))
    )
    return out


# Average question scores within each setup, then round to four decimals.
def aggregate_to_system_level(per_question_df):
    grouping_columns = [
        "dataset_normalized",
        "method_normalized",
        "generator_normalized",
        "top_k",
    ]
    summary_df = (
        per_question_df.groupby(grouping_columns, dropna=False)
        .agg(
            dataset=("dataset", "first"),
            method=("method", "first"),
            generator_model=("generator_model", "first"),
            faithfulness=("faithfulness", "mean"),
            context_recall=("context_recall", "mean"),
            grounding_mean=("grounding_mean", "mean"),
            n_rows=("test_case_id", "size"),
            n_questions=("test_case_id", "nunique"),
        )
        .reset_index()
    )
    summary_df[["faithfulness", "context_recall", "grounding_mean"]] = summary_df[
        ["faithfulness", "context_recall", "grounding_mean"]
    ].round(4)
    summary_df["generator_display"] = summary_df["generator_model"].map(
        display_generator
    )
    return summary_df


# Attach retrieval scores to the supplied generation-best configurations.
def lookup_best_system_retrieval_scores(
    metric_df, best_systems, title, *, dataset_name
):
    rows = []
    for config in best_systems:
        method_normalized = normalize_text(config["method"])
        generator_normalized = normalize_generator(config["generator_model"])
        top_k = int(config["top_k"])
        matched = metric_df[
            metric_df["dataset_normalized"].eq(normalize_text(dataset_name))
            & metric_df["method_normalized"].eq(method_normalized)
            & metric_df["generator_normalized"].eq(generator_normalized)
            & metric_df["top_k"].astype("Int64").eq(top_k)
        ].copy()
        if matched.empty:
            faithfulness = np.nan
            context_recall = np.nan
            grounding_mean = np.nan
            n_rows = 0
            n_questions = 0
            was_matched = False
        else:
            faithfulness = matched["faithfulness"].mean()
            context_recall = matched["context_recall"].mean()
            grounding_mean = matched["grounding_mean"].mean()
            n_rows = int(matched["n_rows"].sum())
            n_questions = int(matched["n_questions"].max())
            was_matched = True
        rows.append(
            {
                "method": config["method_label"],
                "retrieval_method": config["method"],
                "generator_model": display_generator(config["generator_model"]),
                "top_k": top_k,
                "MMR": config["MMR"],
                "rouge1_f1": config["rouge1_f1"],
                "rouge2_f1": config["rouge2_f1"],
                "rougeL_f1": config["rougeL_f1"],
                "bertscore_f1": config["bertscore_f1"],
                "simcse_f1": config["simcse_f1"],
                "ecs_bge_m3": config["ecs_bge_m3"],
                "matcha_f1": config["matcha_f1"],
                "faithfulness": faithfulness,
                "context_recall": context_recall,
                "grounding_mean": grounding_mean,
                "n_rows": n_rows,
                "n_questions": n_questions,
                "matched": was_matched,
            }
        )
    result = pd.DataFrame(rows)
    metric_columns = [
        "MMR",
        "rouge1_f1",
        "rouge2_f1",
        "rougeL_f1",
        "bertscore_f1",
        "simcse_f1",
        "ecs_bge_m3",
        "matcha_f1",
        "faithfulness",
        "context_recall",
        "grounding_mean",
    ]
    result[metric_columns] = result[metric_columns].round(4)
    print_table(title, result)
    return result



# Average setup-level scores into method/top-k summaries.
def print_metric_table_by_method_topk(df, metric, title, mean_column_name=None):
    if mean_column_name is None:
        mean_column_name = f"mean_{metric}"
    metric_df = df[["dataset", "method", "top_k", "generator_display", metric]].copy()
    metric_df = metric_df.dropna(subset=["top_k", metric])
    metric_summary = (
        metric_df.groupby(["dataset", "method", "top_k"], dropna=False)
        .agg(
            metric_value=(metric, "mean"),
            n_generators=("generator_display", "nunique"),
            n_rows=("generator_display", "size"),
        )
        .reset_index()
    )
    metric_summary["metric_value"] = metric_summary["metric_value"].round(4)
    output_tables = {}
    for dataset_name, dataset_part in metric_summary.groupby("dataset", dropna=False):
        table = dataset_part.pivot(
            index="method", columns="top_k", values="metric_value"
        ).reset_index()
        top_k_columns = [column for column in table.columns if column != "method"]
        table[mean_column_name] = table[top_k_columns].mean(axis=1).round(4)
        table = table.sort_values(mean_column_name, ascending=False).reset_index(
            drop=True
        )
        table = table.rename(
            columns={column: f"top_k_{int(column)}" for column in top_k_columns}
        )
        print_table(f"{title} - {dataset_name}", table)
        output_tables[dataset_name] = table
    return output_tables


# 2.1 Tender: overall retrieval evaluation
def evaluate_tender(data_root):
    """Evaluate all Tender rows and match the fixed MMR-best systems."""

    RESULTS_DIR = data_root / "tender_dataset/results/retrieval_analysis"
    # Fixed selections from the generation-quality analysis; no reselection here.
    MMR_BEST_OVERALL = [
        {
            "method_label": "HPC",
            "method": "hybrid_parent_child",
            "generator_model": "meta-llama/Llama-3.1-70B-Instruct",
            "top_k": 15,
            "mmr": 3.86,
        },
        {
            "method_label": "SP",
            "method": "sparse_parent",
            "generator_model": "Qwen/Qwen2.5-72B-Instruct",
            "top_k": 20,
            "mmr": 6.86,
        },
        {
            "method_label": "DP",
            "method": "dense_parent",
            "generator_model": "meta-llama/Llama-3.1-70B-Instruct",
            "top_k": 20,
            "mmr": 7.43,
        },
        {
            "method_label": "DPC",
            "method": "dense_parent_child",
            "generator_model": "meta-llama/Llama-3.1-70B-Instruct",
            "top_k": 15,
            "mmr": 7.57,
        },
        {
            "method_label": "SPC",
            "method": "sparse_parent_child",
            "generator_model": "meta-llama/Llama-3.1-70B-Instruct",
            "top_k": 10,
            "mmr": 14.14,
        },
        {
            "method_label": "SPL",
            "method": "sparse_parent_L",
            "generator_model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
            "top_k": 10,
            "mmr": 106.29,
        },
    ]
    METHOD_LABELS = {
        "dense_parent": "DP",
        "sparse_parent": "SP",
        "dense_parent_child": "DPC",
        "hybrid_parent_child": "HPC",
        "sparse_parent_child": "SPC",
        "sparse_parent_L": "SPL",
    }

    # Normalize Tender method aliases while retaining BM25L as a separate method.
    def normalize_method(value):
        if pd.isna(value):
            return ""
        return re.sub("[^a-z0-9]+", "", str(value).casefold())

    # Match Tender model aliases, vendor prefixes, and quantization suffixes.

    # Load only matching per-question CSV files in the Tender folder.
    files = sorted(
        (
            path
            for path in RESULTS_DIR.iterdir()
            if path.is_file()
            and path.name.startswith("per_question_retrieval_metrics_")
            and ("faithfulness_context_recall" in path.name)
            and (path.suffix.lower() == ".csv")
        )
    )
    if not files:
        raise RuntimeError("No per-question retrieval metric CSV files found.")
    print(f"Per-question files found: {len(files)}")
    for path in files:
        print("-", path.name)
    required_columns = list(RETRIEVAL_COLUMNS)
    frames = []
    for path in files:
        temp = pd.read_csv(path)
        temp.columns = [str(column).strip() for column in temp.columns]
        missing = [column for column in required_columns if column not in temp.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in {path.name}")
        temp["source_file"] = path.name
        frames.append(temp)
    # Combine files, keep Tender rows, and normalize numeric fields.
    df = pd.concat(frames, ignore_index=True)
    df = df[df["dataset"].astype(str).str.strip().str.casefold().eq("tender")].copy()
    if df.empty:
        raise RuntimeError("No Tender rows found.")
    df["top_k"] = pd.to_numeric(df["top_k"], errors="coerce").astype("Int64")
    df["test_case_id"] = pd.to_numeric(df["test_case_id"], errors="coerce").astype(
        "Int64"
    )
    for metric in ["faithfulness", "context_recall"]:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    # Calculate grounding per question before averaging.
    df["grounding_mean"] = df[["faithfulness", "context_recall"]].mean(axis=1)
    df["method_normalized"] = df["method"].map(normalize_method)
    df["generator_normalized"] = df["generator_model"].map(normalize_tender_generator)
    print("\nLoaded per-question rows:", len(df))
    print("Unique test cases:", df["test_case_id"].nunique())
    generator_check = (
        df[["generator_model", "generator_normalized"]]
        .drop_duplicates()
        .sort_values(["generator_normalized", "generator_model"])
        .reset_index(drop=True)
    )
    print("\nGenerator normalization:")
    display(generator_check)

    # Match the fixed configurations and average their per-question scores directly.
    def get_mmr_best_retrieval_scores(df, configs):
        rows = []
        for config in configs:
            method_norm = normalize_method(config["method"])
            generator_norm = normalize_tender_generator(config["generator_model"])
            top_k = int(config["top_k"])
            matched = df[
                (df["method_normalized"] == method_norm)
                & (df["generator_normalized"] == generator_norm)
                & (df["top_k"] == top_k)
            ].copy()
            rows.append(
                {
                    "Method": config["method_label"],
                    "Generator": config["generator_model"],
                    "Top-k": top_k,
                    "MMR": config["mmr"],
                    "Faithfulness": (
                        matched["faithfulness"].mean() if not matched.empty else np.nan
                    ),
                    "Context Recall": (
                        matched["context_recall"].mean()
                        if not matched.empty
                        else np.nan
                    ),
                    "Grounding Mean": (
                        matched["grounding_mean"].mean()
                        if not matched.empty
                        else np.nan
                    ),
                    "Questions": (
                        matched["test_case_id"].nunique() if not matched.empty else 0
                    ),
                    "Rows": len(matched),
                    "Matched": not matched.empty,
                }
            )
        result = pd.DataFrame(rows)
        result = result.sort_values("MMR", ascending=True).reset_index(drop=True)
        return result

    mmr_best_results = get_mmr_best_retrieval_scores(df, MMR_BEST_OVERALL)
    print("\n" + "=" * 100)
    print("Retrieval metrics for MMR-best individual configurations — OVERALL")
    print("=" * 100)
    display(
        mmr_best_results.style.format(
            {
                "MMR": "{:.2f}",
                "Faithfulness": "{:.4f}",
                "Context Recall": "{:.4f}",
                "Grounding Mean": "{:.4f}",
            }
        )
    )
    # Keep the original warnings and diagnostics for unmatched configurations.
    unmatched = mmr_best_results[mmr_best_results["Matched"] == False]
    if not unmatched.empty:
        print("\nWARNING: Some MMR-best configurations could not be matched.")
        display(unmatched)
        print("\nAvailable configurations:")
        display(
            df[["method", "generator_model", "generator_normalized", "top_k"]]
            .drop_duplicates()
            .sort_values(["method", "generator_normalized", "top_k"])
            .reset_index(drop=True)
        )



    # Tender tables average question rows directly, rather than rounded setup means.
    def method_topk_table(df, metric):
        summary = (
            df.groupby(["method", "top_k"], dropna=False)[metric].mean().reset_index()
        )
        table = summary.pivot(
            index="method", columns="top_k", values=metric
        ).reset_index()
        table["method"] = table["method"].map(lambda x: METHOD_LABELS.get(x, x))
        top_k_columns = [column for column in table.columns if column != "method"]
        table[f"mean_{metric}"] = table[top_k_columns].mean(axis=1)
        table = table.sort_values(f"mean_{metric}", ascending=False).reset_index(
            drop=True
        )
        table = table.rename(
            columns={column: f"top_k_{int(column)}" for column in top_k_columns}
        )
        return table

    faithfulness_table = method_topk_table(df, "faithfulness")
    print("\n" + "=" * 100)
    print("Faithfulness by retrieval method and top-k")
    print("=" * 100)
    display(faithfulness_table.style.format(precision=4))
    context_recall_table = method_topk_table(df, "context_recall")
    print("\n" + "=" * 100)
    print("Context recall by retrieval method and top-k")
    print("=" * 100)
    display(context_recall_table.style.format(precision=4))
    grounding_table = method_topk_table(df, "grounding_mean")
    print("\n" + "=" * 100)
    print("Grounding mean by retrieval method and top-k")
    print("=" * 100)
    display(grounding_table.style.format(precision=4))
    print("\nFinished retrieval metric evaluation.")
    print("Per-question files used:", len(files))
    print("Per-question rows used:", len(df))
    print("Unique questions:", df["test_case_id"].nunique())
    print("Top-k values:", sorted(df["top_k"].dropna().astype(int).unique()))

    return locals()


# 2.2 Tender: hard-case DP/DPC comparison
def evaluate_tender_hard(data_root):
    """Compare the two fixed Qwen3-235B configurations on the 16 hard cases."""

    RESULTS_DIR = data_root / "tender_dataset/results/retrieval_analysis"
    HARD_TEST_CASE_IDS = {6, 9, 10, 12, 13, 14, 15, 17, 18, 20, 22, 23, 25, 26, 28, 30}
    # Preserve the supplied generation scores for the two hard-case systems.
    TARGET_SYSTEMS = [
        {
            "method": "dense_parent_child",
            "generator_model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
            "top_k": 10,
            "n_rows": 16,
            "n_test_cases": 16,
            "semantic_mean": 0.7503,
            "rouge1_f1": 0.3469,
            "rouge2_f1": 0.0662,
            "rougeL_f1": 0.1579,
            "bertscore_f1": 0.7005,
            "simcse_f1": 0.5839,
            "matcha_f1": 0.9603,
        },
        {
            "method": "dense_parent",
            "generator_model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
            "top_k": 10,
            "n_rows": 16,
            "n_test_cases": 16,
            "semantic_mean": 0.7454,
            "rouge1_f1": 0.3468,
            "rouge2_f1": 0.0739,
            "rougeL_f1": 0.1553,
            "bertscore_f1": 0.6949,
            "simcse_f1": 0.6002,
            "matcha_f1": 0.9546,
        },
    ]

    # Method matching retains the original whitespace/underscore normalization.
    # Generator matching uses the same CSV/Hugging Face aliases as the overall analysis.
    def normalize_text(value):
        value = "" if pd.isna(value) else str(value)
        value = value.lower().strip()
        value = re.sub("[\\s_]+", "", value)
        return value

    def find_per_question_files(results_dir: Path):
        if not results_dir.exists():
            raise FileNotFoundError(f"Folder does not exist: {results_dir}")
        return sorted(
            [
                path
                for path in results_dir.iterdir()
                if path.is_file()
                and path.name.startswith("per_question_retrieval_metrics_")
                and ("faithfulness_context_recall" in path.name)
                and (path.suffix.lower() == ".csv")
            ]
        )

    # Load hard-case inputs using the original validation and numeric conversion.
    def load_per_question_files(per_question_files):
        frames = []
        required_columns = list(RETRIEVAL_COLUMNS)
        for file_path in per_question_files:
            df = pd.read_csv(file_path)
            missing_columns = [
                column for column in required_columns if column not in df.columns
            ]
            if missing_columns:
                raise ValueError(
                    f"Missing columns {missing_columns} in file: {file_path}"
                )
            df["source_file"] = file_path.name
            frames.append(df)
        if not frames:
            raise RuntimeError("No per-question retrieval metric files found.")
        combined = pd.concat(frames, ignore_index=True)
        combined["top_k"] = pd.to_numeric(combined["top_k"], errors="coerce").astype(
            "Int64"
        )
        combined["test_case_id"] = pd.to_numeric(
            combined["test_case_id"], errors="coerce"
        ).astype("Int64")
        combined["faithfulness"] = pd.to_numeric(
            combined["faithfulness"], errors="coerce"
        )
        combined["context_recall"] = pd.to_numeric(
            combined["context_recall"], errors="coerce"
        )
        combined["grounding_mean"] = combined[["faithfulness", "context_recall"]].mean(
            axis=1
        )
        combined["method_normalized"] = combined["method"].map(normalize_text)
        combined["generator_normalized"] = combined["generator_model"].map(
            normalize_tender_generator
        )
        return combined

    per_question_files = find_per_question_files(RESULTS_DIR)
    print("Per-question retrieval metric files found:")
    for file in per_question_files:
        print("-", file.name)
    per_question_df = load_per_question_files(per_question_files)
    # Filter by the fixed 16 hard-case IDs before selecting the target systems.
    hard_retrieval_df = per_question_df[
        per_question_df["test_case_id"].isin(HARD_TEST_CASE_IDS)
    ].copy()
    print("\nHard-case retrieval rows available:", len(hard_retrieval_df))
    rows = []
    for target in TARGET_SYSTEMS:
        method_norm = normalize_text(target["method"])
        generator_norm = normalize_tender_generator(target["generator_model"])
        top_k = int(target["top_k"])
        matched = hard_retrieval_df[
            (hard_retrieval_df["method_normalized"] == method_norm)
            & (hard_retrieval_df["generator_normalized"] == generator_norm)
            & (hard_retrieval_df["top_k"].astype("Int64") == top_k)
        ].copy()
        row = {
            "method": target["method"],
            "generator_model": target["generator_model"],
            "top_k": top_k,
            "n_rows": target["n_rows"],
            "n_test_cases": target["n_test_cases"],
            "semantic_mean": target["semantic_mean"],
            "rouge1_f1": target["rouge1_f1"],
            "rouge2_f1": target["rouge2_f1"],
            "rougeL_f1": target["rougeL_f1"],
            "bertscore_f1": target["bertscore_f1"],
            "simcse_f1": target["simcse_f1"],
            "matcha_f1": target["matcha_f1"],
        }
        if matched.empty:
            row.update(
                {
                    "faithfulness": np.nan,
                    "context_recall": np.nan,
                    "grounding_mean": np.nan,
                    "retrieval_n_rows": 0,
                    "retrieval_n_test_cases": 0,
                    "matched": False,
                }
            )
        else:
            row.update(
                {
                    "faithfulness": matched["faithfulness"].mean(),
                    "context_recall": matched["context_recall"].mean(),
                    "grounding_mean": matched["grounding_mean"].mean(),
                    "retrieval_n_rows": len(matched),
                    "retrieval_n_test_cases": matched["test_case_id"].nunique(),
                    "matched": True,
                }
            )
        rows.append(row)
    result_table = pd.DataFrame(rows)
    score_columns = [
        "semantic_mean",
        "rouge1_f1",
        "rouge2_f1",
        "rougeL_f1",
        "bertscore_f1",
        "simcse_f1",
        "matcha_f1",
        "faithfulness",
        "context_recall",
        "grounding_mean",
    ]
    result_table[score_columns] = result_table[score_columns].round(4)
    result_table = result_table[
        [
            "method",
            "generator_model",
            "top_k",
            "n_rows",
            "n_test_cases",
            "semantic_mean",
            "rouge1_f1",
            "rouge2_f1",
            "rougeL_f1",
            "bertscore_f1",
            "simcse_f1",
            "matcha_f1",
            "faithfulness",
            "context_recall",
            "grounding_mean",
            "retrieval_n_rows",
            "retrieval_n_test_cases",
            "matched",
        ]
    ]
    print("\nSelected hard-case systems with generation and retrieval metrics:")
    display(result_table)
    # Build the compact comparison table without changing the stored generation scores.
    compact_table = result_table[
        [
            "method",
            "generator_model",
            "top_k",
            "n_rows",
            "n_test_cases",
            "semantic_mean",
            "rouge1_f1",
            "rouge2_f1",
            "rougeL_f1",
            "bertscore_f1",
            "simcse_f1",
            "matcha_f1",
            "faithfulness",
            "context_recall",
            "grounding_mean",
        ]
    ].copy()
    print("\nCompact table:")
    display(compact_table)
    # Keep the original warnings and diagnostics for unmatched configurations.
    unmatched = result_table[result_table["matched"] == False]
    if not unmatched.empty:
        print(
            "\nWarning: At least one selected system was not found in the retrieval metric files."
        )
        display(unmatched)
        print("\nAvailable method-generator-top_k combinations for hard cases:")
        display(
            hard_retrieval_df[["method", "generator_model", "top_k"]]
            .drop_duplicates()
            .sort_values(["method", "generator_model", "top_k"])
            .reset_index(drop=True)
        )

    return locals()


# 3. LEDGAR: per-question CSV evaluation
def evaluate_ledgar(data_root):
    """Evaluate LEDGAR setups and match the fixed generation-best systems."""

    RESULTS_DIR = data_root / "ledgar_dataset/results/retrieval_analysis"
    DATASET_NAME = "ledgar"
    # Preselected generation-best configurations and scores.
    BEST_SYSTEMS = [
        {
            "method_label": "HPC",
            "method": "hybrid_parent_child",
            "generator_model": "Qwen/Qwen2.5-32B-Instruct",
            "top_k": 10,
            "MMR": 4.5714,
            "rouge1_f1": 0.7661,
            "rouge2_f1": 0.5983,
            "rougeL_f1": 0.6326,
            "bertscore_f1": 0.8309,
            "simcse_f1": 0.8044,
            "ecs_bge_m3": 0.9254,
            "matcha_f1": 0.8826,
        },
        {
            "method_label": "SPC",
            "method": "sparse_parent_child",
            "generator_model": "Qwen/Qwen2.5-32B-Instruct",
            "top_k": 20,
            "MMR": 6.0,
            "rouge1_f1": 0.7658,
            "rouge2_f1": 0.5963,
            "rougeL_f1": 0.6271,
            "bertscore_f1": 0.8276,
            "simcse_f1": 0.7985,
            "ecs_bge_m3": 0.9272,
            "matcha_f1": 0.8794,
        },
        {
            "method_label": "DP",
            "method": "dense_parent",
            "generator_model": "Qwen/Qwen2.5-32B-Instruct",
            "top_k": 20,
            "MMR": 10.5714,
            "rouge1_f1": 0.7591,
            "rouge2_f1": 0.5813,
            "rougeL_f1": 0.6122,
            "bertscore_f1": 0.8229,
            "simcse_f1": 0.7947,
            "ecs_bge_m3": 0.9253,
            "matcha_f1": 0.8815,
        },
        {
            "method_label": "SP",
            "method": "sparse_parent",
            "generator_model": "Qwen/Qwen2.5-32B-Instruct",
            "top_k": 5,
            "MMR": 17.1429,
            "rouge1_f1": 0.7501,
            "rouge2_f1": 0.5741,
            "rougeL_f1": 0.6016,
            "bertscore_f1": 0.8195,
            "simcse_f1": 0.7834,
            "ecs_bge_m3": 0.9244,
            "matcha_f1": 0.8666,
        },
        {
            "method_label": "DPC",
            "method": "dense_parent_child",
            "generator_model": "Qwen/Qwen2.5-32B-Instruct",
            "top_k": 20,
            "MMR": 18.7143,
            "rouge1_f1": 0.7476,
            "rouge2_f1": 0.5643,
            "rougeL_f1": 0.5952,
            "bertscore_f1": 0.8169,
            "simcse_f1": 0.7856,
            "ecs_bge_m3": 0.9213,
            "matcha_f1": 0.8733,
        },
        {
            "method_label": "SPL",
            "method": "sparse_parent_L",
            "generator_model": "Qwen/Qwen2.5-32B-Instruct",
            "top_k": 20,
            "MMR": 57.2857,
            "rouge1_f1": 0.711,
            "rouge2_f1": 0.5088,
            "rougeL_f1": 0.5407,
            "bertscore_f1": 0.8,
            "simcse_f1": 0.7647,
            "ecs_bge_m3": 0.9091,
            "matcha_f1": 0.8499,
        },
    ]
    PER_QUESTION_REQUIRED_COLUMNS = list(RETRIEVAL_COLUMNS)

    # Find eligible LEDGAR CSV files recursively.
    def find_csv_files(results_dir):
        results_dir = Path(results_dir)
        if not results_dir.exists():
            raise FileNotFoundError(f"Folder does not exist: {results_dir}")
        return sorted(
            [
                path
                for path in results_dir.rglob("*.csv")
                if path.is_file()
                and "per_question" in path.name.lower()
                and ("summary" not in path.name.lower())
            ],
            key=lambda path: natural_sort_key(str(path)),
        )

    # Retry a single-column CSV with a semicolon delimiter.
    def read_csv_flexible(file_path):
        df = pd.read_csv(file_path)
        if len(df.columns) == 1:
            retry = pd.read_csv(file_path, sep=";")
            if len(retry.columns) > 1:
                df = retry
        return df

    # Validate LEDGAR rows, distinguish BM25L, and reject conflicting duplicates.
    def load_per_question_retrieval_csvs(files, dataset_name, results_dir):
        frames = []
        skipped = []
        for file_path in files:
            try:
                temp = read_csv_flexible(file_path)
            except Exception as exc:
                skipped.append((file_path, f"read error: {exc}"))
                continue
            temp.columns = [normalize_column_name(column) for column in temp.columns]
            missing_columns = [
                column
                for column in PER_QUESTION_REQUIRED_COLUMNS
                if column not in temp.columns
            ]
            if missing_columns:
                skipped.append(
                    (file_path, f"missing required columns: {missing_columns}")
                )
                continue
            temp = temp[PER_QUESTION_REQUIRED_COLUMNS].copy()
            if "sparse_parent_bm25l" in file_path.name.lower():
                temp["method"] = "sparse_parent_L"
            dataset_mask = (
                temp["dataset"]
                .astype("string")
                .str.strip()
                .str.lower()
                .eq(str(dataset_name).strip().lower())
            )
            temp = temp.loc[dataset_mask].copy()
            if temp.empty:
                skipped.append((file_path, f"no rows for dataset '{dataset_name}'"))
                continue
            temp["source_file"] = str(file_path.relative_to(results_dir))
            frames.append(temp)
        if not frames:
            print("\nSkipped CSV files:")
            for file_path, reason in skipped:
                print(f"- {file_path}: {reason}")
            raise RuntimeError(
                "No CSV file containing the required LEDGAR per-question retrieval columns was found."
            )
        combined = pd.concat(frames, ignore_index=True, sort=False)
        for column in ["dataset", "method", "generator_model"]:
            combined[column] = combined[column].astype("string").str.strip()
        combined["test_case_id"] = canonicalize_question_ids(combined["test_case_id"])
        combined["top_k"] = parse_numeric_series(combined["top_k"]).astype("Int64")
        for metric in ["faithfulness", "context_recall"]:
            combined[metric] = parse_numeric_series(combined[metric])
        required_non_null = list(RETRIEVAL_COLUMNS)
        rows_before_drop = len(combined)
        # Require both retrieval metrics and all setup/question identifiers.
        combined = combined.dropna(subset=required_non_null).copy()
        rows_dropped = rows_before_drop - len(combined)
        if rows_dropped:
            warnings.warn(
                f"Removed {rows_dropped} rows with missing identifiers or retrieval metric values."
            )
        combined["dataset_normalized"] = combined["dataset"].map(normalize_text)
        combined["method_normalized"] = combined["method"].map(normalize_text)
        combined["generator_normalized"] = combined["generator_model"].map(
            normalize_generator
        )
        combined["grounding_mean"] = combined[["faithfulness", "context_recall"]].mean(
            axis=1
        )
        # Remove identical copies; differing scores for the same question remain errors.
        exact_duplicate_columns = [
            "dataset_normalized",
            "method_normalized",
            "top_k",
            "generator_normalized",
            "test_case_id",
            "faithfulness",
            "context_recall",
        ]
        rows_before_deduplication = len(combined)
        combined = combined.drop_duplicates(subset=exact_duplicate_columns).reset_index(
            drop=True
        )
        print(
            "\nPer-question rows before exact de-duplication:",
            rows_before_deduplication,
        )
        print("Per-question rows after exact de-duplication:", len(combined))
        setup_question_columns = [
            "dataset_normalized",
            "method_normalized",
            "top_k",
            "generator_normalized",
            "test_case_id",
        ]
        conflicting_mask = combined.duplicated(
            subset=setup_question_columns, keep=False
        )
        if conflicting_mask.any():
            conflicts = (
                combined.loc[
                    conflicting_mask,
                    setup_question_columns
                    + ["faithfulness", "context_recall", "source_file"],
                ]
                .sort_values(setup_question_columns)
                .reset_index(drop=True)
            )
            print_table("Conflicting duplicate setup-question rows", conflicts)
            raise ValueError(
                "Different retrieval scores were found for the same dataset, method, generator, top-k, and question."
            )
        loaded_files = (
            combined[["source_file"]]
            .drop_duplicates()
            .sort_values("source_file")
            .reset_index(drop=True)
        )
        print_table("Loaded per-question retrieval CSV files", loaded_files)
        print(
            f"\nSkipped CSV files that were not usable per-question LEDGAR retrieval files: {len(skipped)}"
        )
        return combined

    csv_files = find_csv_files(RESULTS_DIR)
    print("Per-question CSV files found recursively:", len(csv_files))
    for file_path in csv_files:
        print("-", file_path.relative_to(RESULTS_DIR))
    per_question_df = load_per_question_retrieval_csvs(
        files=csv_files, dataset_name=DATASET_NAME, results_dir=RESULTS_DIR
    )
    # Aggregate available questions to setups before building the overview tables.
    summary_df = aggregate_to_system_level(per_question_df)
    print("\nLoaded LEDGAR per-question rows:", len(per_question_df))
    print("Aggregated LEDGAR system rows:", len(summary_df))
    # Match the supplied generation-best configurations to the retrieval results.
    best_system_grounding = lookup_best_system_retrieval_scores(
        metric_df=summary_df,
        best_systems=BEST_SYSTEMS,
        title="Generation and retrieval metrics for the best individual LEDGAR systems",
        dataset_name=DATASET_NAME,
    )
    # Report faithfulness, context recall, and grounding separately.
    faithfulness_tables_overall = print_metric_table_by_method_topk(
        df=summary_df,
        metric="faithfulness",
        title="Faithfulness by retrieval method and top-k - OVERALL",
        mean_column_name="mean_faithfulness",
    )
    context_recall_tables_overall = print_metric_table_by_method_topk(
        df=summary_df,
        metric="context_recall",
        title="Context recall by retrieval method and top-k - OVERALL",
        mean_column_name="mean_context_recall",
    )
    grounding_tables_overall = print_metric_table_by_method_topk(
        df=summary_df,
        metric="grounding_mean",
        title="Grounding mean by retrieval method and top-k - OVERALL",
        mean_column_name="mean_grounding",
    )
    # Keep the original warnings and diagnostics for unmatched configurations.
    unmatched = best_system_grounding[best_system_grounding["matched"] == False].copy()
    if not unmatched.empty:
        print_table("Warning: unmatched selected best systems", unmatched)
        print_table(
            "Available method-generator-top-k combinations",
            summary_df[["method", "generator_display", "top_k", "n_questions"]]
            .drop_duplicates()
            .sort_values(["method", "generator_display", "top_k"])
            .reset_index(drop=True),
        )
    print(
        "\nFinished retrieval metric evaluation for the selected best LEDGAR systems."
    )
    print("Per-question CSV files inspected:", len(csv_files))
    print("Per-question retrieval rows used:", len(per_question_df))
    print("Aggregated system rows:", len(summary_df))
    print("\nMethods:")
    print(sorted(summary_df["method"].dropna().unique(), key=natural_sort_key))
    print("\nGenerators:")
    print(
        sorted(summary_df["generator_display"].dropna().unique(), key=natural_sort_key)
    )
    print("\nTop-k values:")
    print(sorted(summary_df["top_k"].dropna().astype(int).unique()))

    return locals()


# 4. HUPD: CSV-text Excel evaluation
def evaluate_hupd(data_root):
    """Evaluate HUPD setups and match the fixed generation-best systems."""

    RESULTS_DIR = data_root / "hupd_dataset/results/retrieval_analysis"
    DATASET_NAME = "hupd"
    # Preselected generation-best configurations and scores.
    BEST_SYSTEMS = [
        {
            "method_label": "SPC",
            "method": "sparse_parent_child",
            "generator_model": "Qwen/Qwen2.5-72B-Instruct",
            "top_k": 15,
            "MMR": 1.14,
            "rouge1_f1": 0.5762,
            "rouge2_f1": 0.2936,
            "rougeL_f1": 0.3352,
            "bertscore_f1": 0.7659,
            "simcse_f1": 0.7073,
            "ecs_bge_m3": 0.8713,
            "matcha_f1": 0.9389,
        },
        {
            "method_label": "HPC",
            "method": "hybrid_parent_child",
            "generator_model": "Qwen/Qwen2.5-72B-Instruct",
            "top_k": 15,
            "MMR": 4.14,
            "rouge1_f1": 0.5699,
            "rouge2_f1": 0.2855,
            "rougeL_f1": 0.3291,
            "bertscore_f1": 0.7631,
            "simcse_f1": 0.7039,
            "ecs_bge_m3": 0.8691,
            "matcha_f1": 0.933,
        },
        {
            "method_label": "SP",
            "method": "sparse_parent",
            "generator_model": "Qwen/Qwen2.5-72B-Instruct",
            "top_k": 15,
            "MMR": 6.57,
            "rouge1_f1": 0.561,
            "rouge2_f1": 0.2798,
            "rougeL_f1": 0.3193,
            "bertscore_f1": 0.7624,
            "simcse_f1": 0.7025,
            "ecs_bge_m3": 0.867,
            "matcha_f1": 0.9381,
        },
        {
            "method_label": "DPC",
            "method": "dense_parent_child",
            "generator_model": "Qwen/Qwen2.5-72B-Instruct",
            "top_k": 10,
            "MMR": 12.14,
            "rouge1_f1": 0.5662,
            "rouge2_f1": 0.2752,
            "rougeL_f1": 0.3237,
            "bertscore_f1": 0.7583,
            "simcse_f1": 0.7018,
            "ecs_bge_m3": 0.8659,
            "matcha_f1": 0.9278,
        },
        {
            "method_label": "DP",
            "method": "dense_parent",
            "generator_model": "Qwen/Qwen2.5-72B-Instruct",
            "top_k": 15,
            "MMR": 16.57,
            "rouge1_f1": 0.562,
            "rouge2_f1": 0.2678,
            "rougeL_f1": 0.3178,
            "bertscore_f1": 0.7557,
            "simcse_f1": 0.6992,
            "ecs_bge_m3": 0.8658,
            "matcha_f1": 0.9242,
        },
        {
            "method_label": "SPL",
            "method": "sparse_parent_L",
            "generator_model": "Qwen/Qwen2.5-72B-Instruct",
            "top_k": 5,
            "MMR": 26.43,
            "rouge1_f1": 0.5515,
            "rouge2_f1": 0.2586,
            "rougeL_f1": 0.3097,
            "bertscore_f1": 0.7538,
            "simcse_f1": 0.694,
            "ecs_bge_m3": 0.8609,
            "matcha_f1": 0.926,
        },
    ]
    PER_QUESTION_REQUIRED_COLUMNS = list(RETRIEVAL_COLUMNS)

    def find_per_question_files(results_dir):
        results_dir = Path(results_dir)
        if not results_dir.exists():
            raise FileNotFoundError(f"Folder does not exist: {results_dir}")
        return sorted(
            [
                path
                for path in results_dir.rglob("*.xlsx")
                if path.is_file()
                and (not path.name.startswith("~$"))
                and ("per_question" in path.name.lower())
                and ("summary" not in path.name.lower())
            ],
            key=lambda path: natural_sort_key(str(path)),
        )

    # HUPD workbooks store one complete CSV row per cell in a single Excel column.
    def read_per_question_file(file_path):
        raw = pd.read_excel(file_path, engine="openpyxl", header=None)
        if raw.empty:
            raise ValueError(f"Workbook is empty: {file_path}")
        if raw.shape[1] != 1:
            raise ValueError(
                f"Expected one CSV-text column in {file_path}, but found {raw.shape[1]} Excel columns."
            )
        csv_rows = raw.iloc[:, 0].dropna().astype(str).tolist()
        if not csv_rows:
            raise ValueError(f"No CSV rows found in workbook: {file_path}")
        csv_text = "\n".join(csv_rows)
        df = pd.read_csv(StringIO(csv_text))
        df.columns = [str(column).strip() for column in df.columns]
        return df

    # Validate HUPD rows and reject conflicting setup/question duplicates.
    def load_per_question_retrieval_files(files, dataset_name, results_dir):
        frames = []
        skipped = []
        for file_path in files:
            try:
                temp = read_per_question_file(file_path)
            except Exception as exc:
                skipped.append((file_path, f"read error: {exc}"))
                continue
            temp.columns = [normalize_column_name(column) for column in temp.columns]
            missing_columns = [
                column
                for column in PER_QUESTION_REQUIRED_COLUMNS
                if column not in temp.columns
            ]
            if missing_columns:
                skipped.append(
                    (file_path, f"missing required columns: {missing_columns}")
                )
                continue
            temp = temp[PER_QUESTION_REQUIRED_COLUMNS].copy()
            dataset_mask = (
                temp["dataset"]
                .astype("string")
                .str.strip()
                .str.lower()
                .eq(str(dataset_name).strip().lower())
            )
            temp = temp.loc[dataset_mask].copy()
            if temp.empty:
                skipped.append((file_path, f"no rows for dataset '{dataset_name}'"))
                continue
            temp["source_file"] = str(file_path.relative_to(results_dir))
            frames.append(temp)
        if not frames:
            print("\nSkipped per-question files:")
            for file_path, reason in skipped:
                print(f"- {file_path}: {reason}")
            raise RuntimeError(
                "No per-question file containing the required HUPD retrieval columns was found."
            )
        combined = pd.concat(frames, ignore_index=True, sort=False)
        for column in ["dataset", "method", "generator_model"]:
            combined[column] = combined[column].astype("string").str.strip()
        combined["test_case_id"] = canonicalize_question_ids(combined["test_case_id"])
        combined["top_k"] = parse_numeric_series(combined["top_k"]).astype("Int64")
        for metric in ["faithfulness", "context_recall"]:
            combined[metric] = parse_numeric_series(combined[metric])
        required_non_null = list(RETRIEVAL_COLUMNS)
        rows_before_drop = len(combined)
        # Require both retrieval metrics and all setup/question identifiers.
        combined = combined.dropna(subset=required_non_null).copy()
        rows_dropped = rows_before_drop - len(combined)
        if rows_dropped:
            warnings.warn(
                f"Removed {rows_dropped} rows with missing identifiers or retrieval metric values."
            )
        combined["dataset_normalized"] = combined["dataset"].map(normalize_text)
        combined["method_normalized"] = combined["method"].map(normalize_text)
        combined["generator_normalized"] = combined["generator_model"].map(
            normalize_generator
        )
        combined["grounding_mean"] = combined[["faithfulness", "context_recall"]].mean(
            axis=1
        )
        # Remove identical copies; differing scores for the same question remain errors.
        exact_duplicate_columns = [
            "dataset_normalized",
            "method_normalized",
            "top_k",
            "generator_normalized",
            "test_case_id",
            "faithfulness",
            "context_recall",
        ]
        rows_before_deduplication = len(combined)
        combined = combined.drop_duplicates(subset=exact_duplicate_columns).reset_index(
            drop=True
        )
        print(
            "\nPer-question rows before exact de-duplication:",
            rows_before_deduplication,
        )
        print("Per-question rows after exact de-duplication:", len(combined))
        setup_question_columns = [
            "dataset_normalized",
            "method_normalized",
            "top_k",
            "generator_normalized",
            "test_case_id",
        ]
        conflicting_mask = combined.duplicated(
            subset=setup_question_columns, keep=False
        )
        if conflicting_mask.any():
            conflicts = (
                combined.loc[
                    conflicting_mask,
                    setup_question_columns
                    + ["faithfulness", "context_recall", "source_file"],
                ]
                .sort_values(setup_question_columns)
                .reset_index(drop=True)
            )
            print_table("Conflicting duplicate setup-question rows", conflicts)
            raise ValueError(
                "Different retrieval scores were found for the same dataset, method, generator, top-k, and question."
            )
        loaded_files = (
            combined[["source_file"]]
            .drop_duplicates()
            .sort_values("source_file")
            .reset_index(drop=True)
        )
        print_table("Loaded per-question retrieval files", loaded_files)
        print(
            f"\nSkipped CSV files that were not usable per-question HUPD retrieval files: {len(skipped)}"
        )
        return combined

    per_question_files = find_per_question_files(RESULTS_DIR)
    print("Per-question files found recursively:", len(per_question_files))
    for file_path in per_question_files:
        print("-", file_path.relative_to(RESULTS_DIR))
    per_question_df = load_per_question_retrieval_files(
        files=per_question_files, dataset_name=DATASET_NAME, results_dir=RESULTS_DIR
    )
    # Aggregate available questions to setups before building the overview tables.
    summary_df = aggregate_to_system_level(per_question_df)
    print("\nLoaded HUPD per-question rows:", len(per_question_df))
    print("Aggregated HUPD system rows:", len(summary_df))
    # Match the supplied generation-best configurations to the retrieval results.
    best_system_grounding = lookup_best_system_retrieval_scores(
        metric_df=summary_df,
        best_systems=BEST_SYSTEMS,
        title="Generation and retrieval metrics for the best individual HUPD systems",
        dataset_name=DATASET_NAME,
    )
    # Report faithfulness, context recall, and grounding separately.
    faithfulness_tables_overall = print_metric_table_by_method_topk(
        df=summary_df,
        metric="faithfulness",
        title="Faithfulness by retrieval method and top-k - OVERALL",
        mean_column_name="mean_faithfulness",
    )
    context_recall_tables_overall = print_metric_table_by_method_topk(
        df=summary_df,
        metric="context_recall",
        title="Context recall by retrieval method and top-k - OVERALL",
        mean_column_name="mean_context_recall",
    )
    grounding_tables_overall = print_metric_table_by_method_topk(
        df=summary_df,
        metric="grounding_mean",
        title="Grounding mean by retrieval method and top-k - OVERALL",
        mean_column_name="mean_grounding",
    )
    # Keep the original warnings and diagnostics for unmatched configurations.
    unmatched = best_system_grounding[best_system_grounding["matched"] == False].copy()
    if not unmatched.empty:
        print_table("Warning: unmatched selected best systems", unmatched)
        print_table(
            "Available method-generator-top-k combinations",
            summary_df[["method", "generator_display", "top_k", "n_questions"]]
            .drop_duplicates()
            .sort_values(["method", "generator_display", "top_k"])
            .reset_index(drop=True),
        )
    print("\nFinished retrieval metric evaluation for the selected best HUPD systems.")
    print("Per-question files inspected:", len(per_question_files))
    print("Per-question retrieval rows used:", len(per_question_df))
    print("Aggregated system rows:", len(summary_df))
    print("\nMethods:")
    print(sorted(summary_df["method"].dropna().unique(), key=natural_sort_key))
    print("\nGenerators:")
    print(
        sorted(summary_df["generator_display"].dropna().unique(), key=natural_sort_key)
    )
    print("\nTop-k values:")
    print(sorted(summary_df["top_k"].dropna().astype(int).unique()))

    return locals()


# --- Run selected dataset analyses ---
def main(data_root=PROJECT_ROOT, datasets=DATASETS):
    """Evaluate selected datasets, including hard cases when Tender is selected."""
    datasets = normalize_datasets(datasets)
    data_root = Path(data_root)
    evaluations = (
        ("tender", "tender", evaluate_tender),
        ("tender_hard", "tender", evaluate_tender_hard),
        ("ledgar", "ledgar", evaluate_ledgar),
        ("hupd", "hupd", evaluate_hupd),
    )
    return {
        name: evaluate(data_root)
        for name, dataset, evaluate in evaluations
        if dataset in datasets
    }


if __name__ == "__main__":
    results = main(args.root, datasets=args.datasets)
