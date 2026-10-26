from pathlib import Path
import json
import re
import pandas as pd
from transformers import AutoTokenizer


# ---
# CONFIGURATION
# ---

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_NAMES = ("tender", "ledgar", "hupd")
PROJECT_ROOT = next(
    (
        parent
        for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents)
        if all(
            (parent / f"{dataset}_dataset" / "results").is_dir()
            for dataset in DATASET_NAMES
        )
    ),
    None,
)
if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Could not locate experiment_runner_final above this script."
    )

DATASET_PATHS = {
    dataset: PROJECT_ROOT / f"{dataset}_dataset" / "results"
    for dataset in DATASET_NAMES
}


# Parent-child methods: one representative generator JSON per dataset/method.
METHODS = {
    "DPC": "dense_parent_child",
    "SPC": "sparse_parent_child",
    "HPC": "hybrid_parent_child",
}


TOP_K_VALUES = {5, 10, 15, 20}
METHOD_CODES = {method_name: code for code, method_name in METHODS.items()}
EXPERIMENT_PATTERN = re.compile(
    rf"^({'|'.join(re.escape(name) for name in METHODS.values())})_top_"
    rf"({'|'.join(str(value) for value in sorted(TOP_K_VALUES))})$"
)
REQUIRED_RESULT_FIELDS = {
    "dataset",
    "experiment",
    "test_id",
    "retrieved_parent_ids",
}


# One persistent file for later RQ2 significance testing.
OUTPUT_XLSX = SCRIPT_DIR / "effective_k_DPC_SPC_HPC_question_level.xlsx"


# Tokenizer only: no embedding model or GPU is required.
# Install dependencies once:
#   pip install pandas openpyxl transformers sentencepiece
BGE_M3_TOKENIZER = "BAAI/bge-m3"
_tokenizer = None


def count_generator_context_tokens(record):
    """Count the exact generator_context string, without truncation/special tokens.

    Missing/null contexts are NA, not zero. Empty strings correctly yield zero.
    Non-string values are rejected rather than serialized into misleading text.
    """
    global _tokenizer
    context = record.get("generator_context")
    if context is None:
        return pd.NA
    if not isinstance(context, str):
        raise TypeError(
            f"generator_context must be a string; got {type(context).__name__} "
            f"for test_id={record.get('test_id')}, "
            f"experiment={record.get('experiment')}"
        )
    if _tokenizer is None:
        print(f"Loading tokenizer: {BGE_M3_TOKENIZER}", flush=True)
        _tokenizer = AutoTokenizer.from_pretrained(BGE_M3_TOKENIZER, use_fast=True)
    return len(_tokenizer.encode(
        context,
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    ))


# ---
# 1. PARSE EXPERIMENT
# ---

def parse_experiment(experiment):
    """
    Relevant formats:

        dense_parent_child_top_5
        dense_parent_child_top_10
        sparse_parent_child_top_15
        sparse_parent_child_top_20
        hybrid_parent_child_top_10

    Returns:
        ("DPC", 5)
        ("SPC", 10)

    Irrelevant methods return:
        (None, None)
    """

    if experiment is None:
        return None, None

    experiment = str(experiment).strip().lower()

    match = EXPERIMENT_PATTERN.fullmatch(experiment)

    if not match:
        return None, None

    method_name = match.group(1)
    top_k = int(match.group(2))
    return METHOD_CODES[method_name], top_k


# ---
# 2. LOAD JSON
# ---

def load_json(json_path):

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ---
# 3. FIND RESULT RECORDS
# ---

def find_result_records(obj):
    """
    Recursively finds result records containing:

        dataset
        experiment
        test_id
        retrieved_parent_ids
    """

    records = []

    if isinstance(obj, dict):

        if REQUIRED_RESULT_FIELDS.issubset(obj):

            records.append(obj)

            # Already a full result instance.
            return records

        for value in obj.values():

            if isinstance(value, (dict, list)):

                records.extend(
                    find_result_records(value)
                )

    elif isinstance(obj, list):

        for item in obj:

            if isinstance(item, (dict, list)):

                records.extend(
                    find_result_records(item)
                )

    return records


# ---
# 4. NORMALIZE PARENT ID
# ---

def normalize_parent_id(parent_id):
    """
    Convert parent IDs to a stable representation so that
    duplicate parent IDs can be identified reliably.
    """

    if isinstance(
        parent_id,
        (str, int, float, bool)
    ):
        return str(parent_id)

    return json.dumps(
        parent_id,
        ensure_ascii=False,
        sort_keys=True
    )


# ---
# 5. GET UNIQUE PARENT IDS
# ---

def get_unique_parent_ids(parent_ids):
    """
    Example:

        retrieved_parent_ids:
            [P1, P1, P2, P3, P3]

        unique parent IDs:
            {P1, P2, P3}

        nominal top-k = 5
        effective_k   = 3
    """

    unique_ids = set()

    for parent_id in parent_ids:

        if parent_id is None:
            continue

        unique_ids.add(
            normalize_parent_id(parent_id)
        )

    return unique_ids


# ---
# 6. CHECK WHETHER JSON CONTAINS THE REQUIRED METHOD
# ---

def inspect_json_for_method(
    json_path,
    dataset_name,
    expected_method
):
    """
    Determine whether a JSON actually contains usable retrieval
    records for the requested dataset and method.

    Returns:
        number of usable records
        set of top-k values found

    This makes file selection safer than selecting a JSON only
    because its filename contains the method name.
    """

    try:
        data = load_json(json_path)
    except Exception:
        return 0, set()

    records = find_result_records(data)

    n_records = 0
    topks_found = set()

    for record in records:

        record_dataset = str(
            record.get("dataset", "")
        ).strip().lower()

        if record_dataset != dataset_name.lower():
            continue

        method_code, top_k = parse_experiment(
            record.get("experiment")
        )

        if method_code != expected_method:
            continue

        if top_k not in TOP_K_VALUES:
            continue

        n_records += 1
        topks_found.add(top_k)

    return n_records, topks_found


# ---
# 7. SELECT ONE REPRESENTATIVE JSON
# ---

def select_one_json_per_method(
    dataset_name,
    result_dir,
    method_code,
    method_name
):
    """
    Retrieval is generator-independent, so only one generator JSON
    is needed for each:

        dataset × retrieval method

    The selected JSON must actually contain the requested method
    and all four top-k values.

    This avoids accidentally selecting similarly named files.
    """

    if not result_dir.exists():

        print(
            f"\nERROR: Folder does not exist:\n{result_dir}"
        )

        return None

    all_json_files = sorted(
        result_dir.rglob("*.json"),
        key=lambda path: str(path).lower()
    )

    candidates = []

    for json_file in all_json_files:

        filename = json_file.stem.lower()

        if method_name.lower() not in filename:
            continue

        # Avoid obviously different experimental variants.
        if "reversed" in filename:
            continue

        if "agentic" in filename:
            continue

        candidates.append(json_file)

    if not candidates:

        print(
            f"\nWARNING: No filename candidate found for "
            f"{dataset_name.upper()} / {method_code}"
        )

        return None

    # Candidates are already sorted. Stop at the first valid file;
    # do not inspect additional generator instances after finding it.
    for json_file in candidates:
        n_records, topks_found = inspect_json_for_method(
            json_path=json_file,
            dataset_name=dataset_name,
            expected_method=method_code,
        )
        if n_records > 0 and topks_found == TOP_K_VALUES:
            return json_file

    print(
        f"\nWARNING: Candidates existed, but none contained "
        f"all required top-k values for "
        f"{dataset_name.upper()} / {method_code}"
    )
    return None


# ---
# 8. PROCESS ONE REPRESENTATIVE JSON
# ---

def process_selected_json(
    dataset_name,
    expected_method,
    json_path
):
    """
    Produce exactly one observation per:

        dataset
        × method
        × top_k
        × test_id

    effective_k =
        number of UNIQUE retrieved parent IDs
    """

    data = load_json(json_path)

    records = find_result_records(data)

    rows = []

    for record in records:

        # ---
        # Dataset
        # ---

        record_dataset = str(
            record.get("dataset", "")
        ).strip().lower()

        if record_dataset != dataset_name.lower():
            continue


        # ---
        # Method + nominal top-k
        # ---

        method_code, top_k = parse_experiment(
            record.get("experiment")
        )

        if method_code != expected_method:
            continue

        if top_k not in TOP_K_VALUES:
            continue


        # ---
        # Question ID
        # ---

        test_id = record.get("test_id")

        if test_id is None:
            continue

        test_id = str(test_id).strip()

        if not test_id:
            continue


        # ---
        # Retrieved parent IDs
        # ---

        parent_ids = record.get(
            "retrieved_parent_ids"
        )

        if not isinstance(parent_ids, list):
            continue

        unique_parent_ids = (
            get_unique_parent_ids(parent_ids)
        )


        # ---
        # Effective k
        # ---

        n_parent_id_entries = len(
            parent_ids
        )

        effective_k = len(
            unique_parent_ids
        )

        duplicate_parent_hits = (
            n_parent_id_entries
            - effective_k
        )

        effective_k_ratio = (
            effective_k / top_k
        )

        consolidation_pct = (
            1 - effective_k_ratio
        ) * 100


        # ---
        # Save one question-level observation
        # ---

        rows.append(
            {
                "dataset": dataset_name,

                # Short thesis label
                "method": expected_method,

                # Full retrieval method name
                "method_full": METHODS[
                    expected_method
                ],

                "test_id": test_id,

                # Search-layer k
                "top_k": top_k,

                # THIS IS THE IMPORTANT RQ2 VARIABLE:
                "effective_k": effective_k,

                "generator_context_tokens_bge_m3":
                    count_generator_context_tokens(record),

                # Additional audit variables
                "n_parent_id_entries":
                    n_parent_id_entries,

                "duplicate_parent_hits":
                    duplicate_parent_hits,

                "effective_k_ratio":
                    effective_k_ratio,

                "consolidation_pct":
                    consolidation_pct,

                "source_json":
                    json_path.name,
            }
        )

    return rows


# ---
# 9. PROCESS ALL THREE DATASETS
# ---

all_rows = []

selected_files = []


print()
print("=" * 100)
print("QUESTION-LEVEL EFFECTIVE K AND CONTEXT TOKEN EXTRACTION")
print("=" * 100)


for dataset_name, result_dir in DATASET_PATHS.items():

    for method_code, method_name in METHODS.items():

        selected_json = select_one_json_per_method(
            dataset_name=dataset_name,
            result_dir=result_dir,
            method_code=method_code,
            method_name=method_name,
        )

        if selected_json is None:
            continue

        print(
            f"{dataset_name.upper():8s} | "
            f"{method_code:3s} | "
            f"{selected_json.name}"
        )

        selected_files.append(
            {
                "dataset": dataset_name,
                "method": method_code,
                "method_full": method_name,
                "selected_file":
                    selected_json.name,
                "selected_path":
                    str(selected_json),
            }
        )

        rows = process_selected_json(
            dataset_name=dataset_name,
            expected_method=method_code,
            json_path=selected_json,
        )

        all_rows.extend(rows)


# ---
# 10. BUILD DATAFRAME
# ---

if not all_rows:

    raise RuntimeError(
        "\nNo usable observations found.\n"
        "Check paths, filenames and JSON structure."
    )


df = pd.DataFrame(
    all_rows
)

df["generator_context_tokens_bge_m3"] = pd.array(
    df["generator_context_tokens_bge_m3"], dtype="Int64"
)
missing_contexts = int(df["generator_context_tokens_bge_m3"].isna().sum())
if missing_contexts:
    print(
        f"WARNING: {missing_contexts} observations lack generator_context. "
        "Their token counts remain blank and are excluded from token statistics."
    )

selected_files_df = pd.DataFrame(
    selected_files
)


# ---
# 11. DUPLICATE CHECK
# ---
#
# Required uniqueness:
#
# dataset × method × top_k × test_id
#
# ---

KEY_COLUMNS = [
    "dataset",
    "method",
    "top_k",
    "test_id",
]


duplicate_mask = df.duplicated(
    subset=KEY_COLUMNS,
    keep=False
)

duplicates = df[
    duplicate_mask
].copy()


print()
print("=" * 100)
print("DUPLICATE CHECK")
print("=" * 100)


if duplicates.empty:

    print(
        "OK: Every dataset × method × top-k × question "
        "combination occurs exactly once."
    )

else:

    print(
        "ERROR: Duplicate observations found."
    )

    print(
        duplicates[
            KEY_COLUMNS
            + [
                "effective_k",
                "source_json",
            ]
        ].to_string(
            index=False
        )
    )

    raise RuntimeError(
        "Duplicate question-level effective-k "
        "observations found. File not saved."
    )


# ---
# 12. SANITY CHECK:
# EFFECTIVE K MUST NOT EXCEED NOMINAL K
# ---

impossible = df[
    df["effective_k"]
    >
    df["top_k"]
].copy()


print()
print("=" * 100)
print("SANITY CHECK: EFFECTIVE K <= NOMINAL TOP-K")
print("=" * 100)


if impossible.empty:

    print(
        "OK: effective_k never exceeds nominal top_k."
    )

else:

    print(
        impossible[
            KEY_COLUMNS
            + [
                "effective_k",
                "n_parent_id_entries",
            ]
        ].to_string(
            index=False
        )
    )

    raise RuntimeError(
        "Impossible effective-k observations found. "
        "File not saved."
    )


# ---
# 13. TOP-K COMPLETENESS PER QUESTION
# ---

topk_completeness = (
    df
    .groupby(
        [
            "dataset",
            "method",
            "test_id",
        ]
    )["top_k"]
    .apply(
        lambda values:
        set(values)
        == TOP_K_VALUES
    )
    .reset_index(
        name="complete"
    )
)


incomplete_questions = (
    topk_completeness[
        ~topk_completeness["complete"]
    ]
    .copy()
)


print()
print("=" * 100)
print("TOP-K COMPLETENESS CHECK")
print("=" * 100)


if incomplete_questions.empty:

    print(
        "OK: Every available question contains "
        "k = 5, 10, 15 and 20."
    )

else:

    print(
        f"ERROR: {len(incomplete_questions)} "
        f"question/method combinations do not contain "
        f"all four top-k values."
    )

    print(
        incomplete_questions.to_string(
            index=False
        )
    )

    raise RuntimeError(
        "Incomplete top-k observations found. "
        "File not saved."
    )


# ---
# 14. QUESTION COUNTS
# ---

question_counts = (
    df
    .groupby(
        [
            "dataset",
            "method",
            "top_k",
        ],
        as_index=False
    )
    .agg(
        n_questions=(
            "test_id",
            "nunique"
        ),

        n_contexts_counted=("generator_context_tokens_bge_m3", "count"),
        n_contexts_missing=("generator_context_tokens_bge_m3", lambda x: x.isna().sum()),
        mean_context_tokens_bge_m3=("generator_context_tokens_bge_m3", "mean"),
        median_context_tokens_bge_m3=("generator_context_tokens_bge_m3", "median"),
        std_context_tokens_bge_m3=("generator_context_tokens_bge_m3", "std"),
        min_context_tokens_bge_m3=("generator_context_tokens_bge_m3", "min"),
        max_context_tokens_bge_m3=("generator_context_tokens_bge_m3", "max"),

        mean_effective_k=(
            "effective_k",
            "mean"
        ),

        median_effective_k=(
            "effective_k",
            "median"
        ),

        std_effective_k=(
            "effective_k",
            "std"
        ),

        min_effective_k=(
            "effective_k",
            "min"
        ),

        max_effective_k=(
            "effective_k",
            "max"
        ),

        mean_effective_k_ratio=(
            "effective_k_ratio",
            "mean"
        ),
    )
)


print()
print("=" * 120)
print("SUMMARY BY DATASET × METHOD × NOMINAL TOP-K")
print("=" * 120)

print(
    question_counts
    .round(3)
    .to_string(
        index=False
    )
)


# ---
# 15. SORT MAIN QUESTION-LEVEL TABLE
# ---

dataset_order = {
    "tender": 0,
    "ledgar": 1,
    "hupd": 2,
}

method_order = {
    "DPC": 0,
    "SPC": 1,
    "HPC": 2,
}


df["_dataset_order"] = (
    df["dataset"]
    .map(dataset_order)
)

df["_method_order"] = (
    df["method"]
    .map(method_order)
)


# Numerical sorting where possible, while still supporting
# non-numeric LEDGAR test IDs.
df["_test_id_numeric"] = pd.to_numeric(
    df["test_id"],
    errors="coerce"
)


df = (
    df
    .sort_values(
        [
            "_dataset_order",
            "_method_order",
            "top_k",
            "_test_id_numeric",
            "test_id",
        ],
        na_position="last"
    )
    .drop(
        columns=[
            "_dataset_order",
            "_method_order",
            "_test_id_numeric",
        ]
    )
    .reset_index(
        drop=True
    )
)


# Put the merge-relevant columns first.
main_columns = [
    "dataset",
    "method",
    "method_full",
    "top_k",
    "test_id",

    # Main variable needed later
    "effective_k",

    "generator_context_tokens_bge_m3",

    # Audit / descriptive variables
    "effective_k_ratio",
    "n_parent_id_entries",
    "duplicate_parent_hits",
    "consolidation_pct",
    "source_json",
]


df = df[
    main_columns
]


# ---
# 16. SAVE EXCEL
# ---
#
# Main sheet:
#   effective_k_question_level
#
# This is the sheet the later significance-testing code should
# simply load and merge using:
#
#   dataset
#   method
#   top_k
#   test_id
#
# ---

OUTPUT_XLSX.parent.mkdir(
    parents=True,
    exist_ok=True
)


with pd.ExcelWriter(
    OUTPUT_XLSX,
    engine="openpyxl"
) as writer:

    # FIRST SHEET = file later loaded for RQ2.
    df.to_excel(
        writer,
        sheet_name="effective_k_question_level",
        index=False
    )

    # Useful audit sheet: which generator JSON was selected?
    selected_files_df.to_excel(
        writer,
        sheet_name="selected_json_files",
        index=False
    )

    # Descriptive summary only.
    question_counts.to_excel(
        writer,
        sheet_name="summary",
        index=False
    )


# ---
# 17. FINAL VERIFICATION
# ---

saved_df = pd.read_excel(
    OUTPUT_XLSX,
    sheet_name="effective_k_question_level"
)


if len(saved_df) != len(df):

    raise RuntimeError(
        "Saved Excel row count does not match "
        "the extracted dataframe."
    )


if saved_df.duplicated(
    subset=KEY_COLUMNS
).any():

    raise RuntimeError(
        "Duplicates found after writing Excel."
    )


# ---
# 18. FINAL OUTPUT
# ---

print()
print("=" * 100)
print("EXTRACTION FINISHED")
print("=" * 100)

print(
    f"\nQuestion-level observations saved: "
    f"{len(df):,}"
)

print(
    f"\nOutput file:\n{OUTPUT_XLSX}"
)

print(
    "\nMain sheet:"
    "\n  effective_k_question_level"
)

print(
    "\nMerge key for later RQ2 significance testing:"
    "\n  dataset + method + top_k + test_id"
)

print(
    "\nEffective-k variable:"
    "\n  effective_k"
)

print("\nContext-size variable:\n  generator_context_tokens_bge_m3")
print("Tokenizer: BAAI/bge-m3; no truncation; no added special tokens.")
print("Context statistics describe the selected representative JSON per dataset/method.")
