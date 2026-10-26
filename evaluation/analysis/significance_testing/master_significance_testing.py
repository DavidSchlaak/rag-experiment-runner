"""Run RQ2 and RQ3 significance testing in a normal PyCharm Python configuration.

Script parameters:
    --analysis rq2 --dataset tender
    --analysis rq3 --datasets ledgar,hupd

With no parameters, run both analyses across all datasets. Install the
project requirements before running."""

import argparse
from pathlib import Path

try:
    from .significance_io import make_local_readers as _make_local_readers
except ImportError:
    from significance_io import make_local_readers as _make_local_readers

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASETS = ("tender", "ledgar", "hupd")


def normalize_datasets(datasets):
    if isinstance(datasets, str):
        datasets = datasets.split(",")
    selected = tuple(name.strip().lower() for name in datasets)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(DATASETS):
        raise ValueError("Datasets must contain unique names from tender,ledgar,hupd.")
    return selected


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--analysis", choices=("all", "rq2", "rq3"), default="all")
    parser.add_argument(
        "--dataset", "--datasets", dest="datasets", default=DATASETS,
        metavar="NAME[,NAME...]",
        help="Datasets to analyze: tender, ledgar, hupd (default: all three).",
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Project data root.")
    parser.add_argument(
        "--output-dir", type=Path,
        help="Workbook destination (default: this project's significance_testing folder).",
    )
    parser.add_argument(
        "--effective-k-file", type=Path, help="Optional RQ2 effective-k workbook override.",
    )
    args = parser.parse_args(argv)
    try:
        args.datasets = normalize_datasets(args.datasets)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def display(*objects):
    """Print tables in scripts and preserve rich display in IPython sessions."""
    import pandas as pd
    from IPython import get_ipython
    from IPython.display import display as ipython_display
    from pandas.io.formats.style import Styler
    if getattr(get_ipython(), "kernel", None) is not None:
        ipython_display(*objects)
        return
    for obj in objects:
        if isinstance(obj, Styler):
            print(obj.to_string(delimiter=" | "))
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            print(obj.to_string())
        else:
            print(obj)


def output_file(output_dir, filename, datasets):
    path = Path(output_dir) / filename
    if set(datasets) != set(DATASETS):
        suffix = "_".join(name for name in DATASETS if name in datasets)
        path = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
    return path


def default_effective_k_file(data_root):
    return Path(data_root) / (
        "evaluation/analysis/effective_k_analysis/"
        "effective_k_DPC_SPC_question_level.xlsx"
    )




def run_rq2(data_root=PROJECT_ROOT, datasets=DATASETS, output_dir=None, effective_k_file=None):
    """Run the original RQ2 analysis with independent local state."""
    # ---
    # RQ2 SIGNIFICANCE TESTING
    # ---
    #
    # RQ2:
    # Parent-child retrieval vs direct-parent retrieval
    #
    # Dense:  DPC vs DP
    # Sparse: SPC vs SP
    #
    # LEVEL 1 — PRIMARY NOMINAL-k ANALYSIS
    # ---
    # Paired two-sided Wilcoxon signed-rank tests:
    #   DPC_k vs DP_k
    #   SPC_k vs SP_k
    # for k ∈ {5, 10, 15, 20}
    #
    # LEVEL 2 — EFFECTIVE-PARENT-ADJUSTED SENSITIVITY ANALYSIS
    # ---
    # Question fixed-effects OLS regression with quadratic adjustment
    # for actual effective parent breadth E:
    #
    #   score = alpha_question
    #           + beta_1 * parent_child
    #           + gamma_1 * E_centered
    #           + gamma_2 * E_centered^2
    #           + error
    #
    # Inference uses question-clustered sandwich standard errors.
    # No normality assumption is imposed on metric scores/residuals for
    # the cluster-robust significance test; inference relies on independent
    # question clusters and large-cluster asymptotics.
    #
    # beta_1 > 0  -> parent-child retrieval higher after E adjustment
    # beta_1 < 0  -> direct-parent retrieval higher after E adjustment
    #
    # METRIC FAMILIES
    # ---
    # Generation:
    #   ROUGE-1, ROUGE-2, ROUGE-L, BERTScore, SimCSE, BGE-M3 ECS, MATCHA
    # Evidence:
    #   Faithfulness, Context Recall
    #
    # MULTIPLE TESTING
    # ---
    # Holm correction separately within:
    #   dataset × analysis level × metric family
    #
    # DATA ACCESS / OUTPUT
    # ---
    # All input files are read locally from experiment_runner_final. Output workbooks are saved in the significance_testing folder.
    # ---


    # ---
    # 1. IMPORTS
    # ---

    import warnings
    from pathlib import Path
    import sys

    import numpy as np
    import pandas as pd

    from scipy.stats import (
        wilcoxon,
        rankdata,
    )

    from statsmodels.stats.multitest import (
        multipletests,
    )

    import statsmodels.formula.api as smf

    # Use the script's console-aware display helper.






    # ---
    # 2. CONFIG
    # ---

    DATASETS = list(normalize_datasets(datasets))


    TOP_K_VALUES = [
        5,
        10,
        15,
        20,
    ]


    ALPHA = 0.05


    PROJECT_ROOT = Path(data_root).resolve()
    SIGNIFICANCE_DIR = Path(output_dir).resolve() if output_dir is not None else (
        PROJECT_ROOT / "evaluation/analysis/significance_testing"
    )
    SIGNIFICANCE_DIR.mkdir(parents=True, exist_ok=True)
    list_result_children, read_result_file, FOLDER_MIME = make_local_readers(PROJECT_ROOT)

    # Read inputs from each dataset's existing result folders.
    RESULT_FOLDERS = {
        dataset: {
            "generation": PROJECT_ROOT / f"{dataset}_dataset" / "results" / "metric_analysis",
            "evidence": PROJECT_ROOT / f"{dataset}_dataset" / "results" / "retrieval_analysis",
        }
        for dataset in DATASETS
    }

    OUTPUT_FILE = output_file(SIGNIFICANCE_DIR, "RQ2_significance_results_effective_k_adjusted.xlsx", DATASETS)



    # ---
    # 5. METHOD DEFINITIONS
    # ---

    PAIR_SPECS = {

        "dense": {

            "parent_child":
                "DPC",

            "direct":
                "DP",
        },


        "sparse": {

            "parent_child":
                "SPC",

            "direct":
                "SP",
        },
    }


    RAW_METHOD_MAP = {

        "dense_parent_child":
            "DPC",

        "dense_parent":
            "DP",

        "sparse_parent_child":
            "SPC",

        "sparse_parent":
            "SP",

        # Legacy internal name
        "sparse_parent_reversed":
            "SP",
    }


    # ---
    # 6. METRICS
    # ---

    GENERATION_METRICS = [

        "rouge1_f1",
        "rouge2_f1",
        "rougeL_f1",

        "bertscore_f1",
        "simcse_f1",

        "ecs_bge_m3",
        "matcha_f1",
    ]


    EVIDENCE_METRICS = [

        "faithfulness",
        "context_recall",
    ]


    METRIC_LABELS = {

        "rouge1_f1":
            "ROUGE-1",

        "rouge2_f1":
            "ROUGE-2",

        "rougeL_f1":
            "ROUGE-L",

        "bertscore_f1":
            "BERTScore",

        "simcse_f1":
            "SimCSE",

        "ecs_bge_m3":
            "BGE-M3 ECS",

        "matcha_f1":
            "MATCHA",

        "faithfulness":
            "Faithfulness",

        "context_recall":
            "Context Recall",
    }


    # ---
    # 7. EXPECTED GENERATORS
    # ---

    EXPECTED_GENERATORS = {

        "tender": {

            "meta-llama/Llama-3.1-8B-Instruct-instruct",
            "meta-llama/Llama-3.1-70B-Instruct-instruct",

            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",

            "Qwen/Qwen3-235B-A22B-Instruct-2507",
        },


        "ledgar": {

            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
        },


        "hupd": {

            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
        },
    }


    EXPECTED_N_GENERATORS = {

        dataset:
            len(generators)

        for dataset, generators
        in EXPECTED_GENERATORS.items()
    }


    # ---
    # 9-11. SHARED LOCAL FILE HELPERS
    # These helpers only read files under experiment_runner_final.
    # ---

    # ---
    # 12. GENERATION FOLDER CLASSIFICATION
    # ---

    def classify_generation_folder(
        dataset,
        folder_name,
    ):

        name = (
            str(
                folder_name
            )
            .strip()
            .lower()
        )


        # Never descend into these experiment types.
        forbidden_tokens = [

            "agentic",
            "hybrid_parent_child",
            "bm25l",
            "sparse_parent_child_reversed",
        ]


        if any(

            token in name

            for token
            in forbidden_tokens
        ):

            return None


        if (
            f"{dataset}_dense_parent_child_"
            in name
        ):

            return "DPC"


        if (
            f"{dataset}_dense_parent_"
            in name
        ):

            return "DP"


        if (
            f"{dataset}_sparse_parent_child_"
            in name
        ):

            return "SPC"


        if (
            f"{dataset}_sparse_parent_"
            in name
        ):

            return "SP"


        return None


    # ---
    # 13. SELECT GENERATION FILE
    # ---

    def select_per_question_generation_file(
        files_in_folder
    ):

        candidates = []


        for item in files_in_folder:

            if (
                item.get(
                    "mimeType"
                )
                ==
                FOLDER_MIME
            ):

                continue


            name = (
                item.get(
                    "name",
                    ""
                )
                .lower()
            )


            if not name.startswith(
                "per_question_metrics_"
            ):

                continue


            if (
                "_all_generators"
                not in name
            ):

                continue


            if name.endswith(
                ".xlsx"
            ):

                priority = 0


            elif name.endswith(
                ".csv"
            ):

                priority = 1


            else:

                continue


            candidates.append(
                (
                    priority,
                    item,
                )
            )


        if not candidates:

            return None


        candidates.sort(
            key=lambda x: (
                x[0],
                x[1][
                    "name"
                ].lower(),
            )
        )


        best_priority = (
            candidates[0][0]
        )


        best = [

            item

            for priority, item
            in candidates

            if priority
            ==
            best_priority
        ]


        if len(
            best
        ) != 1:

            raise RuntimeError(
                "Ambiguous per-question generation file."
            )


        return best[0]


    # ---
    # 14. DISCOVER GENERATION FILES
    # ---

    def discover_generation_files():

        selected = []


        for dataset in DATASETS:

            root = (
                RESULT_FOLDERS[
                    dataset
                ][
                    "generation"
                ]
            )


            root_children = (
                list_result_children(
                    root
                )
            )


            for child in root_children:

                if (
                    child.get(
                        "mimeType"
                    )
                    !=
                    FOLDER_MIME
                ):

                    continue


                child_id = (
                    child[
                        "id"
                    ]
                )
                method = (
                    classify_generation_folder(

                        dataset,

                        child[
                            "name"
                        ],
                    )
                )


                # Crucially: skip before entering folder.
                if method is None:

                    continue


                contents = (
                    list_result_children(
                        child_id
                    )
                )


                selected_file = (
                    select_per_question_generation_file(
                        contents
                    )
                )


                if selected_file is None:

                    raise RuntimeError(
                        "No per-question file found in "
                        f"{child['name']}"
                    )


                selected.append(
                    {
                        "dataset":
                            dataset,

                        "method_expected":
                            method,

                        "generator_folder":
                            child[
                                "name"
                            ],

                        "file_id":
                            selected_file[
                                "id"
                            ],

                        "file_name":
                            selected_file[
                                "name"
                            ],
                    }
                )


        return pd.DataFrame(
            selected
        )


    GENERATION_FILE_INDEX = (
        discover_generation_files()
    )


    # ---
    # 15. EVIDENCE FILE DISCOVERY
    # ---

    EVIDENCE_METHOD_NAMES = {

        "DPC":
            "dense_parent_child",

        "DP":
            "dense_parent",

        "SPC":
            "sparse_parent_child",

        "SP":
            "sparse_parent",
    }


    def select_evidence_file(

        dataset,

        method_code,

        files,
    ):

        method_name = (
            EVIDENCE_METHOD_NAMES[
                method_code
            ]
        )


        expected = (

            f"per_question_retrieval_metrics_"
            f"{dataset}_"
            f"{method_name}_"
            f"faithfulness_context_recall"

        ).lower()


        candidates = []


        for item in files:

            if (
                item.get(
                    "mimeType"
                )
                ==
                FOLDER_MIME
            ):

                continue


            name = (
                item.get(
                    "name",
                    ""
                )
                .lower()
            )


            if not name.startswith(
                expected
            ):

                continue


            if name.endswith(
                ".xlsx"
            ):

                priority = 0


            elif name.endswith(
                ".csv"
            ):

                priority = 1


            else:

                continue


            candidates.append(
                (
                    priority,
                    item,
                )
            )


        if not candidates:

            raise RuntimeError(
                f"No evidence file for "
                f"{dataset}/{method_code}"
            )


        candidates.sort(
            key=lambda x: (
                x[0],
                x[1][
                    "name"
                ].lower(),
            )
        )


        best_priority = (
            candidates[0][0]
        )


        best = [

            item

            for priority, item
            in candidates

            if priority
            ==
            best_priority
        ]


        if len(
            best
        ) != 1:

            raise RuntimeError(
                f"Ambiguous evidence file for "
                f"{dataset}/{method_code}"
            )


        return best[0]


    def discover_evidence_files():

        selected = []


        for dataset in DATASETS:

            root = (
                RESULT_FOLDERS[
                    dataset
                ][
                    "evidence"
                ]
            )


            files = (
                list_result_children(
                    root
                )
            )


            for method in [

                "DPC",
                "DP",
                "SPC",
                "SP",
            ]:

                item = (
                    select_evidence_file(

                        dataset,

                        method,

                        files,
                    )
                )


                selected.append(
                    {
                        "dataset":
                            dataset,

                        "method_expected":
                            method,

                        "file_id":
                            item[
                                "id"
                            ],

                        "file_name":
                            item[
                                "name"
                            ],
                    }
                )


        return pd.DataFrame(
            selected
        )


    EVIDENCE_FILE_INDEX = (
        discover_evidence_files()
    )


    # ---
    # 16. NORMALIZATION HELPERS
    # ---

    def normalize_question_id(
        series
    ):

        return (
            series
            .astype(str)
            .str.strip()
            .str.replace(
                r"\.0$",
                "",
                regex=True,
            )
        )


    def canonicalize_generator(
        value
    ):

        if pd.isna(
            value
        ):

            return np.nan


        original = str(
            value
        )


        s = (
            original
            .strip()
            .lower()
            .replace(
                "_",
                ""
            )
            .replace(
                " ",
                ""
            )
        )


        # Llama 3.1
        if (
            "llama3.1"
            in s
            or
            "llama-3.1"
            in s
        ):

            if "70b" in s:

                return (
                    "meta-llama/Llama-3.1-70B-Instruct-instruct"
                )

            if "8b" in s:

                return (
                    "meta-llama/Llama-3.1-8B-Instruct-instruct"
                )


        # Qwen 2.5
        if (
            "qwen2.5"
            in s
            or
            "qwen-2.5"
            in s
        ):

            if "72b" in s:

                return (
                    "Qwen/Qwen2.5-72B-Instruct"
                )

            if "32b" in s:

                return (
                    "Qwen/Qwen2.5-32B-Instruct"
                )

            if "7b" in s:

                return (
                    "Qwen/Qwen2.5-7B-Instruct"
                )


        # Qwen 3
        if (
            "qwen3"
            in s
            or
            "qwen-3"
            in s
        ):

            if "235b" in s:

                return (
                    "Qwen/Qwen3-235B-A22B-Instruct-2507"
                )


        return (
            "UNKNOWN::"
            +
            original
        )


    def first_non_null(
        series
    ):

        values = (
            series
            .dropna()
        )


        if len(
            values
        ) == 0:

            return np.nan


        return (
            values.iloc[0]
        )


    # ---
    # 17. PREPARE METRIC TABLE
    # ---

    def prepare_metric_dataframe(

        df,

        dataset_expected,

        method_expected,

        metrics,

        source_name,
    ):

        # Remove Excel index columns.
        df = df.loc[
            :,
            ~df.columns
            .astype(str)
            .str.startswith(
                "Unnamed:"
            )
        ].copy()


        required = {

            "dataset",
            "method",
            "top_k",
            "generator_model",
        }


        missing = (
            required
            -
            set(
                df.columns
            )
        )


        if missing:

            raise RuntimeError(
                f"Missing columns in {source_name}: "
                f"{sorted(missing)}"
            )


        # Question ID
        if (
            "test_id"
            in df.columns
        ):

            qid = (
                df[
                    "test_id"
                ]
                .copy()
            )


            if (
                "test_case_id"
                in df.columns
            ):

                qid = qid.fillna(
                    df[
                        "test_case_id"
                    ]
                )


        elif (
            "test_case_id"
            in df.columns
        ):

            qid = (
                df[
                    "test_case_id"
                ]
                .copy()
            )


        else:

            raise RuntimeError(
                f"No question ID in {source_name}"
            )


        df[
            "question_id"
        ] = (
            normalize_question_id(
                qid
            )
        )


        df[
            "dataset"
        ] = (
            df[
                "dataset"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
        )


        df[
            "method_raw"
        ] = (
            df[
                "method"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
        )


        df[
            "method_label"
        ] = (
            df[
                "method_raw"
            ]
            .map(
                RAW_METHOD_MAP
            )
        )


        df[
            "generator"
        ] = (
            df[
                "generator_model"
            ]
            .apply(
                canonicalize_generator
            )
        )


        df[
            "top_k"
        ] = pd.to_numeric(

            df[
                "top_k"
            ],

            errors="coerce",
        )


        missing_metrics = (
            set(
                metrics
            )
            -
            set(
                df.columns
            )
        )


        if missing_metrics:

            raise RuntimeError(
                f"Missing metrics in {source_name}: "
                f"{sorted(missing_metrics)}"
            )


        for metric in metrics:

            df[
                metric
            ] = pd.to_numeric(

                df[
                    metric
                ],

                errors="coerce",
            )


        df = df[
            (
                df[
                    "dataset"
                ]
                ==
                dataset_expected
            )
            &
            (
                df[
                    "method_label"
                ]
                ==
                method_expected
            )
            &
            (
                df[
                    "top_k"
                ]
                .isin(
                    TOP_K_VALUES
                )
            )
            &
            (
                df[
                    "generator"
                ]
                .isin(
                    EXPECTED_GENERATORS[
                        dataset_expected
                    ]
                )
            )
        ].copy()


        df[
            "top_k"
        ] = (
            df[
                "top_k"
            ]
            .astype(int)
        )


        return df


    # ---
    # 18. LOAD GENERATION
    # ---

    def load_generation_metrics():

        frames = []


        print(
            "Loading generation metrics..."
        )


        for _, row in (
            GENERATION_FILE_INDEX
            .iterrows()
        ):

            raw = (
                read_result_file(
                    {
                        "id":
                            row[
                                "file_id"
                            ],

                        "name":
                            row[
                                "file_name"
                            ],
                    }
                )
            )


            prepared = (
                prepare_metric_dataframe(

                    raw,

                    row[
                        "dataset"
                    ],

                    row[
                        "method_expected"
                    ],

                    GENERATION_METRICS,

                    row[
                        "file_name"
                    ],
                )
            )


            frames.append(
                prepared
            )


        return pd.concat(

            frames,

            ignore_index=True,

            sort=False,
        )


    RAW_GENERATION = (
        load_generation_metrics()
    )


    # ---
    # 19. LOAD EVIDENCE
    # ---

    def load_evidence_metrics():

        frames = []


        print(
            "Loading evidence-support metrics..."
        )


        for _, row in (
            EVIDENCE_FILE_INDEX
            .iterrows()
        ):

            raw = (
                read_result_file(
                    {
                        "id":
                            row[
                                "file_id"
                            ],

                        "name":
                            row[
                                "file_name"
                            ],
                    }
                )
            )


            prepared = (
                prepare_metric_dataframe(

                    raw,

                    row[
                        "dataset"
                    ],

                    row[
                        "method_expected"
                    ],

                    EVIDENCE_METRICS,

                    row[
                        "file_name"
                    ],
                )
            )


            frames.append(
                prepared
            )


        return pd.concat(

            frames,

            ignore_index=True,

            sort=False,
        )


    RAW_EVIDENCE = (
        load_evidence_metrics()
    )


    # ---
    # 20. VALIDATE GLOBAL GENERATOR SETS
    # ---

    def validate_generator_sets(
        raw
    ):

        for dataset in DATASETS:

            expected = (
                EXPECTED_GENERATORS[
                    dataset
                ]
            )


            for method in [

                "DPC",
                "DP",
                "SPC",
                "SP",
            ]:

                found = set(

                    raw.loc[
                        (
                            raw[
                                "dataset"
                            ]
                            ==
                            dataset
                        )
                        &
                        (
                            raw[
                                "method_label"
                            ]
                            ==
                            method
                        ),

                        "generator",
                    ]
                    .dropna()
                    .unique()
                )


                if found != expected:

                    raise RuntimeError(
                        f"Generator-set mismatch for "
                        f"{dataset}/{method}.\n"
                        f"Expected: {sorted(expected)}\n"
                        f"Found: {sorted(found)}"
                    )


    validate_generator_sets(
        RAW_GENERATION
    )


    validate_generator_sets(
        RAW_EVIDENCE
    )


    # ---
    # 21. DEDUPLICATE IDENTICAL COPIES
    # ---

    def deduplicate_raw(

        raw,

        metrics,
    ):

        key = [

            "dataset",
            "method_label",
            "top_k",
            "generator",
            "question_id",
        ]


        # Different values for same observation -> stop.
        for metric in metrics:

            counts = (
                raw
                .groupby(
                    key,
                    dropna=False,
                )[
                    metric
                ]
                .nunique(
                    dropna=True
                )
            )


            if (
                counts
                >
                1
            ).any():

                raise RuntimeError(
                    f"Conflicting duplicate values found "
                    f"for metric {metric}."
                )


        return (
            raw
            .groupby(
                key,
                as_index=False,
                dropna=False,
            )[
                metrics
            ]
            .agg(
                first_non_null
            )
        )


    RAW_GENERATION = (
        deduplicate_raw(

            RAW_GENERATION,

            GENERATION_METRICS,
        )
    )


    RAW_EVIDENCE = (
        deduplicate_raw(

            RAW_EVIDENCE,

            EVIDENCE_METRICS,
        )
    )


    # ---
    # 22. GENERATOR AVERAGING
    # ---
    #
    # Only observations with the COMPLETE generator set are retained.
    #
    # Incomplete configurations are silently excluded from testing,
    # but a compact audit summary is stored in the output workbook.
    #
    # ---

    def generator_average(

        raw,

        metrics,

        family,
    ):

        valid_parts = []

        issue_parts = []


        for dataset in DATASETS:

            expected_n = (
                EXPECTED_N_GENERATORS[
                    dataset
                ]
            )


            dataset_raw = raw[
                raw[
                    "dataset"
                ]
                ==
                dataset
            ]


            for metric in metrics:

                grouped = (
                    dataset_raw
                    .groupby(
                        [
                            "dataset",
                            "method_label",
                            "top_k",
                            "question_id",
                        ],
                        as_index=False,
                    )
                    .agg(

                        score=(
                            metric,
                            "mean",
                        ),

                        n_metric_values=(
                            metric,
                            "count",
                        ),

                        n_generators=(
                            "generator",
                            "nunique",
                        ),
                    )
                )


                grouped[
                    "metric"
                ] = (
                    metric
                )


                grouped[
                    "metric_family"
                ] = (
                    family
                )


                grouped[
                    "expected_generators"
                ] = (
                    expected_n
                )


                complete = (
                    (
                        grouped[
                            "n_metric_values"
                        ]
                        ==
                        expected_n
                    )
                    &
                    (
                        grouped[
                            "n_generators"
                        ]
                        ==
                        expected_n
                    )
                )


                valid_parts.append(
                    grouped[
                        complete
                    ].copy()
                )


                incomplete = grouped[
                    ~complete
                ].copy()


                if not incomplete.empty:

                    issue_parts.append(
                        incomplete
                    )


        averaged = pd.concat(

            valid_parts,

            ignore_index=True,
        )


        if issue_parts:

            issues = pd.concat(

                issue_parts,

                ignore_index=True,
            )


        else:

            issues = pd.DataFrame()


        return (
            averaged,
            issues,
        )


    AVG_GENERATION, GENERATION_ISSUES = (
        generator_average(

            RAW_GENERATION,

            GENERATION_METRICS,

            "generation",
        )
    )


    AVG_EVIDENCE, EVIDENCE_ISSUES = (
        generator_average(

            RAW_EVIDENCE,

            EVIDENCE_METRICS,

            "evidence",
        )
    )


    AVG_ALL = pd.concat(

        [
            AVG_GENERATION,
            AVG_EVIDENCE,
        ],

        ignore_index=True,
    )


    ALL_GENERATOR_ISSUES = pd.concat(

        [
            GENERATION_ISSUES,
            EVIDENCE_ISSUES,
        ],

        ignore_index=True,
    )


    # ---
    # 23. COMPACT DATA QUALITY SUMMARY
    # ---

    if not ALL_GENERATOR_ISSUES.empty:

        INCOMPLETE_CONFIG_SUMMARY = (

            ALL_GENERATOR_ISSUES[
                [
                    "dataset",
                    "method_label",
                    "top_k",
                    "question_id",
                ]
            ]
            .drop_duplicates()
            .groupby(
                [
                    "dataset",
                    "method_label",
                    "top_k",
                ],
                as_index=False,
            )
            .size()
            .rename(
                columns={
                    "size":
                        "n_incomplete_question_configs"
                }
            )
        )


    else:

        INCOMPLETE_CONFIG_SUMMARY = pd.DataFrame(
            columns=[
                "dataset",
                "method_label",
                "top_k",
                "n_incomplete_question_configs",
            ]
        )


    # ---
    # 24. LOAD EFFECTIVE-K WORKBOOK FROM THE PROJECT
    # ---

    EFFECTIVE_K_FILE = (Path(effective_k_file).resolve() if effective_k_file is not None else default_effective_k_file(PROJECT_ROOT))
    if not EFFECTIVE_K_FILE.is_file():
        raise FileNotFoundError(f"Effective-k workbook not found: {EFFECTIVE_K_FILE}")

    effective_k = pd.read_excel(
        EFFECTIVE_K_FILE,
        sheet_name="effective_k_question_level",
    )
    print(f"Loaded effective-k workbook: {EFFECTIVE_K_FILE}")

    # ---
    # 25. NORMALIZE EFFECTIVE-K
    # ---

    required_effective = {

        "dataset",
        "method",
        "top_k",
        "test_id",
        "effective_k",
    }


    missing = (
        required_effective
        -
        set(
            effective_k.columns
        )
    )


    if missing:

        raise RuntimeError(
            f"Missing effective-k columns: "
            f"{sorted(missing)}"
        )


    effective_k[
        "dataset"
    ] = (
        effective_k[
            "dataset"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )


    effective_k[
        "method"
    ] = (
        effective_k[
            "method"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )


    effective_k[
        "question_id"
    ] = (
        normalize_question_id(
            effective_k[
                "test_id"
            ]
        )
    )


    effective_k[
        "top_k"
    ] = (
        pd.to_numeric(
            effective_k[
                "top_k"
            ],
            errors="raise",
        )
        .astype(int)
    )


    effective_k[
        "effective_k"
    ] = pd.to_numeric(

        effective_k[
            "effective_k"
        ],

        errors="raise",
    )


    effective_k = effective_k[
        (
            effective_k[
                "dataset"
            ]
            .isin(
                DATASETS
            )
        )
        &
        (
            effective_k[
                "method"
            ]
            .isin(
                [
                    "DPC",
                    "SPC",
                ]
            )
        )
        &
        (
            effective_k[
                "top_k"
            ]
            .isin(
                TOP_K_VALUES
            )
        )
    ].copy()


    EFFECTIVE_KEY = [

        "dataset",
        "method",
        "top_k",
        "question_id",
    ]


    if effective_k.duplicated(
        subset=EFFECTIVE_KEY
    ).any():

        raise RuntimeError(
            "Duplicate effective-k observations found."
        )


    if (
        effective_k[
            "effective_k"
        ]
        >
        effective_k[
            "top_k"
        ]
    ).any():

        raise RuntimeError(
            "effective_k > nominal top_k found."
        )


    # ---
    # 26. EFFECTIVE-K SUMMARY
    # ---

    EFFECTIVE_K_SUMMARY = (

        effective_k

        .groupby(
            [
                "dataset",
                "method",
                "top_k",
            ],
            as_index=False,
        )

        .agg(

            n_questions=(
                "question_id",
                "nunique",
            ),

            mean_effective_k=(
                "effective_k",
                "mean",
            ),

            median_effective_k=(
                "effective_k",
                "median",
            ),

            std_effective_k=(
                "effective_k",
                "std",
            ),

            min_effective_k=(
                "effective_k",
                "min",
            ),

            max_effective_k=(
                "effective_k",
                "max",
            ),
        )
    )


    # ---
    # 27. RANK-BISERIAL EFFECT SIZE
    # ---

    def paired_rank_biserial(
        a,
        b,
    ):

        a = np.asarray(
            a,
            dtype=float,
        )


        b = np.asarray(
            b,
            dtype=float,
        )


        difference = (
            a
            -
            b
        )


        difference = difference[
            difference
            != 0
        ]


        if len(
            difference
        ) == 0:

            return 0.0


        ranks = rankdata(

            np.abs(
                difference
            ),

            method="average",
        )


        w_positive = ranks[
            difference
            >
            0
        ].sum()


        w_negative = ranks[
            difference
            <
            0
        ].sum()


        denominator = (
            w_positive
            +
            w_negative
        )


        if denominator == 0:

            return 0.0


        return float(

            (
                w_positive
                -
                w_negative
            )

            /

            denominator
        )


    # ---
    # 28. WILCOXON
    # ---

    def paired_wilcoxon(
        parent_child,
        direct,
    ):

        a = np.asarray(
            parent_child,
            dtype=float,
        )


        b = np.asarray(
            direct,
            dtype=float,
        )


        difference = (
            a
            -
            b
        )


        n_nonzero = int(
            np.sum(
                difference
                != 0
            )
        )


        if n_nonzero == 0:

            statistic = 0.0
            p = 1.0


        else:

            result = wilcoxon(

                a,
                b,

                alternative=
                    "two-sided",

                zero_method=
                    "wilcox",

                correction=False,

                method="auto",
            )


            statistic = float(
                result.statistic
            )


            p = float(
                result.pvalue
            )


        return {

            "n":
                len(
                    a
                ),

            "n_nonzero":
                n_nonzero,

            "wilcoxon_W":
                statistic,

            "p_raw":
                p,

            "rank_biserial_r":
                paired_rank_biserial(
                    a,
                    b,
                ),

            "mean_parent_child":
                float(
                    np.mean(
                        a
                    )
                ),

            "mean_direct":
                float(
                    np.mean(
                        b
                    )
                ),

            "mean_difference_PC_minus_direct":
                float(
                    np.mean(
                        difference
                    )
                ),

            "median_parent_child":
                float(
                    np.median(
                        a
                    )
                ),

            "median_direct":
                float(
                    np.median(
                        b
                    )
                ),

            "median_difference_PC_minus_direct":
                float(
                    np.median(
                        difference
                    )
                ),
        }


    # ---
    # 29. HOLM
    # ---

    def add_holm(
        dataframe
    ):

        result = (
            dataframe.copy()
        )


        result[
            "p_holm"
        ] = np.nan


        result[
            "significant_holm"
        ] = False


        for _, indices in (

            result
            .groupby(
                [
                    "dataset",
                    "analysis_level",
                    "metric_family",
                ]
            )
            .groups
            .items()
        ):

            indices = list(
                indices
            )


            valid = [

                idx

                for idx
                in indices

                if pd.notna(
                    result.loc[
                        idx,
                        "p_raw"
                    ]
                )
            ]


            if not valid:

                continue


            reject, adjusted, _, _ = (
                multipletests(

                    result.loc[
                        valid,
                        "p_raw"
                    ]
                    .to_numpy(
                        dtype=float
                    ),

                    alpha=
                        ALPHA,

                    method=
                        "holm",
                )
            )


            result.loc[
                valid,
                "p_holm"
            ] = (
                adjusted
            )


            result.loc[
                valid,
                "significant_holm"
            ] = (
                reject
            )


        return result


    # ---
    # 30. LEVEL 1 — NOMINAL K
    # ---

    def run_level1(
        averaged
    ):

        rows = []


        for dataset in DATASETS:

            for family, metrics in [

                (
                    "generation",
                    GENERATION_METRICS,
                ),

                (
                    "evidence",
                    EVIDENCE_METRICS,
                ),
            ]:

                for metric in metrics:

                    metric_data = averaged[
                        (
                            averaged[
                                "dataset"
                            ]
                            ==
                            dataset
                        )
                        &
                        (
                            averaged[
                                "metric_family"
                            ]
                            ==
                            family
                        )
                        &
                        (
                            averaged[
                                "metric"
                            ]
                            ==
                            metric
                        )
                    ]


                    for retrieval_type, pair in (
                        PAIR_SPECS.items()
                    ):

                        pc_method = (
                            pair[
                                "parent_child"
                            ]
                        )


                        direct_method = (
                            pair[
                                "direct"
                            ]
                        )


                        for k in TOP_K_VALUES:

                            pc = metric_data[
                                (
                                    metric_data[
                                        "method_label"
                                    ]
                                    ==
                                    pc_method
                                )
                                &
                                (
                                    metric_data[
                                        "top_k"
                                    ]
                                    ==
                                    k
                                )
                            ][
                                [
                                    "question_id",
                                    "score",
                                ]
                            ].rename(
                                columns={
                                    "score":
                                        "pc_score"
                                }
                            )


                            direct = metric_data[
                                (
                                    metric_data[
                                        "method_label"
                                    ]
                                    ==
                                    direct_method
                                )
                                &
                                (
                                    metric_data[
                                        "top_k"
                                    ]
                                    ==
                                    k
                                )
                            ][
                                [
                                    "question_id",
                                    "score",
                                ]
                            ].rename(
                                columns={
                                    "score":
                                        "direct_score"
                                }
                            )


                            paired = pc.merge(

                                direct,

                                on=
                                    "question_id",

                                how=
                                    "inner",
                            )


                            if paired.empty:

                                continue


                            stats = (
                                paired_wilcoxon(

                                    paired[
                                        "pc_score"
                                    ],

                                    paired[
                                        "direct_score"
                                    ],
                                )
                            )


                            r = (
                                stats[
                                    "rank_biserial_r"
                                ]
                            )


                            if r > 0:

                                direction = (
                                    f"{pc_method} > "
                                    f"{direct_method}"
                                )

                            elif r < 0:

                                direction = (
                                    f"{direct_method} > "
                                    f"{pc_method}"
                                )

                            else:

                                direction = (
                                    "no direction"
                                )


                            rows.append(
                                {
                                    "analysis_level":
                                        "L1_nominal_k",

                                    "dataset":
                                        dataset,

                                    "metric_family":
                                        family,

                                    "metric":
                                        metric,

                                    "retrieval_type":
                                        retrieval_type,

                                    "parent_child_method":
                                        pc_method,

                                    "direct_method":
                                        direct_method,

                                    "parent_child_k":
                                        k,

                                    "direct_k":
                                        k,

                                    "direction":
                                        direction,

                                    **stats,
                                }
                            )


        return add_holm(
            pd.DataFrame(
                rows
            )
        )


    LEVEL1 = (
        run_level1(
            AVG_ALL
        )
    )


    # ---
    # 31. LEVEL 2 — EFFECTIVE-PARENT-ADJUSTED REGRESSION DATA
    # ---
    #
    # Model fitted separately for each:
    #   dataset × metric × retrieval comparison
    #
    # Dense:  DPC vs DP
    # Sparse: SPC vs SP
    #
    # Direct-parent effective_k = nominal top_k.
    # Parent-child effective_k = measured question-level effective_k.
    #
    # Questions are retained only if both compared methods are represented.
    # All available k observations for those questions are then used jointly.
    # ---

    def build_level2_dataset(
        averaged,
        effective_table,
        dataset,
        family,
        metric,
        pc_method,
        direct_method,
    ):

        metric_data = averaged[
            (
                averaged["dataset"] == dataset
            )
            &
            (
                averaged["metric_family"] == family
            )
            &
            (
                averaged["metric"] == metric
            )
            &
            (
                averaged["method_label"].isin(
                    [pc_method, direct_method]
                )
            )
        ][
            [
                "method_label",
                "top_k",
                "question_id",
                "score",
            ]
        ].copy()

        # ---
        # Direct-parent observations:
        # one retrieved parent per nominal retrieval slot, therefore
        # effective_k equals nominal top_k.
        # ---
        direct = metric_data[
            metric_data["method_label"] == direct_method
        ].copy()

        direct["effective_k"] = (
            direct["top_k"].astype(float)
        )

        # ---
        # Parent-child observations:
        # merge actual unique-parent counts measured per question and k.
        # ---
        pc = metric_data[
            metric_data["method_label"] == pc_method
        ].copy()

        eff = effective_table[
            (
                effective_table["dataset"] == dataset
            )
            &
            (
                effective_table["method"] == pc_method
            )
        ][
            [
                "question_id",
                "top_k",
                "effective_k",
            ]
        ].copy()

        pc = pc.merge(
            eff,
            on=["question_id", "top_k"],
            how="inner",
            validate="one_to_one",
        )

        model_data = pd.concat(
            [pc, direct],
            ignore_index=True,
            sort=False,
        )

        model_data = (
            model_data
            .dropna(
                subset=[
                    "score",
                    "effective_k",
                    "question_id",
                ]
            )
            .copy()
        )

        # ---
        # Question fixed effects identify the method coefficient from
        # within-question information. Remove questions for which only
        # one of the two compared methods is available.
        # ---
        methods_per_question = (
            model_data
            .groupby("question_id")["method_label"]
            .nunique()
        )

        valid_questions = methods_per_question[
            methods_per_question == 2
        ].index

        model_data = model_data[
            model_data["question_id"].isin(valid_questions)
        ].copy()

        if model_data.empty:
            return model_data

        model_data["is_parent_child"] = (
            model_data["method_label"] == pc_method
        ).astype(int)

        # ---
        # Center E before squaring.
        # This leaves the fitted quadratic curve unchanged but makes
        # the linear/quadratic terms less collinear and easier to estimate.
        # ---
        effective_k_mean = float(
            model_data["effective_k"].mean()
        )

        model_data["effective_k_centered"] = (
            model_data["effective_k"] - effective_k_mean
        )

        model_data["effective_k_centered_sq"] = (
            model_data["effective_k_centered"] ** 2
        )

        model_data.attrs["effective_k_mean"] = effective_k_mean

        return model_data


    # ---
    # 32. LEVEL 2 — QUESTION FIXED-EFFECTS OLS
    #     WITH QUESTION-CLUSTERED SANDWICH INFERENCE
    # ---
    #
    # score ~ parent_child
    #         + centered effective_k
    #         + centered effective_k^2
    #         + question fixed effects
    #
    # Coefficients are estimated by OLS.
    # Standard errors / CIs / p-values use a cluster-robust sandwich
    # covariance estimator with question_id as the cluster.
    #
    # The coefficient of interest is is_parent_child (beta_1).
    # ---

    def fit_effective_k_adjusted_model(
        model_data
    ):

        formula = (
            "score ~ "
            "is_parent_child + "
            "effective_k_centered + "
            "effective_k_centered_sq + "
            "C(question_id)"
        )

        model = smf.ols(
            formula,
            data=model_data,
        )

        # Small-sample covariance correction is enabled.
        # use_t=True makes coefficient tests use t-based inference.
        fit = model.fit(
            cov_type="cluster",
            cov_kwds={
                "groups": model_data["question_id"],
                "use_correction": True,
                "df_correction": True,
            },
            use_t=True,
        )

        beta_name = "is_parent_child"
        linear_name = "effective_k_centered"
        quadratic_name = "effective_k_centered_sq"

        beta_ci = fit.conf_int().loc[beta_name]
        linear_ci = fit.conf_int().loc[linear_name]
        quadratic_ci = fit.conf_int().loc[quadratic_name]

        beta = float(fit.params[beta_name])
        gamma1 = float(fit.params[linear_name])
        gamma2 = float(fit.params[quadratic_name])

        effective_k_mean = float(
            model_data.attrs.get(
                "effective_k_mean",
                model_data["effective_k"].mean(),
            )
        )

        # Turning point of the fitted quadratic on the ORIGINAL E scale:
        # E* = mean(E) - gamma1 / (2*gamma2)
        if np.isclose(gamma2, 0.0):
            turning_point = np.nan
        else:
            turning_point = (
                effective_k_mean
                -
                gamma1 / (2.0 * gamma2)
            )

        observed_min = float(
            model_data["effective_k"].min()
        )

        observed_max = float(
            model_data["effective_k"].max()
        )

        turning_point_in_range = bool(
            pd.notna(turning_point)
            and
            observed_min <= turning_point <= observed_max
        )

        return {
            "model_type":
                "OLS_question_FE_clustered_SE_quadratic_E",

            "parent_child_beta":
                beta,

            "parent_child_se_cluster":
                float(fit.bse[beta_name]),

            "parent_child_ci_low":
                float(beta_ci.iloc[0]),

            "parent_child_ci_high":
                float(beta_ci.iloc[1]),

            "p_raw":
                float(fit.pvalues[beta_name]),

            "effective_k_linear_gamma1":
                gamma1,

            "effective_k_linear_se_cluster":
                float(fit.bse[linear_name]),

            "effective_k_linear_ci_low":
                float(linear_ci.iloc[0]),

            "effective_k_linear_ci_high":
                float(linear_ci.iloc[1]),

            "effective_k_linear_p":
                float(fit.pvalues[linear_name]),

            "effective_k_quadratic_gamma2":
                gamma2,

            "effective_k_quadratic_se_cluster":
                float(fit.bse[quadratic_name]),

            "effective_k_quadratic_ci_low":
                float(quadratic_ci.iloc[0]),

            "effective_k_quadratic_ci_high":
                float(quadratic_ci.iloc[1]),

            "effective_k_quadratic_p":
                float(fit.pvalues[quadratic_name]),

            "effective_k_center_mean":
                effective_k_mean,

            "effective_k_turning_point":
                float(turning_point)
                if pd.notna(turning_point)
                else np.nan,

            "turning_point_within_observed_range":
                turning_point_in_range,

            "observed_effective_k_min":
                observed_min,

            "observed_effective_k_max":
                observed_max,

            "n_clusters_questions":
                int(model_data["question_id"].nunique()),

            "n_observations":
                int(len(model_data)),

            "df_resid":
                float(fit.df_resid),
        }


    # ---
    # 33. RUN LEVEL 2
    # ---

    def run_level2(
        averaged,
        effective_table,
    ):

        rows = []

        for dataset in DATASETS:

            for family, metrics in [
                (
                    "generation",
                    GENERATION_METRICS,
                ),
                (
                    "evidence",
                    EVIDENCE_METRICS,
                ),
            ]:

                for metric in metrics:

                    for retrieval_type, pair in PAIR_SPECS.items():

                        pc_method = pair["parent_child"]
                        direct_method = pair["direct"]

                        model_data = build_level2_dataset(
                            averaged=averaged,
                            effective_table=effective_table,
                            dataset=dataset,
                            family=family,
                            metric=metric,
                            pc_method=pc_method,
                            direct_method=direct_method,
                        )

                        if (
                            model_data.empty
                            or
                            model_data["question_id"].nunique() < 2
                        ):
                            continue

                        stats = fit_effective_k_adjusted_model(
                            model_data
                        )

                        beta = stats["parent_child_beta"]

                        if beta > 0:
                            direction = f"{pc_method} > {direct_method}"
                        elif beta < 0:
                            direction = f"{direct_method} > {pc_method}"
                        else:
                            direction = "no direction"

                        rows.append(
                            {
                                "analysis_level":
                                    "L2_effective_k_adjusted",

                                "dataset":
                                    dataset,

                                "metric_family":
                                    family,

                                "metric":
                                    metric,

                                "retrieval_type":
                                    retrieval_type,

                                "parent_child_method":
                                    pc_method,

                                "direct_method":
                                    direct_method,

                                "mean_effective_k_parent_child":
                                    float(
                                        model_data.loc[
                                            model_data["is_parent_child"] == 1,
                                            "effective_k",
                                        ].mean()
                                    ),

                                "mean_effective_k_direct":
                                    float(
                                        model_data.loc[
                                            model_data["is_parent_child"] == 0,
                                            "effective_k",
                                        ].mean()
                                    ),

                                "direction":
                                    direction,

                                **stats,
                            }
                        )

        results = pd.DataFrame(rows)

        if results.empty:
            return results

        return add_holm(results)


    LEVEL2 = run_level2(
        AVG_ALL,
        effective_k,
    )


    # ---
    # 34. PER-METRIC PATTERN SUMMARY — LEVEL 1
    # ---

    def summarize_level1_by_metric(
        results
    ):

        rows = []

        grouping = [
            "dataset",
            "analysis_level",
            "metric_family",
            "retrieval_type",
            "parent_child_method",
            "direct_method",
            "metric",
        ]

        for keys, group in results.groupby(grouping):

            (
                dataset,
                level,
                family,
                retrieval_type,
                pc_method,
                direct_method,
                metric,
            ) = keys

            pc_direction = f"{pc_method} > {direct_method}"
            direct_direction = f"{direct_method} > {pc_method}"

            significant = group[
                group["significant_holm"]
            ]

            pc_sig = significant[
                significant["direction"] == pc_direction
            ]

            direct_sig = significant[
                significant["direction"] == direct_direction
            ]

            pc_k = sorted(
                pc_sig["parent_child_k"]
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )

            direct_k = sorted(
                direct_sig["parent_child_k"]
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )

            rows.append(
                {
                    "dataset": dataset,
                    "analysis_level": level,
                    "metric_family": family,
                    "retrieval_type": retrieval_type,
                    "parent_child_method": pc_method,
                    "direct_method": direct_method,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],

                    "significant_k_parent_child":
                        ",".join(map(str, pc_k)),

                    "significant_k_direct":
                        ",".join(map(str, direct_k)),

                    "n_significant_parent_child":
                        len(pc_k),

                    "n_significant_direct":
                        len(direct_k),

                    "all_k_significant_parent_child":
                        set(pc_k) == set(TOP_K_VALUES),

                    "all_k_significant_direct":
                        set(direct_k) == set(TOP_K_VALUES),

                    "no_significant_result":
                        len(significant) == 0,

                    "minimum_holm_p":
                        float(group["p_holm"].min()),

                    "maximum_holm_p":
                        float(group["p_holm"].max()),
                }
            )

        return pd.DataFrame(rows)


    L1_PATTERN_BY_METRIC = summarize_level1_by_metric(
        LEVEL1
    )


    # ---
    # 35. PER-METRIC PATTERN SUMMARY — LEVEL 2
    # ---

    def summarize_level2_by_metric(
        results
    ):

        rows = []

        for _, row in results.iterrows():

            pc_direction = (
                f"{row['parent_child_method']} > "
                f"{row['direct_method']}"
            )

            direct_direction = (
                f"{row['direct_method']} > "
                f"{row['parent_child_method']}"
            )

            rows.append(
                {
                    "dataset":
                        row["dataset"],

                    "analysis_level":
                        row["analysis_level"],

                    "metric_family":
                        row["metric_family"],

                    "retrieval_type":
                        row["retrieval_type"],

                    "parent_child_method":
                        row["parent_child_method"],

                    "direct_method":
                        row["direct_method"],

                    "metric":
                        row["metric"],

                    "metric_label":
                        METRIC_LABELS[row["metric"]],

                    "significant_parent_child":
                        bool(
                            row["significant_holm"]
                            and
                            row["direction"] == pc_direction
                        ),

                    "significant_direct":
                        bool(
                            row["significant_holm"]
                            and
                            row["direction"] == direct_direction
                        ),

                    "no_significant_result":
                        not bool(row["significant_holm"]),

                    "parent_child_beta":
                        row["parent_child_beta"],

                    "parent_child_se_cluster":
                        row["parent_child_se_cluster"],

                    "parent_child_ci_low":
                        row["parent_child_ci_low"],

                    "parent_child_ci_high":
                        row["parent_child_ci_high"],

                    "p_raw":
                        row["p_raw"],

                    "p_holm":
                        row["p_holm"],

                    "gamma1_linear_E":
                        row["effective_k_linear_gamma1"],

                    "gamma2_quadratic_E":
                        row["effective_k_quadratic_gamma2"],

                    "effective_k_turning_point":
                        row["effective_k_turning_point"],

                    "turning_point_within_observed_range":
                        row["turning_point_within_observed_range"],
                }
            )

        return pd.DataFrame(rows)


    L2_PATTERN_BY_METRIC = summarize_level2_by_metric(
        LEVEL2
    )


    # ---
    # 36. BUILD COMPACT KEY SUMMARY
    # ---

    def format_level1_significance(
        subset,
        side,
    ):

        entries = []

        for _, row in (
            subset
            .sort_values("metric")
            .iterrows()
        ):

            metric = row["metric_label"]

            if side == "pc":
                k_text = row["significant_k_parent_child"]
                all_k = row["all_k_significant_parent_child"]
            else:
                k_text = row["significant_k_direct"]
                all_k = row["all_k_significant_direct"]

            if not k_text:
                continue

            if all_k:
                entries.append(
                    f"{metric} (all k)"
                )
            else:
                entries.append(
                    f"{metric} (k={k_text})"
                )

        return (
            "; ".join(entries)
            if entries
            else "None"
        )


    def format_no_significance(
        subset
    ):

        metrics = (
            subset.loc[
                subset["no_significant_result"],
                "metric_label",
            ]
            .tolist()
        )

        return (
            ", ".join(metrics)
            if metrics
            else "None"
        )


    def build_key_summary_level1(
        pattern_table
    ):

        rows = []

        grouping = [
            "dataset",
            "analysis_level",
            "retrieval_type",
            "parent_child_method",
            "direct_method",
        ]

        for keys, group in pattern_table.groupby(grouping):

            (
                dataset,
                level,
                retrieval_type,
                pc,
                direct,
            ) = keys

            generation = group[
                group["metric_family"] == "generation"
            ]

            evidence = group[
                group["metric_family"] == "evidence"
            ]

            rows.append(
                {
                    "dataset": dataset,
                    "analysis_level": level,
                    "contrast": f"{pc} vs {direct}",

                    "generation_PC_significant":
                        format_level1_significance(
                            generation,
                            "pc",
                        ),

                    "generation_direct_significant":
                        format_level1_significance(
                            generation,
                            "direct",
                        ),

                    "generation_no_significance":
                        format_no_significance(
                            generation
                        ),

                    "evidence_PC_significant":
                        format_level1_significance(
                            evidence,
                            "pc",
                        ),

                    "evidence_direct_significant":
                        format_level1_significance(
                            evidence,
                            "direct",
                        ),

                    "evidence_no_significance":
                        format_no_significance(
                            evidence
                        ),
                }
            )

        return pd.DataFrame(rows)


    def format_level2_significance(
        subset,
        side,
    ):

        if side == "pc":
            selected = subset[
                subset["significant_parent_child"]
            ]
        else:
            selected = subset[
                subset["significant_direct"]
            ]

        if selected.empty:
            return "None"

        entries = []

        for _, row in selected.sort_values("metric").iterrows():
            entries.append(
                f"{row['metric_label']} "
                f"(p_Holm={row['p_holm']:.4g}, "
                f"beta={row['parent_child_beta']:.4g})"
            )

        return "; ".join(entries)


    def build_key_summary_level2(
        pattern_table
    ):

        rows = []

        grouping = [
            "dataset",
            "analysis_level",
            "retrieval_type",
            "parent_child_method",
            "direct_method",
        ]

        for keys, group in pattern_table.groupby(grouping):

            (
                dataset,
                level,
                retrieval_type,
                pc,
                direct,
            ) = keys

            generation = group[
                group["metric_family"] == "generation"
            ]

            evidence = group[
                group["metric_family"] == "evidence"
            ]

            rows.append(
                {
                    "dataset": dataset,
                    "analysis_level": level,
                    "contrast": f"{pc} vs {direct}",

                    "generation_PC_significant":
                        format_level2_significance(
                            generation,
                            "pc",
                        ),

                    "generation_direct_significant":
                        format_level2_significance(
                            generation,
                            "direct",
                        ),

                    "generation_no_significance":
                        format_no_significance(
                            generation
                        ),

                    "evidence_PC_significant":
                        format_level2_significance(
                            evidence,
                            "pc",
                        ),

                    "evidence_direct_significant":
                        format_level2_significance(
                            evidence,
                            "direct",
                        ),

                    "evidence_no_significance":
                        format_no_significance(
                            evidence
                        ),
                }
            )

        return pd.DataFrame(rows)


    KEY_SUMMARY = pd.concat(
        [
            build_key_summary_level1(
                L1_PATTERN_BY_METRIC
            ),
            build_key_summary_level2(
                L2_PATTERN_BY_METRIC
            ),
        ],
        ignore_index=True,
    )

    KEY_SUMMARY["dataset_order"] = (
        KEY_SUMMARY["dataset"].map(
            {
                "tender": 0,
                "ledgar": 1,
                "hupd": 2,
            }
        )
    )

    KEY_SUMMARY["level_order"] = (
        KEY_SUMMARY["analysis_level"].map(
            {
                "L1_nominal_k": 0,
                "L2_effective_k_adjusted": 1,
            }
        )
    )

    KEY_SUMMARY = (
        KEY_SUMMARY
        .sort_values(
            [
                "dataset_order",
                "level_order",
                "contrast",
            ]
        )
        .drop(
            columns=[
                "dataset_order",
                "level_order",
            ]
        )
        .reset_index(drop=True)
    )


    # ---
    # 37. SIGNIFICANT-ONLY TABLES
    # ---

    LEVEL1_SIGNIFICANT = LEVEL1[
        LEVEL1["significant_holm"]
    ].copy()

    LEVEL2_SIGNIFICANT = LEVEL2[
        LEVEL2["significant_holm"]
    ].copy()


    # ---
    # 38. README SHEET
    # ---

    README = pd.DataFrame(
        {
            "Item": [
                "RQ2",
                "Level 1",
                "Level 2",
                "Level 2 formula",
                "Level 2 inference",
                "Generator averaging",
                "Multiplicity correction",
                "Direction",
                "Effective-k source",
                "KEY_SUMMARY",
                "Local input scope",
                "Output destination",
            ],

            "Description": [
                "Parent-child retrieval vs direct-parent retrieval: DPC vs DP and SPC vs SP.",

                "Primary analysis: paired two-sided Wilcoxon signed-rank tests at identical nominal top-k values (5, 10, 15, 20).",

                "Sensitivity analysis: question fixed-effects OLS using all k observations jointly and adjusting continuously for actual effective parent count.",

                "score = question fixed effect + beta_PC*parent_child + gamma1*(E-mean(E)) + gamma2*(E-mean(E))^2 + error.",

                "Coefficient covariance, standard errors, confidence intervals and p-values use the question-clustered sandwich estimator with small-sample covariance correction and t-based inference. No bootstrap is used.",

                "Metrics are averaged across all expected generators before inferential testing; incomplete generator configurations are excluded.",

                "Holm correction separately within dataset × analysis level × metric family. For Level 2, the correction is applied to the parent-child coefficient tests across DPC-vs-DP and SPC-vs-SP metrics in the family.",

                "For Level 1, positive rank-biserial means parent-child is higher. For Level 2, positive parent_child_beta means parent-child is higher after effective-k adjustment.",

                "The local effective_k_DPC_SPC_question_level.xlsx workbook is read from evaluation/analysis/effective_k_analysis_results. For direct-parent retrieval, effective_k is defined as nominal top_k.",

                "Compact overview for thesis interpretation. Level 1 cells show significant k values; Level 2 cells show significant metrics with Holm p-values and adjusted parent-child beta estimates.",

                "Input data are read only from the dataset results folders inside experiment_runner_final.",

                "The significance workbook is saved to evaluation/analysis/significance_testing.",
            ],
        }
    )


    # ---
    # 39. WRITE LOCAL TEMPORARY XLSX
    # ---

    print()
    print(
        "Creating local result workbook..."
    )

    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl",
    ) as writer:

        KEY_SUMMARY.to_excel(
            writer,
            sheet_name="KEY_SUMMARY",
            index=False,
        )

        L1_PATTERN_BY_METRIC.to_excel(
            writer,
            sheet_name="L1_metric_patterns",
            index=False,
        )

        L2_PATTERN_BY_METRIC.to_excel(
            writer,
            sheet_name="L2_metric_patterns",
            index=False,
        )

        LEVEL1_SIGNIFICANT.to_excel(
            writer,
            sheet_name="L1_significant",
            index=False,
        )

        LEVEL2_SIGNIFICANT.to_excel(
            writer,
            sheet_name="L2_significant",
            index=False,
        )

        LEVEL1.to_excel(
            writer,
            sheet_name="L1_full",
            index=False,
        )

        LEVEL2.to_excel(
            writer,
            sheet_name="L2_full",
            index=False,
        )

        EFFECTIVE_K_SUMMARY.to_excel(
            writer,
            sheet_name="effective_k_summary",
            index=False,
        )

        INCOMPLETE_CONFIG_SUMMARY.to_excel(
            writer,
            sheet_name="data_quality",
            index=False,
        )

        GENERATION_FILE_INDEX.to_excel(
            writer,
            sheet_name="generation_files",
            index=False,
        )

        EVIDENCE_FILE_INDEX.to_excel(
            writer,
            sheet_name="evidence_files",
            index=False,
        )

        README.to_excel(
            writer,
            sheet_name="README",
            index=False,
        )


    # ---
    # 40. IMPROVE WORKBOOK READABILITY
    # ---

    from openpyxl import load_workbook
    from openpyxl.styles import Font, Alignment

    workbook = load_workbook(
        OUTPUT_FILE
    )

    for sheet in workbook.worksheets:

        sheet.freeze_panes = "A2"

        if sheet.max_row > 1:
            sheet.auto_filter.ref = sheet.dimensions

        for cell in sheet[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(
                vertical="top",
                wrap_text=True,
            )

        for column_cells in sheet.columns:

            column_letter = column_cells[0].column_letter
            max_length = 0

            for cell in column_cells:
                if cell.value is None:
                    continue

                max_length = max(
                    max_length,
                    len(str(cell.value)),
                )

            sheet.column_dimensions[column_letter].width = min(
                max(max_length + 2, 12),
                60,
            )

    workbook.save(
        OUTPUT_FILE
    )


    # ---
    # 41. PRINT ONLY THE IMPORTANT SUMMARY
    # ---

    print()
    print("=" * 120)
    print("RQ2 — KEY SIGNIFICANCE SUMMARY")
    print("=" * 120)
    print()

    display(
        KEY_SUMMARY
    )

    print()
    print("Interpretation:")
    print(
        "  Level 1: '(all k)' = Holm-significant at nominal "
        "k = 5, 10, 15, and 20; '(k=...)' lists only significant k values."
    )
    print(
        "  Level 2: beta is the adjusted parent-child minus direct-parent difference "
        "after question fixed effects and quadratic effective-k adjustment."
    )
    print(
        "  Positive beta = DPC/SPC higher; negative beta = DP/SP higher."
    )
    print(
        "  Level 2 p-values use question-clustered sandwich standard errors and Holm correction."
    )


    print(f"\n\nSaved significance workbook to: {OUTPUT_FILE}")
    return OUTPUT_FILE


def run_rq3(data_root=PROJECT_ROOT, datasets=DATASETS, output_dir=None):
    """Run the original RQ3 analysis with independent local state."""
    # ---
    # RQ3 SIGNIFICANCE TESTING
    # ---
    #
    # RQ3:
    # How do different retrieval strategies, namely sparse, dense,
    # and hybrid retrieval, affect evidence support and generation
    # quality of hierarchical RAG systems for RTUG?
    #
    # METHODS:
    #     DPC = dense parent-child
    #     SPC = sparse parent-child
    #     HPC = hybrid parent-child
    #
    # PRIMARY ANALYSIS:
    #
    # For each dataset × metric × nominal top-k:
    #
    #     DPC_k vs SPC_k vs HPC_k
    #
    # k ∈ {5, 10, 15, 20}
    #
    # 1. Average each metric over all expected generators
    #    for each question × method × top-k.
    #
    # 2. Restrict each Friedman test to the intersection of
    #    question IDs available for DPC, SPC and HPC.
    #
    # 3. Friedman omnibus test:
    #
    #       H0:
    #       DPC, SPC and HPC have the same within-question
    #       rank distribution.
    #
    # 4. Kendall's W:
    #
    #       W = Friedman_chi2 / [N * (K - 1)]
    #
    #       with K = 3 methods.
    #
    # 5. Holm correction for omnibus tests separately within:
    #
    #       dataset × metric family
    #
    #    Generation:
    #       7 metrics × 4 k = 28 tests / dataset
    #
    #    Evidence:
    #       2 metrics × 4 k = 8 tests / dataset
    #
    # 6. ONLY if the Holm-adjusted Friedman test is significant:
    #
    #       DPC vs SPC
    #       DPC vs HPC
    #       SPC vs HPC
    #
    #    using paired two-sided Wilcoxon signed-rank tests.
    #
    # 7. Holm correction across the 3 Wilcoxon tests within
    #    that one dataset × metric × k matrix.
    #
    # 8. Pairwise direction is determined AFTER the two-sided test
    #    from the rank-biserial effect size / paired score difference.
    #
    #
    # SIGNIFICANCE:
    #
    #     alpha = .05
    #
    #     Final inference uses Holm-adjusted p-values.
    #
    #
    # METRIC FAMILIES:
    #
    # Generation:
    #     ROUGE-1
    #     ROUGE-2
    #     ROUGE-L
    #     BERTScore
    #     SimCSE
    #     BGE-M3 ECS
    #     MATCHA
    #
    # Evidence:
    #     Faithfulness
    #     Context Recall
    #
    #
    # OUTPUT:
    # The workbook is saved in the significance_testing folder.
    # ---


    # ---
    # 1. IMPORTS
    # ---

    import warnings
    from pathlib import Path
    import sys

    import numpy as np
    import pandas as pd

    from scipy.stats import (
        friedmanchisquare,
        wilcoxon,
        rankdata,
    )

    from statsmodels.stats.multitest import multipletests

    # Use the script's console-aware display helper.




    from openpyxl import load_workbook
    from openpyxl.styles import Font, Alignment


    # ---
    # 2. CONFIG
    # ---

    DATASETS = list(normalize_datasets(datasets))

    METHODS = [
        "DPC",
        "SPC",
        "HPC",
    ]

    TOP_K_VALUES = [
        5,
        10,
        15,
        20,
    ]

    ALPHA = 0.05

    PROJECT_ROOT = Path(data_root).resolve()
    SIGNIFICANCE_DIR = Path(output_dir).resolve() if output_dir is not None else (
        PROJECT_ROOT / "evaluation/analysis/significance_testing"
    )
    SIGNIFICANCE_DIR.mkdir(parents=True, exist_ok=True)
    list_result_children, read_result_file, FOLDER_MIME = make_local_readers(PROJECT_ROOT)

    # Read inputs from each dataset's existing result folders.
    RESULT_FOLDERS = {
        dataset: {
            "generation": PROJECT_ROOT / f"{dataset}_dataset" / "results" / "metric_analysis",
            "evidence": PROJECT_ROOT / f"{dataset}_dataset" / "results" / "retrieval_analysis",
        }
        for dataset in DATASETS
    }

    OUTPUT_FILE = output_file(SIGNIFICANCE_DIR, "RQ3_significance_results.xlsx", DATASETS)



    # ---
    # 5. METHOD DEFINITIONS
    # ---

    RAW_METHOD_MAP = {

        "dense_parent_child":
            "DPC",

        "sparse_parent_child":
            "SPC",

        "hybrid_parent_child":
            "HPC",
    }


    EVIDENCE_METHOD_NAMES = {

        "DPC":
            "dense_parent_child",

        "SPC":
            "sparse_parent_child",

        "HPC":
            "hybrid_parent_child",
    }


    PAIR_SPECS = [

        ("DPC", "SPC"),
        ("DPC", "HPC"),
        ("SPC", "HPC"),
    ]


    # ---
    # 6. METRICS
    # ---

    GENERATION_METRICS = [

        "rouge1_f1",
        "rouge2_f1",
        "rougeL_f1",

        "bertscore_f1",
        "simcse_f1",

        "ecs_bge_m3",
        "matcha_f1",
    ]


    EVIDENCE_METRICS = [

        "faithfulness",
        "context_recall",
    ]


    METRIC_LABELS = {

        "rouge1_f1":
            "ROUGE-1",

        "rouge2_f1":
            "ROUGE-2",

        "rougeL_f1":
            "ROUGE-L",

        "bertscore_f1":
            "BERTScore",

        "simcse_f1":
            "SimCSE",

        "ecs_bge_m3":
            "BGE-M3 ECS",

        "matcha_f1":
            "MATCHA",

        "faithfulness":
            "Faithfulness",

        "context_recall":
            "Context Recall",
    }


    # ---
    # 7. EXPECTED GENERATORS
    # ---

    EXPECTED_GENERATORS = {

        "tender": {

            "meta-llama/Llama-3.1-8B-Instruct-instruct",
            "meta-llama/Llama-3.1-70B-Instruct-instruct",

            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",

            "Qwen/Qwen3-235B-A22B-Instruct-2507",
        },

        "ledgar": {

            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
        },

        "hupd": {

            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
        },
    }


    EXPECTED_N_GENERATORS = {

        dataset:
            len(generators)

        for dataset, generators
        in EXPECTED_GENERATORS.items()
    }


    # ---
    # 9-11. SHARED LOCAL FILE HELPERS
    # These helpers only read files under experiment_runner_final.
    # ---

    # ---
    # 12. GENERATION FOLDER CLASSIFICATION
    # ---

    def classify_generation_folder(
        dataset,
        folder_name,
    ):

        name = (
            str(folder_name)
            .strip()
            .lower()
        )

        # Never enter irrelevant experiment families.
        forbidden_tokens = [

            "agentic",
            "bm25l",
            "sparse_parent_child_reversed",
        ]

        if any(
            token in name
            for token in forbidden_tokens
        ):
            return None

        if (
            f"{dataset}_dense_parent_child_"
            in name
        ):
            return "DPC"

        if (
            f"{dataset}_sparse_parent_child_"
            in name
        ):
            return "SPC"

        if (
            f"{dataset}_hybrid_parent_child_"
            in name
        ):
            return "HPC"

        return None


    # ---
    # 13. SELECT PER-QUESTION GENERATION FILE
    # ---

    def select_per_question_generation_file(
        files_in_folder
    ):

        candidates = []

        for item in files_in_folder:

            if (
                item.get("mimeType")
                ==
                FOLDER_MIME
            ):
                continue

            name = (
                item
                .get(
                    "name",
                    ""
                )
                .lower()
            )

            if not name.startswith(
                "per_question_metrics_"
            ):
                continue

            if (
                "_all_generators"
                not in name
            ):
                continue

            if name.endswith(".xlsx"):
                priority = 0

            elif name.endswith(".csv"):
                priority = 1

            else:
                continue

            candidates.append(
                (
                    priority,
                    item,
                )
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda x: (
                x[0],
                x[1]["name"].lower(),
            )
        )

        best_priority = candidates[0][0]

        best = [

            item

            for priority, item
            in candidates

            if priority == best_priority
        ]

        if len(best) != 1:

            raise RuntimeError(
                "Ambiguous per-question generation file."
            )

        return best[0]


    # ---
    # 14. DISCOVER GENERATION FILES
    # ---

    def discover_generation_files():

        selected = []

        for dataset in DATASETS:

            root = (
                RESULT_FOLDERS[
                    dataset
                ][
                    "generation"
                ]
            )

            root_children = list_result_children(
                root
            )

            for child in root_children:

                if (
                    child.get("mimeType")
                    !=
                    FOLDER_MIME
                ):
                    continue

                child_id = child["id"]

                method = classify_generation_folder(
                    dataset,
                    child["name"],
                )

                # Skip BEFORE entering.
                if method is None:
                    continue

                contents = list_result_children(
                    child_id
                )

                selected_file = (
                    select_per_question_generation_file(
                        contents
                    )
                )

                if selected_file is None:

                    raise RuntimeError(
                        "No per-question generation file "
                        f"found in {child['name']}"
                    )

                selected.append(
                    {
                        "dataset":
                            dataset,

                        "method_expected":
                            method,

                        "generator_folder":
                            child["name"],

                        "file_id":
                            selected_file["id"],

                        "file_name":
                            selected_file["name"],
                    }
                )

        result = pd.DataFrame(
            selected
        )

        if result.empty:

            raise RuntimeError(
                "No generation files were discovered."
            )

        return result


    GENERATION_FILE_INDEX = (
        discover_generation_files()
    )


    # ---
    # 15. DISCOVER EVIDENCE FILES
    # ---

    def select_evidence_file(
        dataset,
        method_code,
        files,
    ):

        method_name = (
            EVIDENCE_METHOD_NAMES[
                method_code
            ]
        )

        expected = (
            f"per_question_retrieval_metrics_"
            f"{dataset}_"
            f"{method_name}_"
            f"faithfulness_context_recall"
        ).lower()

        candidates = []

        for item in files:

            if (
                item.get("mimeType")
                ==
                FOLDER_MIME
            ):
                continue

            name = (
                item
                .get(
                    "name",
                    ""
                )
                .lower()
            )

            if not name.startswith(
                expected
            ):
                continue

            if name.endswith(".xlsx"):
                priority = 0

            elif name.endswith(".csv"):
                priority = 1

            else:
                continue

            candidates.append(
                (
                    priority,
                    item,
                )
            )

        if not candidates:

            raise RuntimeError(
                f"No evidence file found for "
                f"{dataset}/{method_code}"
            )

        candidates.sort(
            key=lambda x: (
                x[0],
                x[1]["name"].lower(),
            )
        )

        best_priority = candidates[0][0]

        best = [

            item

            for priority, item
            in candidates

            if priority == best_priority
        ]

        if len(best) != 1:

            raise RuntimeError(
                f"Ambiguous evidence file for "
                f"{dataset}/{method_code}"
            )

        return best[0]


    def discover_evidence_files():

        selected = []

        for dataset in DATASETS:

            root = (
                RESULT_FOLDERS[
                    dataset
                ][
                    "evidence"
                ]
            )

            files = list_result_children(
                root
            )

            for method in METHODS:

                item = select_evidence_file(
                    dataset,
                    method,
                    files,
                )

                selected.append(
                    {
                        "dataset":
                            dataset,

                        "method_expected":
                            method,

                        "file_id":
                            item["id"],

                        "file_name":
                            item["name"],
                    }
                )

        return pd.DataFrame(
            selected
        )


    EVIDENCE_FILE_INDEX = (
        discover_evidence_files()
    )


    # ---
    # 16. NORMALIZATION HELPERS
    # ---

    def normalize_question_id(series):

        return (
            series
            .astype(str)
            .str.strip()
            .str.replace(
                r"\.0$",
                "",
                regex=True,
            )
        )


    def canonicalize_generator(value):

        if pd.isna(value):
            return np.nan

        original = str(value)

        s = (
            original
            .strip()
            .lower()
            .replace("_", "")
            .replace(" ", "")
        )

        # Llama 3.1
        if (
            "llama3.1" in s
            or
            "llama-3.1" in s
        ):

            if "70b" in s:
                return "meta-llama/Llama-3.1-70B-Instruct-instruct"

            if "8b" in s:
                return "meta-llama/Llama-3.1-8B-Instruct-instruct"

        # Qwen 2.5
        if (
            "qwen2.5" in s
            or
            "qwen-2.5" in s
        ):

            if "72b" in s:
                return "Qwen/Qwen2.5-72B-Instruct"

            if "32b" in s:
                return "Qwen/Qwen2.5-32B-Instruct"

            if "7b" in s:
                return "Qwen/Qwen2.5-7B-Instruct"

        # Qwen 3
        if (
            "qwen3" in s
            or
            "qwen-3" in s
        ):

            if "235b" in s:
                return "Qwen/Qwen3-235B-A22B-Instruct-2507"

        return (
            "UNKNOWN::"
            +
            original
        )


    def first_non_null(series):

        values = series.dropna()

        if len(values) == 0:
            return np.nan

        return values.iloc[0]


    # ---
    # 17. PREPARE METRIC DATAFRAME
    # ---

    def prepare_metric_dataframe(
        df,
        dataset_expected,
        method_expected,
        metrics,
        source_name,
    ):

        # Remove accidental Excel index columns.
        df = df.loc[
            :,
            ~df.columns
            .astype(str)
            .str.startswith(
                "Unnamed:"
            )
        ].copy()

        required = {
            "dataset",
            "method",
            "top_k",
            "generator_model",
        }

        missing = (
            required
            -
            set(df.columns)
        )

        if missing:

            raise RuntimeError(
                f"Missing columns in {source_name}: "
                f"{sorted(missing)}"
            )

        # Question ID
        if "test_id" in df.columns:

            qid = (
                df["test_id"]
                .copy()
            )

            if "test_case_id" in df.columns:

                qid = qid.fillna(
                    df["test_case_id"]
                )

        elif "test_case_id" in df.columns:

            qid = (
                df["test_case_id"]
                .copy()
            )

        else:

            raise RuntimeError(
                f"No question ID in {source_name}"
            )

        df["question_id"] = (
            normalize_question_id(
                qid
            )
        )

        df["dataset"] = (
            df["dataset"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df["method_raw"] = (
            df["method"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df["method_label"] = (
            df["method_raw"]
            .map(
                RAW_METHOD_MAP
            )
        )

        df["generator"] = (
            df["generator_model"]
            .apply(
                canonicalize_generator
            )
        )

        df["top_k"] = pd.to_numeric(
            df["top_k"],
            errors="coerce",
        )

        missing_metrics = (
            set(metrics)
            -
            set(df.columns)
        )

        if missing_metrics:

            raise RuntimeError(
                f"Missing metrics in {source_name}: "
                f"{sorted(missing_metrics)}"
            )

        for metric in metrics:

            df[metric] = pd.to_numeric(
                df[metric],
                errors="coerce",
            )

        df = df[
            (
                df["dataset"]
                ==
                dataset_expected
            )
            &
            (
                df["method_label"]
                ==
                method_expected
            )
            &
            (
                df["top_k"]
                .isin(
                    TOP_K_VALUES
                )
            )
            &
            (
                df["generator"]
                .isin(
                    EXPECTED_GENERATORS[
                        dataset_expected
                    ]
                )
            )
        ].copy()

        df["top_k"] = (
            df["top_k"]
            .astype(int)
        )

        return df


    # ---
    # 18. LOAD GENERATION METRICS
    # ---

    def load_generation_metrics():

        frames = []

        print(
            "Loading generation metrics..."
        )

        for _, row in (
            GENERATION_FILE_INDEX
            .iterrows()
        ):

            raw = (
                read_result_file(
                    {
                        "id":
                            row["file_id"],

                        "name":
                            row["file_name"],
                    }
                )
            )

            prepared = (
                prepare_metric_dataframe(
                    raw,
                    row["dataset"],
                    row["method_expected"],
                    GENERATION_METRICS,
                    row["file_name"],
                )
            )

            frames.append(
                prepared
            )

        return pd.concat(
            frames,
            ignore_index=True,
            sort=False,
        )


    RAW_GENERATION = (
        load_generation_metrics()
    )


    # ---
    # 19. LOAD EVIDENCE METRICS
    # ---

    def load_evidence_metrics():

        frames = []

        print(
            "Loading evidence-support metrics..."
        )

        for _, row in (
            EVIDENCE_FILE_INDEX
            .iterrows()
        ):

            raw = (
                read_result_file(
                    {
                        "id":
                            row["file_id"],

                        "name":
                            row["file_name"],
                    }
                )
            )

            prepared = (
                prepare_metric_dataframe(
                    raw,
                    row["dataset"],
                    row["method_expected"],
                    EVIDENCE_METRICS,
                    row["file_name"],
                )
            )

            frames.append(
                prepared
            )

        return pd.concat(
            frames,
            ignore_index=True,
            sort=False,
        )


    RAW_EVIDENCE = (
        load_evidence_metrics()
    )


    # ---
    # 20. VALIDATE GLOBAL GENERATOR SETS
    # ---
    #
    # This verifies that every retrieval strategy has observations
    # from the complete expected generator set somewhere in the data.
    #
    # Individual incomplete question/configuration combinations are
    # handled later by complete-generator averaging.
    #
    # ---

    def validate_generator_sets(raw):

        for dataset in DATASETS:

            expected = (
                EXPECTED_GENERATORS[
                    dataset
                ]
            )

            for method in METHODS:

                found = set(

                    raw.loc[
                        (
                            raw["dataset"]
                            ==
                            dataset
                        )
                        &
                        (
                            raw["method_label"]
                            ==
                            method
                        ),
                        "generator",
                    ]
                    .dropna()
                    .unique()
                )

                if found != expected:

                    raise RuntimeError(
                        f"Generator-set mismatch for "
                        f"{dataset}/{method}.\n"
                        f"Expected: {sorted(expected)}\n"
                        f"Found: {sorted(found)}"
                    )


    validate_generator_sets(
        RAW_GENERATION
    )

    validate_generator_sets(
        RAW_EVIDENCE
    )


    # ---
    # 21. DEDUPLICATE IDENTICAL COPIES
    # ---
    #
    # Some all_generators workbooks may be duplicated across
    # generator-named folders.
    #
    # Exact duplicates are collapsed.
    #
    # Conflicting values for the same observation stop execution.
    #
    # ---

    def deduplicate_raw(
        raw,
        metrics,
    ):

        key = [
            "dataset",
            "method_label",
            "top_k",
            "generator",
            "question_id",
        ]

        for metric in metrics:

            counts = (
                raw
                .groupby(
                    key,
                    dropna=False,
                )[metric]
                .nunique(
                    dropna=True
                )
            )

            if (
                counts
                >
                1
            ).any():

                problematic = (
                    counts[
                        counts > 1
                    ]
                    .reset_index()
                )

                raise RuntimeError(
                    f"Conflicting duplicate values found "
                    f"for metric {metric}.\n"
                    f"{problematic.head(20)}"
                )

        return (
            raw
            .groupby(
                key,
                as_index=False,
                dropna=False,
            )[metrics]
            .agg(
                first_non_null
            )
        )


    RAW_GENERATION = (
        deduplicate_raw(
            RAW_GENERATION,
            GENERATION_METRICS,
        )
    )


    RAW_EVIDENCE = (
        deduplicate_raw(
            RAW_EVIDENCE,
            EVIDENCE_METRICS,
        )
    )


    # ---
    # 22. AVERAGE OVER GENERATORS
    # ---
    #
    # Inferential unit = QUESTION.
    #
    # Before statistical testing:
    #
    #     question × method × top-k × metric
    #
    # is averaged over ALL expected generators.
    #
    # A question/configuration is retained for a metric only if all
    # expected generator values are present:
    #
    #     Tender = 6
    #     LEDGAR = 3
    #     HUPD   = 3
    #
    # ---

    def generator_average(
        raw,
        metrics,
        family,
    ):

        valid_parts = []
        issue_parts = []

        for dataset in DATASETS:

            expected_n = (
                EXPECTED_N_GENERATORS[
                    dataset
                ]
            )

            dataset_raw = raw[
                raw["dataset"]
                ==
                dataset
            ]

            for metric in metrics:

                grouped = (
                    dataset_raw
                    .groupby(
                        [
                            "dataset",
                            "method_label",
                            "top_k",
                            "question_id",
                        ],
                        as_index=False,
                    )
                    .agg(

                        score=(
                            metric,
                            "mean",
                        ),

                        n_metric_values=(
                            metric,
                            "count",
                        ),

                        n_generators=(
                            "generator",
                            "nunique",
                        ),
                    )
                )

                grouped["metric"] = metric
                grouped["metric_family"] = family
                grouped["expected_generators"] = expected_n

                complete = (
                    (
                        grouped["n_metric_values"]
                        ==
                        expected_n
                    )
                    &
                    (
                        grouped["n_generators"]
                        ==
                        expected_n
                    )
                )

                valid_parts.append(
                    grouped[
                        complete
                    ].copy()
                )

                incomplete = (
                    grouped[
                        ~complete
                    ].copy()
                )

                if not incomplete.empty:

                    issue_parts.append(
                        incomplete
                    )

        averaged = pd.concat(
            valid_parts,
            ignore_index=True,
        )

        if issue_parts:

            issues = pd.concat(
                issue_parts,
                ignore_index=True,
            )

        else:

            issues = pd.DataFrame()

        return (
            averaged,
            issues,
        )


    AVG_GENERATION, GENERATION_ISSUES = (
        generator_average(
            RAW_GENERATION,
            GENERATION_METRICS,
            "generation",
        )
    )


    AVG_EVIDENCE, EVIDENCE_ISSUES = (
        generator_average(
            RAW_EVIDENCE,
            EVIDENCE_METRICS,
            "evidence",
        )
    )


    AVG_ALL = pd.concat(
        [
            AVG_GENERATION,
            AVG_EVIDENCE,
        ],
        ignore_index=True,
    )


    ALL_GENERATOR_ISSUES = pd.concat(
        [
            GENERATION_ISSUES,
            EVIDENCE_ISSUES,
        ],
        ignore_index=True,
    )


    # ---
    # 23. COMPACT DATA QUALITY SUMMARY
    # ---

    if not ALL_GENERATOR_ISSUES.empty:

        INCOMPLETE_CONFIG_SUMMARY = (

            ALL_GENERATOR_ISSUES[
                [
                    "dataset",
                    "method_label",
                    "top_k",
                    "question_id",
                ]
            ]
            .drop_duplicates()

            .groupby(
                [
                    "dataset",
                    "method_label",
                    "top_k",
                ],
                as_index=False,
            )

            .size()

            .rename(
                columns={
                    "size":
                        "n_incomplete_question_configs"
                }
            )
        )

    else:

        INCOMPLETE_CONFIG_SUMMARY = pd.DataFrame(
            columns=[
                "dataset",
                "method_label",
                "top_k",
                "n_incomplete_question_configs",
            ]
        )


    # ---
    # 24. MATCHED THREE-METHOD MATRIX
    # ---
    #
    # For every dataset × metric × k:
    #
    # rows    = matched questions
    # columns = DPC, SPC, HPC
    #
    # Only questions available for ALL THREE methods are retained.
    #
    # ---

    def get_matched_matrix(
        averaged,
        dataset,
        family,
        metric,
        top_k,
    ):

        subset = averaged[
            (
                averaged["dataset"]
                ==
                dataset
            )
            &
            (
                averaged["metric_family"]
                ==
                family
            )
            &
            (
                averaged["metric"]
                ==
                metric
            )
            &
            (
                averaged["top_k"]
                ==
                top_k
            )
            &
            (
                averaged["method_label"]
                .isin(
                    METHODS
                )
            )
        ][
            [
                "question_id",
                "method_label",
                "score",
            ]
        ].copy()

        if subset.empty:
            return pd.DataFrame(
                columns=METHODS
            )

        if subset.duplicated(
            subset=[
                "question_id",
                "method_label",
            ]
        ).any():

            raise RuntimeError(
                "Duplicate question × method values found "
                f"for {dataset}/{family}/{metric}/k={top_k}"
            )

        matrix = (
            subset
            .pivot(
                index="question_id",
                columns="method_label",
                values="score",
            )
        )

        for method in METHODS:

            if method not in matrix.columns:

                return pd.DataFrame(
                    columns=METHODS
                )

        matrix = (
            matrix[
                METHODS
            ]
            .dropna(
                how="any"
            )
            .sort_index()
        )

        return matrix


    # ---
    # 25. MEAN RANKS
    # ---
    #
    # Higher metric scores are better.
    #
    # Therefore:
    #
    #     rank 1 = best
    #     rank 2 = middle
    #     rank 3 = worst
    #
    # Ties receive average ranks.
    #
    # ---

    def calculate_mean_ranks(matrix):

        scores = matrix[
            METHODS
        ].to_numpy(
            dtype=float
        )

        ranks = np.vstack(
            [
                rankdata(
                    -row,
                    method="average",
                )

                for row
                in scores
            ]
        )

        result = {}

        for index, method in enumerate(
            METHODS
        ):

            result[
                f"{method}_mean_rank"
            ] = float(
                ranks[
                    :,
                    index
                ].mean()
            )

        return result


    # ---
    # 26. FRIEDMAN TEST
    # ---

    def run_friedman(
        matrix
    ):

        n = len(matrix)

        if n < 2:

            return {
                "n":
                    n,

                "friedman_chi2":
                    np.nan,

                "p_raw":
                    np.nan,

                "kendall_W":
                    np.nan,
            }

        values = (
            matrix[
                METHODS
            ]
            .to_numpy(
                dtype=float
            )
        )

        # If all three methods are identical for every question,
        # SciPy's tie correction can become undefined.
        if np.all(
            np.ptp(
                values,
                axis=1,
            )
            ==
            0
        ):

            chi2 = 0.0
            p = 1.0
            W = 0.0

        else:

            with warnings.catch_warnings():

                warnings.simplefilter(
                    "ignore"
                )

                result = friedmanchisquare(
                    matrix["DPC"].to_numpy(),
                    matrix["SPC"].to_numpy(),
                    matrix["HPC"].to_numpy(),
                )

            chi2 = float(
                result.statistic
            )

            p = float(
                result.pvalue
            )

            # K = 3 methods
            W = (
                chi2
                /
                (
                    n
                    *
                    (
                        len(METHODS)
                        -
                        1
                    )
                )
            )

            # Numerical protection
            W = float(
                np.clip(
                    W,
                    0.0,
                    1.0,
                )
            )

        return {
            "n":
                n,

            "friedman_chi2":
                chi2,

            "p_raw":
                p,

            "kendall_W":
                W,
        }


    # ---
    # 27. RUN ALL FRIEDMAN TESTS
    # ---

    def run_all_friedman(
        averaged
    ):

        rows = []

        for dataset in DATASETS:

            for family, metrics in [

                (
                    "generation",
                    GENERATION_METRICS,
                ),

                (
                    "evidence",
                    EVIDENCE_METRICS,
                ),
            ]:

                for metric in metrics:

                    for top_k in TOP_K_VALUES:

                        matrix = (
                            get_matched_matrix(
                                averaged,
                                dataset,
                                family,
                                metric,
                                top_k,
                            )
                        )

                        if matrix.empty:
                            continue

                        stats = (
                            run_friedman(
                                matrix
                            )
                        )

                        ranks = (
                            calculate_mean_ranks(
                                matrix
                            )
                        )

                        rows.append(
                            {
                                "analysis_level":
                                    "RQ3_Friedman",

                                "dataset":
                                    dataset,

                                "metric_family":
                                    family,

                                "metric":
                                    metric,

                                "metric_label":
                                    METRIC_LABELS[
                                        metric
                                    ],

                                "top_k":
                                    top_k,

                                "DPC_mean":
                                    float(
                                        matrix["DPC"].mean()
                                    ),

                                "SPC_mean":
                                    float(
                                        matrix["SPC"].mean()
                                    ),

                                "HPC_mean":
                                    float(
                                        matrix["HPC"].mean()
                                    ),

                                "DPC_median":
                                    float(
                                        matrix["DPC"].median()
                                    ),

                                "SPC_median":
                                    float(
                                        matrix["SPC"].median()
                                    ),

                                "HPC_median":
                                    float(
                                        matrix["HPC"].median()
                                    ),

                                **ranks,
                                **stats,
                            }
                        )

        return pd.DataFrame(
            rows
        )


    FRIEDMAN = (
        run_all_friedman(
            AVG_ALL
        )
    )


    # ---
    # 28. HOLM CORRECTION — FRIEDMAN
    # ---
    #
    # Separate families per:
    #
    #     dataset × metric family
    #
    # Generation:
    #     28 omnibus tests
    #
    # Evidence:
    #     8 omnibus tests
    #
    # ---

    def add_friedman_holm(
        dataframe
    ):

        result = dataframe.copy()

        result["p_holm"] = np.nan
        result["significant_holm"] = False
        result["holm_family_size"] = np.nan

        for _, indices in (

            result
            .groupby(
                [
                    "dataset",
                    "metric_family",
                ]
            )
            .groups
            .items()
        ):

            indices = list(
                indices
            )

            valid = [

                idx

                for idx
                in indices

                if pd.notna(
                    result.loc[
                        idx,
                        "p_raw"
                    ]
                )
            ]

            if not valid:
                continue

            p_values = (
                result.loc[
                    valid,
                    "p_raw"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            reject, adjusted, _, _ = (
                multipletests(
                    p_values,
                    alpha=ALPHA,
                    method="holm",
                )
            )

            result.loc[
                valid,
                "p_holm"
            ] = adjusted

            result.loc[
                valid,
                "significant_holm"
            ] = reject

            result.loc[
                valid,
                "holm_family_size"
            ] = len(valid)

        result[
            "holm_family_size"
        ] = (
            result[
                "holm_family_size"
            ]
            .astype(
                "Int64"
            )
        )

        return result


    FRIEDMAN = (
        add_friedman_holm(
            FRIEDMAN
        )
    )


    FRIEDMAN_SIGNIFICANT = (
        FRIEDMAN[
            FRIEDMAN[
                "significant_holm"
            ]
        ]
        .copy()
    )


    # ---
    # 29. PAIRED RANK-BISERIAL EFFECT SIZE
    # ---
    #
    # d = first - second
    #
    # Positive:
    #     first method performs better
    #
    # Negative:
    #     second method performs better
    #
    # ---

    def paired_rank_biserial(
        first,
        second,
    ):

        first = np.asarray(
            first,
            dtype=float,
        )

        second = np.asarray(
            second,
            dtype=float,
        )

        difference = (
            first
            -
            second
        )

        # Wilcoxon removes zero differences.
        difference = (
            difference[
                difference
                !=
                0
            ]
        )

        if len(difference) == 0:
            return 0.0

        ranks = rankdata(
            np.abs(
                difference
            ),
            method="average",
        )

        w_positive = (
            ranks[
                difference > 0
            ]
            .sum()
        )

        w_negative = (
            ranks[
                difference < 0
            ]
            .sum()
        )

        denominator = (
            w_positive
            +
            w_negative
        )

        if denominator == 0:
            return 0.0

        return float(
            (
                w_positive
                -
                w_negative
            )
            /
            denominator
        )


    # ---
    # 30. PAIRED WILCOXON
    # ---

    def paired_wilcoxon(
        first,
        second,
    ):

        a = np.asarray(
            first,
            dtype=float,
        )

        b = np.asarray(
            second,
            dtype=float,
        )

        difference = (
            a
            -
            b
        )

        n_nonzero = int(
            np.sum(
                difference
                !=
                0
            )
        )

        if n_nonzero == 0:

            statistic = 0.0
            p = 1.0

        else:

            with warnings.catch_warnings():

                warnings.simplefilter(
                    "ignore"
                )

                result = wilcoxon(
                    a,
                    b,
                    zero_method="wilcox",
                    correction=False,
                    alternative="two-sided",
                    method="auto",
                )

            statistic = float(
                result.statistic
            )

            p = float(
                result.pvalue
            )

        return {

            "n":
                len(a),

            "n_nonzero":
                n_nonzero,

            "wilcoxon_W":
                statistic,

            "p_raw":
                p,

            "rank_biserial_r":
                paired_rank_biserial(
                    a,
                    b,
                ),

            "mean_first":
                float(
                    np.mean(a)
                ),

            "mean_second":
                float(
                    np.mean(b)
                ),

            "mean_difference_first_minus_second":
                float(
                    np.mean(
                        difference
                    )
                ),

            "median_first":
                float(
                    np.median(a)
                ),

            "median_second":
                float(
                    np.median(b)
                ),

            "median_difference_first_minus_second":
                float(
                    np.median(
                        difference
                    )
                ),
        }


    # ---
    # 31. POST-HOC WILCOXON TESTS
    # ---
    #
    # IMPORTANT:
    #
    # These tests are performed ONLY for Friedman matrices that
    # remain significant AFTER the omnibus Holm correction.
    #
    # Each pairwise comparison uses exactly the same matched
    # question set as the corresponding Friedman test.
    #
    # ---

    def run_posthoc(
        averaged,
        friedman_results,
    ):

        rows = []

        significant_omnibus = (
            friedman_results[
                friedman_results[
                    "significant_holm"
                ]
            ]
        )

        for _, omnibus in (
            significant_omnibus
            .iterrows()
        ):

            dataset = omnibus["dataset"]
            family = omnibus["metric_family"]
            metric = omnibus["metric"]
            top_k = int(
                omnibus["top_k"]
            )

            matrix = (
                get_matched_matrix(
                    averaged,
                    dataset,
                    family,
                    metric,
                    top_k,
                )
            )

            if matrix.empty:
                continue

            for first_method, second_method in PAIR_SPECS:

                stats = (
                    paired_wilcoxon(
                        matrix[
                            first_method
                        ],
                        matrix[
                            second_method
                        ],
                    )
                )

                r = (
                    stats[
                        "rank_biserial_r"
                    ]
                )

                mean_diff = (
                    stats[
                        "mean_difference_first_minus_second"
                    ]
                )

                if r > 0:

                    direction = (
                        f"{first_method} > "
                        f"{second_method}"
                    )

                elif r < 0:

                    direction = (
                        f"{second_method} > "
                        f"{first_method}"
                    )

                elif mean_diff > 0:

                    direction = (
                        f"{first_method} > "
                        f"{second_method}"
                    )

                elif mean_diff < 0:

                    direction = (
                        f"{second_method} > "
                        f"{first_method}"
                    )

                else:

                    direction = (
                        "no direction"
                    )

                rows.append(
                    {
                        "analysis_level":
                            "RQ3_posthoc",

                        "dataset":
                            dataset,

                        "metric_family":
                            family,

                        "metric":
                            metric,

                        "metric_label":
                            METRIC_LABELS[
                                metric
                            ],

                        "top_k":
                            top_k,

                        "first_method":
                            first_method,

                        "second_method":
                            second_method,

                        "contrast":
                            (
                                f"{first_method} vs "
                                f"{second_method}"
                            ),

                        "direction":
                            direction,

                        "friedman_chi2":
                            omnibus[
                                "friedman_chi2"
                            ],

                        "friedman_kendall_W":
                            omnibus[
                                "kendall_W"
                            ],

                        "friedman_p_raw":
                            omnibus[
                                "p_raw"
                            ],

                        "friedman_p_holm":
                            omnibus[
                                "p_holm"
                            ],

                        **stats,
                    }
                )

        if not rows:

            return pd.DataFrame()

        return pd.DataFrame(
            rows
        )


    POSTHOC = (
        run_posthoc(
            AVG_ALL,
            FRIEDMAN,
        )
    )


    # ---
    # 32. HOLM CORRECTION — POST-HOC
    # ---
    #
    # Exactly 3 comparisons are corrected within each significant:
    #
    #     dataset × family × metric × k
    #
    # matrix.
    #
    # ---

    def add_posthoc_holm(
        dataframe
    ):

        if dataframe.empty:

            result = dataframe.copy()

            result["p_holm"] = []
            result["significant_holm"] = []
            result["holm_family_size"] = []

            return result

        result = dataframe.copy()

        result["p_holm"] = np.nan
        result["significant_holm"] = False
        result["holm_family_size"] = np.nan

        group_columns = [
            "dataset",
            "metric_family",
            "metric",
            "top_k",
        ]

        for _, indices in (

            result
            .groupby(
                group_columns
            )
            .groups
            .items()
        ):

            indices = list(
                indices
            )

            valid = [

                idx

                for idx
                in indices

                if pd.notna(
                    result.loc[
                        idx,
                        "p_raw"
                    ]
                )
            ]

            if not valid:
                continue

            # Exactly three pairwise comparisons should exist.
            if len(valid) != 3:

                raise RuntimeError(
                    "Expected exactly 3 post-hoc tests "
                    "within each significant Friedman matrix, "
                    f"but found {len(valid)}."
                )

            reject, adjusted, _, _ = (
                multipletests(
                    result.loc[
                        valid,
                        "p_raw"
                    ]
                    .to_numpy(
                        dtype=float
                    ),
                    alpha=ALPHA,
                    method="holm",
                )
            )

            result.loc[
                valid,
                "p_holm"
            ] = adjusted

            result.loc[
                valid,
                "significant_holm"
            ] = reject

            result.loc[
                valid,
                "holm_family_size"
            ] = len(valid)

        result[
            "holm_family_size"
        ] = (
            result[
                "holm_family_size"
            ]
            .astype(
                "Int64"
            )
        )

        return result


    POSTHOC = (
        add_posthoc_holm(
            POSTHOC
        )
    )


    if POSTHOC.empty:

        POSTHOC_SIGNIFICANT = (
            POSTHOC.copy()
        )

    else:

        POSTHOC_SIGNIFICANT = (
            POSTHOC[
                POSTHOC[
                    "significant_holm"
                ]
            ]
            .copy()
        )


    # ---
    # 33. SIGNIFICANCE PATTERN TABLE
    # ---
    #
    # One row per:
    #
    #     dataset × family × metric
    #
    # Summarizes which k values had significant Friedman results
    # and which pairwise directions were significant.
    #
    # ---

    def format_k_list(values):

        values = sorted(
            {
                int(v)
                for v in values
            }
        )

        if not values:
            return ""

        if set(values) == set(
            TOP_K_VALUES
        ):
            return "all k"

        return ",".join(
            str(v)
            for v in values
        )


    def format_p_value(p):

        if pd.isna(p):
            return ""

        p = float(p)

        if p < 0.0001:
            return f"{p:.2e}"

        return f"{p:.4f}"


    def build_metric_pattern_table(
        friedman,
        posthoc,
    ):

        rows = []

        for dataset in DATASETS:

            for family, metrics in [

                (
                    "generation",
                    GENERATION_METRICS,
                ),

                (
                    "evidence",
                    EVIDENCE_METRICS,
                ),
            ]:

                for metric in metrics:

                    omni = friedman[
                        (
                            friedman["dataset"]
                            ==
                            dataset
                        )
                        &
                        (
                            friedman["metric_family"]
                            ==
                            family
                        )
                        &
                        (
                            friedman["metric"]
                            ==
                            metric
                        )
                    ]

                    sig_omni = omni[
                        omni[
                            "significant_holm"
                        ]
                    ]

                    sig_ks = (
                        sig_omni[
                            "top_k"
                        ]
                        .tolist()
                    )

                    if not posthoc.empty:

                        ph = posthoc[
                            (
                                posthoc["dataset"]
                                ==
                                dataset
                            )
                            &
                            (
                                posthoc["metric_family"]
                                ==
                                family
                            )
                            &
                            (
                                posthoc["metric"]
                                ==
                                metric
                            )
                            &
                            (
                                posthoc[
                                    "significant_holm"
                                ]
                            )
                        ]

                    else:

                        ph = pd.DataFrame()

                    row = {

                        "dataset":
                            dataset,

                        "metric_family":
                            family,

                        "metric":
                            metric,

                        "metric_label":
                            METRIC_LABELS[
                                metric
                            ],

                        "significant_friedman_k":
                            format_k_list(
                                sig_ks
                            ),

                        "n_significant_friedman":
                            len(
                                sig_ks
                            ),

                        "minimum_friedman_p_holm":
                            (
                                float(
                                    sig_omni[
                                        "p_holm"
                                    ]
                                    .min()
                                )
                                if not sig_omni.empty
                                else np.nan
                            ),

                        "maximum_friedman_p_holm":
                            (
                                float(
                                    sig_omni[
                                        "p_holm"
                                    ]
                                    .max()
                                )
                                if not sig_omni.empty
                                else np.nan
                            ),
                    }

                    for first, second in PAIR_SPECS:

                        if ph.empty:

                            pair = pd.DataFrame()

                        else:

                            pair = ph[
                                (
                                    ph["first_method"]
                                    ==
                                    first
                                )
                                &
                                (
                                    ph["second_method"]
                                    ==
                                    second
                                )
                            ]

                        first_direction = (
                            f"{first} > {second}"
                        )

                        second_direction = (
                            f"{second} > {first}"
                        )

                        if pair.empty:

                            first_ks = []
                            second_ks = []

                        else:

                            first_ks = (
                                pair.loc[
                                    pair["direction"]
                                    ==
                                    first_direction,
                                    "top_k",
                                ]
                                .tolist()
                            )

                            second_ks = (
                                pair.loc[
                                    pair["direction"]
                                    ==
                                    second_direction,
                                    "top_k",
                                ]
                                .tolist()
                            )

                        row[
                            f"{first}_gt_{second}_k"
                        ] = (
                            format_k_list(
                                first_ks
                            )
                        )

                        row[
                            f"{second}_gt_{first}_k"
                        ] = (
                            format_k_list(
                                second_ks
                            )
                        )

                    rows.append(
                        row
                    )

        return pd.DataFrame(
            rows
        )


    METRIC_PATTERNS = (
        build_metric_pattern_table(
            FRIEDMAN,
            POSTHOC,
        )
    )


    # ---
    # 34. COMPACT KEY SUMMARY
    # ---
    #
    # One row per dataset × metric family × k.
    #
    # Designed for quick thesis interpretation.
    #
    # ---

    def build_key_summary(
        friedman,
        posthoc,
    ):

        rows = []

        for dataset in DATASETS:

            for family in [
                "generation",
                "evidence",
            ]:

                for top_k in TOP_K_VALUES:

                    omni = friedman[
                        (
                            friedman["dataset"]
                            ==
                            dataset
                        )
                        &
                        (
                            friedman["metric_family"]
                            ==
                            family
                        )
                        &
                        (
                            friedman["top_k"]
                            ==
                            top_k
                        )
                    ]

                    sig_omni = omni[
                        omni[
                            "significant_holm"
                        ]
                    ].copy()

                    omnibus_entries = []

                    for _, r in (
                        sig_omni
                        .sort_values(
                            "metric_label"
                        )
                        .iterrows()
                    ):

                        omnibus_entries.append(
                            (
                                f"{r['metric_label']} "
                                f"(p={format_p_value(r['p_holm'])}, "
                                f"W={r['kendall_W']:.3f})"
                            )
                        )

                    if omnibus_entries:

                        omnibus_text = (
                            "; ".join(
                                omnibus_entries
                            )
                        )

                    else:

                        omnibus_text = "None"

                    if posthoc.empty:

                        ph = pd.DataFrame()

                    else:

                        ph = posthoc[
                            (
                                posthoc["dataset"]
                                ==
                                dataset
                            )
                            &
                            (
                                posthoc["metric_family"]
                                ==
                                family
                            )
                            &
                            (
                                posthoc["top_k"]
                                ==
                                top_k
                            )
                            &
                            (
                                posthoc[
                                    "significant_holm"
                                ]
                            )
                        ].copy()

                    pair_entries = []

                    if not ph.empty:

                        for _, r in (
                            ph
                            .sort_values(
                                [
                                    "metric_label",
                                    "contrast",
                                ]
                            )
                            .iterrows()
                        ):

                            pair_entries.append(
                                (
                                    f"{r['metric_label']}: "
                                    f"{r['direction']} "
                                    f"(p={format_p_value(r['p_holm'])}, "
                                    f"r_rb={r['rank_biserial_r']:.3f})"
                                )
                            )

                    if pair_entries:

                        pair_text = (
                            "; ".join(
                                pair_entries
                            )
                        )

                    else:

                        pair_text = "None"

                    rows.append(
                        {
                            "dataset":
                                dataset,

                            "metric_family":
                                family,

                            "top_k":
                                top_k,

                            "n_metrics_tested":
                                len(
                                    omni
                                ),

                            "n_significant_friedman":
                                len(
                                    sig_omni
                                ),

                            "significant_friedman":
                                omnibus_text,

                            "n_significant_posthoc":
                                len(
                                    ph
                                ),

                            "significant_posthoc":
                                pair_text,
                        }
                    )

        return pd.DataFrame(
            rows
        )


    KEY_SUMMARY = (
        build_key_summary(
            FRIEDMAN,
            POSTHOC,
        )
    )


    # ---
    # 35. CHECK EXPECTED TEST COUNTS
    # ---
    #
    # Per dataset:
    #
    # generation = 28 Friedman tests
    # evidence   =  8 Friedman tests
    #
    # If fewer exist, the workbook still gets created, but this
    # diagnostic makes the missing matrix explicit.
    #
    # ---

    TEST_COUNT_SUMMARY = (

        FRIEDMAN
        .groupby(
            [
                "dataset",
                "metric_family",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size":
                    "n_friedman_tests"
            }
        )
    )


    TEST_COUNT_SUMMARY[
        "expected_tests"
    ] = (
        TEST_COUNT_SUMMARY[
            "metric_family"
        ]
        .map(
            {
                "generation":
                    28,

                "evidence":
                    8,
            }
        )
    )


    TEST_COUNT_SUMMARY[
        "complete"
    ] = (
        TEST_COUNT_SUMMARY[
            "n_friedman_tests"
        ]
        ==
        TEST_COUNT_SUMMARY[
            "expected_tests"
        ]
    )


    # ---
    # 36. README
    # ---

    README = pd.DataFrame(
        {
            "Item": [

                "RQ3",

                "Compared methods",

                "Inferential unit",

                "Generator handling",

                "Omnibus test",

                "Omnibus multiplicity",

                "Post-hoc gate",

                "Post-hoc tests",

                "Post-hoc multiplicity",

                "Direction",

                "Kendall W",

                "Mean ranks",

                "Significance",

                "P-values to report",

                "Local input scope",

                "Output destination",
            ],

            "Description": [

                (
                    "How do sparse, dense and hybrid retrieval "
                    "affect evidence support and generation quality "
                    "of hierarchical RAG systems?"
                ),

                (
                    "DPC (dense parent-child), "
                    "SPC (sparse parent-child), "
                    "HPC (hybrid parent-child)."
                ),

                (
                    "Question. Tests are paired within question."
                ),

                (
                    "Each metric is first averaged across all expected "
                    "generators for that dataset. A question/configuration "
                    "is retained only when the complete expected generator "
                    "set is available for that metric."
                ),

                (
                    "Friedman test separately for every "
                    "dataset × metric × top-k, comparing DPC, SPC and HPC."
                ),

                (
                    "Holm correction separately within each "
                    "dataset × metric family across all metric × top-k "
                    "Friedman tests: 28 generation and 8 evidence tests."
                ),

                (
                    "Post-hoc tests are performed only when the "
                    "corresponding Friedman test remains significant "
                    "after the omnibus Holm correction."
                ),

                (
                    "Paired two-sided Wilcoxon signed-rank tests: "
                    "DPC-SPC, DPC-HPC and SPC-HPC."
                ),

                (
                    "The three pairwise Wilcoxon p-values are Holm-corrected "
                    "within each significant dataset × metric × top-k matrix."
                ),

                (
                    "Wilcoxon tests are two-sided. Direction is reported "
                    "afterward from rank-biserial sign / paired differences. "
                    "Positive rank-biserial means first_method > second_method."
                ),

                (
                    "Kendall W = Friedman chi-square / [N × (3-1)]."
                ),

                (
                    "Higher metric scores are ranked better; rank 1 is best."
                ),

                (
                    "alpha = .05."
                ),

                (
                    "Use p_holm for final significance claims. "
                    "p_raw is retained for transparency."
                ),

                (
                    "Input data are read only from the dataset results folders inside experiment_runner_final."
                ),

                (
                    "The significance workbook is saved to evaluation/analysis/significance_testing."
                ),
            ],
        }
    )


    # ---
    # 37. WRITE RESULT WORKBOOK
    # ---

    print()
    print(
        "Creating RQ3 result workbook..."
    )


    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl",
    ) as writer:

        # ---
        # Most useful interpretation sheets
        # ---

        KEY_SUMMARY.to_excel(
            writer,
            sheet_name="KEY_SUMMARY",
            index=False,
        )

        METRIC_PATTERNS.to_excel(
            writer,
            sheet_name="METRIC_PATTERNS",
            index=False,
        )

        # ---
        # Omnibus tests
        # ---

        FRIEDMAN_SIGNIFICANT.to_excel(
            writer,
            sheet_name="FRIEDMAN_SIGNIFICANT",
            index=False,
        )

        FRIEDMAN.to_excel(
            writer,
            sheet_name="FRIEDMAN_FULL",
            index=False,
        )

        # ---
        # Post-hoc tests
        # ---

        POSTHOC_SIGNIFICANT.to_excel(
            writer,
            sheet_name="POSTHOC_SIGNIFICANT",
            index=False,
        )

        POSTHOC.to_excel(
            writer,
            sheet_name="POSTHOC_FULL",
            index=False,
        )

        # ---
        # Reproducibility / diagnostics
        # ---

        TEST_COUNT_SUMMARY.to_excel(
            writer,
            sheet_name="TEST_COUNTS",
            index=False,
        )

        INCOMPLETE_CONFIG_SUMMARY.to_excel(
            writer,
            sheet_name="DATA_QUALITY",
            index=False,
        )

        GENERATION_FILE_INDEX.to_excel(
            writer,
            sheet_name="GENERATION_FILES",
            index=False,
        )

        EVIDENCE_FILE_INDEX.to_excel(
            writer,
            sheet_name="EVIDENCE_FILES",
            index=False,
        )

        README.to_excel(
            writer,
            sheet_name="README",
            index=False,
        )


    # ---
    # 38. FORMAT WORKBOOK
    # ---

    workbook = load_workbook(
        OUTPUT_FILE
    )


    for sheet in workbook.worksheets:

        sheet.freeze_panes = "A2"

        if sheet.max_row > 1:

            sheet.auto_filter.ref = (
                sheet.dimensions
            )

        # Bold headers
        for cell in sheet[1]:

            cell.font = Font(
                bold=True
            )

            cell.alignment = Alignment(
                vertical="top",
                wrap_text=True,
            )

        # Sensible column widths
        for column_cells in sheet.columns:

            column_letter = (
                column_cells[0]
                .column_letter
            )

            max_length = 0

            for cell in column_cells:

                if cell.value is None:
                    continue

                length = len(
                    str(
                        cell.value
                    )
                )

                max_length = max(
                    max_length,
                    length,
                )

            sheet.column_dimensions[
                column_letter
            ].width = min(
                max(
                    max_length + 2,
                    12,
                ),
                70,
            )


    workbook.save(
        OUTPUT_FILE
    )


    # ---
    # 39. PRINT ONLY COMPACT RESULTS
    # ---

    print()
    print("=" * 120)
    print("RQ3 — KEY SIGNIFICANCE SUMMARY")
    print("=" * 120)
    print()

    display(
        KEY_SUMMARY
    )


    print()
    print("=" * 120)
    print("TEST COMPLETENESS")
    print("=" * 120)
    print()

    display(
        TEST_COUNT_SUMMARY
    )


    print()
    print(
        "Interpretation rules:"
    )

    print(
        "  • Final significance uses Holm-adjusted p-values."
    )

    print(
        "  • Friedman p_holm < .05 means at least one of "
        "DPC/SPC/HPC differs at that metric and k."
    )

    print(
        "  • Post-hoc Wilcoxon tests are run only for such "
        "significant Friedman matrices."
    )

    print(
        "  • Pairwise p_holm < .05 identifies the differing pair."
    )

    print(
        "  • Positive rank-biserial means first_method > second_method."
    )

    print(
        "  • Mean rank 1 = best."
    )


    print(f"\n\nSaved significance workbook to: {OUTPUT_FILE}")
    return OUTPUT_FILE



def main(
    data_root=PROJECT_ROOT, datasets=DATASETS, analysis="all",
    output_dir=None, effective_k_file=None,
):
    """Run selected analyses and return their saved workbook paths."""
    datasets = normalize_datasets(datasets)
    if analysis not in ("all", "rq2", "rq3"):
        raise ValueError("Analysis must be all, rq2, or rq3.")
    data_root = Path(data_root).resolve()
    for dataset in datasets:
        for family in ("metric_analysis", "retrieval_analysis"):
            folder = data_root / f"{dataset}_dataset" / "results" / family
            if not folder.is_dir():
                raise FileNotFoundError(f"Required input folder not found: {folder}")
    if analysis in ("all", "rq2"):
        effective_k_file = (
            Path(effective_k_file).resolve() if effective_k_file is not None
            else default_effective_k_file(data_root)
        )
        if not effective_k_file.is_file():
            raise FileNotFoundError(f"RQ2 requires the effective-k workbook: {effective_k_file}")
    destination = Path(output_dir).resolve() if output_dir is not None else (
        data_root / "evaluation/analysis/significance_testing"
    )
    results = {}
    for name, evaluate in (("rq2", run_rq2), ("rq3", run_rq3)):
        if analysis in ("all", name):
            print("\n" + "=" * 120)
            print(f"{name.upper()} SIGNIFICANCE TESTING | Datasets: {', '.join(datasets)}")
            print("=" * 120, flush=True)
            kwargs = {"datasets": datasets, "output_dir": destination}
            if name == "rq2":
                kwargs["effective_k_file"] = effective_k_file
            results[name] = evaluate(data_root, **kwargs)
    return results


if __name__ == "__main__":
    args = parse_args()
    results = main(
        args.root, datasets=args.datasets, analysis=args.analysis,
        output_dir=args.output_dir, effective_k_file=args.effective_k_file,
    )
