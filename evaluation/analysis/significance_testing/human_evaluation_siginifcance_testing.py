"""Human evaluation significance testing (three blinded evaluator workbooks).

Install once: pip install numpy scipy krippendorff

The anonymization mapping XLSX is required to identify DPC, DP and GT from the
Response A/B/C labels.
"""

from __future__ import annotations

import csv
import itertools
import math
import posixpath
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

try:
    import krippendorff
    import numpy as np
    from scipy.stats import rankdata, wilcoxon
except ImportError as exc:
    raise SystemExit("Install the dependencies with: pip install numpy scipy krippendorff") from exc


# ---
# Configuration: paths are relative to this script's location
# ---

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = PROJECT_ROOT / "tender_dataset" / "results" / "human_evaluation"
OUTPUT_DIR = SCRIPT_DIR

EVALUATOR_FILES = [
    "Human_Evaluation_Qwen3-235B_top10_(1).xlsx",
    "Human_Evaluation_Qwen3-235B_top10_(2).xlsx",
    "Human_Evaluation_Qwen3-235B_top10_(3).xlsx",
]

# The administrator mapping workbook is required to
# decode which blinded response belongs to which retrieval configuration.
MAPPING_FILENAME = "Human_Evaluation_Admin_Mapping_Qwen3-235B_top10.xlsx"

SOURCE_ORDER = ("Dense parent-child", "Dense parent", "Ground truth")
SOURCE_SHORT = {
    "Dense parent-child": "DPC",
    "Dense parent": "DP",
    "Ground truth": "GT",
}
DIMENSIONS = (
    "Coverage (Broad)",
    "Coherence (Logical)",
    "User Intent Alignment",
    "Cyclicality",
    "Helpfulness",
)
RESPONSE_LABELS = ("Response A", "Response B", "Response C")
ALPHA = 0.05

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = {"m": MAIN_NS}
CELL_REF = re.compile(r"([A-Z]+)(\d+)")


# ---
# Read the existing XLSX layout without modifying the evaluator files
# ---

def read_xlsx_cells(path: Path) -> dict[str, dict[str, object]]:
    """Read spreadsheet cell values directly from the XLSX XML."""
    with ZipFile(path) as archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = [
                "".join(node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t"))
                for item in root.findall("m:si", NS)
            ]

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {node.attrib["Id"]: node.attrib["Target"] for node in relationships}
        sheets = {}

        for sheet in workbook.find("m:sheets", NS):
            name = sheet.attrib["name"]
            relation_id = sheet.attrib[f"{{{REL_NS}}}id"]
            target = targets[relation_id].replace("\\", "/")
            if target.startswith("/"):
                target = target.lstrip("/")
            elif not target.startswith("xl/"):
                target = posixpath.normpath(posixpath.join("xl", target))

            root = ET.fromstring(archive.read(target))
            cells = {}
            for cell in root.findall(".//m:c", NS):
                reference = cell.attrib["r"]
                cell_type = cell.attrib.get("t")
                value_node = cell.find("m:v", NS)
                raw = value_node.text if value_node is not None else None

                if cell_type == "s":
                    value = shared_strings[int(raw)] if raw is not None else ""
                elif cell_type == "inlineStr":
                    inline = cell.find("m:is", NS)
                    value = (
                        "".join(node.text or "" for node in inline.iter(f"{{{MAIN_NS}}}t"))
                        if inline is not None else ""
                    )
                elif cell_type == "b":
                    value = bool(int(raw)) if raw is not None else False
                elif cell_type in ("str", "e"):
                    value = raw or ""
                elif raw is None:
                    value = None
                else:
                    try:
                        number = float(raw)
                        value = int(number) if number.is_integer() else number
                    except ValueError:
                        value = raw
                cells[reference] = value

            sheets[name] = cells

        return sheets


def normalize_test_id(value: object) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        return str(int(number)) if number.is_integer() else text
    except ValueError:
        return text


def test_id_sort_key(test_id: str):
    return (0, int(test_id)) if test_id.isdigit() else (1, test_id)


def load_mapping(path: Path) -> dict[tuple[str, str], str]:
    """Load the response anonymization mapping from the first worksheet."""
    mapping = {}
    sheets = read_xlsx_cells(path)
    if not sheets:
        raise ValueError(f"Mapping workbook contains no worksheets: {path}")

    cells = next(iter(sheets.values()))
    rows: dict[int, dict[str, object]] = defaultdict(dict)
    for reference, value in cells.items():
        match = CELL_REF.fullmatch(reference)
        if match:
            rows[int(match.group(2))][match.group(1)] = value

    required = {"test_id", "response_label", "actual_source"}
    header_row = None
    columns = {}
    for row_number, row in rows.items():
        normalized = {
            str(value).strip().lower(): column
            for column, value in row.items()
            if value is not None and str(value).strip()
        }
        if required.issubset(normalized):
            header_row = row_number
            columns = {name: normalized[name] for name in required}
            break

    if header_row is None:
        raise ValueError(
            "Mapping workbook must contain columns: "
            + ", ".join(sorted(required))
        )

    for row_number, row in rows.items():
        if row_number <= header_row:
            continue
        values = {name: row.get(column) for name, column in columns.items()}
        if all(value is None or not str(value).strip() for value in values.values()):
            continue
        if any(value is None for value in values.values()):
            raise ValueError(f"Incomplete mapping row in {path.name}, row {row_number}")

        key = (
            normalize_test_id(values["test_id"]),
            str(values["response_label"]).strip(),
        )
        source = str(values["actual_source"]).strip()
        if key[1] not in RESPONSE_LABELS or source not in SOURCE_ORDER:
            raise ValueError(f"Invalid mapping entry: {key} -> {source!r}")
        if key in mapping and mapping[key] != source:
            raise ValueError(f"Conflicting mapping for {key}")
        mapping[key] = source

    if not mapping:
        raise ValueError(f"No mapping rows found in {path}")
    return mapping


def section_rows(cells: dict, accepted: tuple[str, ...]) -> dict[str, int]:
    rows = {}
    for reference, value in cells.items():
        match = CELL_REF.fullmatch(reference)
        if match and match.group(1) == "A" and value in accepted:
            if value in rows:
                raise ValueError(f"Repeated section label: {value}")
            rows[value] = int(match.group(2))
    return rows


def extract_workbook(path: Path, mapping: dict, fallback_id: str):
    sheets = read_xlsx_cells(path)
    identifier = sheets.get("Instructions", {}).get("B3")
    evaluator = str(identifier).strip() if identifier is not None and str(identifier).strip() else fallback_id
    likert_rows, ranking_rows = [], []
    test_sheets = sorted(
        (name for name in sheets if re.fullmatch(r"TC\d+", name)),
        key=lambda name: int(name[2:]),
    )
    if not test_sheets:
        raise ValueError(f"{path.name}: no TC sheets found")

    for sheet in test_sheets:
        cells = sheets[sheet]
        test_id = normalize_test_id(cells.get("B3") or sheet[2:])
        response_sections = section_rows(cells, RESPONSE_LABELS)
        if set(response_sections) != set(RESPONSE_LABELS):
            raise ValueError(f"{path.name}/{sheet}: incomplete response sections")

        resolved_sources = {}
        for response, header_row in response_sections.items():
            key = (test_id, response)
            if key not in mapping:
                raise ValueError(f"{path.name}/{sheet}: mapping missing for {key}")
            source = mapping[key]
            resolved_sources[response] = source
            observed_dimensions = set()

            for row_number in range(header_row + 2, header_row + 7):
                dimension = cells.get(f"I{row_number}")
                rating = cells.get(f"M{row_number}")
                if dimension not in DIMENSIONS or dimension in observed_dimensions:
                    raise ValueError(f"{path.name}/{sheet}: invalid dimension in I{row_number}: {dimension!r}")
                if not isinstance(rating, (int, float)) or isinstance(rating, bool):
                    raise ValueError(f"{path.name}/{sheet}: missing numeric rating M{row_number}")
                if not 1 <= rating <= 5 or not float(rating).is_integer():
                    raise ValueError(f"{path.name}/{sheet}: rating must be an integer from 1 to 5 in M{row_number}")
                observed_dimensions.add(dimension)
                likert_rows.append({
                    "evaluator": evaluator, "test_id": test_id, "source": source,
                    "dimension": dimension, "rating": int(rating),
                })

        if set(resolved_sources.values()) != set(SOURCE_ORDER):
            raise ValueError(f"{path.name}/{sheet}: each system must occur once in the mapping")

        ranking_section = section_rows(cells, ("Selected response",))
        if "Selected response" not in ranking_section:
            raise ValueError(f"{path.name}/{sheet}: ranking row missing")
        row_number = ranking_section["Selected response"]
        labels = [cells.get(f"{column}{row_number}") for column in "BDF"]
        if sorted(labels, key=str) != list(RESPONSE_LABELS):
            raise ValueError(f"{path.name}/{sheet}: incomplete or duplicate ranking {labels}")
        for rank, response in enumerate(labels, 1):
            ranking_rows.append({
                "evaluator": evaluator, "test_id": test_id,
                "source": resolved_sources[response], "rank": rank,
            })

    return evaluator, likert_rows, ranking_rows


# ---
# --- Significance tests and effect sizes ---
# ---

def sample_sd(values) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def mean_sd(values) -> str:
    return f"{statistics.mean(values):.2f} ({sample_sd(values):.2f})"


def format_p(value: float) -> str:
    return "< .001" if value < 0.001 else f"{value:.3f}"


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down FWER adjustment, retaining the original comparison order."""
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    for position, index in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        running_max = max(running_max, min(1.0, (len(p_values) - position) * p_values[index]))
        adjusted[index] = running_max
    return adjusted


def rank_biserial(first, second) -> float:
    differences = np.asarray(first, float) - np.asarray(second, float)
    differences = differences[~np.isclose(differences, 0.0)]
    if differences.size == 0:
        return 0.0
    ranks = rankdata(np.abs(differences), method="average")
    positive = float(ranks[differences > 0].sum())
    negative = float(ranks[differences < 0].sum())
    return (positive - negative) / (positive + negative)


def paired_wilcoxon(first, second):
    if np.allclose(np.asarray(first, float) - np.asarray(second, float), 0):
        return 0.0, 1.0, 0.0
    result = wilcoxon(first, second, zero_method="wilcox", alternative="two-sided")
    return float(result.statistic), float(result.pvalue), rank_biserial(first, second)


def exact_friedman(samples) -> tuple[float, float]:
    """Exact permutation p-value for the tie-corrected Friedman statistic.

    Uses dynamic programming to enumerate permutations of
    within-test-case ranks (including multiplicity for tied permutations).
    """
    matrix = np.column_stack([np.asarray(sample, dtype=float) for sample in samples])
    n, k = matrix.shape
    if n < 1 or k < 3:
        raise ValueError("Friedman test requires at least one case and three systems")
    ranks = np.apply_along_axis(rankdata, 1, matrix, method="average")
    tie_sum = sum(
        int(count) ** 3 - int(count)
        for row in ranks for count in np.unique(row, return_counts=True)[1] if count > 1
    )
    correction = 1.0 - tie_sum / (n * (k**3 - k))
    if np.isclose(correction, 0.0):
        return 0.0, 1.0

    def statistic(sums):
        return float((12.0 * np.sum(np.asarray(sums, dtype=float) ** 2) / (n * k * (k + 1))
                      - 3.0 * n * (k + 1)) / correction)

    observed = statistic(ranks.sum(axis=0))
    states = Counter({(0,) * k: 1})
    for row in ranks:
        permutations = Counter(itertools.permutations(tuple(int(round(2 * value)) for value in row)))
        next_states = Counter()
        for current, current_count in states.items():
            for permutation, multiplicity in permutations.items():
                total = tuple(a + b for a, b in zip(current, permutation))
                next_states[total] += current_count * multiplicity
        states = next_states

    extreme = sum(count for sums, count in states.items()
                  if statistic(np.asarray(sums, float) / 2) >= observed - 1e-12)
    return observed, float(extreme / (math.factorial(k) ** n))


def case_level_values(rows: list[dict], value_key: str, dimension: str | None):
    """Average the three evaluators for each case/system before testing."""
    grouped = defaultdict(list)
    for row in rows:
        if dimension is None or row.get("dimension") == dimension:
            grouped[(row["test_id"], row["source"])].append(float(row[value_key]))

    test_ids = sorted({test_id for test_id, _ in grouped}, key=test_id_sort_key)
    arrays = {source: [] for source in SOURCE_ORDER}
    for test_id in test_ids:
        for source in SOURCE_ORDER:
            values = grouped.get((test_id, source), [])
            if not values:
                raise ValueError(f"Missing {value_key} values for case {test_id}, source {source}")
            arrays[source].append(statistics.mean(values))
    return arrays, len(test_ids)


def test_outcome(rows: list[dict], value_key: str, outcome: str, dimension: str | None):
    arrays, n_cases = case_level_values(rows, value_key, dimension)
    q, p = exact_friedman([arrays[source] for source in SOURCE_ORDER])
    omnibus = {
        "outcome": outcome, "friedman_q": q, "df": len(SOURCE_ORDER) - 1,
        "exact_p": p, "kendalls_w": q / (n_cases * (len(SOURCE_ORDER) - 1)),
        "n_cases": n_cases,
    }
    posthoc = []
    for first, second in itertools.combinations(SOURCE_ORDER, 2):
        statistic, raw_p, effect_size = paired_wilcoxon(arrays[first], arrays[second])
        posthoc.append({
            "outcome": outcome,
            "comparison": f"{SOURCE_SHORT[first]} vs {SOURCE_SHORT[second]}",
            "wilcoxon_w": statistic, "raw_p": raw_p,
            "rank_biserial_r": effect_size,
        })
    for row, p_adjusted in zip(posthoc, holm_adjust([row["raw_p"] for row in posthoc])):
        row["holm_p"] = p_adjusted
    return omnibus, posthoc


def kendalls_w_for_raters(rank_matrix) -> float:
    ranks = np.asarray(rank_matrix, dtype=float)
    n_raters, n_objects = ranks.shape
    sums = ranks.sum(axis=0)
    expected = n_raters * (n_objects + 1) / 2
    denominator = n_raters**2 * (n_objects**3 - n_objects)
    return float(12 * np.sum((sums - expected) ** 2) / denominator)


def reliability(likert_rows, ranking_rows, evaluators, test_ids):
    alphas = {}
    for dimension in DIMENSIONS:
        lookup = {
            (row["evaluator"], row["test_id"], row["source"]): row["rating"]
            for row in likert_rows if row["dimension"] == dimension
        }
        matrix = np.array([
            [lookup.get((evaluator, test_id, source), np.nan)
             for test_id in test_ids for source in SOURCE_ORDER]
            for evaluator in evaluators
        ], dtype=float)
        alphas[dimension] = float(krippendorff.alpha(reliability_data=matrix,
                                                     level_of_measurement="ordinal"))

    rank_lookup = {
        (row["evaluator"], row["test_id"], row["source"]): row["rank"]
        for row in ranking_rows
    }
    concordances = [kendalls_w_for_raters([
        [rank_lookup[(evaluator, test_id, source)] for source in SOURCE_ORDER]
        for evaluator in evaluators
    ]) for test_id in test_ids]
    return alphas, {
        "mean": statistics.mean(concordances), "sd": sample_sd(concordances),
        "median": statistics.median(concordances),
        "min": min(concordances), "max": max(concordances),
    }


# ---
# --- Report and save test results ---
# ---

def md_table(headers, rows) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
    ])


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)


def validate_input_files():
    paths = [INPUT_DIR / name for name in EVALUATOR_FILES]
    mapping_path = INPUT_DIR / MAPPING_FILENAME
    missing = [str(path) for path in [*paths, mapping_path] if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required evaluation inputs are missing:\n- " + "\n- ".join(missing)
            + "\nThe administrator mapping workbook is essential for decoding the blinded responses."
        )
    return paths, mapping_path


def main() -> None:
    paths, mapping_path = validate_input_files()
    mapping = load_mapping(mapping_path)

    evaluator_ids = []
    likert_rows, ranking_rows = [], []
    test_sets = []
    for index, path in enumerate(paths, 1):
        evaluator, likert, rankings = extract_workbook(path, mapping, f"Evaluator {index}")
        evaluator_ids.append(evaluator)
        test_sets.append({row["test_id"] for row in rankings})
        likert_rows.extend(likert)
        ranking_rows.extend(rankings)

    if len(set(evaluator_ids)) != len(paths):
        raise ValueError(f"Evaluator IDs must be unique: {evaluator_ids}")
    if not test_sets[0] or any(tests != test_sets[0] for tests in test_sets[1:]):
        raise ValueError("All three evaluators must have the same complete test-case set")
    test_ids = sorted(test_sets[0], key=test_id_sort_key)
    evaluators = sorted(evaluator_ids)
    alphas, rank_agreement = reliability(likert_rows, ranking_rows, evaluators, test_ids)

    likert_tests, posthoc = [], []
    for dimension in DIMENSIONS:
        omnibus, comparisons = test_outcome(likert_rows, "rating", dimension, dimension)
        likert_tests.append(omnibus)
        posthoc.extend(comparisons)

    # Adjust the five Likert omnibus tests with Holm correction.
    for test, corrected in zip(likert_tests, holm_adjust([test["exact_p"] for test in likert_tests])):
        test["holm_p"] = corrected
        test["significant"] = corrected < ALPHA

    ranking_test, ranking_posthoc = test_outcome(ranking_rows, "rank", "Ranking", None)
    ranking_test["holm_p"] = ""  # separate ranking omnibus, not in the 5-test Holm family
    ranking_test["significant"] = ranking_test["exact_p"] < ALPHA
    posthoc.extend(ranking_posthoc)

    outcomes = {test["outcome"]: test for test in [*likert_tests, ranking_test]}
    for row in posthoc:
        row["omnibus_significant"] = outcomes[row["outcome"]]["significant"]
        row["holm_significant"] = row["omnibus_significant"] and row["holm_p"] < ALPHA

    likert_display = []
    for dimension in DIMENSIONS:
        test = outcomes[dimension]
        means = [mean_sd([r["rating"] for r in likert_rows
                          if r["dimension"] == dimension and r["source"] == source])
                 for source in SOURCE_ORDER]
        likert_display.append([
            dimension, *means, f"{alphas[dimension]:.3f}",
            f"{test['friedman_q']:.3f}", format_p(test["holm_p"]),
            f"{test['kendalls_w']:.3f}",
        ])

    ranking_display = []
    for source in SOURCE_ORDER:
        values = [r["rank"] for r in ranking_rows if r["source"] == source]
        first, last = values.count(1), values.count(3)
        ranking_display.append([
            SOURCE_SHORT[source], mean_sd(values),
            f"{first} ({100 * first / len(values):.1f}%)",
            f"{last} ({100 * last / len(values):.1f}%)",
        ])

    report = [
        "# Human evaluation: significance testing",
        "",
        "## Input files",
        *(f"- {path.name}" for path in paths),
        f"- {mapping_path.name} (response anonymization key)",
        "",
        f"N = {len(test_ids)} paired test cases; evaluators = {len(evaluators)}; alpha = {ALPHA}.",
        "Within each test case, ratings/ranks are averaged across evaluators before Friedman/Wilcoxon.",
        "Likert Friedman p-values are exact, then Holm-adjusted across five dimensions.",
        "Pairwise Wilcoxon p-values are Holm-adjusted within each outcome.",
        "Post-hoc comparisons are interpreted only following a significant omnibus test.",
        "",
        "## Table 1. Likert ratings, reliability, and Friedman tests",
        "",
        md_table([
            "Dimension", "DPC M (SD)", "DP M (SD)", "GT M (SD)",
            "Krippendorff alpha", "Friedman Q", "Exact Holm-p", "Kendall W",
        ], likert_display),
        "",
        "## Table 2. Ranking results",
        "",
        md_table(["System", "Mean rank (SD)", "First place n (%)", "Last place n (%)"], ranking_display),
        "",
        "## Ranking: inter-rater reliability",
        "",
        f"Mean Kendall W = {rank_agreement['mean']:.3f} "
        f"(SD = {rank_agreement['sd']:.3f}; Median = {rank_agreement['median']:.3f}; "
        f"Min-Max = {rank_agreement['min']:.3f}-{rank_agreement['max']:.3f}).",
        "",
        "## Ranking: Friedman test",
        "",
        f"Q = {ranking_test['friedman_q']:.3f}, exact p = {format_p(ranking_test['exact_p'])}, "
        f"Kendall W = {ranking_test['kendalls_w']:.3f}.",
        "",
        "## Pairwise post-hoc tests for significant Friedman outcomes",
        "",
    ]

    significant = [test["outcome"] for test in [*likert_tests, ranking_test] if test["significant"]]
    if not significant:
        report.append("No Friedman test was significant; pairwise results are exploratory only.")
    else:
        for outcome in significant:
            report.extend([f"### {outcome}", ""])
            for row in posthoc:
                if row["outcome"] == outcome:
                    report.append(
                        f"- {row['comparison']}: Wilcoxon W = {row['wilcoxon_w']:.3f}; "
                        f"Holm-p = {format_p(row['holm_p'])}; "
                        f"rank-biserial r = {row['rank_biserial_r']:.3f}."
                    )
            report.append("")

    report.extend([
        "## Statistical notes", "",
        "- Higher Likert ratings are better; lower ranking positions are better.",
        "- The sign of rank-biserial r follows first system minus second system.",
        "- CSV outputs retain unrounded statistics, including gated/non-significant comparisons.",
    ])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "human_evaluation_friedman_tests.csv"
    posthoc_path = OUTPUT_DIR / "human_evaluation_posthoc_tests.csv"
    report_path = OUTPUT_DIR / "human_evaluation_significance_report.md"

    write_csv(summary_path, [*likert_tests, ranking_test], [
        "outcome", "n_cases", "friedman_q", "df", "exact_p", "holm_p", "kendalls_w", "significant",
    ])
    write_csv(posthoc_path, posthoc, [
        "outcome", "comparison", "wilcoxon_w", "raw_p", "holm_p",
        "rank_biserial_r", "omnibus_significant", "holm_significant",
    ])
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(report))
    print(f"\nSaved report and two CSV result tables to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
