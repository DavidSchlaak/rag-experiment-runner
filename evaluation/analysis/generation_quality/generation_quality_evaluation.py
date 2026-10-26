"""Evaluate Tender, LEDGAR, and HUPD using the original dataset-specific rules.

Run from any working directory with `python generation_quality_evaluation.py`.
Select a dataset with `--dataset tender` (or ledgar/hupd), or select several
with `--datasets tender,ledgar`. Omit the option to evaluate all three.

Stages:
    1. Shared configuration, parsing, and ranking helpers.
    2. Tender similarity, difficulty, standard-system, and agentic evaluations.
    3. LEDGAR and HUPD standard-system and agentic evaluations.
    4. Ordered execution from the local project data folders.

The root contains tender_dataset, ledgar_dataset, and hupd_dataset. Use an
IPython session for styled tables and inline figures. PyCharm script runs
print section headings and formatted tables as plain text. Excel
exports keep their original locations. Tender question-gold diagnostics and
easy/hard comparisons are saved to Excel only, without CSV or console tables.
Results are also returned by main().
The similarity analysis requires the BAAI/bge-m3 model and may use a GPU.
"""

import argparse
from pathlib import Path
import re
import subprocess
import sys
import warnings


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _style_metric_column(col, metric_cols, lower_is_better_cols):
    """Highlight the best and second-best numeric values in a metric column."""
    import numpy as np
    import pandas as pd

    styles = pd.Series("", index=col.index)
    if col.name not in metric_cols:
        return styles
    values = pd.to_numeric(col, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return styles
    unique_values = sorted(valid.unique())
    if col.name not in lower_is_better_cols:
        unique_values.reverse()
    best_value = unique_values[0]
    styles.loc[values[np.isclose(values, best_value)].index] = "font-weight: bold"
    if len(unique_values) >= 2:
        second_value = unique_values[1]
        second_idx = values[np.isclose(values, second_value)].index
        for idx in second_idx:
            if styles.loc[idx]:
                styles.loc[idx] += "; text-decoration: underline"
            else:
                styles.loc[idx] = "text-decoration: underline"
    return styles


DATASETS = ("tender", "ledgar", "hupd")


def normalize_datasets(datasets):
    """Validate dataset names supplied by the CLI or a Python caller."""
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
    parser = argparse.ArgumentParser(description=__doc__)
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
        help="Install dependencies once before evaluation.",
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
                "pandas",
                "numpy",
                "scipy",
                "matplotlib",
                "openpyxl",
                "scikit-learn",
                "sentence-transformers",
                "rouge-score",
                "torch",
                "ipython",
                "jinja2",
            ],
            check=True,
        )

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import get_ipython
from IPython.display import display as ipython_display, Markdown
from pandas.io.formats.style import Styler


def display(*objects):
    """Print tables in scripts and preserve rich display in IPython sessions."""
    if __name__ != "__main__" and getattr(get_ipython(), "kernel", None) is not None:
        ipython_display(*objects)
        return
    for obj in objects:
        if isinstance(obj, Markdown):
            text = re.sub(r"(?m)^\s*#{1,6}\s+", "", obj.data).replace("**", "").replace(chr(96), "")
            if re.search(r"(?m)^\s*#{1,6}\s+", obj.data):
                print("\n" + "=" * 100)
                print(text)
                print("=" * 100, flush=True)
            else:
                print(text)
        elif isinstance(obj, Styler):
            print(obj.to_string(delimiter=" | ", max_rows=pd.get_option("display.max_rows")))
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            print(obj.to_string(max_rows=pd.get_option("display.max_rows")))
        else:
            print(obj)


# 1. Shared configuration and helpers
# Lists are copied at use sites so dataset settings remain independent.

GENERATION_METRICS = (
    "rouge1_f1",
    "rouge2_f1",
    "rougeL_f1",
    "bertscore_f1",
    "simcse_f1",
    "ecs_bge_m3",
    "matcha_f1",
)
SEMANTIC_SCORE_METRICS = ("bertscore_f1", "simcse_f1", "ecs_bge_m3", "matcha_f1")
LEXICAL_SCORE_METRICS = ("rouge1_f1", "rouge2_f1", "rougeL_f1")
ROUGE_SCORE_METRICS = (
    "rouge1_precision",
    "rouge1_recall",
    "rouge1_f1",
    "rouge2_precision",
    "rouge2_recall",
    "rouge2_f1",
    "rougeL_precision",
    "rougeL_recall",
    "rougeL_f1",
)
BERT_SCORE_METRICS = ("bertscore_precision", "bertscore_recall", "bertscore_f1")
SIMCSE_SCORE_METRICS = ("simcse_precision", "simcse_recall", "simcse_f1")
MATCHA_SCORE_METRICS = ("matcha_precision", "matcha_recall", "matcha_f1")


def load_test_cases(path: Path) -> pd.DataFrame:
    read_attempts = [
        {"sep": ";", "encoding": "utf-8-sig", "engine": "python"},
        {"sep": ",", "encoding": "utf-8-sig", "engine": "python"},
        {"sep": ";", "encoding": "utf-8", "engine": "python"},
        {"sep": ",", "encoding": "utf-8", "engine": "python"},
    ]
    last_error = None
    for kwargs in read_attempts:
        try:
            df = pd.read_csv(path, **kwargs)
            if len(df.columns) > 1:
                return df
        except Exception as e:
            last_error = e
    raise ValueError(f"Could not load CSV correctly. Last error: {last_error}")


def find_column(df: pd.DataFrame, candidates):
    normalized = {c.strip().lower(): c for c in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    raise ValueError(
        f"None of the candidate columns were found: {candidates}. Available columns: {list(df.columns)}"
    )


def clean_for_metric(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    text = text.replace("-\n", "")
    text = text.replace("\n", " ")
    text = re.sub("\\s+", " ", text).strip()
    return text


def safe_ratio(a, b):
    if b == 0:
        return 0.0
    return a / b


def normalize_column_name(col):
    return str(col).strip()


def make_safe_display_value(x):
    if pd.isna(x):
        return ""
    return str(x)


def parse_numeric_series(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .pipe(pd.to_numeric, errors="coerce")
    )


def display_section(title, text=None):
    display(Markdown(f"## {title}"))
    if text:
        display(Markdown(text))


def show_plot():
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_correlation_heatmap(corr, title):
    if corr.empty:
        return
    plt.figure(figsize=(12, 10))
    heatmap = plt.imshow(corr.values, vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    # Annotate finite correlation values with contrast against the cell color.
    for row in range(len(corr.index)):
        for column in range(len(corr.columns)):
            value = corr.iat[row, column]
            if pd.isna(value):
                continue
            red, green, blue, _ = heatmap.cmap(heatmap.norm(value))
            brightness = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            plt.text(
                column, row, f"{value:.2f}",
                ha="center", va="center", fontsize=10,
                color="black" if brightness > 0.5 else "white",
            )
    plt.title(title)
    show_plot()



def parse_numeric(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})
        .pipe(pd.to_numeric, errors="coerce")
    )


def calculate_metric_ranks(frame, metrics, semantic_metrics):
    """Average per-metric ranks (MMR) and semantic scores (SM); ties share ranks."""
    frame = frame.copy()
    rank_columns = []
    for metric in metrics:
        rank_column = f"_rank_{metric}"
        frame[rank_column] = frame[metric].rank(method="average", ascending=False)
        rank_columns.append(rank_column)
    frame["MMR"] = frame[rank_columns].mean(axis=1)
    frame["SM"] = frame[semantic_metrics].mean(axis=1)
    return frame.drop(columns=rank_columns)


def matches_qwen25_32b(value):
    text = str(value).strip().lower()
    qwen_match = re.search("qwen[\\s/_:-]*2[._-]?5", text)
    size_match = re.search("(?<!\\d)32[\\s_-]*b\\b", text)
    return bool(qwen_match and size_match)


def select_best_per_method(frame):
    frame = frame.copy()
    frame = frame.sort_values(
        ["method", "MMR", "SM"], ascending=[True, True, False], kind="stable"
    )
    return frame.groupby("method", as_index=False, dropna=False).first()


def add_composite_columns(df, *, lexical_metrics, semantic_metrics, primary_metrics):
    out = df.copy()
    out["lexical_mean_f1"] = out[lexical_metrics].mean(axis=1)
    out["semantic_mean"] = out[semantic_metrics].mean(axis=1)
    out["primary_metric_mean"] = out[primary_metrics].mean(axis=1)
    rank_cols = []
    for metric in primary_metrics:
        rank_col = f"rank_{metric}"
        out[rank_col] = out.groupby("dataset")[metric].rank(
            method="average", ascending=False
        )
        rank_cols.append(rank_col)
    out["mean_metric_rank"] = out[rank_cols].mean(axis=1)
    out["overall_rank_within_dataset"] = (
        out.groupby("dataset")["mean_metric_rank"]
        .rank(method="dense", ascending=True)
        .astype("Int64")
    )
    return out


def rank_aggregate_table(df, group_cols, *, aggregate_fn):
    out = aggregate_fn(df, group_cols)
    if "dataset" in group_cols:
        out["rank_within_dataset"] = (
            out.groupby("dataset")["mean_metric_rank"]
            .rank(method="dense", ascending=True)
            .astype("Int64")
        )
    return out


def conditional_consistency_analysis(
    df, competitor_col, fixed_col, condition_cols, metrics, *, consistency_fn
):
    outputs = []
    for fixed_value, subset in df.groupby(fixed_col, dropna=False):
        _, summary = consistency_fn(
            subset,
            competitor_col=competitor_col,
            condition_cols=condition_cols,
            metrics=metrics,
        )
        if summary.empty:
            continue
        summary.insert(0, fixed_col, fixed_value)
        outputs.append(summary)
    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True)


def tokenize_words(text: str):
    text = clean_for_metric(text).lower()
    return re.findall("(?u)\\b\\w[\\wäöüß]+\\b", text)


# 2. Tender evaluation
def evaluate_tender(data_root):
    """Run the original analyses in order and return their result variables."""

    # 2.1 Question–gold similarity: lexical overlap, ROUGE, and BGE-M3 scores.
    # Results are saved to Excel without console tables or CSV exports.

    import torch
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    from rouge_score import rouge_scorer
    BASE_DIR = data_root / 'tender_dataset/test_cases.csv'
    CSV_PATH = BASE_DIR
    OUTPUT_DIR = CSV_PATH.parent / 'question_gold_similarity_analysis'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_XLSX = OUTPUT_DIR / 'question_gold_similarity_scores.xlsx'
    EMBEDDING_MODEL = 'BAAI/bge-m3'
    df = load_test_cases(CSV_PATH)
    QUESTION_COL = find_column(df, ['Example Question', 'example_question', 'question', 'Query', 'query', 'original_question'])
    GOLD_COL = find_column(df, ['Gold Text', 'gold_text', 'ground_truth', 'Ground Truth', 'answer', 'reference', 'Reference'])

    def unique_word_jaccard(a: str, b: str) -> float:
        a_set = set(tokenize_words(a))
        b_set = set(tokenize_words(b))
        if not a_set and (not b_set):
            return 1.0
        if not a_set or not b_set:
            return 0.0
        return len(a_set & b_set) / len(a_set | b_set)

    def word_overlap_recall_question_covers_gold(question: str, gold: str) -> float:
        q_set = set(tokenize_words(question))
        g_set = set(tokenize_words(gold))
        if not g_set:
            return 0.0
        return len(q_set & g_set) / len(g_set)

    def word_overlap_precision_question_terms_in_gold(question: str, gold: str) -> float:
        q_set = set(tokenize_words(question))
        g_set = set(tokenize_words(gold))
        if not q_set:
            return 0.0
        return len(q_set & g_set) / len(q_set)
    questions = df[QUESTION_COL].fillna('').astype(str).map(clean_for_metric).tolist()
    gold_texts = df[GOLD_COL].fillna('').astype(str).map(clean_for_metric).tolist()
    df['question_char_count'] = [len(q) for q in questions]
    df['gold_char_count'] = [len(g) for g in gold_texts]
    df['question_word_count'] = [len(tokenize_words(q)) for q in questions]
    df['gold_word_count'] = [len(tokenize_words(g)) for g in gold_texts]
    df['question_to_gold_char_ratio'] = [safe_ratio(df.loc[i, 'question_char_count'], df.loc[i, 'gold_char_count']) for i in df.index]
    df['question_to_gold_word_ratio'] = [safe_ratio(df.loc[i, 'question_word_count'], df.loc[i, 'gold_word_count']) for i in df.index]
    all_texts = questions + gold_texts
    word_tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), token_pattern='(?u)\\b\\w[\\wäöüß]+\\b', min_df=1)
    word_tfidf_matrix = word_tfidf.fit_transform(all_texts)
    q_word_tfidf = word_tfidf_matrix[:len(questions)]
    g_word_tfidf = word_tfidf_matrix[len(questions):]
    df['tfidf_word_1_2gram_cosine'] = [float(cosine_similarity(q_word_tfidf[i], g_word_tfidf[i])[0, 0]) for i in range(len(df))]
    char_tfidf = TfidfVectorizer(lowercase=True, analyzer='char_wb', ngram_range=(3, 5), min_df=1)
    char_tfidf_matrix = char_tfidf.fit_transform(all_texts)
    q_char_tfidf = char_tfidf_matrix[:len(questions)]
    g_char_tfidf = char_tfidf_matrix[len(questions):]
    df['tfidf_char_3_5gram_cosine'] = [float(cosine_similarity(q_char_tfidf[i], g_char_tfidf[i])[0, 0]) for i in range(len(df))]
    df['unique_word_jaccard'] = [unique_word_jaccard(questions[i], gold_texts[i]) for i in range(len(df))]
    df['question_terms_covered_by_gold_precision'] = [word_overlap_precision_question_terms_in_gold(questions[i], gold_texts[i]) for i in range(len(df))]
    df['gold_terms_covered_by_question_recall'] = [word_overlap_recall_question_covers_gold(questions[i], gold_texts[i]) for i in range(len(df))]
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    rouge1_p = []
    rouge1_r = []
    rouge1_f = []
    rouge2_p = []
    rouge2_r = []
    rouge2_f = []
    rougeL_p = []
    rougeL_r = []
    rougeL_f = []
    for q, g in zip(questions, gold_texts):
        scores = rouge.score(g, q)
        rouge1_p.append(scores['rouge1'].precision)
        rouge1_r.append(scores['rouge1'].recall)
        rouge1_f.append(scores['rouge1'].fmeasure)
        rouge2_p.append(scores['rouge2'].precision)
        rouge2_r.append(scores['rouge2'].recall)
        rouge2_f.append(scores['rouge2'].fmeasure)
        rougeL_p.append(scores['rougeL'].precision)
        rougeL_r.append(scores['rougeL'].recall)
        rougeL_f.append(scores['rougeL'].fmeasure)
    df['rouge1_precision_question_vs_gold'] = rouge1_p
    df['rouge1_recall_question_vs_gold'] = rouge1_r
    df['rouge1_f1_question_vs_gold'] = rouge1_f
    df['rouge2_precision_question_vs_gold'] = rouge2_p
    df['rouge2_recall_question_vs_gold'] = rouge2_r
    df['rouge2_f1_question_vs_gold'] = rouge2_f
    df['rougeL_precision_question_vs_gold'] = rougeL_p
    df['rougeL_recall_question_vs_gold'] = rougeL_r
    df['rougeL_f1_question_vs_gold'] = rougeL_f
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    question_embeddings = embedding_model.encode(questions, batch_size=8, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    gold_embeddings = embedding_model.encode(gold_texts, batch_size=8, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    df['bge_m3_semantic_cosine'] = np.sum(question_embeddings * gold_embeddings, axis=1).astype(float)
    df['rank_by_bge_m3_similarity_desc'] = df['bge_m3_semantic_cosine'].rank(method='min', ascending=False).astype(int)
    df['rank_by_tfidf_word_similarity_desc'] = df['tfidf_word_1_2gram_cosine'].rank(method='min', ascending=False).astype(int)
    df['rank_by_gold_term_coverage_desc'] = df['gold_terms_covered_by_question_recall'].rank(method='min', ascending=False).astype(int)
    metadata_cols = []
    for candidate in ['ID', 'id', 'Project Name', 'project_name', 'Theme', 'theme', 'Chapter', 'chapter']:
        if candidate in df.columns:
            metadata_cols.append(candidate)
    score_cols = ['question_word_count', 'gold_word_count', 'question_to_gold_word_ratio', 'question_char_count', 'gold_char_count', 'question_to_gold_char_ratio', 'tfidf_word_1_2gram_cosine', 'tfidf_char_3_5gram_cosine', 'unique_word_jaccard', 'question_terms_covered_by_gold_precision', 'gold_terms_covered_by_question_recall', 'rouge1_precision_question_vs_gold', 'rouge1_recall_question_vs_gold', 'rouge1_f1_question_vs_gold', 'rouge2_precision_question_vs_gold', 'rouge2_recall_question_vs_gold', 'rouge2_f1_question_vs_gold', 'rougeL_precision_question_vs_gold', 'rougeL_recall_question_vs_gold', 'rougeL_f1_question_vs_gold', 'bge_m3_semantic_cosine', 'rank_by_bge_m3_similarity_desc', 'rank_by_tfidf_word_similarity_desc', 'rank_by_gold_term_coverage_desc']
    text_cols = [QUESTION_COL, GOLD_COL]
    ordered_cols = metadata_cols + score_cols + text_cols
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    df_out = df[ordered_cols + remaining_cols].copy()
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_columns', 200)
    pd.set_option('display.width', 200)
    display_cols = metadata_cols + score_cols
    df_out.to_excel(OUTPUT_XLSX, index=False)
    summary_metrics = ['tfidf_word_1_2gram_cosine', 'tfidf_char_3_5gram_cosine', 'unique_word_jaccard', 'question_terms_covered_by_gold_precision', 'gold_terms_covered_by_question_recall', 'rouge1_f1_question_vs_gold', 'rouge2_f1_question_vs_gold', 'rougeL_f1_question_vs_gold', 'bge_m3_semantic_cosine']
    summary = df_out[summary_metrics].describe().T
    summary_path = OUTPUT_DIR / 'question_gold_similarity_summary.xlsx'
    summary.to_excel(summary_path)

    # 2.2 Lexical difficulty: IDF coverage and difficulty scores, without embeddings.
    # Results are saved to Excel without console tables or CSV exports.

    BASE_DIR = data_root / 'tender_dataset/test_cases.csv'
    CSV_PATH = BASE_DIR
    OUTPUT_DIR = CSV_PATH.parent / 'question_gold_lexical_difficulty_analysis'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_XLSX = OUTPUT_DIR / 'question_gold_lexical_scores.xlsx'
    OUTPUT_SUMMARY_XLSX = OUTPUT_DIR / 'question_gold_lexical_scores_summary.xlsx'
    df = load_test_cases(CSV_PATH)
    QUESTION_COL = find_column(df, ['Example Question', 'example_question', 'question', 'Query', 'query', 'original_question'])
    GOLD_COL = find_column(df, ['Gold Text', 'gold_text', 'ground_truth', 'Ground Truth', 'reference', 'Reference'])
    questions = df[QUESTION_COL].fillna('').astype(str).map(clean_for_metric).tolist()
    gold_texts = df[GOLD_COL].fillna('').astype(str).map(clean_for_metric).tolist()
    df['question_word_count'] = [len(tokenize_words(q)) for q in questions]
    df['gold_word_count'] = [len(tokenize_words(g)) for g in gold_texts]
    df['question_to_gold_word_ratio'] = [safe_ratio(df.loc[i, 'question_word_count'], df.loc[i, 'gold_word_count']) for i in df.index]
    all_texts = questions + gold_texts
    tfidf_vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), token_pattern='(?u)\\b\\w[\\wäöüß]+\\b', min_df=1)
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)
    q_tfidf = tfidf_matrix[:len(questions)]
    g_tfidf = tfidf_matrix[len(questions):]
    df['tfidf_word_1_2gram_cosine'] = [float(cosine_similarity(q_tfidf[i], g_tfidf[i])[0, 0]) for i in range(len(df))]
    idf_vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 1), token_pattern='(?u)\\b\\w[\\wäöüß]+\\b', min_df=1, use_idf=True, smooth_idf=True)
    idf_vectorizer.fit(all_texts)
    idf_by_term = dict(zip(idf_vectorizer.get_feature_names_out(), idf_vectorizer.idf_))

    def idf_weighted_gold_coverage(question: str, gold: str, idf_lookup: dict) -> float:
        question_terms = set(tokenize_words(question))
        gold_terms = set(tokenize_words(gold))
        if not gold_terms:
            return 0.0
        denominator = sum((idf_lookup.get(term, 0.0) for term in gold_terms))
        if denominator == 0:
            return 0.0
        numerator = sum((idf_lookup.get(term, 0.0) for term in gold_terms if term in question_terms))
        return numerator / denominator
    df['idf_weighted_gold_coverage_by_question'] = [idf_weighted_gold_coverage(questions[i], gold_texts[i], idf_by_term) for i in range(len(df))]
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)
    rouge1_recall = []
    rougeL_recall = []
    for q, g in zip(questions, gold_texts):
        scores = rouge.score(g, q)
        rouge1_recall.append(scores['rouge1'].recall)
        rougeL_recall.append(scores['rougeL'].recall)
    df['rouge1_recall_question_vs_gold'] = rouge1_recall
    df['rougeL_recall_question_vs_gold'] = rougeL_recall
    df['rank_by_idf_weighted_gold_coverage_desc'] = df['idf_weighted_gold_coverage_by_question'].rank(method='min', ascending=False).astype(int)
    df['rank_by_question_to_gold_word_ratio_desc'] = df['question_to_gold_word_ratio'].rank(method='min', ascending=False).astype(int)
    df['rank_by_tfidf_word_cosine_desc'] = df['tfidf_word_1_2gram_cosine'].rank(method='min', ascending=False).astype(int)
    df['rank_by_rouge1_recall_desc'] = df['rouge1_recall_question_vs_gold'].rank(method='min', ascending=False).astype(int)
    df['rank_by_rougeL_recall_desc'] = df['rougeL_recall_question_vs_gold'].rank(method='min', ascending=False).astype(int)
    metadata_cols = []
    for candidate in ['ID', 'id', 'Project Name', 'project_name', 'Theme', 'theme', 'Chapter', 'chapter']:
        if candidate in df.columns:
            metadata_cols.append(candidate)
    metric_cols = ['question_word_count', 'gold_word_count', 'question_to_gold_word_ratio', 'idf_weighted_gold_coverage_by_question', 'tfidf_word_1_2gram_cosine', 'rouge1_recall_question_vs_gold', 'rougeL_recall_question_vs_gold', 'rank_by_idf_weighted_gold_coverage_desc', 'rank_by_question_to_gold_word_ratio_desc', 'rank_by_tfidf_word_cosine_desc', 'rank_by_rouge1_recall_desc', 'rank_by_rougeL_recall_desc']
    text_cols = [QUESTION_COL, GOLD_COL]
    ordered_cols = metadata_cols + metric_cols + text_cols
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    df_out = df[ordered_cols + remaining_cols].copy()
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_columns', 200)
    pd.set_option('display.width', 200)
    df_out.to_excel(OUTPUT_XLSX, index=False)
    summary_metrics = ['question_to_gold_word_ratio', 'idf_weighted_gold_coverage_by_question', 'tfidf_word_1_2gram_cosine', 'rouge1_recall_question_vs_gold', 'rougeL_recall_question_vs_gold']
    summary = df_out[summary_metrics].describe().T
    summary.to_excel(OUTPUT_SUMMARY_XLSX)

    # 2.3 Standard systems: load, aggregate, rank, compare, and plot.
    display_section("Tender - standard-system generation quality")

    BASE_DIR = data_root / "tender_dataset/results/metric_analysis"
    ID_COLS = ["dataset", "method", "top_k", "generator_model"]
    COUNT_COLS = ["n_rows", "n_questions", "n_empty_answers", "n_empty_ground_truth"]
    ROUGE_METRICS = list(ROUGE_SCORE_METRICS)
    BERTSCORE_METRICS = list(BERT_SCORE_METRICS)
    SIMCSE_METRICS = list(SIMCSE_SCORE_METRICS)
    ECS_METRICS = ["ecs_bge_m3"]
    MATCHA_METRICS = list(MATCHA_SCORE_METRICS)
    ALL_METRICS = (
        ROUGE_METRICS
        + BERTSCORE_METRICS
        + SIMCSE_METRICS
        + ECS_METRICS
        + MATCHA_METRICS
    )
    PRIMARY_METRICS = list(GENERATION_METRICS)
    LEXICAL_METRICS = list(LEXICAL_SCORE_METRICS)
    SEMANTIC_METRICS = list(SEMANTIC_SCORE_METRICS)
    PER_QUESTION_REQUIRED_COLS = ID_COLS + ["test_case_id"] + PRIMARY_METRICS
    AGENTIC_EXCLUDE_TOKEN = (
        "tender_agentic_tender_answers"
    )
    AGENTIC_EXCLUDE_METHODS = {"agentic_tender_answers"}
    SPARSE_PARENT_L_PATH_TOKEN = "sparse_parent_bm25l"
    SPARSE_PARENT_L_METHOD = "sparse_parent_L"
    TOP_N_SETUPS_TO_SHOW = 30
    TOP_N_INTERACTION_ROWS = 50
    ROUND_DECIMALS = 4

    def is_sparse_parent_L_path(path, base_dir):
        relative_path = str(Path(path).relative_to(base_dir)).lower()
        return SPARSE_PARENT_L_PATH_TOKEN in relative_path

    # Discover eligible Tender files and exclude agentic results.
    def find_valid_per_question_files(base_dir):
        base_dir = Path(base_dir)
        general_candidates = {
            p
            for p in base_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() == ".xlsx"
            and p.name.startswith("per_question")
            and (not p.name.startswith("~$"))
        }
        sparse_parent_L_folders = [
            p
            for p in base_dir.rglob("*")
            if p.is_dir() and SPARSE_PARENT_L_PATH_TOKEN in p.name.lower()
        ]
        sparse_parent_L_candidates = set()
        for folder in sparse_parent_L_folders:
            for path in folder.rglob("per_question*.xlsx"):
                if path.is_file() and (not path.name.startswith("~$")):
                    sparse_parent_L_candidates.add(path)
        candidates = sorted(general_candidates | sparse_parent_L_candidates)
        valid_files = []
        skipped_files = []
        excluded_files = []
        for path in candidates:
            relative_path = str(path.relative_to(base_dir))
            normalized_path = relative_path.lower()
            if AGENTIC_EXCLUDE_TOKEN.lower() in normalized_path:
                excluded_files.append(
                    (relative_path, "explicitly excluded agentic result set")
                )
                continue
            try:
                preview = pd.read_excel(path, nrows=5)
                preview.columns = [normalize_column_name(c) for c in preview.columns]
            except Exception as exc:
                skipped_files.append((relative_path, f"read error: {exc}"))
                continue
            missing = [
                c for c in PER_QUESTION_REQUIRED_COLS if c not in preview.columns
            ]
            if missing:
                skipped_files.append(
                    (relative_path, f"missing required columns: {missing[:5]}")
                )
                continue
            if is_sparse_parent_L_path(path, base_dir):
                preview["method"] = SPARSE_PARENT_L_METHOD
            method_values = preview["method"].astype(str).str.strip().str.lower()
            if method_values.isin(
                {value.lower() for value in AGENTIC_EXCLUDE_METHODS}
            ).any():
                excluded_files.append(
                    (relative_path, "explicitly excluded agentic method")
                )
                continue
            valid_files.append(path)
        return (valid_files, skipped_files, excluded_files)

    # Load and validate per-question metrics before setup-level aggregation.
    def load_per_question_metrics(base_dir):
        valid_files, skipped_files, excluded_files = find_valid_per_question_files(
            base_dir
        )
        display(
            Markdown(
                f"### Loaded files\nValid per-question metric files found: **{len(valid_files)}**"
            )
        )
        for path in valid_files:
            print(path.relative_to(base_dir))
        if excluded_files:
            display(Markdown("### Explicitly excluded files"))
            for name, reason in excluded_files:
                print(f"{name}: {reason}")
        if skipped_files:
            display(Markdown("### Skipped files"))
            for name, reason in skipped_files:
                print(f"{name}: {reason}")
        if not valid_files:
            raise ValueError(
                f"No valid per_question .xlsx metric files found in {base_dir}"
            )
        frames = []
        for path in valid_files:
            temp = pd.read_excel(path)
            temp.columns = [normalize_column_name(c) for c in temp.columns]
            temp["source_file"] = path.name
            temp["source_path"] = str(path.relative_to(base_dir))
            if is_sparse_parent_L_path(path, base_dir):
                temp["method"] = SPARSE_PARENT_L_METHOD
            frames.append(temp)
        per_question_df = pd.concat(frames, ignore_index=True, sort=False)
        sparse_parent_L_source_mask = (
            per_question_df["source_path"]
            .astype(str)
            .str.lower()
            .str.contains(SPARSE_PARENT_L_PATH_TOKEN, regex=False, na=False)
        )
        n_sparse_parent_L_rows = int(sparse_parent_L_source_mask.sum())
        if n_sparse_parent_L_rows:
            invalid_sparse_parent_L_rows = (
                per_question_df.loc[sparse_parent_L_source_mask, "method"]
                .astype(str)
                .str.strip()
                .ne(SPARSE_PARENT_L_METHOD)
            )
            if invalid_sparse_parent_L_rows.any():
                raise AssertionError(
                    "At least one sparse_parent_BM25L row was not mapped to method='sparse_parent_L'."
                )
            display(
                Markdown(
                    f"Loaded sparse_parent_L (BM25L) per-question rows: **{n_sparse_parent_L_rows}**"
                )
            )
        for col in ID_COLS:
            per_question_df[col] = per_question_df[col].astype(str).str.strip()
        per_question_df["test_case_id"] = (
            per_question_df["test_case_id"].astype(str).str.strip()
        )
        per_question_df["top_k"] = parse_numeric_series(
            per_question_df["top_k"]
        ).astype("Int64")
        per_question_df["top_k"] = per_question_df["top_k"].astype("Int64")
        for col in COUNT_COLS + ALL_METRICS:
            if col in per_question_df.columns:
                per_question_df[col] = parse_numeric_series(per_question_df[col])
        agentic_mask = per_question_df["source_path"].astype(
            str
        ).str.lower().str.contains(
            AGENTIC_EXCLUDE_TOKEN.lower(), regex=False, na=False
        ) | per_question_df[
            "method"
        ].astype(
            str
        ).str.strip().str.lower().isin(
            {value.lower() for value in AGENTIC_EXCLUDE_METHODS}
        )
        n_agentic_rows = int(agentic_mask.sum())
        if n_agentic_rows:
            display(
                Markdown(f"Excluded agentic per-question rows: **{n_agentic_rows}**")
            )
            per_question_df = per_question_df.loc[~agentic_mask].copy()
        before = len(per_question_df)
        dedup_cols = (
            ID_COLS
            + ["test_case_id"]
            + [c for c in COUNT_COLS + ALL_METRICS if c in per_question_df.columns]
        )
        per_question_df = per_question_df.drop_duplicates(
            subset=dedup_cols
        ).reset_index(drop=True)
        after = len(per_question_df)
        display(
            Markdown(
                f"Per-question rows before exact de-duplication: **{before}**  \nPer-question rows after exact de-duplication: **{after}**"
            )
        )
        aggregation = {
            metric: (metric, "mean")
            for metric in ALL_METRICS
            if metric in per_question_df.columns
        }
        if "n_rows" in per_question_df.columns:
            aggregation["n_rows"] = ("n_rows", "sum")
        else:
            aggregation["n_rows"] = ("test_case_id", "size")
        aggregation["n_questions"] = ("test_case_id", "nunique")
        if "n_empty_answers" in per_question_df.columns:
            aggregation["n_empty_answers"] = ("n_empty_answers", "sum")
        if "n_empty_ground_truth" in per_question_df.columns:
            aggregation["n_empty_ground_truth"] = ("n_empty_ground_truth", "sum")
        df = (
            per_question_df.groupby(ID_COLS, dropna=False)
            .agg(**aggregation)
            .reset_index()
        )
        for col in ID_COLS:
            if col != "top_k":
                df[col] = df[col].astype(str).str.strip()
        df["top_k"] = df["top_k"].astype("Int64")
        remaining_agentic = (
            df["method"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({value.lower() for value in AGENTIC_EXCLUDE_METHODS})
        )
        if remaining_agentic.any():
            raise AssertionError("Agentic results remain after exclusion.")
        if n_sparse_parent_L_rows:
            sparse_parent_L_present = (
                df["method"].astype(str).str.strip().eq(SPARSE_PARENT_L_METHOD).any()
            )
            if not sparse_parent_L_present:
                raise AssertionError(
                    "sparse_parent_BM25L files were loaded, but sparse_parent_L is missing after aggregation."
                )
        display(
            Markdown(
                f"Aggregated setup rows used for results presentation: **{len(df)}**"
            )
        )
        return df

    # Tender overviews use question-count weights; keep this rule dataset-specific.
    def weighted_mean(values, weights):
        values = pd.to_numeric(values, errors="coerce")
        weights = pd.to_numeric(weights, errors="coerce")
        mask = values.notna() & weights.notna() & (weights > 0)
        if mask.sum() == 0:
            return np.nan
        return float(np.average(values[mask], weights=weights[mask]))

    def aggregate_metrics(df, group_cols, metrics=None):
        if metrics is None:
            metrics = PRIMARY_METRICS + [
                "lexical_mean_f1",
                "semantic_mean",
                "primary_metric_mean",
                "mean_metric_rank",
            ]
        rows = []
        for keys, group in df.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_cols, keys))
            if "n_questions" in group.columns and group["n_questions"].notna().any():
                weights = group["n_questions"].fillna(1)
            else:
                weights = pd.Series(np.ones(len(group)), index=group.index)
            row["n_experiment_rows"] = len(group)
            if "n_rows" in group.columns:
                row["n_rows_total"] = group["n_rows"].sum()
            if "n_questions" in group.columns:
                row["n_questions_total"] = group["n_questions"].sum()
            for metric in metrics:
                if metric in group.columns:
                    row[metric] = weighted_mean(group[metric], weights)
            rows.append(row)
        out = pd.DataFrame(rows)
        if "mean_metric_rank" in out.columns:
            out = out.sort_values(group_cols + ["mean_metric_rank"], kind="stable")
        else:
            out = out.sort_values(group_cols, kind="stable")
        return out.reset_index(drop=True)

    # Presentation helpers: highlight scores without changing the data.
    def style_best_second(df, metric_cols, lower_is_better_cols=None):
        if lower_is_better_cols is None:
            lower_is_better_cols = []
        metric_cols = [c for c in metric_cols if c in df.columns]

        return df.style.format(precision=ROUND_DECIMALS).apply(
            lambda col: _style_metric_column(col, metric_cols, lower_is_better_cols),
            axis=0,
        )

    # Compare competitors under matched conditions and summarize wins and ranks.
    def consistency_analysis(
        df,
        competitor_col,
        condition_cols,
        metrics=PRIMARY_METRICS,
        score_col_for_summary="semantic_mean",
    ):
        needed = [competitor_col] + condition_cols + metrics
        temp = df[needed].copy()
        agg_cols = condition_cols + [competitor_col]
        temp = temp.groupby(agg_cols, dropna=False)[metrics].mean().reset_index()
        long_rows = []
        for metric in metrics:
            metric_df = temp[condition_cols + [competitor_col, metric]].dropna()
            for condition_keys, group in metric_df.groupby(
                condition_cols, dropna=False
            ):
                if not isinstance(condition_keys, tuple):
                    condition_keys = (condition_keys,)
                if group[competitor_col].nunique() < 2:
                    continue
                group = group.copy()
                group["rank"] = group[metric].rank(method="average", ascending=False)
                max_score = group[metric].max()
                for _, row in group.iterrows():
                    out = dict(zip(condition_cols, condition_keys))
                    out[competitor_col] = row[competitor_col]
                    out["metric"] = metric
                    out["score"] = row[metric]
                    out["rank"] = row["rank"]
                    out["is_best"] = bool(np.isclose(row[metric], max_score))
                    long_rows.append(out)
        long_df = pd.DataFrame(long_rows)
        if long_df.empty:
            return (pd.DataFrame(), pd.DataFrame())
        summary = (
            long_df.groupby([competitor_col], dropna=False)
            .agg(
                compared_cases=("rank", "size"),
                n_best=("is_best", "sum"),
                avg_rank=("rank", "mean"),
                avg_score=("score", "mean"),
            )
            .reset_index()
        )
        summary["win_rate"] = summary["n_best"] / summary["compared_cases"]
        metric_win_counts = (
            long_df[long_df["is_best"]]
            .groupby([competitor_col, "metric"], dropna=False)
            .size()
            .unstack("metric", fill_value=0)
            .reset_index()
        )
        summary = summary.merge(
            metric_win_counts, on=competitor_col, how="left"
        ).fillna(0)
        summary = summary.sort_values(
            ["win_rate", "avg_rank", "avg_score"], ascending=[False, True, False]
        ).reset_index(drop=True)
        return (long_df, summary)

    def identify_best_by_constraint(
        df, group_cols, competitor_col, target_metric="semantic_mean"
    ):
        temp = (
            df.groupby(group_cols + [competitor_col], dropna=False)[target_metric]
            .mean()
            .reset_index()
        )
        rows = []
        for keys, group in temp.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            if group[competitor_col].nunique() < 2:
                continue
            max_score = group[target_metric].max()
            winners = group[group[target_metric].eq(max_score)]
            for _, row in winners.iterrows():
                out = dict(zip(group_cols, keys))
                out[f"best_{competitor_col}"] = row[competitor_col]
                out[target_metric] = row[target_metric]
                rows.append(out)
        return pd.DataFrame(rows)

    METHOD_DISPLAY_NAMES = {
        "dense_parent": "DP",
        "dp": "DP",
        "dense_parent_child": "DPC",
        "dpc": "DPC",
        "sparse_parent_child": "SPC",
        "spc": "SPC",
        "sparse_parent": "SP",
        "sp": "SP",
        "sparse_parent_l": "SPL",
        "spl": "SPL",
        "hybrid_parent_child": "HPC",
        "hpc": "HPC",
    }
    METHOD_DISPLAY_ORDER = ["DP", "DPC", "SPC", "SP", "SPL", "HPC"]
    METHOD_COLORS = {
        "DP": "tab:blue",
        "DPC": "tab:orange",
        "SPC": "tab:brown",
        "SP": "tab:red",
        "SPL": "tab:purple",
        "HPC": "tab:green",
    }

    def get_method_color(method_label):
        return METHOD_COLORS.get(str(method_label).strip(), None)

    def clean_method_label(method_name):
        label = str(method_name).strip()
        return METHOD_DISPLAY_NAMES.get(label.lower(), label)

    def order_method_labels(labels):
        labels = list(labels)
        known = [label for label in METHOD_DISPLAY_ORDER if label in labels]
        unknown = sorted(
            (label for label in labels if label not in METHOD_DISPLAY_ORDER)
        )
        return known + unknown

    GENERATOR_DISPLAY_ORDER = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
    ]
    GENERATOR_ORDER_INDEX = {
        label: index for index, label in enumerate(GENERATOR_DISPLAY_ORDER)
    }

    def generator_sort_key(label):
        label = str(label).strip()
        return (
            GENERATOR_ORDER_INDEX.get(label, len(GENERATOR_DISPLAY_ORDER)),
            label.lower(),
        )

    # Plot helpers: preserve the original series, ordering, and formatting.
    def plot_topk_trends_by_method(df, dataset, metric="mean_metric_rank"):
        plot_df = df[df["dataset"] == dataset].copy()
        if plot_df.empty:
            return
        summary = (
            plot_df.groupby(["method", "top_k"], dropna=False)[metric]
            .mean()
            .reset_index()
            .sort_values("top_k")
        )
        if summary.empty:
            return
        summary["method_display"] = summary["method"].map(clean_method_label)
        ordered_methods = order_method_labels(
            summary["method_display"].dropna().unique()
        )
        plt.figure(figsize=(10, 5))
        for method_display in ordered_methods:
            group = summary[summary["method_display"] == method_display].sort_values(
                "top_k"
            )
            plt.plot(
                group["top_k"],
                group[metric],
                marker="o",
                label=method_display,
                color=get_method_color(method_display),
            )
        plt.title(f"{metric} across top-k by method ({dataset})")
        plt.xlabel("Top-k")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.legend()
        show_plot()


    def plot_heatmap_from_pivot(pivot, title, xlabel, ylabel, value_format="{:.3f}"):
        if pivot.empty:
            return
        plt.figure(
            figsize=(max(8, 0.8 * len(pivot.columns)), max(5, 0.45 * len(pivot.index)))
        )
        plt.imshow(pivot.values, aspect="auto")
        plt.colorbar(label="score")
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.values[i, j]
                if not pd.isna(value):
                    plt.text(
                        j,
                        i,
                        value_format.format(value),
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
        show_plot()


    def plot_method_generator_heatmap(df, dataset, metric="mean_metric_rank"):
        plot_df = df[df["dataset"] == dataset].copy()
        if plot_df.empty:
            return
        plot_df["method_display"] = plot_df["method"].map(clean_method_label)
        pivot = (
            plot_df.groupby(["method_display", "generator_model"], dropna=False)[metric]
            .mean()
            .unstack("generator_model")
        )
        pivot = pivot.reindex(order_method_labels(pivot.index))
        ordered_generators = sorted(pivot.columns, key=generator_sort_key)
        pivot = pivot.reindex(columns=ordered_generators)
        if pivot.empty:
            return
        plt.figure(
            figsize=(max(8, 0.8 * len(pivot.columns)), max(5, 0.45 * len(pivot.index)))
        )
        plt.imshow(pivot.values, aspect="auto", cmap="viridis_r")
        plt.colorbar(label="MMR (lower is better)")
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.title(f"MMR: method by generator ({dataset})")
        plt.xlabel("Generator")
        plt.ylabel("Method")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.values[i, j]
                if not pd.isna(value):
                    plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
        show_plot()



    # Run the Tender loader, then derive composite scores and within-dataset ranks.
    df = load_per_question_metrics(BASE_DIR)
    df = add_composite_columns(
        df,
        lexical_metrics=LEXICAL_METRICS,
        semantic_metrics=SEMANTIC_METRICS,
        primary_metrics=PRIMARY_METRICS,
    )
    display_section(
        "Data overview",
        "The following table shows the loaded datasets, methods, top-k values, and generators.",
    )
    # Show coverage, best setups, and method/top-k/generator summaries.
    overview_counts = {
        "n_rows": len(df),
        "datasets": ", ".join(sorted(df["dataset"].dropna().unique())),
        "methods": ", ".join(sorted(df["method"].dropna().unique())),
        "top_k_values": ", ".join(map(str, sorted(df["top_k"].dropna().unique()))),
        "generators": ", ".join(
            sorted(df["generator_model"].dropna().unique(), key=generator_sort_key)
        ),
    }
    display(pd.DataFrame([overview_counts]))
    display_section(
        "Best individual setups",
        "Rows are ranked within each dataset by the average rank across the primary metrics. Lower mean metric rank is better.",
    )
    setup_cols = [
        "dataset",
        "method",
        "top_k",
        "generator_model",
        "n_rows",
        "n_questions",
        "overall_rank_within_dataset",
        "mean_metric_rank",
        "lexical_mean_f1",
        "semantic_mean",
        "primary_metric_mean",
    ] + PRIMARY_METRICS
    setup_table = (
        df[setup_cols]
        .sort_values(["dataset", "overall_rank_within_dataset", "mean_metric_rank"])
        .reset_index(drop=True)
    )
    display(
        style_best_second(
            setup_table.head(TOP_N_SETUPS_TO_SHOW),
            metric_cols=PRIMARY_METRICS
            + ["lexical_mean_f1", "semantic_mean", "primary_metric_mean"],
            lower_is_better_cols=["mean_metric_rank"],
        )
    )
    display_section(
        "Overall method overview",
        "This table averages each method across available generators and top-k values. Consistency tables below compare methods under matched constraints.",
    )
    method_overview = rank_aggregate_table(
        df, ["dataset", "method"], aggregate_fn=aggregate_metrics
    )
    display(
        style_best_second(
            method_overview,
            metric_cols=PRIMARY_METRICS
            + ["lexical_mean_f1", "semantic_mean", "primary_metric_mean"],
            lower_is_better_cols=["mean_metric_rank"],
        )
    )
    retrieval_df = df.copy()
    display_section(
        "Overall generator overview",
        "This table averages each generator across methods and top-k values.",
    )
    generator_overview = rank_aggregate_table(
        df, ["dataset", "generator_model"], aggregate_fn=aggregate_metrics
    )
    display(
        style_best_second(
            generator_overview,
            metric_cols=PRIMARY_METRICS
            + ["lexical_mean_f1", "semantic_mean", "primary_metric_mean"],
            lower_is_better_cols=["mean_metric_rank"],
        )
    )
    display_section(
        "Consistency of retrieval methods under matched constraints",
        "This compares methods only within the same dataset, generator, and top-k value.",
    )
    method_long, method_consistency = consistency_analysis(
        retrieval_df,
        competitor_col="method",
        condition_cols=["dataset", "generator_model", "top_k"],
        metrics=PRIMARY_METRICS,
    )
    display(method_consistency)
    display_section(
        "Method consistency under each top-k constraint",
        "This shows whether the best retrieval method changes depending on context size.",
    )
    method_by_topk_consistency = conditional_consistency_analysis(
        retrieval_df,
        competitor_col="method",
        fixed_col="top_k",
        condition_cols=["dataset", "generator_model"],
        metrics=PRIMARY_METRICS,
        consistency_fn=consistency_analysis,
    )
    display(method_by_topk_consistency)
    display_section(
        "Method consistency under each generator constraint",
        "This shows whether some retrieval methods work better with specific generator models.",
    )
    method_by_generator_consistency = conditional_consistency_analysis(
        retrieval_df,
        competitor_col="method",
        fixed_col="generator_model",
        condition_cols=["dataset", "top_k"],
        metrics=PRIMARY_METRICS,
        consistency_fn=consistency_analysis,
    )
    display(method_by_generator_consistency)
    display_section(
        "Best method per matched constraint",
        "This identifies the best retrieval method for each dataset, generator, and top-k combination according to semantic_mean.",
    )
    best_method_by_constraint = identify_best_by_constraint(
        retrieval_df,
        group_cols=["dataset", "generator_model", "top_k"],
        competitor_col="method",
        target_metric="semantic_mean",
    )
    display(best_method_by_constraint.head(TOP_N_INTERACTION_ROWS))
    display_section(
        "Detailed generator by method by top-k overview",
        "This table supports direct comparison of model behavior across retrieval methods and top-k values.",
    )
    generator_method_topk = rank_aggregate_table(
        df,
        ["dataset", "generator_model", "method", "top_k"],
        aggregate_fn=aggregate_metrics,
    )
    generator_method_topk = generator_method_topk.sort_values(
        ["dataset", "generator_model", "mean_metric_rank"]
    ).reset_index(drop=True)
    display(
        style_best_second(
            generator_method_topk.head(TOP_N_INTERACTION_ROWS),
            metric_cols=PRIMARY_METRICS
            + ["lexical_mean_f1", "semantic_mean", "primary_metric_mean"],
            lower_is_better_cols=["mean_metric_rank"],
        )
    )
    display_section(
        "Metric correlation matrix",
        "Spearman and Pearson correlations compare F1 metrics and the standalone ECS score; precision, recall, and derived means are excluded.",
    )
    available_metric_cols = [metric for metric in GENERATION_METRICS if metric in df.columns]
    spearman_corr = (
        df[available_metric_cols].corr(method="spearman").round(ROUND_DECIMALS)
    )
    pearson_corr = (
        df[available_metric_cols].corr(method="pearson").round(ROUND_DECIMALS)
    )
    display_section("Spearman correlation")
    display(spearman_corr)
    display_section("Pearson correlation")
    display(pearson_corr)
    display_section("Plots", "The plots are shown inline and are not saved.")
    # Render MMR comparison figures for every loaded dataset.
    for dataset in sorted(df["dataset"].dropna().unique()):
        display_section(f"Plots for dataset: {dataset}")
        plot_topk_trends_by_method(
            df=retrieval_df, dataset=dataset, metric="mean_metric_rank"
        )
        plot_method_generator_heatmap(df=df, dataset=dataset, metric="mean_metric_rank")
    plot_correlation_heatmap(
        spearman_corr, "Tender - Spearman correlation matrix across evaluation metrics"
    )


    # 2.4 Focused view: Qwen2.5-32B sparse-parent scores by retrieval depth.
    display_section("Tender - Qwen2.5-32B sparse-parent scores by top-k")

    qwen32b_sparse_parent = df[
        df["generator_model"]
        .astype(str)
        .str.contains(
            "qwen.*2\\.5.*32b|qwen2\\.5.*32b", case=False, regex=True, na=False
        )
        & df["method"]
        .astype(str)
        .str.lower()
        .isin(["sparse_parent", "sparse parent", "sp"])
    ].copy()
    qwen32b_sparse_parent_table = (
        qwen32b_sparse_parent[["top_k"] + PRIMARY_METRICS]
        .groupby("top_k", as_index=False)
        .mean(numeric_only=True)
        .sort_values("top_k")
        .reset_index(drop=True)
    )
    display(
        qwen32b_sparse_parent_table.style.format(
            {metric: "{:.4f}" for metric in PRIMARY_METRICS}
        )
    )

    # 2.5 Select the best individual configuration for each retrieval method.

    display_section(
        "Best individual system per method",
        "For each method, this table shows the best-performing individual setup, including its generator model and top-k. Selection is based on the lowest mean_metric_rank. Only F1 variants are shown for metrics with precision/recall/F1 splits.",
    )
    BEST_PER_METHOD_METRICS = list(GENERATION_METRICS)
    best_per_method_cols = [
        "dataset",
        "method",
        "generator_model",
        "top_k",
        "n_rows",
        "n_questions",
        "mean_metric_rank",
        "lexical_mean_f1",
        "semantic_mean",
        "primary_metric_mean",
    ] + BEST_PER_METHOD_METRICS
    best_individual_system_per_method = (
        df[best_per_method_cols]
        .sort_values(
            ["dataset", "method", "mean_metric_rank", "primary_metric_mean"],
            ascending=[True, True, True, False],
        )
        .groupby(["dataset", "method"], as_index=False, dropna=False)
        .first()
        .sort_values(["dataset", "mean_metric_rank"], ascending=[True, True])
        .reset_index(drop=True)
    )
    display(
        style_best_second(
            best_individual_system_per_method,
            metric_cols=BEST_PER_METHOD_METRICS
            + ["lexical_mean_f1", "semantic_mean", "primary_metric_mean"],
            lower_is_better_cols=["mean_metric_rank"],
        )
    )

    # 2.6 Easy/hard generation quality: save tables to Excel only.

    BASE_DIR = data_root / 'tender_dataset/results/metric_analysis'
    EASY_TEST_CASE_IDS = {1, 2, 3, 4, 5, 7, 8, 11, 16, 19, 21, 24, 27, 29}
    HARD_TEST_CASE_IDS = {6, 9, 10, 12, 13, 14, 15, 17, 18, 20, 22, 23, 25, 26, 28, 30}
    PRIMARY_METRICS = list(GENERATION_METRICS)
    SEMANTIC_METRICS = list(SEMANTIC_SCORE_METRICS)
    AGENTIC_EXCLUDE_TOKEN = 'tender_agentic_tender_answers'
    METHOD_LABELS = {'sparse_parent': 'SP', 'sparse_parent_L': 'SPL', 'sparse_parent_child': 'SPC', 'dense_parent': 'DP', 'dense_parent_child': 'DPC', 'hybrid_parent_child': 'HPC'}
    METHOD_ORDER = ['SP', 'SPL', 'SPC', 'DP', 'DPC', 'HPC']
    METHOD_COLORS = {'SP': 'tab:red', 'SPL': 'tab:purple', 'SPC': 'tab:brown', 'DP': 'tab:blue', 'DPC': 'tab:orange', 'HPC': 'tab:green'}

    def read_csv_robust(path):
        attempts = [{'sep': ',', 'encoding': 'utf-8-sig'}, {'sep': ';', 'encoding': 'utf-8-sig'}, {'sep': ',', 'encoding': 'utf-8'}, {'sep': ';', 'encoding': 'utf-8'}]
        last_error = None
        for kwargs in attempts:
            try:
                temp = pd.read_csv(path, **kwargs)
                if len(temp.columns) > 1:
                    return temp
            except Exception as exc:
                last_error = exc
        raise ValueError(f'Could not read {path}. Last error: {last_error}')

    def infer_dataset_from_filename(filename):
        name = Path(filename).name.lower()
        match = re.search('per_question_metrics_([^_]+)', name)
        if match:
            return match.group(1)
        return np.nan

    def contains_explicit_sparse_parent_L(text):
        return bool(re.search('(?<![a-z0-9])sparse_parent_l(?![a-z0-9])', str(text).lower()))

    def normalize_method(method, source_path=''):
        value = str(method).strip()
        lower = value.lower()
        path_lower = str(source_path).lower()
        if 'sparse_parent_bm25l' in path_lower or contains_explicit_sparse_parent_L(path_lower) or lower in {'sparse_parent_bm25l', 'sparse_parent_l'}:
            return 'sparse_parent_L'
        if 'sparse_parent_reversed' in path_lower or lower in {'sparse_parent', 'sparse_parent_reversed'}:
            return 'sparse_parent'
        return value

    def infer_method_from_filename(filename):
        name = str(filename).lower()
        if 'sparse_parent_bm25l' in name or contains_explicit_sparse_parent_L(name):
            return 'sparse_parent_L'
        if 'sparse_parent_reversed' in name:
            return 'sparse_parent'
        methods = ['dense_parent_child', 'hybrid_parent_child', 'sparse_parent_child', 'dense_parent', 'sparse_parent', 'sparse_parent_L']
        for method in methods:
            if method in name:
                return method
        return np.nan

    def extract_top_k(filename):
        match = re.search('_top_(\\d+)', str(filename))
        if match:
            return int(match.group(1))
        return np.nan

    def extract_tender_case_number(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            if np.isfinite(value) and float(value).is_integer():
                return int(value)
        text = str(value).strip()
        try:
            numeric = float(text)
            if np.isfinite(numeric) and numeric.is_integer():
                return int(numeric)
        except (ValueError, TypeError):
            pass
        match = re.search('(\\d+)$', text)
        if match:
            return int(match.group(1))
        return np.nan

    def assign_difficulty(case_number):
        if pd.isna(case_number):
            return 'unknown'
        case_number = int(case_number)
        if case_number in EASY_TEST_CASE_IDS:
            return 'easy'
        if case_number in HARD_TEST_CASE_IDS:
            return 'hard'
        return 'unknown'
    files = sorted((path for path in BASE_DIR.rglob('per_question*.csv') if path.is_file() and (not path.name.startswith('~$')) and (AGENTIC_EXCLUDE_TOKEN.lower() not in str(path).lower())))
    if not files:
        raise ValueError(f'No per-question CSV files found under {BASE_DIR}')
    frames = []
    for path in files:
        temp = read_csv_robust(path)
        temp.columns = [str(column).strip() for column in temp.columns]
        temp['source_file'] = path.name
        temp['source_path'] = str(path)
        if 'dataset' not in temp.columns:
            temp['dataset'] = infer_dataset_from_filename(path.name)
        if 'method' not in temp.columns:
            temp['method'] = infer_method_from_filename(path.name)
        temp['method'] = [normalize_method(method, path) for method in temp['method']]
        if 'top_k' not in temp.columns:
            temp['top_k'] = extract_top_k(path.name)
        frames.append(temp)
    df = pd.concat(frames, ignore_index=True)
    required_columns = ['dataset', 'method', 'top_k', 'generator_model', 'test_case_id'] + PRIMARY_METRICS
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    df['dataset'] = df['dataset'].astype(str).str.strip().str.lower()
    df['generator_model'] = df['generator_model'].astype(str).str.strip()
    df['top_k'] = pd.to_numeric(df['top_k'], errors='coerce').astype('Int64')
    for metric in PRIMARY_METRICS:
        df[metric] = pd.to_numeric(df[metric], errors='coerce')
    if 'n_rows' in df.columns:
        df['n_rows'] = pd.to_numeric(df['n_rows'], errors='coerce')
    df = df[df['dataset'].eq('tender')].copy()
    if df.empty:
        raise ValueError('No tender rows found.')
    df['tender_case_number'] = df['test_case_id'].map(extract_tender_case_number)
    if 'test_id' in df.columns:
        fallback = df['test_id'].map(extract_tender_case_number)
        df['tender_case_number'] = df['tender_case_number'].fillna(fallback)
    df['tender_case_number'] = df['tender_case_number'].astype('Int64')
    df['difficulty_group'] = df['tender_case_number'].map(assign_difficulty)
    df = df[df['difficulty_group'].isin(['easy', 'hard'])].copy()
    if df.empty:
        raise ValueError('No easy/hard tender rows remain. Check test_case_id/test_id parsing.')
    df['semantic_mean'] = df[SEMANTIC_METRICS].mean(axis=1)

    def aggregate_setup(group):
        if 'n_rows' in group.columns:
            weights = group['n_rows'].fillna(1)
        else:
            weights = pd.Series(1.0, index=group.index)
        result = {'n_test_cases': group['tender_case_number'].nunique()}
        for metric in PRIMARY_METRICS + ['semantic_mean']:
            values = pd.to_numeric(group[metric], errors='coerce')
            valid = values.notna() & weights.notna() & (weights > 0)
            if valid.any():
                result[metric] = float(np.average(values[valid], weights=weights[valid]))
            else:
                result[metric] = np.nan
        return pd.Series(result)
    setup_df = df.groupby(['difficulty_group', 'method', 'generator_model', 'top_k'], dropna=False).apply(aggregate_setup, include_groups=False).reset_index()
    if setup_df.empty:
        raise ValueError('No setup-level configurations could be created.')
    rank_columns = []
    for metric in PRIMARY_METRICS:
        rank_column = f'rank_{metric}'
        setup_df[rank_column] = setup_df.groupby('difficulty_group')[metric].rank(method='average', ascending=False)
        rank_columns.append(rank_column)
    setup_df['mean_metric_rank'] = setup_df[rank_columns].mean(axis=1)

    def get_best_configurations(setup_df, difficulty):
        subset = setup_df[setup_df['difficulty_group'] == difficulty].copy()
        if subset.empty:
            return pd.DataFrame()
        subset = subset.sort_values(['method', 'mean_metric_rank', 'semantic_mean'], ascending=[True, True, False])
        best = subset.groupby('method', as_index=False, dropna=False).first()
        best['Method'] = best['method'].map(METHOD_LABELS).fillna(best['method'])
        best = best.sort_values(['mean_metric_rank', 'semantic_mean'], ascending=[True, False]).reset_index(drop=True)
        columns = ['Method', 'generator_model', 'top_k', 'n_test_cases', 'mean_metric_rank', 'semantic_mean', 'rouge1_f1', 'rouge2_f1', 'rougeL_f1', 'bertscore_f1', 'simcse_f1', 'ecs_bge_m3', 'matcha_f1']
        result = best[columns].rename(columns={'generator_model': 'Generator', 'top_k': 'Top-k', 'n_test_cases': 'Cases', 'mean_metric_rank': 'MMR', 'semantic_mean': 'Semantic Mean', 'rouge1_f1': 'ROUGE-1 F1', 'rouge2_f1': 'ROUGE-2 F1', 'rougeL_f1': 'ROUGE-L F1', 'bertscore_f1': 'BERTScore F1', 'simcse_f1': 'SimCSE F1', 'ecs_bge_m3': 'BGE-M3 ECS', 'matcha_f1': 'MATCHA F1'})
        return result

    DIFFICULTY_OUTPUT_XLSX = (
        data_root
        / "tender_dataset/question_gold_lexical_difficulty_analysis"
        / "tender_easy_hard_generation_quality.xlsx"
    )
    difficulty_outputs = {
        "case_groups": pd.DataFrame(
            [
                {"tender_case_number": case_id, "difficulty_group": group}
                for group, case_ids in (("easy", EASY_TEST_CASE_IDS), ("hard", HARD_TEST_CASE_IDS))
                for case_id in sorted(case_ids)
            ]
        ),
        "question_scores": df.copy(),
        "setup_scores": setup_df.copy(),
        "best_easy": get_best_configurations(setup_df, "easy"),
        "best_hard": get_best_configurations(setup_df, "hard"),
    }
    mmr_by_method_topk = (
        setup_df.groupby(["difficulty_group", "method", "top_k"], as_index=False, dropna=False)
        ["mean_metric_rank"].mean()
    )
    mmr_by_method_topk["method_short"] = (
        mmr_by_method_topk["method"].map(METHOD_LABELS).fillna(mmr_by_method_topk["method"])
    )
    difficulty_outputs["mmr_by_method_topk"] = mmr_by_method_topk

    # 2.7 Hard-case DP/DPC comparison: save tables to Excel only.

    TARGET_DIFFICULTY = 'hard'
    TARGET_METHODS = ['dense_parent', 'dense_parent_child']
    TARGET_GENERATOR_CONTAINS = 'Qwen/Qwen3-235B-A22B-Instruct-2507'
    TARGET_TOP_K = 10
    ROUND_DECIMALS = 4
    F1_METRICS = ['rouge1_f1', 'rouge2_f1', 'rougeL_f1', 'bertscore_f1', 'simcse_f1', 'matcha_f1']
    SEMANTIC_METRICS_FOR_MEAN = list(SEMANTIC_SCORE_METRICS)
    DISPLAY_METRICS = ['semantic_mean'] + F1_METRICS
    required_cols = ['difficulty_group', 'method', 'top_k', 'generator_model', 'tender_case_number'] + F1_METRICS
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f'The following required columns are missing: {missing_cols}. Run the easy/hard per-question analysis code first.')
    work_df = df.copy()
    if 'semantic_mean' not in work_df.columns:
        missing_semantic_cols = [c for c in SEMANTIC_METRICS_FOR_MEAN if c not in work_df.columns]
        if missing_semantic_cols:
            raise ValueError(f'semantic_mean is missing and cannot be computed because these columns are missing: {missing_semantic_cols}')
        work_df['semantic_mean'] = work_df[SEMANTIC_METRICS_FOR_MEAN].mean(axis=1)
    subset = work_df[work_df['difficulty_group'].eq(TARGET_DIFFICULTY) & work_df['method'].isin(TARGET_METHODS) & work_df['generator_model'].astype(str).str.contains(TARGET_GENERATOR_CONTAINS, case=False, regex=False) & work_df['top_k'].astype('Int64').eq(TARGET_TOP_K)].copy()
    if subset.empty:
        available_debug = work_df[work_df['difficulty_group'].eq(TARGET_DIFFICULTY)][['method', 'top_k', 'generator_model']].drop_duplicates().sort_values(['method', 'top_k', 'generator_model'])
    else:
        method_overview = subset.groupby(['method', 'generator_model', 'top_k'], dropna=False).agg(n_rows=('tender_case_number', 'size'), n_test_cases=('tender_case_number', lambda x: x.dropna().nunique()), **{metric: (metric, 'mean') for metric in DISPLAY_METRICS}).reset_index().sort_values('semantic_mean', ascending=False)
        id_cols = ['tender_case_number']
        optional_id_cols = ['test_case_id', 'test_id']
        display_cols = [c for c in optional_id_cols if c in subset.columns] + id_cols + ['method', 'generator_model', 'top_k'] + DISPLAY_METRICS
        per_case_table = subset[display_cols].sort_values(['tender_case_number', 'method']).reset_index(drop=True)
        pair_index_cols = ['tender_case_number', 'generator_model', 'top_k']
        paired = subset[pair_index_cols + ['method'] + DISPLAY_METRICS].pivot_table(index=pair_index_cols, columns='method', values=DISPLAY_METRICS, aggfunc='mean')
        paired.columns = [f'{metric}__{method}' for metric, method in paired.columns]
        paired = paired.reset_index()
        for metric in DISPLAY_METRICS:
            dense_col = f'{metric}__dense_parent'
            dpc_col = f'{metric}__dense_parent_child'
            delta_col = f'delta_dpc_minus_dense__{metric}'
            if dense_col in paired.columns and dpc_col in paired.columns:
                paired[delta_col] = paired[dpc_col] - paired[dense_col]
        delta_cols = [c for c in paired.columns if c.startswith('delta_dpc_minus_dense__')]
        if delta_cols:
            delta_summary = paired[delta_cols].agg(['mean', 'median', 'min', 'max']).T.reset_index().rename(columns={'index': 'metric'})
            delta_summary['metric'] = delta_summary['metric'].str.replace('delta_dpc_minus_dense__', '', regex=False)
            win_summary_rows = []
            for delta_col in delta_cols:
                metric_name = delta_col.replace('delta_dpc_minus_dense__', '')
                values = paired[delta_col].dropna()
                win_summary_rows.append({'metric': metric_name, 'n_paired_cases': len(values), 'dense_parent_child_better': int((values > 0).sum()), 'dense_parent_better': int((values < 0).sum()), 'tie': int((values == 0).sum()), 'dense_parent_child_win_rate': (values > 0).mean() if len(values) > 0 else np.nan, 'mean_delta_dpc_minus_dense': values.mean() if len(values) > 0 else np.nan})
            win_summary = pd.DataFrame(win_summary_rows)

    difficulty_outputs["hard_case_filter"] = pd.DataFrame(
        [{
            "difficulty": TARGET_DIFFICULTY,
            "methods": ", ".join(TARGET_METHODS),
            "generator": TARGET_GENERATOR_CONTAINS,
            "top_k": TARGET_TOP_K,
            "n_matching_rows": len(subset),
        }]
    )
    if subset.empty:
        difficulty_outputs["hard_available_setups"] = available_debug.copy()
    else:
        difficulty_outputs["hard_method_comparison"] = method_overview.copy()
        difficulty_outputs["hard_case_scores"] = per_case_table.copy()
        difficulty_outputs["hard_paired_scores"] = paired.copy()
        if delta_cols:
            difficulty_outputs["hard_paired_deltas"] = (
                paired[pair_index_cols + delta_cols].sort_values("tender_case_number").copy()
            )
            difficulty_outputs["hard_delta_summary"] = delta_summary.copy()
            difficulty_outputs["hard_win_summary"] = win_summary.copy()
        else:
            difficulty_outputs["hard_pair_status"] = pd.DataFrame(
                [{"status": "Paired comparison unavailable: one method is missing for the matched hard cases."}]
            )

    # 2.8 Agentic overview: aggregate the per-question metric file.

    TENDER_AGENTIC_DIR = (
        data_root / "tender_dataset/multi_agent/multi_agent_results"
    )
    AGENTIC_PER_QUESTION_FILE = (
        TENDER_AGENTIC_DIR
        / "per_question_metrics_tender_agentic_tender_answers_all_generators.csv"
    )
    if not AGENTIC_PER_QUESTION_FILE.is_file():
        raise FileNotFoundError(
            f"Agentic per-question CSV not found: {AGENTIC_PER_QUESTION_FILE}"
        )
    DISPLAY_METRICS = list(GENERATION_METRICS)
    SEMANTIC_METRICS = list(SEMANTIC_SCORE_METRICS)
    agentic_question_df = pd.read_csv(AGENTIC_PER_QUESTION_FILE)
    agentic_question_df.columns = agentic_question_df.columns.astype(str).str.strip()
    if "dataset" in agentic_question_df.columns:
        agentic_question_df = agentic_question_df[
            agentic_question_df["dataset"].astype(str).str.strip().str.lower().eq("tender")
        ].copy()
    missing_metrics = [metric for metric in DISPLAY_METRICS if metric not in agentic_question_df.columns]
    if missing_metrics:
        raise ValueError(f"Missing expected agentic metric columns: {missing_metrics}")
    for metric in DISPLAY_METRICS:
        agentic_question_df[metric] = pd.to_numeric(
            agentic_question_df[metric], errors="coerce"
        )
    summary_columns = [
        column for column in ("method", "generator_model", "top_k")
        if column in agentic_question_df.columns
    ]
    agentic_summary = (
        agentic_question_df.groupby(summary_columns, dropna=False)[DISPLAY_METRICS]
        .mean()
        .reset_index()
    )
    agentic_summary["semantic_mean"] = agentic_summary[SEMANTIC_METRICS].mean(axis=1)
    agentic_summary["Method"] = "MA"
    if "top_k" in agentic_summary.columns:
        agentic_summary["Top-k"] = pd.to_numeric(
            agentic_summary["top_k"], errors="coerce"
        ).astype("Int64")
    else:
        agentic_summary["Top-k"] = pd.NA
    overall_agentic_results = (
        agentic_summary[
            [
                "Method",
                "Top-k",
                "semantic_mean",
                "rouge1_f1",
                "rouge2_f1",
                "rougeL_f1",
                "bertscore_f1",
                "simcse_f1",
                "ecs_bge_m3",
                "matcha_f1",
            ]
        ]
        .rename(
            columns={
                "semantic_mean": "SM",
                "rouge1_f1": "R-1",
                "rouge2_f1": "R-2",
                "rougeL_f1": "R-L",
                "bertscore_f1": "BERT",
                "simcse_f1": "SimCSE",
                "ecs_bge_m3": "ECS",
                "matcha_f1": "MATCHA",
            }
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )
    display_section("Tender - agentic generation-quality overview")
    display(
        overall_agentic_results.style.format(
            {
                "SM": "{:.4f}",
                "R-1": "{:.4f}",
                "R-2": "{:.4f}",
                "R-L": "{:.4f}",
                "BERT": "{:.4f}",
                "SimCSE": "{:.4f}",
                "ECS": "{:.4f}",
                "MATCHA": "{:.4f}",
            },
            na_rep="-",
        )
    )

    # 2.9 Compare Qwen2.5-32B standard and agentic systems overall and by difficulty.

    BASE_DIR = data_root / "tender_dataset/results/metric_analysis"
    TENDER_AGENTIC_DIR = data_root / "tender_dataset/multi_agent/multi_agent_results"
    AGENTIC_PER_QUESTION_FILE = (
        TENDER_AGENTIC_DIR
        / "per_question_metrics_tender_agentic_tender_answers_all_generators.csv"
    )
    EASY_TEST_CASE_IDS = {1, 2, 3, 4, 5, 7, 8, 11, 16, 19, 21, 24, 27, 29}
    HARD_TEST_CASE_IDS = {6, 9, 10, 12, 13, 14, 15, 17, 18, 20, 22, 23, 25, 26, 28, 30}
    METRICS = list(GENERATION_METRICS)
    SEMANTIC_METRICS = list(SEMANTIC_SCORE_METRICS)
    CONFIG_COLS = ["method", "generator_model", "top_k"]
    METHOD_ABBREVIATIONS = {
        "dense_parent": "DP",
        "sparse_parent": "SP",
        "dense_parent_child": "DPC",
        "sparse_parent_child": "SPC",
        "hybrid_parent_child": "HPC",
        "agentic": "MA",
    }

    def normalize_metrics(frame):
        frame = frame.copy()
        frame.columns = frame.columns.astype(str).str.strip()
        missing = [metric for metric in METRICS if metric not in frame.columns]
        if missing:
            raise ValueError(f"Missing metric columns: {missing}")
        for metric in METRICS:
            frame[metric] = parse_numeric(frame[metric])
        return frame

    def normalize_config_columns(frame):
        frame = frame.copy()
        required = ["method", "generator_model"]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing configuration columns: {missing}")
        frame["method"] = frame["method"].astype(str).str.strip()
        frame["generator_model"] = frame["generator_model"].astype(str).str.strip()
        if "top_k" in frame.columns:
            frame["top_k"] = parse_numeric(frame["top_k"]).astype("Int64")
        else:
            frame["top_k"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
        return frame

    def select_best_configuration_per_method(frame):
        frame = frame.copy()
        frame = frame.sort_values(
            ["method", "MMR", "SM"], ascending=[True, True, False]
        )
        return frame.groupby("method", as_index=False, dropna=False).first()

    def abbreviate_method(value):
        text = str(value).strip()
        key = text.lower()
        return METHOD_ABBREVIATIONS.get(key, text)

    def short_generator_label(value):
        text = str(value).strip()
        if text.lower() in {"", "nan", "none", "<na>"}:
            return "–"
        if matches_qwen25_32b(text):
            return "Qwen/Qwen2.5-32B-Instruct"
        return text

    def format_final_table(frame):
        frame = frame.copy()
        frame["Method"] = frame["method"].apply(abbreviate_method)
        frame["Generator"] = frame["generator_model"].apply(short_generator_label)

        def format_top_k(row):
            if row["Method"] == "MA":
                return "–"
            if pd.isna(row["top_k"]):
                return "–"
            return str(int(row["top_k"]))

        frame["Top-k"] = frame.apply(format_top_k, axis=1)
        return (
            frame[
                [
                    "Method",
                    "Generator",
                    "Top-k",
                    "MMR",
                    "SM",
                    "rouge1_f1",
                    "rouge2_f1",
                    "rougeL_f1",
                    "bertscore_f1",
                    "simcse_f1",
                    "ecs_bge_m3",
                    "matcha_f1",
                ]
            ]
            .rename(
                columns={
                    "rouge1_f1": "R-1",
                    "rouge2_f1": "R-2",
                    "rougeL_f1": "R-L",
                    "bertscore_f1": "BERT",
                    "simcse_f1": "SimCSE",
                    "ecs_bge_m3": "ECS",
                    "matcha_f1": "MATCHA",
                }
            )
            .sort_values(["MMR", "SM"], ascending=[True, False])
            .reset_index(drop=True)
        )

    def display_result_table(frame):
        display(
            frame.style.format(
                {
                    "MMR": "{:.2f}",
                    "SM": "{:.4f}",
                    "R-1": "{:.4f}",
                    "R-2": "{:.4f}",
                    "R-L": "{:.4f}",
                    "BERT": "{:.4f}",
                    "SimCSE": "{:.4f}",
                    "ECS": "{:.4f}",
                    "MATCHA": "{:.4f}",
                },
                na_rep="–",
            )
        )

    def add_test_case_id(frame):
        frame = frame.copy()
        candidate_columns = [
            "test_case_id",
            "question_id",
            "test_id",
            "ID",
            "id",
            "row_index",
        ]
        id_column = next(
            (column for column in candidate_columns if column in frame.columns), None
        )
        if id_column is None:
            raise ValueError(
                f"Could not identify the test-case ID column. Available columns: {list(frame.columns)}"
            )
        if id_column == "row_index":
            numeric = pd.to_numeric(frame[id_column], errors="coerce")
            if numeric.notna().any() and numeric.min() == 0 and (numeric.max() <= 29):
                numeric = numeric + 1
            frame["_test_case_id"] = numeric.astype("Int64")
            return frame

        def extract_id(value):
            if pd.isna(value):
                return pd.NA
            text = str(value).strip()
            try:
                numeric_value = float(text)
                if numeric_value.is_integer():
                    return int(numeric_value)
            except ValueError:
                pass
            match = re.search("(\\d+)$", text)
            if match:
                return int(match.group(1))
            return pd.NA

        frame["_test_case_id"] = frame[id_column].apply(extract_id).astype("Int64")
        return frame

    # Aggregate standard-system per-question CSVs for the overall comparison.
    standard_per_question_files = sorted(
        path for path in BASE_DIR.rglob("per_question*.csv")
        if path.is_file() and not path.name.startswith("~$")
    )
    standard_pq_frames = []
    for path in standard_per_question_files:
        temp = pd.read_csv(path)
        temp.columns = temp.columns.astype(str).str.strip()
        if "dataset" in temp.columns:
            temp = temp[temp["dataset"].astype(str).str.strip().str.lower().eq("tender")].copy()
        elif "tender" not in path.name.lower():
            continue
        if temp.empty:
            continue
        try:
            temp = normalize_metrics(temp)
            temp = normalize_config_columns(temp)
            temp = add_test_case_id(temp)
        except ValueError:
            continue
        temp = temp[temp["generator_model"].apply(matches_qwen25_32b)].copy()
        if not temp.empty:
            standard_pq_frames.append(temp[CONFIG_COLS + ["_test_case_id"] + METRICS])
    if not standard_pq_frames:
        raise ValueError("No standard Tender Qwen2.5-32B per-question CSVs found.")
    standard_pq = pd.concat(standard_pq_frames, ignore_index=True)
    standard_pq = standard_pq.drop_duplicates(
        subset=CONFIG_COLS + ["_test_case_id"] + METRICS
    ).reset_index(drop=True)
    overall_standard = (
        standard_pq.groupby(CONFIG_COLS, dropna=False)[METRICS]
        .mean()
        .reset_index()
    )
    # Aggregate the agentic per-question CSV as one workflow configuration.
    if not AGENTIC_PER_QUESTION_FILE.is_file():
        raise FileNotFoundError(
            f"Agentic per-question CSV not found: {AGENTIC_PER_QUESTION_FILE}"
        )
    agentic_source = pd.read_csv(AGENTIC_PER_QUESTION_FILE)
    agentic_source.columns = agentic_source.columns.astype(str).str.strip()
    if "dataset" in agentic_source.columns:
        agentic_source = agentic_source[
            agentic_source["dataset"].astype(str).str.strip().str.lower().eq("tender")
        ].copy()
    agentic_source = normalize_metrics(agentic_source)
    agentic_source = normalize_config_columns(agentic_source)
    agentic_source = agentic_source[
        agentic_source["generator_model"].apply(matches_qwen25_32b)
    ].copy()
    if agentic_source.empty:
        raise ValueError("No Tender Qwen2.5-32B agentic per-question rows found.")
    agentic_source = add_test_case_id(agentic_source)
    agentic_source = agentic_source.drop_duplicates(
        subset=CONFIG_COLS + ["_test_case_id"] + METRICS
    ).reset_index(drop=True)
    overall_agentic = pd.DataFrame(
        [
            {
                "method": "agentic",
                "generator_model": "Qwen/Qwen2.5-32B-Instruct",
                "top_k": pd.NA,
                **{metric: overall_value for metric in METRICS
                   for overall_value in [agentic_source[metric].mean()]},
            }
        ]
    )
    overall_agentic["top_k"] = overall_agentic["top_k"].astype("Int64")
    overall_all_configurations = pd.concat(
        [overall_standard, overall_agentic], ignore_index=True
    )
    # Rank all eligible configurations before selecting one winner per method.
    overall_ranked = calculate_metric_ranks(
        overall_all_configurations, METRICS, SEMANTIC_METRICS
    )
    overall_best_per_method = select_best_configuration_per_method(overall_ranked)
    overall_table = format_final_table(overall_best_per_method)
    # Load per-question data for the separate easy and hard comparisons.
    standard_per_question_files = sorted(
        path for path in BASE_DIR.rglob("per_question*.csv")
        if path.is_file() and not path.name.startswith("~$")
    )
    standard_pq_frames = []
    for path in standard_per_question_files:
        temp = pd.read_csv(path)
        temp.columns = temp.columns.astype(str).str.strip()
        if "dataset" in temp.columns:
            temp = temp[
                temp["dataset"].astype(str).str.strip().str.lower().eq("tender")
            ].copy()
        elif "tender" not in path.name.lower():
            continue
        if temp.empty:
            continue
        try:
            temp = normalize_metrics(temp)
            temp = normalize_config_columns(temp)
            temp = add_test_case_id(temp)
        except ValueError:
            continue
        temp = temp[temp["generator_model"].apply(matches_qwen25_32b)].copy()
        if temp.empty:
            continue
        standard_pq_frames.append(temp[CONFIG_COLS + ["_test_case_id"] + METRICS])
    if not standard_pq_frames:
        raise ValueError("No standard Tender Qwen2.5-32B per-question results found.")
    standard_pq = pd.concat(standard_pq_frames, ignore_index=True)
    standard_pq = standard_pq.drop_duplicates(
        subset=CONFIG_COLS + ["_test_case_id"] + METRICS
    ).reset_index(drop=True)
    if not AGENTIC_PER_QUESTION_FILE.exists():
        raise FileNotFoundError(
            f"Agentic per-question file not found: {AGENTIC_PER_QUESTION_FILE}"
        )
    agentic_pq = agentic_source.copy()
    agentic_pq = (
        agentic_pq.groupby("_test_case_id", dropna=False)[METRICS].mean().reset_index()
    )
    agentic_pq["method"] = "agentic"
    agentic_pq["generator_model"] = "Qwen/Qwen2.5-32B-Instruct"
    agentic_pq["top_k"] = pd.Series(pd.NA, index=agentic_pq.index, dtype="Int64")
    agentic_pq = agentic_pq[CONFIG_COLS + ["_test_case_id"] + METRICS]
    all_per_question = pd.concat([standard_pq, agentic_pq], ignore_index=True)

    # Recompute configuration scores and ranks within the requested case subset.
    def build_best_per_method_split(per_question_df, test_case_ids):
        subset = per_question_df[
            per_question_df["_test_case_id"].isin(test_case_ids)
        ].copy()
        if subset.empty:
            raise ValueError("No rows found for the requested test-case split.")
        expected_n = len(test_case_ids)
        config_counts = (
            subset.groupby(CONFIG_COLS, dropna=False)["_test_case_id"]
            .nunique()
            .reset_index(name="n_test_cases")
        )
        incomplete = config_counts[config_counts["n_test_cases"] != expected_n]
        if not incomplete.empty:
            raise ValueError(
                f"At least one configuration does not contain all {expected_n} required test cases:\n{incomplete.to_string(index=False)}"
            )
        configuration_scores = (
            subset.groupby(CONFIG_COLS, dropna=False)[METRICS].mean().reset_index()
        )
        ranked = calculate_metric_ranks(configuration_scores, METRICS, SEMANTIC_METRICS)
        best_per_method = select_best_configuration_per_method(ranked)
        return format_final_table(best_per_method)

    hard_table = build_best_per_method_split(all_per_question, HARD_TEST_CASE_IDS)
    easy_table = build_best_per_method_split(all_per_question, EASY_TEST_CASE_IDS)
    display(Markdown("## Overall — best Qwen2.5-32B configuration per method"))
    display_result_table(overall_table)
    difficulty_outputs["qwen32_overall"] = overall_table.copy()
    difficulty_outputs["qwen32_hard"] = hard_table.copy()
    difficulty_outputs["qwen32_easy"] = easy_table.copy()
    difficulty_outputs["qwen32_question_scores"] = all_per_question.copy()
    difficulty_outputs["README"] = pd.DataFrame(
        [
            {"analysis": "Easy/hard groups", "description": "Fixed Tender test-case IDs; see case_groups."},
            {"analysis": "Best configurations", "description": "Ranks are computed separately within easy and hard groups; lower MMR is better."},
            {"analysis": "MMR by method/top-k", "description": "Underlying data for the former easy/hard line plots."},
            {"analysis": "Hard DP/DPC comparison", "description": "Fixed Qwen3-235B configurations at top-k 10; deltas are DPC minus DP."},
            {"analysis": "Qwen2.5-32B comparison", "description": "Standard and agentic configurations are ranked within each test-case subset."},
        ]
    )
    with pd.ExcelWriter(DIFFICULTY_OUTPUT_XLSX, engine="openpyxl") as difficulty_writer:
        for sheet_name, table in difficulty_outputs.items():
            table.to_excel(
                difficulty_writer, sheet_name=sheet_name, index=False, freeze_panes=(1, 0)
            )

    return locals()


# 3. LEDGAR and HUPD evaluation
# HUPD reuses the LEDGAR loader; both retain their own data and rankings.
def evaluate_ledgar_hupd(data_root, datasets=("ledgar", "hupd")):
    """Run the selected LEDGAR/HUPD analyses with shared helpers only."""
    datasets = normalize_datasets(datasets)
    if "tender" in datasets:
        raise ValueError("Use evaluate_tender() for the Tender dataset.")

    # 3.1 LEDGAR: load available questions, aggregate setups, rank, and compare.

    BASE_DIR = data_root / "ledgar_dataset/results/metric_analysis"
    ID_COLS = ["dataset", "method", "top_k", "generator_model"]
    COUNT_COLS = ["n_empty_answers", "n_empty_ground_truth"]
    METHOD_ABBREVIATIONS = {
        "dense_parent": "DP",
        "dense_parent_child": "DPC",
        "sparse_parent": "SP",
        "sparse_parent_child": "SPC",
        "sparse_parent_L": "SPL",
        "hybrid_parent_child": "HPC",
    }

    def abbreviate_method_value(value):
        if pd.isna(value):
            return value
        value = str(value)
        return METHOD_ABBREVIATIONS.get(value, value)

    def apply_method_abbreviations(df):
        out = df.copy()
        if "method" in out.columns:
            out["method"] = out["method"].map(abbreviate_method_value)
        return out

    ROUGE_METRICS = list(ROUGE_SCORE_METRICS)
    BERTSCORE_METRICS = list(BERT_SCORE_METRICS)
    SIMCSE_METRICS = list(SIMCSE_SCORE_METRICS)
    ECS_METRICS = ["ecs_bge_m3"]
    MATCHA_METRICS = list(MATCHA_SCORE_METRICS)
    ALL_METRICS = (
        ROUGE_METRICS
        + BERTSCORE_METRICS
        + SIMCSE_METRICS
        + ECS_METRICS
        + MATCHA_METRICS
    )
    PRIMARY_METRICS = list(GENERATION_METRICS)
    LEXICAL_METRICS = list(LEXICAL_SCORE_METRICS)
    SEMANTIC_METRICS = list(SEMANTIC_SCORE_METRICS)
    EXPECTED_PER_QUESTION_FILES = {
        "per_question_metrics_ledgar_dense_parent_child_all_generators.xlsx": "dense_parent_child",
        "per_question_metrics_ledgar_sparse_parent_child_all_generators.xlsx": "sparse_parent_child",
        "per_question_metrics_ledgar_sparse_parent_all_generators.xlsx": "sparse_parent",
        "per_question_metrics_ledgar_sparse_parent_reversed_all_generators.xlsx": "sparse_parent",
        "per_question_metrics_ledgar_sparse_parent_L_all_generators.xlsx": "sparse_parent_L",
        "per_question_metrics_ledgar_sparse_parent_BM25L_all_generators.xlsx": "sparse_parent_L",
        "per_question_metrics_ledgar_dense_parent_all_generators.xlsx": "dense_parent",
        "per_question_metrics_ledgar_hybrid_parent_child_all_generators.xlsx": "hybrid_parent_child",
    }
    PER_QUESTION_REQUIRED_COLS = ID_COLS + ["test_case_id"] + ALL_METRICS
    TOP_N_SETUPS_TO_SHOW = 30
    TOP_N_INTERACTION_ROWS = 50
    ROUND_DECIMALS = 4

    def canonicalize_question_ids(series):
        out = series.astype("string").str.strip()
        out = out.mask(out.str.lower().isin(["", "nan", "none", "null", "<na>"]))
        numeric_like = out.str.fullmatch("[+-]?\\d+(?:\\.0+)?", na=False)
        out.loc[numeric_like] = out.loc[numeric_like].map(
            lambda value: str(int(float(value)))
        )
        return out

    def question_id_sort_key(value):
        value = str(value)
        if re.fullmatch("[+-]?\\d+", value):
            return (0, int(value))
        return (1, value)

    def format_question_ids(question_ids):
        ordered = sorted(
            {str(value) for value in question_ids}, key=question_id_sort_key
        )
        return ", ".join(ordered)

    SPARSE_METHOD_BY_FOLDER = {
        **{
            f"ledgar_sparse_parent_qwen2.5-{size}b-instruct": "sparse_parent"
            for size in (7, 32, 72)
        },
        **{
            f"ledgar_sparse_parent_bm25l_qwen2.5-{size}b-instruct": "sparse_parent_L"
            for size in (7, 32, 72)
        },
    }

    def expected_method_for_file(path, base_dir):
        path = Path(path)
        default_method = EXPECTED_PER_QUESTION_FILES[path.name]
        if default_method in {"sparse_parent", "sparse_parent_L"}:
            folder_parts = (Path(base_dir).name,) + path.relative_to(base_dir).parts[
                :-1
            ]
            for folder_name in reversed(folder_parts):
                method = SPARSE_METHOD_BY_FOLDER.get(folder_name.casefold())
                if method is not None:
                    return method
        return default_method

    def find_valid_excel_files(base_dir):
        base_dir = Path(base_dir)
        if not base_dir.exists():
            raise FileNotFoundError(f"BASE_DIR does not exist: {base_dir}")
        folders = [base_dir] + sorted(
            (path for path in base_dir.rglob("*") if path.is_dir())
        )
        candidates = []
        for folder in folders:
            for path in sorted(folder.glob("*.xlsx")):
                if path.name.startswith("~$"):
                    continue
                if path.name in EXPECTED_PER_QUESTION_FILES:
                    candidates.append(path)
        valid_files = []
        skipped_files = []
        for path in candidates:
            expected_method = expected_method_for_file(path, base_dir)
            try:
                preview = pd.read_excel(path, nrows=5)
                preview.columns = [normalize_column_name(c) for c in preview.columns]
            except Exception as exc:
                skipped_files.append((path, f"read error: {exc}"))
                continue
            missing = [
                col for col in PER_QUESTION_REQUIRED_COLS if col not in preview.columns
            ]
            if missing:
                skipped_files.append((path, f"missing required columns: {missing[:8]}"))
                continue
            method_values = (
                preview["method"].dropna().astype(str).str.strip().unique().tolist()
            )
            unexpected_methods = [
                value for value in method_values if value != expected_method
            ]
            if unexpected_methods:
                skipped_files.append(
                    (
                        path,
                        f"filename/method mismatch: expected '{expected_method}', found {unexpected_methods}",
                    )
                )
                continue
            valid_files.append(path)
        return (valid_files, skipped_files)

    # Validate question IDs and duplicate rows before averaging.
    def validate_question_rows(per_question_df):
        temp = per_question_df.copy()
        missing_id_rows = temp["test_case_id"].isna()
        if missing_id_rows.any():
            display(
                Markdown(
                    f"Rows removed because `test_case_id` is missing: **{int(missing_id_rows.sum())}**"
                )
            )
            temp = temp.loc[~missing_id_rows].copy()
        key_cols = ID_COLS + ["test_case_id"]
        duplicate_mask = temp.duplicated(subset=key_cols, keep=False)
        if duplicate_mask.any():
            conflicts = temp.loc[
                duplicate_mask, key_cols + ["source_folder", "source_file"]
            ].sort_values(key_cols)
            display(Markdown("### Conflicting duplicate setup-question rows"))
            display(conflicts)
            raise ValueError(
                "Multiple non-identical rows were found for the same dataset/method/top-k/generator/test_case_id. Resolve these duplicates before aggregation."
            )
        return temp

    # Average each metric over its available, non-missing question scores.
    def aggregate_per_question_to_setup_rows(per_question_df):
        rows = []
        for keys, group in per_question_df.groupby(ID_COLS, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(ID_COLS, keys))
            for col in ["n_empty_answers", "n_empty_ground_truth"]:
                if col in group.columns:
                    row[col] = float(group[col].sum(min_count=1))
                else:
                    row[col] = np.nan
            for metric in ALL_METRICS:
                if metric in group.columns:
                    row[metric] = group[metric].mean()
            if "source_file" in group.columns:
                row["source_file"] = "; ".join(
                    sorted(group["source_file"].dropna().astype(str).unique())
                )
            if "source_folder" in group.columns:
                row["source_folder"] = "; ".join(
                    sorted(group["source_folder"].dropna().astype(str).unique())
                )
            rows.append(row)
        return pd.DataFrame(rows)

    # Load only approved workbooks and report accepted and skipped files.
    def load_metric_summaries(base_dir):
        base_dir = Path(base_dir)
        valid_files, skipped_files = find_valid_excel_files(base_dir)
        display(
            Markdown(
                f"### Loaded files\nValid per-question metric files found: **{len(valid_files)}**"
            )
        )
        for path in valid_files:
            print(path.relative_to(base_dir))
        if skipped_files:
            display(Markdown("### Skipped matching files"))
            for path, reason in skipped_files:
                try:
                    shown_path = path.relative_to(base_dir)
                except ValueError:
                    shown_path = path
                print(f"{shown_path}: {reason}")
        if not valid_files:
            expected_names = "\n".join(
                (f"- {name}" for name in EXPECTED_PER_QUESTION_FILES)
            )
            raise ValueError(
                f"No valid per-question metric files found below {base_dir}.\nExpected one or more of these exact filenames:\n{expected_names}"
            )
        frames = []
        for path in valid_files:
            temp = pd.read_excel(path)
            temp.columns = [normalize_column_name(c) for c in temp.columns]
            expected_method = expected_method_for_file(path, base_dir)
            temp["method"] = temp["method"].astype("string").str.strip()
            blank_method = temp["method"].isna() | temp["method"].eq("")
            temp.loc[blank_method, "method"] = expected_method
            wrong_method = ~temp["method"].eq(expected_method)
            if wrong_method.any():
                wrong_values = sorted(
                    temp.loc[wrong_method, "method"].dropna().astype(str).unique()
                )
                raise ValueError(
                    f"Method mismatch in {path}: expected '{expected_method}', found {wrong_values}"
                )
            temp["source_file"] = path.name
            temp["source_folder"] = str(path.parent.relative_to(base_dir))
            frames.append(temp)
        per_question_df = pd.concat(frames, ignore_index=True, sort=False)
        for col in ["dataset", "method", "generator_model"]:
            per_question_df[col] = per_question_df[col].astype("string").str.strip()
        per_question_df["test_case_id"] = canonicalize_question_ids(
            per_question_df["test_case_id"]
        )
        per_question_df["top_k"] = parse_numeric_series(
            per_question_df["top_k"]
        ).astype("Int64")
        for col in COUNT_COLS + ALL_METRICS:
            if col in per_question_df.columns:
                per_question_df[col] = parse_numeric_series(per_question_df[col])
        before = len(per_question_df)
        dedup_cols = [
            col
            for col in ID_COLS + ["test_case_id"] + ALL_METRICS
            if col in per_question_df.columns
        ]
        per_question_df = per_question_df.drop_duplicates(
            subset=dedup_cols
        ).reset_index(drop=True)
        after = len(per_question_df)
        display(
            Markdown(
                f"Per-question rows before de-duplication: **{before}**  \nPer-question rows after de-duplication: **{after}**"
            )
        )
        per_question_df = validate_question_rows(per_question_df)
        df = aggregate_per_question_to_setup_rows(per_question_df)
        display(
            Markdown(
                f"Aggregated setup rows produced: **{len(df)}** (one row per dataset/method/top-k/generator)."
            )
        )
        return df

    # LEDGAR/HUPD overviews average setups without Tender’s question-count weights.
    def aggregate_metrics(df, group_cols, metrics=None):
        if metrics is None:
            metrics = PRIMARY_METRICS + [
                "lexical_mean_f1",
                "semantic_mean",
                "primary_metric_mean",
                "mean_metric_rank",
            ]
        rows = []
        for keys, group in df.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_cols, keys))
            for metric in metrics:
                if metric in group.columns:
                    row[metric] = pd.to_numeric(group[metric], errors="coerce").mean()
            rows.append(row)
        out = pd.DataFrame(rows)
        if "mean_metric_rank" in out.columns:
            out = out.sort_values(group_cols + ["mean_metric_rank"], kind="stable")
        else:
            out = out.sort_values(group_cols, kind="stable")
        return out.reset_index(drop=True)

    # Presentation helpers: highlight scores without changing the data.
    def style_best_second(df, metric_cols, lower_is_better_cols=None):
        if lower_is_better_cols is None:
            lower_is_better_cols = []
        df = apply_method_abbreviations(df)
        metric_cols = [c for c in metric_cols if c in df.columns]

        return df.style.format(precision=ROUND_DECIMALS).apply(
            lambda col: _style_metric_column(col, metric_cols, lower_is_better_cols),
            axis=0,
        )

    # Compare competitors under matched conditions and summarize wins and ranks.
    def consistency_analysis(
        df, competitor_col, condition_cols, metrics=PRIMARY_METRICS
    ):
        needed = [competitor_col] + condition_cols + metrics
        temp = df[needed].copy()
        agg_cols = condition_cols + [competitor_col]
        temp = temp.groupby(agg_cols, dropna=False)[metrics].mean().reset_index()
        long_rows = []
        for metric in metrics:
            metric_df = temp[condition_cols + [competitor_col, metric]].dropna()
            for condition_keys, group in metric_df.groupby(
                condition_cols, dropna=False
            ):
                if not isinstance(condition_keys, tuple):
                    condition_keys = (condition_keys,)
                if group[competitor_col].nunique() < 2:
                    continue
                group = group.copy()
                group["rank"] = group[metric].rank(method="average", ascending=False)
                max_score = group[metric].max()
                for _, row in group.iterrows():
                    out = dict(zip(condition_cols, condition_keys))
                    out[competitor_col] = row[competitor_col]
                    out["metric"] = metric
                    out["score"] = row[metric]
                    out["rank"] = row["rank"]
                    out["is_best"] = bool(np.isclose(row[metric], max_score))
                    long_rows.append(out)
        long_df = pd.DataFrame(long_rows)
        if long_df.empty:
            return (pd.DataFrame(), pd.DataFrame())
        summary = (
            long_df.groupby([competitor_col], dropna=False)
            .agg(
                compared_cases=("rank", "size"),
                n_best=("is_best", "sum"),
                avg_rank=("rank", "mean"),
                avg_score=("score", "mean"),
            )
            .reset_index()
        )
        summary["win_rate"] = summary["n_best"] / summary["compared_cases"]
        metric_win_counts = (
            long_df[long_df["is_best"]]
            .groupby([competitor_col, "metric"], dropna=False)
            .size()
            .unstack("metric", fill_value=0)
            .reset_index()
        )
        summary = summary.merge(metric_win_counts, on=competitor_col, how="left")
        summary = summary.fillna(0)
        summary = summary.sort_values(
            ["win_rate", "avg_rank", "avg_score"], ascending=[False, True, False]
        ).reset_index(drop=True)
        return (long_df, summary)

    # Plot helpers: preserve the original series, ordering, and formatting.
    def plot_topk_trends_by_method(df, dataset, metric="mean_metric_rank"):
        plot_df = df[df["dataset"] == dataset].copy()
        if plot_df.empty:
            return
        summary = (
            plot_df.groupby(["method", "top_k"], dropna=False)[metric]
            .mean()
            .reset_index()
            .sort_values("top_k")
        )
        if summary.empty:
            return
        plt.figure(figsize=(10, 5))
        for method, group in summary.groupby("method"):
            group = group.sort_values("top_k")
            plt.plot(
                group["top_k"],
                group[metric],
                marker="o",
                label=abbreviate_method_value(method),
            )
        plt.title(f"{metric} across top-k by method ({dataset})")
        plt.xlabel("Top-k")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.legend()
        show_plot()

    def plot_heatmap_from_pivot(
        pivot,
        title,
        xlabel,
        ylabel,
        value_format="{:.3f}",
        cmap=None,
        colorbar_label="score",
    ):
        if pivot.empty:
            return
        plt.figure(
            figsize=(max(8, 0.8 * len(pivot.columns)), max(5, 0.45 * len(pivot.index)))
        )
        plt.imshow(pivot.values, aspect="auto", cmap=cmap)
        plt.colorbar(label=colorbar_label)
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.values[i, j]
                if not pd.isna(value):
                    plt.text(
                        j,
                        i,
                        value_format.format(value),
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
        show_plot()


    def extract_generator_parameter_size_b(model_name):
        matches = re.findall("(\\d+(?:\\.\\d+)?)\\s*[bB](?![A-Za-z])", str(model_name))
        return float(matches[-1]) if matches else np.inf

    def clean_generator_display_name(model_name):
        return str(model_name).split("/")[-1]

    def plot_method_generator_heatmap(df, dataset, metric="mean_metric_rank"):
        plot_df = df[df["dataset"] == dataset].copy()
        if plot_df.empty:
            return
        pivot = (
            plot_df.groupby(["method", "generator_model"], dropna=False)[metric]
            .mean()
            .unstack("generator_model")
        )
        generator_order = sorted(
            pivot.columns,
            key=lambda model: (extract_generator_parameter_size_b(model), str(model)),
        )
        pivot = pivot.reindex(columns=generator_order)
        pivot.index = [abbreviate_method_value(method) for method in pivot.index]
        pivot.columns = [clean_generator_display_name(model) for model in pivot.columns]
        plot_heatmap_from_pivot(
            pivot=pivot,
            title=f"MMR: method by generator ({dataset})",
            xlabel="Generator (ascending parameter size)",
            ylabel="Method",
            value_format="{:.2f}",
            cmap="viridis_r",
            colorbar_label="MMR (lower is better)",
        )


    BEST_PER_METHOD_METRICS = list(GENERATION_METRICS)

    best_per_method_cols = [
        "dataset",
        "method",
        "generator_model",
        "top_k",
        "mean_metric_rank",
        "lexical_mean_f1",
        "semantic_mean",
        "primary_metric_mean",
    ] + BEST_PER_METHOD_METRICS

    if "ledgar" in datasets:
        display_section("LEDGAR - standard-system generation quality")
        # Run the LEDGAR loader using each setup’s available questions.
        df = load_metric_summaries(BASE_DIR)
        df = add_composite_columns(
            df,
            lexical_metrics=LEXICAL_METRICS,
            semantic_metrics=SEMANTIC_METRICS,
            primary_metrics=PRIMARY_METRICS,
        )
        display_section("Data overview")
        # Show coverage, best setups, and method/top-k/generator summaries.
        overview_counts = {
            "datasets": ", ".join(sorted(df["dataset"].dropna().unique())),
            "methods": ", ".join(
                sorted(
                    (
                        abbreviate_method_value(method)
                        for method in df["method"].dropna().unique()
                    )
                )
            ),
            "top_k_values": ", ".join(map(str, sorted(df["top_k"].dropna().unique()))),
            "generators": ", ".join(sorted(df["generator_model"].dropna().unique())),
        }
        display(pd.DataFrame([overview_counts]))
        display_section(
            "Best individual setups",
            "Rows are ranked within each dataset by the average rank across the primary metrics. Lower mean metric rank is better.",
        )
        setup_cols = [
            "dataset",
            "method",
            "top_k",
            "generator_model",
            "overall_rank_within_dataset",
            "mean_metric_rank",
            "lexical_mean_f1",
            "semantic_mean",
            "primary_metric_mean",
        ] + PRIMARY_METRICS
        setup_table = (
            df[setup_cols]
            .sort_values(["dataset", "overall_rank_within_dataset", "mean_metric_rank"])
            .reset_index(drop=True)
        )
        display(
            style_best_second(
                setup_table.head(TOP_N_SETUPS_TO_SHOW),
                metric_cols=PRIMARY_METRICS
                + ["lexical_mean_f1", "semantic_mean", "primary_metric_mean"],
                lower_is_better_cols=["mean_metric_rank"],
            )
        )
        display_section(
            "Overall method overview",
            "This table averages each method across available generators and top-k values.",
        )
        method_overview = rank_aggregate_table(
            df, ["dataset", "method"], aggregate_fn=aggregate_metrics
        )
        display(
            style_best_second(
                method_overview,
                metric_cols=PRIMARY_METRICS
                + ["lexical_mean_f1", "semantic_mean", "primary_metric_mean"],
                lower_is_better_cols=["mean_metric_rank"],
            )
        )
        retrieval_df = df.copy()
        display_section(
            "Overall generator overview",
            "This table averages each generator across methods and top-k values.",
        )
        generator_overview = rank_aggregate_table(
            df, ["dataset", "generator_model"], aggregate_fn=aggregate_metrics
        )
        display(
            style_best_second(
                generator_overview,
                metric_cols=PRIMARY_METRICS
                + ["lexical_mean_f1", "semantic_mean", "primary_metric_mean"],
                lower_is_better_cols=["mean_metric_rank"],
            )
        )
        display_section("Consistency of retrieval methods under matched constraints")
        method_long, method_consistency = consistency_analysis(
            retrieval_df,
            competitor_col="method",
            condition_cols=["dataset", "generator_model", "top_k"],
            metrics=PRIMARY_METRICS,
        )
        display(apply_method_abbreviations(method_consistency))
        display_section("Method consistency under each top-k constraint")
        method_by_topk_consistency = conditional_consistency_analysis(
            retrieval_df,
            competitor_col="method",
            fixed_col="top_k",
            condition_cols=["dataset", "generator_model"],
            metrics=PRIMARY_METRICS,
            consistency_fn=consistency_analysis,
        )
        display(apply_method_abbreviations(method_by_topk_consistency))
        display_section("Method consistency under each generator constraint")
        method_by_generator_consistency = conditional_consistency_analysis(
            retrieval_df,
            competitor_col="method",
            fixed_col="generator_model",
            condition_cols=["dataset", "top_k"],
            metrics=PRIMARY_METRICS,
            consistency_fn=consistency_analysis,
        )
        display(apply_method_abbreviations(method_by_generator_consistency))
        display_section("Detailed generator by method by top-k overview")
        generator_method_topk = rank_aggregate_table(
            df,
            ["dataset", "generator_model", "method", "top_k"],
            aggregate_fn=aggregate_metrics,
        )
        generator_method_topk = generator_method_topk.sort_values(
            ["dataset", "generator_model", "mean_metric_rank"]
        ).reset_index(drop=True)
        display(
            style_best_second(
                generator_method_topk.head(TOP_N_INTERACTION_ROWS),
                metric_cols=PRIMARY_METRICS
                + ["lexical_mean_f1", "semantic_mean", "primary_metric_mean"],
                lower_is_better_cols=["mean_metric_rank"],
            )
        )
        # Render MMR comparison figures for every loaded dataset.
        for dataset in sorted(df["dataset"].dropna().unique()):
            display_section(f"Visualizations for dataset: {dataset}")
            plot_topk_trends_by_method(
                df=retrieval_df, dataset=dataset, metric="mean_metric_rank"
            )
            plot_method_generator_heatmap(df=df, dataset=dataset, metric="mean_metric_rank")


        # Use the same F1 metrics and standalone ECS score as the Tender matrix.
        ledgar_correlation_metrics = [
            metric for metric in GENERATION_METRICS if metric in df.columns
        ]
        ledgar_spearman_corr = (
            df[ledgar_correlation_metrics].corr(method="spearman").round(ROUND_DECIMALS)
        )
        display_section("LEDGAR Spearman correlation")
        display(ledgar_spearman_corr)
        plot_correlation_heatmap(
            ledgar_spearman_corr,
            "LEDGAR - Spearman correlation matrix across evaluation metrics",
        )

        # 3.2 LEDGAR: select the best individual configuration per method.

        display_section(
            "Best individual system per method",
            "For each method, this table shows the best-performing individual setup, including its generator model and top-k. Selection is based on the lowest mean_metric_rank. Only F1 variants are shown for metrics with precision/recall/F1 splits.",
        )
        best_individual_system_per_method = (
            df[best_per_method_cols]
            .sort_values(
                ["dataset", "method", "mean_metric_rank", "primary_metric_mean"],
                ascending=[True, True, True, False],
            )
            .groupby(["dataset", "method"], as_index=False, dropna=False)
            .first()
            .sort_values(["dataset", "mean_metric_rank"])
            .reset_index(drop=True)
        )
        display(
            style_best_second(
                best_individual_system_per_method,
                metric_cols=BEST_PER_METHOD_METRICS
                + ["lexical_mean_f1", "semantic_mean", "primary_metric_mean"],
                lower_is_better_cols=["mean_metric_rank"],
            )
        )

        # 3.3 LEDGAR: rank Qwen2.5-32B standard and agentic configurations together.

        LEDGAR_BASE_DIR = data_root / "ledgar_dataset/results/metric_analysis"
        LEDGAR_AGENTIC_DIR = data_root / "ledgar_dataset/multi_agent/multi_agent_results"
        LEDGAR_AGENTIC_PER_QUESTION_FILE = (
            LEDGAR_AGENTIC_DIR
            / "per_question_metrics_ledgar_agentic_ledgar_answers_all_generators.csv"
        )
        LEDGAR_MMR_METRICS = list(GENERATION_METRICS)
        LEDGAR_SEMANTIC_METRICS = list(SEMANTIC_SCORE_METRICS)
        LEDGAR_CONFIG_COLS = ["method", "generator_model", "top_k"]
        LEDGAR_METHOD_ABBREVIATIONS = {
            "dense_parent": "DP",
            "sparse_parent": "SP",
            "dense_parent_child": "DPC",
            "sparse_parent_child": "SPC",
            "hybrid_parent_child": "HPC",
            "agentic": "MA",
        }

        def ledgar_format_result_table(frame):
            frame = frame.copy()
            frame["Method"] = (
                frame["method"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(LEDGAR_METHOD_ABBREVIATIONS)
                .fillna(frame["method"])
            )
            frame["Generator"] = "Qwen/Qwen2.5-32B-Instruct"
            frame["Top-k"] = frame.apply(
                lambda row: (
                    "–"
                    if row["Method"] == "MA" or pd.isna(row["top_k"])
                    else str(int(row["top_k"]))
                ),
                axis=1,
            )
            return (
                frame[
                    [
                        "Method",
                        "Generator",
                        "Top-k",
                        "MMR",
                        "SM",
                        "rouge1_f1",
                        "rouge2_f1",
                        "rougeL_f1",
                        "bertscore_f1",
                        "simcse_f1",
                        "ecs_bge_m3",
                        "matcha_f1",
                    ]
                ]
                .rename(
                    columns={
                        "rouge1_f1": "R-1",
                        "rouge2_f1": "R-2",
                        "rougeL_f1": "R-L",
                        "bertscore_f1": "BERT",
                        "simcse_f1": "SimCSE",
                        "ecs_bge_m3": "ECS",
                        "matcha_f1": "MATCHA",
                    }
                )
                .sort_values(["MMR", "SM"], ascending=[True, False], kind="stable")
                .reset_index(drop=True)
            )

        required_standard_columns = LEDGAR_CONFIG_COLS + LEDGAR_MMR_METRICS
        missing_standard_columns = [
            column for column in required_standard_columns if column not in df.columns
        ]
        if missing_standard_columns:
            raise ValueError(f"Missing required columns in df: {missing_standard_columns}")
        ledgar_standard_32b = df[df["generator_model"].apply(matches_qwen25_32b)].copy()
        if ledgar_standard_32b.empty:
            raise ValueError("No Qwen2.5-32B configurations found in df.")
        print(
            "Qwen2.5-32B generator values found:",
            ledgar_standard_32b["generator_model"].dropna().unique().tolist(),
        )
        for metric in LEDGAR_MMR_METRICS:
            ledgar_standard_32b[metric] = parse_numeric(ledgar_standard_32b[metric])
        ledgar_standard_32b["top_k"] = parse_numeric(ledgar_standard_32b["top_k"]).astype(
            "Int64"
        )
        ledgar_standard_32b = (
            ledgar_standard_32b.groupby(LEDGAR_CONFIG_COLS, dropna=False)[
                LEDGAR_MMR_METRICS
            ]
            .mean()
            .reset_index()
        )
        display_section("LEDGAR - standard Qwen2.5-32B configurations")
        print("Standard Qwen2.5-32B configurations found:", len(ledgar_standard_32b))
        display(
            ledgar_standard_32b[["method", "generator_model", "top_k"]]
            .sort_values(["method", "top_k"])
            .reset_index(drop=True)
        )
        if not LEDGAR_AGENTIC_PER_QUESTION_FILE.exists():
            raise FileNotFoundError(
                f"LEDGAR agentic per-question file not found:\n{LEDGAR_AGENTIC_PER_QUESTION_FILE}"
            )
        ledgar_agentic_pq = pd.read_csv(
            LEDGAR_AGENTIC_PER_QUESTION_FILE, sep=None, engine="python"
        )
        ledgar_agentic_pq.columns = ledgar_agentic_pq.columns.astype(str).str.strip()
        print("\nAgentic CSV rows loaded:", len(ledgar_agentic_pq))
        print("Agentic CSV columns:", list(ledgar_agentic_pq.columns))
        if "dataset" in ledgar_agentic_pq.columns:
            ledgar_agentic_pq = ledgar_agentic_pq[
                ledgar_agentic_pq["dataset"]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("ledgar")
            ].copy()
        generator_column = next(
            (
                column
                for column in ["generator_model", "generator", "model"]
                if column in ledgar_agentic_pq.columns
            ),
            None,
        )
        if generator_column is None:
            raise ValueError(
                f"Could not identify the generator column in the LEDGAR agentic CSV.\nAvailable columns: {list(ledgar_agentic_pq.columns)}"
            )
        print("Agentic generator column:", generator_column)
        ledgar_agentic_pq = ledgar_agentic_pq[
            ledgar_agentic_pq[generator_column].apply(matches_qwen25_32b)
        ].copy()
        if ledgar_agentic_pq.empty:
            raise ValueError("No Qwen2.5-32B rows found in the LEDGAR agentic CSV.")
        print("Qwen2.5-32B agentic rows:", len(ledgar_agentic_pq))
        missing_agentic_metrics = [
            metric
            for metric in LEDGAR_MMR_METRICS
            if metric not in ledgar_agentic_pq.columns
        ]
        if missing_agentic_metrics:
            raise ValueError(
                f"Missing expected metrics in LEDGAR agentic CSV: {missing_agentic_metrics}"
            )
        for metric in LEDGAR_MMR_METRICS:
            ledgar_agentic_pq[metric] = parse_numeric(ledgar_agentic_pq[metric])
        agentic_metric_counts = (
            ledgar_agentic_pq[LEDGAR_MMR_METRICS]
            .notna()
            .sum()
            .rename("non_missing_rows")
            .to_frame()
        )
        display(Markdown("### LEDGAR Multi-Agent metric availability"))
        display(agentic_metric_counts)
        ledgar_agentic_configuration = pd.DataFrame(
            [
                {
                    "method": "agentic",
                    "generator_model": "Qwen/Qwen2.5-32B-Instruct",
                    "top_k": pd.NA,
                    **{
                        metric: ledgar_agentic_pq[metric].mean()
                        for metric in LEDGAR_MMR_METRICS
                    },
                }
            ]
        )
        ledgar_agentic_configuration["top_k"] = ledgar_agentic_configuration[
            "top_k"
        ].astype("Int64")
        display(Markdown("### LEDGAR Multi-Agent aggregated configuration"))
        display(ledgar_agentic_configuration)
        ledgar_all_32b_configurations = pd.concat(
            [ledgar_standard_32b, ledgar_agentic_configuration], ignore_index=True
        )
        if ledgar_all_32b_configurations[LEDGAR_MMR_METRICS].isna().any().any():
            problematic = ledgar_all_32b_configurations[
                ["method", "generator_model", "top_k"] + LEDGAR_MMR_METRICS
            ].loc[ledgar_all_32b_configurations[LEDGAR_MMR_METRICS].isna().any(axis=1)]
            display(Markdown("### Configurations with missing metrics"))
            display(problematic)
            raise ValueError(
                "At least one configuration contains missing evaluation metrics. MMR calculation stopped."
            )
        # Compute ranks across standard and agentic configurations, then select winners.
        ledgar_all_32b_ranked = calculate_metric_ranks(
            ledgar_all_32b_configurations, LEDGAR_MMR_METRICS, LEDGAR_SEMANTIC_METRICS
        )
        display(Markdown("## LEDGAR — all Qwen2.5-32B configurations included in MMR"))
        display(
            ledgar_all_32b_ranked[["method", "generator_model", "top_k", "MMR", "SM"]]
            .sort_values(["MMR", "SM"], ascending=[True, False])
            .reset_index(drop=True)
            .style.format({"MMR": "{:.2f}", "SM": "{:.4f}"}, na_rep="–")
        )
        ledgar_best_32b_per_method = select_best_per_method(ledgar_all_32b_ranked)
        ledgar_agentic_result_table = ledgar_format_result_table(ledgar_best_32b_per_method)
        display(Markdown("## LEDGAR — best Qwen2.5-32B configuration per method"))
        display(
            ledgar_agentic_result_table.style.format(
                {
                    "MMR": "{:.2f}",
                    "SM": "{:.4f}",
                    "R-1": "{:.4f}",
                    "R-2": "{:.4f}",
                    "R-L": "{:.4f}",
                    "BERT": "{:.4f}",
                    "SimCSE": "{:.4f}",
                    "ECS": "{:.4f}",
                    "MATCHA": "{:.4f}",
                },
                na_rep="–",
            )
        )
        print(
            "\nConfigurations included in MMR ranking:", len(ledgar_all_32b_configurations)
        )
        print("Standard Qwen2.5-32B configurations:", len(ledgar_standard_32b))
        print("Multi-Agent configurations: 1")

    if "hupd" in datasets:
        display_section("HUPD - standard-system generation quality")
        # 3.4 HUPD: reuse the loader with HUPD file mappings and repeat evaluation.

        HUPD_BASE_DIR = data_root / "hupd_dataset/results/metric_analysis"
        HUPD_EXPECTED_PER_QUESTION_FILES = {
            "per_question_metrics_hupd_dense_parent_child_all_generators.xlsx": "dense_parent_child",
            "per_question_metrics_hupd_sparse_parent_child_all_generators.xlsx": "sparse_parent_child",
            "per_question_metrics_hupd_sparse_parent_all_generators.xlsx": "sparse_parent",
            "per_question_metrics_hupd_sparse_parent_reversed_all_generators.xlsx": "sparse_parent",
            "per_question_metrics_hupd_sparse_parent_L_all_generators.xlsx": "sparse_parent_L",
            "per_question_metrics_hupd_sparse_parent_BM25L_all_generators.xlsx": "sparse_parent_L",
            "per_question_metrics_hupd_dense_parent_all_generators.xlsx": "dense_parent",
            "per_question_metrics_hupd_hybrid_parent_child_all_generators.xlsx": "hybrid_parent_child",
        }
        HUPD_SPARSE_METHOD_BY_FOLDER = {
            **{
                f"hupd_sparse_parent_qwen2.5-{size}b-instruct": "sparse_parent"
                for size in (7, 32, 72)
            },
            **{
                f"hupd_sparse_parent_bm25l_qwen2.5-{size}b-instruct": "sparse_parent_L"
                for size in (7, 32, 72)
            },
        }
        # Temporarily switch loader mappings; restore LEDGAR settings even on failure.
        _original_expected_files = EXPECTED_PER_QUESTION_FILES
        _original_sparse_method_folders = SPARSE_METHOD_BY_FOLDER
        try:
            EXPECTED_PER_QUESTION_FILES = HUPD_EXPECTED_PER_QUESTION_FILES
            SPARSE_METHOD_BY_FOLDER = HUPD_SPARSE_METHOD_BY_FOLDER
            hupd_df = load_metric_summaries(HUPD_BASE_DIR)
        finally:
            EXPECTED_PER_QUESTION_FILES = _original_expected_files
            SPARSE_METHOD_BY_FOLDER = _original_sparse_method_folders
        hupd_df = add_composite_columns(
            hupd_df,
            lexical_metrics=LEXICAL_METRICS,
            semantic_metrics=SEMANTIC_METRICS,
            primary_metrics=PRIMARY_METRICS,
        )
        display_section("HUPD data overview")
        hupd_overview_counts = {
            "datasets": ", ".join(sorted(hupd_df["dataset"].dropna().astype(str).unique())),
            "methods": ", ".join(sorted(hupd_df["method"].dropna().astype(str).unique())),
            "top_k_values": ", ".join(
                map(str, sorted(hupd_df["top_k"].dropna().astype(int).unique()))
            ),
            "generators": ", ".join(
                sorted(hupd_df["generator_model"].dropna().astype(str).unique())
            ),
        }
        display(pd.DataFrame([hupd_overview_counts]))
        display_section(
            "HUPD best individual setups",
            "Rows are ranked within the dataset by the average rank across the primary metrics. Lower mean metric rank is better.",
        )
        hupd_setup_cols = [
            "dataset",
            "method",
            "top_k",
            "generator_model",
            "overall_rank_within_dataset",
            "mean_metric_rank",
            "lexical_mean_f1",
            "semantic_mean",
            "primary_metric_mean",
        ] + PRIMARY_METRICS
        hupd_setup_table = (
            hupd_df[hupd_setup_cols]
            .sort_values(
                ["dataset", "overall_rank_within_dataset", "mean_metric_rank"],
                kind="stable",
            )
            .reset_index(drop=True)
        )
        display(
            style_best_second(
                hupd_setup_table.head(TOP_N_SETUPS_TO_SHOW),
                metric_cols=PRIMARY_METRICS
                + ["lexical_mean_f1", "semantic_mean", "primary_metric_mean"],
                lower_is_better_cols=["mean_metric_rank"],
            )
        )
        display_section(
            "HUPD overall method overview",
            "This table averages each retrieval method across available generators and top-k values.",
        )
        hupd_method_overview = rank_aggregate_table(
            hupd_df, ["dataset", "method"], aggregate_fn=aggregate_metrics
        )
        display(
            style_best_second(
                hupd_method_overview,
                metric_cols=PRIMARY_METRICS
                + ["lexical_mean_f1", "semantic_mean", "primary_metric_mean"],
                lower_is_better_cols=["mean_metric_rank"],
            )
        )
        hupd_retrieval_df = hupd_df.copy()
        display_section(
            "HUPD overall generator overview",
            "This table averages each generator across retrieval methods and top-k values.",
        )
        hupd_generator_overview = rank_aggregate_table(
            hupd_df, ["dataset", "generator_model"], aggregate_fn=aggregate_metrics
        )
        display(
            style_best_second(
                hupd_generator_overview,
                metric_cols=PRIMARY_METRICS
                + ["lexical_mean_f1", "semantic_mean", "primary_metric_mean"],
                lower_is_better_cols=["mean_metric_rank"],
            )
        )
        display_section("HUPD consistency of retrieval methods under matched constraints")
        hupd_method_long, hupd_method_consistency = consistency_analysis(
            hupd_retrieval_df,
            competitor_col="method",
            condition_cols=["dataset", "generator_model", "top_k"],
            metrics=PRIMARY_METRICS,
        )
        display(hupd_method_consistency)
        display_section("HUPD method consistency under each top-k constraint")
        hupd_method_by_topk_consistency = conditional_consistency_analysis(
            hupd_retrieval_df,
            competitor_col="method",
            fixed_col="top_k",
            condition_cols=["dataset", "generator_model"],
            metrics=PRIMARY_METRICS,
            consistency_fn=consistency_analysis,
        )
        display(hupd_method_by_topk_consistency)
        display_section("HUPD method consistency under each generator constraint")
        hupd_method_by_generator_consistency = conditional_consistency_analysis(
            hupd_retrieval_df,
            competitor_col="method",
            fixed_col="generator_model",
            condition_cols=["dataset", "top_k"],
            metrics=PRIMARY_METRICS,
            consistency_fn=consistency_analysis,
        )
        display(hupd_method_by_generator_consistency)
        display_section("HUPD detailed generator by method by top-k overview")
        hupd_generator_method_topk = rank_aggregate_table(
            hupd_df,
            ["dataset", "generator_model", "method", "top_k"],
            aggregate_fn=aggregate_metrics,
        )
        hupd_generator_method_topk = hupd_generator_method_topk.sort_values(
            ["dataset", "generator_model", "mean_metric_rank"], kind="stable"
        ).reset_index(drop=True)
        display(
            style_best_second(
                hupd_generator_method_topk.head(TOP_N_INTERACTION_ROWS),
                metric_cols=PRIMARY_METRICS
                + ["lexical_mean_f1", "semantic_mean", "primary_metric_mean"],
                lower_is_better_cols=["mean_metric_rank"],
            )
        )

        def plot_hupd_method_generator_heatmap(df, dataset, metric="mean_metric_rank"):
            plot_df = df[df["dataset"] == dataset].copy()
            if plot_df.empty:
                return

            def clean_hupd_generator_label(model_name):
                label = str(model_name).strip()
                label = re.sub("^Qwen/", "", label, flags=re.IGNORECASE)
                label = re.sub("^meta-llama/", "", label, flags=re.IGNORECASE)
                label = re.sub("[-_:]?q4[_-]?k[_-]?m\\b", "", label, flags=re.IGNORECASE)
                return label.rstrip("-_: ")

            def generator_parameter_size(label):
                normalized = str(label).lower()
                size_match = re.search("(?<!\\d)(7|32|72)\\s*b\\b", normalized)
                if size_match:
                    return int(size_match.group(1))
                return float("inf")

            plot_df["generator_display"] = plot_df["generator_model"].map(
                clean_hupd_generator_label
            )
            pivot = (
                plot_df.groupby(["method", "generator_display"], dropna=False)[metric]
                .mean()
                .unstack("generator_display")
            )
            ordered_generators = sorted(pivot.columns, key=generator_parameter_size)
            pivot = pivot.reindex(columns=ordered_generators)
            pivot.index = [abbreviate_method_value(method) for method in pivot.index]
            if pivot.empty:
                return
            plt.figure(
                figsize=(max(8, 0.8 * len(pivot.columns)), max(5, 0.45 * len(pivot.index)))
            )
            plt.imshow(pivot.values, aspect="auto", cmap="viridis_r")
            plt.colorbar(label="MMR (lower is better)")
            plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
            plt.yticks(range(len(pivot.index)), pivot.index)
            plt.title(f"MMR: method by generator ({dataset})")
            plt.xlabel("Generator")
            plt.ylabel("Method")
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    value = pivot.values[i, j]
                    if not pd.isna(value):
                        plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
            show_plot()

        for hupd_dataset in sorted(hupd_df["dataset"].dropna().astype(str).unique()):
            display_section(f"HUPD visualizations for dataset: {hupd_dataset}")
            plot_topk_trends_by_method(
                df=hupd_retrieval_df, dataset=hupd_dataset, metric="mean_metric_rank"
            )
            plot_hupd_method_generator_heatmap(
                df=hupd_df, dataset=hupd_dataset, metric="mean_metric_rank"
            )

        # Use the same F1 metrics and standalone ECS score as the Tender matrix.
        hupd_correlation_metrics = [
            metric for metric in GENERATION_METRICS if metric in hupd_df.columns
        ]
        hupd_spearman_corr = (
            hupd_df[hupd_correlation_metrics].corr(method="spearman").round(ROUND_DECIMALS)
        )
        display_section("HUPD Spearman correlation")
        display(hupd_spearman_corr)
        plot_correlation_heatmap(
            hupd_spearman_corr,
            "HUPD - Spearman correlation matrix across evaluation metrics",
        )

        # 3.5 HUPD: select the best individual configuration per method.

        display_section(
            "HUPD best individual system per method",
            "For each method, this table shows the best-performing individual setup, including its generator model and top-k. Selection is based on the lowest mean_metric_rank. Only F1 variants are shown for metrics with precision/recall/F1 splits.",
        )
        hupd_best_individual_system_per_method = (
            hupd_df[best_per_method_cols]
            .sort_values(
                ["dataset", "method", "mean_metric_rank", "primary_metric_mean"],
                ascending=[True, True, True, False],
                kind="stable",
            )
            .groupby(["dataset", "method"], as_index=False, dropna=False)
            .first()
            .sort_values(["dataset", "mean_metric_rank"], kind="stable")
            .reset_index(drop=True)
        )
        display(
            style_best_second(
                hupd_best_individual_system_per_method,
                metric_cols=BEST_PER_METHOD_METRICS
                + ["lexical_mean_f1", "semantic_mean", "primary_metric_mean"],
                lower_is_better_cols=["mean_metric_rank"],
            )
        )

        # 3.6 HUPD: rank Qwen2.5-32B standard and agentic configurations together.

        HUPD_AGENTIC_DIR = data_root / "hupd_dataset/multi_agent/multi_agent_results"
        HUPD_AGENTIC_PER_QUESTION_FILE = (
            HUPD_AGENTIC_DIR
            / "per_question_metrics_hupd_agentic_hupd_answers_all_generators.csv"
        )
        HUPD_MMR_METRICS = list(GENERATION_METRICS)
        HUPD_SEMANTIC_METRICS = list(SEMANTIC_SCORE_METRICS)
        HUPD_CONFIG_COLS = ["method", "generator_model", "top_k"]
        HUPD_METHOD_ABBREVIATIONS = {
            "dense_parent": "DP",
            "sparse_parent": "SP",
            "dense_parent_child": "DPC",
            "sparse_parent_child": "SPC",
            "hybrid_parent_child": "HPC",
            "agentic": "MA",
        }

        def hupd_format_result_table(frame):
            frame = frame.copy()
            frame["Method"] = (
                frame["method"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(HUPD_METHOD_ABBREVIATIONS)
                .fillna(frame["method"])
            )
            frame["Generator"] = "Qwen/Qwen2.5-32B-Instruct"
            frame["Top-k"] = frame.apply(
                lambda row: (
                    "–"
                    if row["Method"] == "MA" or pd.isna(row["top_k"])
                    else str(int(row["top_k"]))
                ),
                axis=1,
            )
            return (
                frame[
                    [
                        "Method",
                        "Generator",
                        "Top-k",
                        "MMR",
                        "SM",
                        "rouge1_f1",
                        "rouge2_f1",
                        "rougeL_f1",
                        "bertscore_f1",
                        "simcse_f1",
                        "ecs_bge_m3",
                        "matcha_f1",
                    ]
                ]
                .rename(
                    columns={
                        "rouge1_f1": "R-1",
                        "rouge2_f1": "R-2",
                        "rougeL_f1": "R-L",
                        "bertscore_f1": "BERT",
                        "simcse_f1": "SimCSE",
                        "ecs_bge_m3": "ECS",
                        "matcha_f1": "MATCHA",
                    }
                )
                .sort_values(["MMR", "SM"], ascending=[True, False], kind="stable")
                .reset_index(drop=True)
            )

        hupd_standard_32b = hupd_df[
            hupd_df["generator_model"].apply(matches_qwen25_32b)
        ].copy()
        if hupd_standard_32b.empty:
            raise ValueError("No Qwen2.5-32B configurations found in hupd_df.")
        required_standard_columns = HUPD_CONFIG_COLS + HUPD_MMR_METRICS
        missing_standard_columns = [
            column
            for column in required_standard_columns
            if column not in hupd_standard_32b.columns
        ]
        if missing_standard_columns:
            raise ValueError(
                f"Missing required columns in hupd_df: {missing_standard_columns}"
            )
        for metric in HUPD_MMR_METRICS:
            hupd_standard_32b[metric] = parse_numeric(hupd_standard_32b[metric])
        hupd_standard_32b["top_k"] = parse_numeric(hupd_standard_32b["top_k"]).astype(
            "Int64"
        )
        hupd_standard_32b = (
            hupd_standard_32b.groupby(HUPD_CONFIG_COLS, dropna=False)[HUPD_MMR_METRICS]
            .mean()
            .reset_index()
        )
        if not HUPD_AGENTIC_PER_QUESTION_FILE.exists():
            raise FileNotFoundError(
                f"HUPD agentic per-question file not found: {HUPD_AGENTIC_PER_QUESTION_FILE}"
            )
        hupd_agentic_pq = pd.read_csv(
            HUPD_AGENTIC_PER_QUESTION_FILE, sep=None, engine="python"
        )
        hupd_agentic_pq.columns = hupd_agentic_pq.columns.astype(str).str.strip()
        if "dataset" in hupd_agentic_pq.columns:
            hupd_agentic_pq = hupd_agentic_pq[
                hupd_agentic_pq["dataset"].astype(str).str.strip().str.lower().eq("hupd")
            ].copy()
        generator_column = next(
            (
                column
                for column in ["generator_model", "generator", "model"]
                if column in hupd_agentic_pq.columns
            ),
            None,
        )
        if generator_column is None:
            raise ValueError(
                f"Could not identify the generator column in the HUPD agentic CSV. Available columns: {list(hupd_agentic_pq.columns)}"
            )
        hupd_agentic_pq = hupd_agentic_pq[
            hupd_agentic_pq[generator_column].apply(matches_qwen25_32b)
        ].copy()
        if hupd_agentic_pq.empty:
            raise ValueError("No Qwen2.5-32B rows found in the HUPD agentic CSV.")
        missing_agentic_metrics = [
            metric for metric in HUPD_MMR_METRICS if metric not in hupd_agentic_pq.columns
        ]
        if missing_agentic_metrics:
            raise ValueError(
                f"Missing expected metrics in HUPD agentic CSV: {missing_agentic_metrics}"
            )
        for metric in HUPD_MMR_METRICS:
            hupd_agentic_pq[metric] = parse_numeric(hupd_agentic_pq[metric])
        hupd_agentic_configuration = pd.DataFrame(
            [
                {
                    "method": "agentic",
                    "generator_model": "Qwen/Qwen2.5-32B-Instruct",
                    "top_k": pd.NA,
                    **{
                        metric: hupd_agentic_pq[metric].mean()
                        for metric in HUPD_MMR_METRICS
                    },
                }
            ]
        )
        hupd_agentic_configuration["top_k"] = hupd_agentic_configuration["top_k"].astype(
            "Int64"
        )
        hupd_all_32b_configurations = pd.concat(
            [hupd_standard_32b, hupd_agentic_configuration], ignore_index=True
        )
        # Compute ranks across standard and agentic configurations, then select winners.
        hupd_all_32b_ranked = calculate_metric_ranks(
            hupd_all_32b_configurations, HUPD_MMR_METRICS, HUPD_SEMANTIC_METRICS
        )
        hupd_best_32b_per_method = select_best_per_method(hupd_all_32b_ranked)
        hupd_agentic_result_table = hupd_format_result_table(hupd_best_32b_per_method)
        display(Markdown("## HUPD — best Qwen2.5-32B configuration per method"))
        display(
            hupd_agentic_result_table.style.format(
                {
                    "MMR": "{:.2f}",
                    "SM": "{:.4f}",
                    "R-1": "{:.4f}",
                    "R-2": "{:.4f}",
                    "R-L": "{:.4f}",
                    "BERT": "{:.4f}",
                    "SimCSE": "{:.4f}",
                    "ECS": "{:.4f}",
                    "MATCHA": "{:.4f}",
                },
                na_rep="–",
            )
        )
        print("\nConfigurations included in MMR ranking:", len(hupd_all_32b_configurations))
        print("Standard Qwen2.5-32B configurations:", len(hupd_standard_32b))
        print("Multi-Agent configurations: 1")

    return locals()


# 4. Run the selected datasets.
def main(data_root=PROJECT_ROOT, datasets=DATASETS):
    """Evaluate selected datasets with separate state."""
    datasets = normalize_datasets(datasets)
    data_root = Path(data_root)
    results = {}
    if "tender" in datasets:
        results["tender"] = evaluate_tender(data_root)
    ledgar_hupd = tuple(name for name in datasets if name in ("ledgar", "hupd"))
    if ledgar_hupd:
        results["ledgar_hupd"] = evaluate_ledgar_hupd(data_root, datasets=ledgar_hupd)
    return results


if __name__ == "__main__":
    results = main(args.root, datasets=args.datasets)
