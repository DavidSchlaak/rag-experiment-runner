"""Generate synthetic queries for the LEDGAR test cases.

Run from any working directory with the project environment active:
    python dataset_preparation/synthetic_query_generation/ledgar_generate_synthetic_queries.py

Install project dependencies with: python -m pip install -r requirements.txt
"""

import os
import re
import json
import math
import time
import getpass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
from tqdm.auto import tqdm
from openai import OpenAI
try:
    from IPython.display import display
except ImportError:
    display = print



try:
    from .query_utils import (
        build_user_prompt, clean_gold_text_for_prompt, extract_json_object,
        find_project_root, get_word_limit_for_row, has_long_exact_overlap,
        remove_hupd_markers, round_up_to_50, sentence_count,
        strip_chapter_heading, validate_model_output, word_count,
    )
except ImportError:
    from query_utils import (
        build_user_prompt, clean_gold_text_for_prompt, extract_json_object,
        find_project_root, get_word_limit_for_row, has_long_exact_overlap,
        remove_hupd_markers, round_up_to_50, sentence_count,
        strip_chapter_heading, validate_model_output, word_count,
    )

PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)


BASE_DIR = PROJECT_ROOT / "ledgar_dataset"

TEST_CSV = BASE_DIR / "ledgar_test_cases.csv"

if not TEST_CSV.exists():
    raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")

BASE_URL = os.environ.get(
    "QWEN_BASE_URL",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

MODEL = os.environ.get(
    "QWEN_MODEL",
    "qwen3-235b-a22b-instruct",
)

API_KEY = os.environ.get("QWEN_API_KEY")

if not API_KEY:
    API_KEY = getpass.getpass("Enter QWEN_API_KEY, or press Enter for local vLLM: ")

if not API_KEY:
    API_KEY = "EMPTY"

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 350
RETRIES_PER_ROW = 3
SLEEP_BETWEEN_CALLS = 0.15

EXTRA_BODY = {
    "enable_thinking": False
}

USE_JSON_RESPONSE_FORMAT = True

MAX_ROWS_TO_PROCESS = None

SYSTEM_PROMPT = """
You are a synthetic requirement querie generator for a semi-synthetic RAG benchmark.

The benchmark task is:
A RAG system receives a clause type, a sparse set of drafting requirements, and a word limit.
The system must retrieve similar clauses from a knowledge base and generate a new clause close to a held-out gold clause.

You will receive:
- chapter: the clause type label
- theme: a broad inferred legal-domain theme
- word_limit: the rounded maximum word count
- gold_text: the held-out gold clause

Use only the gold_text to infer sparse drafting requirements.
Do not use external knowledge.
Do not infer facts that are not grounded in the gold_text.

The generated query must behave like instructions from a human legal requester:
- The requester knows the essential business/legal facts needed for the clause.
- The requester does not provide a full legal template.
- The requester does not specify formal clause structure, paragraph order, or boilerplate wording.
- The RAG system should infer drafting style, formal structure, and legal phrasing from retrieved similar clauses.

Good requirements:
- Preserve essential facts that cannot be inferred from retrieval alone.
- Include specific numbers, periods, fees, monetary thresholds, governing institutions, committees, courts, jurisdictions, named plans, named agreement types, party roles, or procedural triggers when they are central.
- State the factual skeleton from an outline perspective.
- Be sparse but sufficient to draft the clause.
- Do not list every sentence-level detail.

Bad requirements:
- Do not copy sentences from the gold clause.
- Do not paraphrase the whole clause as a compressed summary.
- Do not reproduce the original order of the clause.
- Do not instruct the model how to structure the clause formally.
- Do not say "include a first paragraph", "include a carve-out section", "use the following wording", or similar.
- Do not include legal boilerplate that could be learned from retrieved examples.
- Do not mention that these requirements were generated from a gold text.

Required output:
Return JSON only with exactly these keys:
{
  "requirements": [
    "...",
    "..."
  ],
  "closing_sentence": "..."
}

Rules:
1. requirements must contain 3 to 6 concise bullet-style requirement sentences.
2. Each requirement must be one sentence.
3. requirements must not exceed 96 words in total.
4. closing_sentence must be exactly: "Write a concise but comprehensive chapter of no more than [word_limit] words."
5. Do not copy any sequence of 7 or more words from the gold_text.

One-shot example:

Input:
chapter:
Interests

theme:
financing_credit_and_payment_terms

length_words_raw:
369

gold_text:
<<<GOLD_TEXT_START
Interest on Advances shall be payable in arrears (a) on the first day of each month with respect to Domestic Rate Loans, (b) with respect to LIBOR Rate Loans having an Interest Period of one, two or three months, at the end of the applicable Interest Period, and (c) with respect to LIBOR Rate Loans having an Interest Period of six months, at the end of each three month period during such Interest Period, provided that all accrued and unpaid interest shall be due and payable at the end of the Term. Interest charges shall be computed on the actual principal amount of Advances outstanding during the applicable month at a rate per annum equal to (i) with respect to Revolving Advances, the applicable Revolving Interest Rate, and (ii) with respect to Swing Loans, the Revolving Interest Rate for Domestic Rate Loans (as applicable, the “ Contract Rate ”). Except as expressly provided otherwise in this Agreement, any Obligations other than the Advances that are not paid when due shall accrue interest at the Revolving Interest Rate for Domestic Rate Loans, subject to the provision of the final sentence of this Section 3.1 regarding the Default Rate. Whenever, subsequent to the Closing Date, the Alternate Base Rate is increased or decreased, the applicable Contract Rate shall be similarly changed without notice or demand of any kind by an amount equal to the amount of such change in the Alternate Base Rate during the time such change or changes remain in effect. The LIBOR Rate shall be adjusted with respect to LIBOR Rate Loans without notice or demand of any kind on the effective date of any change in the Reserve Percentage as of such effective date. Upon and after the occurrence of an Event of Default, and during the continuation thereof, at the option of Agent or at the direction of Required Lenders (or, in the case of any Event of Default under Section 10.7 hereof, immediately and automatically upon the occurrence of any such Event of Default without the requirement of any affirmative action by any party), the Obligations shall bear interest at the applicable Contract Rate plus two percent (2%) per annum (as applicable, the “ Default Rate ”).
GOLD_TEXT_END>>>

Output:
{
  "requirements": [
    "Specify interest payment timing for Advances, distinguishing Domestic Rate Loans and LIBOR Rate Loans by Interest Period length.",
    "State that interest is computed on actual outstanding principal and distinguish Revolving Advances from Swing Loans.",
    "Address interest on overdue Obligations other than Advances, subject to contrary agreement provisions.",
    "Preserve automatic rate adjustments tied to Alternate Base Rate changes and Reserve Percentage changes.",
    "Specify the Default Rate trigger for continuing Events of Default, including Required Lenders or Agent direction and the automatic Section 10.7 trigger."
  ],
  "closing_sentence": "Write a concise but comprehensive chapter of no more than 400 words."
}
"""












def format_example_question(
    chapter: str,
    requirements: List[str],
    word_limit: int
) -> str:
    opening = f"Do write the clause {chapter}."
    closing = f"Write a concise but comprehensive chapter of no more than {word_limit} words."

    req_lines = "\n".join(f"- {r.strip()}" for r in requirements)

    return f"{opening}\n\nRequirements:\n{req_lines}\n\n{closing}"


def call_qwen(messages: List[Dict[str, str]]) -> str:
    kwargs = {
        "model": MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }

    if USE_JSON_RESPONSE_FORMAT:
        kwargs["response_format"] = {"type": "json_object"}

    if EXTRA_BODY is not None:
        kwargs["extra_body"] = EXTRA_BODY

    try:
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    except Exception:
        fallback_kwargs = {
            "model": MODEL,
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_OUTPUT_TOKENS,
        }

        response = client.chat.completions.create(**fallback_kwargs)
        return response.choices[0].message.content

def generate_query_for_row(row: pd.Series) -> Dict[str, Any]:
    chapter = str(row["chapter"]).strip()
    theme = str(row["theme"]).strip()
    block_id = str(row["block_id"]).strip()

    gold_text_full = str(row["Gold Text"]).strip()
    gold_text = clean_gold_text_for_prompt(gold_text_full, chapter)

    word_limit = get_word_limit_for_row(row, gold_text)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                chapter=chapter,
                theme=theme,
                word_limit=word_limit,
                gold_text=gold_text,
            ),
        },
    ]

    last_errors = []
    raw = ""

    for attempt in range(1, RETRIES_PER_ROW + 1):
        raw = call_qwen(messages)

        try:
            obj = extract_json_object(raw)

            valid, errors = validate_model_output(
                obj=obj,
                chapter=chapter,
                word_limit=word_limit,
                gold_text=gold_text,
            )

            if valid:
                requirements = [r.strip() for r in obj["requirements"]]

                example_question = format_example_question(
                    chapter=chapter,
                    requirements=requirements,
                    word_limit=word_limit,
                )

                return {
                    "block_id": block_id,
                    "chapter": chapter,
                    "theme": theme,
                    "success": True,
                    "attempts": attempt,
                    "word_limit_rounded": word_limit,
                    "requirements": requirements,
                    "example_question": example_question,
                    "raw_model_output": raw,
                    "errors": [],
                }

            last_errors = errors

        except Exception as e:
            last_errors = [str(e)]

        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": (
                "The previous output was invalid for these reasons:\n"
                + "\n".join(f"- {err}" for err in last_errors)
                + "\n\nRegenerate valid JSON only. "
                "Use exactly the keys requirements and closing_sentence. "
                "Use 3 to 6 sparse requirement sentences. "
                "Keep the exact closing sentence. "
                "Do not copy any sequence of 7 or more words from the gold text. "
                "Do not include HUPD markers or formal structure instructions."
            ),
        })

        time.sleep(SLEEP_BETWEEN_CALLS)

    return {
        "block_id": block_id,
        "chapter": chapter,
        "theme": theme,
        "success": False,
        "attempts": RETRIES_PER_ROW,
        "word_limit_rounded": word_limit,
        "requirements": [],
        "example_question": "",
        "raw_model_output": raw,
        "errors": last_errors,
    }

df = pd.read_csv(TEST_CSV)

required_columns = [
    "chapter",
    "theme",
    "Example Question",
    "Gold Text",
    "length_tokens",
    "length_words_raw",
    "block_id",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

df["block_id"] = df["block_id"].astype(str)
df["chapter"] = df["chapter"].astype(str)
df["theme"] = df["theme"].astype(str)


print("Loaded test cases:", len(df))
print("Input CSV:", TEST_CSV)
print("Output CSV will overwrite:", TEST_CSV)

print("\nChapter counts:")
print(df["chapter"].value_counts().sort_index())

print("\nTheme counts:")
print(df["theme"].value_counts().sort_index())

success_count = 0
failure_count = 0
processed_count = 0

process_df = df

if MAX_ROWS_TO_PROCESS is not None:
    process_df = df.head(MAX_ROWS_TO_PROCESS)

for idx, row in tqdm(
    process_df.iterrows(),
    total=len(process_df),
    desc="Generating synthetic LEDGAR queries"
):
    result = generate_query_for_row(row)

    df.loc[idx, "Example Question"] = result["example_question"]

    df.loc[idx, "word_limit_rounded"] = result["word_limit_rounded"]
    df.loc[idx, "synthetic_query_success"] = result["success"]

    if result["success"]:
        success_count += 1
    else:
        failure_count += 1

    processed_count += 1

    time.sleep(SLEEP_BETWEEN_CALLS)

empty_questions = (
    df["Example Question"]
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("")
)

if empty_questions.any():
    print("Warning: empty Example Question rows:", int(empty_questions.sum()))

marker_mask = (
    df["Example Question"]
    .fillna("")
    .astype(str)
    .str.contains(r"<SOH>|<EOH>", case=False, regex=True)
)

if marker_mask.any():
    examples = df.loc[
        marker_mask,
        ["block_id", "chapter", "theme", "Example Question"]
    ].head(10)

    raise AssertionError(
        "Some generated Example Question rows still contain <SOH> or <EOH> markers.\n"
        + examples.to_string(index=False)
    )

final_columns = [
    "chapter",
    "theme",
    "Example Question",
    "Gold Text",
    "length_tokens",
    "length_words_raw",
    "block_id",
    "word_limit_rounded",
    "synthetic_query_success",
]

available_final_columns = [c for c in final_columns if c in df.columns]

df = df[available_final_columns]

df.to_csv(TEST_CSV, index=False, encoding="utf-8")

print("\nDone.")
print("Updated CSV:", TEST_CSV)

print("\nGeneration summary:")
print("Rows processed in this run:", processed_count)
print("New successes in this run:", success_count)
print("New failures in this run:", failure_count)
print("Total rows:", len(df))

if "synthetic_query_success" in df.columns:
    print("Rows with synthetic success flag:", int(df["synthetic_query_success"].fillna(False).sum()))

print("\nPreview:")
display(df.head(10))
