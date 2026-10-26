"""Generate synthetic queries for the HUPD test cases.

Run from any working directory with the project environment active:
    python dataset_preparation/synthetic_query_generation/hupd_generate_synthetic_queries.py

Install project dependencies with: python -m pip install -r requirements.txt
"""

import os
import re
import json
import math
import time
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

# Locate the local HUPD dataset inside this project.
BASE_DIR = PROJECT_ROOT / "hupd_dataset"

TEST_CSV = BASE_DIR / "hupd_test_cases.csv"

if not TEST_CSV.exists():
    raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")

# Connect to the running vLLM server.
# Set VLLM_BASE_URL, VLLM_API_KEY, or VLLM_MODEL when needed.
BASE_URL = os.environ.get(
    "VLLM_BASE_URL",
    "http://127.0.0.1:8000/v1",
).rstrip("/")

API_KEY = os.environ.get(
    "VLLM_API_KEY",
    "EMPTY",
)

MODEL = os.environ.get(
    "VLLM_MODEL",
    "",
).strip()

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

try:
    available_models = [
        model.id
        for model in client.models.list().data
    ]
except Exception as exc:
    raise ConnectionError(
        f"Could not connect to the vLLM server at {BASE_URL}. "
        "Make sure the server is running and reachable."
    ) from exc

if not available_models:
    raise RuntimeError(
        f"The vLLM server at {BASE_URL} returned no available models."
    )

if not MODEL:
    if len(available_models) == 1:
        MODEL = available_models[0]
    else:
        raise ValueError(
            "The vLLM server exposes multiple models. "
            "Set VLLM_MODEL to one of: "
            + ", ".join(available_models)
        )

if MODEL not in available_models:
    raise ValueError(
        f"Configured model '{MODEL}' is not exposed by the vLLM server. "
        f"Available models: {available_models}"
    )

print("vLLM server:", BASE_URL)
print("Model:", MODEL)

# Generation settings.
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 350
RETRIES_PER_ROW = 3
SLEEP_BETWEEN_CALLS = 0.15

EXTRA_BODY = {}

USE_JSON_RESPONSE_FORMAT = True

EXPECTED_HUPD_CHAPTERS = {"Background", "Summary", "Claims"}

MAX_ROWS_TO_PROCESS = None

# Instructions for generating synthetic requirements.
SYSTEM_PROMPT = """
You are a synthetic requirement query generator for a semi-synthetic patent-section RAG benchmark.

The benchmark task is:
A RAG system receives a patent chapter type, a sparse set of technical drafting requirements, and a word limit.
The system must retrieve similar patent sections from a knowledge base and generate a new patent section close to a held-out gold section.

You will receive:
- chapter: the patent section type, usually Background, Summary, or Claims
- theme: a broad technology-domain theme
- word_limit: the rounded maximum word count
- gold_text: the held-out gold patent section

Use only the gold_text to infer sparse drafting requirements.
Do not use external knowledge.
Do not infer facts that are not grounded in the gold_text.

The generated query must behave like instructions from a human patent requester:
- The requester knows the essential technical facts needed for the patent chapter.
- The requester does not provide a full patent template.
- The requester does not specify formal section structure, paragraph order, claim numbering, or boilerplate wording.
- The RAG system should infer drafting style, formal structure, and patent phrasing from retrieved similar sections.

Good requirements:
- Preserve essential technical facts that cannot be inferred from retrieval alone.
- Preserve the main technical setting, technical problem, system components, method steps, data flows, matching logic, user interactions, conditions, or technical effects when they are central.
- For Background, preserve the technological context, existing approaches, their limitation, and the resulting technical need.
- For Summary, preserve the core disclosed solution, principal components, operations, and technical effect.
- For Claims, preserve the core inventive subject matter and essential limitations without reproducing claim numbering or full claim structure.
- State the factual skeleton from an outline perspective.
- Be sparse but sufficient to draft the patent chapter.
- Do not list every sentence-level detail.

Bad requirements:
- Do not copy sentences from the gold section.
- Do not paraphrase the whole section as a compressed summary.
- Do not reproduce the original order of the section.
- Do not instruct the model how to structure the section formally.
- Do not say "include a first paragraph", "include claim 1", "include dependent claims", "use the following wording", or similar.
- Do not include patent boilerplate that could be learned from retrieved examples.
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
Background

theme:
business_and_administrative_data_processing

word_limit:
350

gold_text:
<<<GOLD_TEXT_START
With the advent of the internet, a plethora of social networking sites and ecommerce sites have been established. By way of example, Facebook, Amazon and eBay are internet-based-networking and ecommerce sites that allow users to create and participate in social and ecommerce networks. Such social networking and ecommerce sites share several common characteristics in that they allow users to create profiles, find other users or products and share information to users about themselves or products, goods and services. In finding products, goods or services, existing sites allow users to search various fields of categories. For example, Amazon allows users to search through a database to identify products in a specific category like office furniture or bedroom furniture. Alternatively, such sites may display multiple sub-categories within a category. In providing such options, the site is querying or polling the user to match the user with similar products, goods or services. While the sites may offer options of categories to choose from to the user, such sites do not provide a progressive poll query that, in real-time, matches the user with products, goods or services based upon all of the polls taken. In other words, existing sites maintain a database with user profiles. When a new user registers on an existing site, the site may attempt to match the user profile with existing products as stored on a database based on searches. While such existing sites match a user profile against predefined categories, they do not progressively poll and match the user against all other products, goods and services after each query item. Thus, a continuing need exists for a poll-based net-working system that includes matching logic that is operable for matching, in real-time, a user with other products, goods and services based upon all of the polls taken by the user about opinions of products, goods and services.
GOLD_TEXT_END>>>

Output:
{
  "requirements": [
    "Describe internet-based social networking and ecommerce systems that let users create profiles, search products, and share user or product information.",
    "Preserve representative examples such as Facebook, Amazon, and eBay as existing online networking or commerce platforms.",
    "Explain that existing systems match users to products through predefined categories or subcategories.",
    "State the limitation that matching is not progressively updated in real time after each user query.",
    "Identify the need for a poll-based networking system with matching logic based on collected user opinions about products, goods, and services."
  ],
  "closing_sentence": "Write a concise but comprehensive chapter of no more than 350 words."
}
"""

# Build the model input.

# Clean and validate model inputs and outputs.









def format_example_question(
    chapter: str,
    requirements: List[str],
    word_limit: int
) -> str:
    opening = f"Do write the chapter {chapter}."
    closing = f"Write a concise but comprehensive chapter of no more than {word_limit} words."

    req_lines = "\n".join(f"- {r.strip()}" for r in requirements)

    return f"{opening}\n\nRequirements:\n{req_lines}\n\n{closing}"


# Send a request to the vLLM server.
def call_model(messages: List[Dict[str, str]]) -> str:
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

# Generate one query and retry invalid responses.
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
        raw = call_model(messages)

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

# Load and check the test cases.
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

unexpected_chapters = sorted(set(df["chapter"].unique()) - EXPECTED_HUPD_CHAPTERS)

if unexpected_chapters:
    print("Warning: unexpected HUPD chapters found:", unexpected_chapters)
    print("The code will still process them, but the prompt is optimized for Background, Summary, and Claims.")

print("Loaded test cases:", len(df))
print("Input CSV:", TEST_CSV)
print("Output CSV will overwrite:", TEST_CSV)

print("\nChapter counts:")
print(df["chapter"].value_counts().sort_index())

print("\nTheme counts:")
print(df["theme"].value_counts().sort_index())

# Generate the synthetic queries.
success_count = 0
failure_count = 0
processed_count = 0

process_df = df

if MAX_ROWS_TO_PROCESS is not None:
    process_df = df.head(MAX_ROWS_TO_PROCESS)

for idx, row in tqdm(
    process_df.iterrows(),
    total=len(process_df),
    desc="Generating synthetic HUPD queries"
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

# Validate and save the updated CSV.
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
