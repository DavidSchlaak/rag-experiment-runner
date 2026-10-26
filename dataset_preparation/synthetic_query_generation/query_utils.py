"""Shared text and validation helpers for synthetic query generation."""

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

# --- Project and prompt helpers ---


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


def build_user_prompt(chapter: str, theme: str, word_limit: int, gold_text: str) -> str:
    return f"""
chapter:
{chapter}

theme:
{theme}

word_limit:
{word_limit}

gold_text:
<<<GOLD_TEXT_START
{gold_text}
GOLD_TEXT_END>>>

Generate the synthetic requirement query now.
Return JSON only.
"""


def round_up_to_50(n: Any) -> int:
    try:
        n = int(float(n))
    except Exception:
        n = 0

    if n <= 0:
        return 50

    return int(math.ceil(n / 50.0) * 50)


def strip_chapter_heading(text: str, chapter: str) -> str:
    text = str(text or "").strip().replace("\r\n", "\n").replace("\r", "\n")
    chapter = str(chapter or "").strip()

    if chapter and text.startswith(chapter + "\n\n"):
        return text[len(chapter + "\n\n"):].strip()

    if chapter and text.startswith(chapter + "\n"):
        return text[len(chapter + "\n"):].strip()

    return text.strip()


def remove_hupd_markers(text: str) -> str:
    text = str(text or "")

    text = re.sub(
        r"\s*<SOH>\s*[^<>]{0,200}?\s*<EOH>\s*",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_gold_text_for_prompt(text: str, chapter: str) -> str:
    text = strip_chapter_heading(text, chapter)
    text = remove_hupd_markers(text)
    return text.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", str(text)))


def extract_json_object(raw: str) -> Dict[str, Any]:
    raw = str(raw or "").strip()

    raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")

    return json.loads(raw[start:end + 1])


def has_long_exact_overlap(
    requirements_text: str,
    gold_text: str,
    n: int = 7
) -> Tuple[bool, Optional[str]]:
    req_tokens = re.findall(r"[A-Za-z0-9]+", str(requirements_text).lower())
    gold_tokens = re.findall(r"[A-Za-z0-9]+", str(gold_text).lower())

    if len(req_tokens) < n or len(gold_tokens) < n:
        return False, None

    gold_ngrams = {
        " ".join(gold_tokens[i:i + n])
        for i in range(len(gold_tokens) - n + 1)
    }

    for i in range(len(req_tokens) - n + 1):
        phrase = " ".join(req_tokens[i:i + n])

        if phrase in gold_ngrams:
            return True, phrase

    return False, None


def sentence_count(text: str) -> int:
    text = str(text or "").strip()

    if not text:
        return 0

    return len(re.findall(r"[.!?]+(?:\s|$)", text))


def validate_model_output(
    obj: Dict[str, Any],
    chapter: str,
    word_limit: int,
    gold_text: str,
) -> Tuple[bool, List[str]]:
    errors = []

    expected_closing = f"Write a concise but comprehensive chapter of no more than {word_limit} words."

    if not isinstance(obj, dict):
        return False, ["Output is not a JSON object."]

    required_keys = {"requirements", "closing_sentence"}

    if set(obj.keys()) != required_keys:
        errors.append(f"JSON keys must be exactly {sorted(required_keys)}.")

    closing = str(obj.get("closing_sentence", "")).strip()
    requirements = obj.get("requirements", None)

    if closing != expected_closing:
        errors.append("closing_sentence does not match the required exact sentence.")

    if not isinstance(requirements, list):
        errors.append("requirements must be a list.")
        return False, errors

    if not (3 <= len(requirements) <= 6):
        errors.append(f"requirements must contain 3 to 6 items, got {len(requirements)}.")

    if not all(isinstance(r, str) for r in requirements):
        errors.append("All requirements must be strings.")
        return False, errors

    req_text = " ".join(r.strip() for r in requirements)
    req_words = word_count(req_text)

    if req_words > 96:
        errors.append(f"requirements exceed 96 words: {req_words} words.")

    forbidden_phrases = [
        "first paragraph",
        "second paragraph",
        "third paragraph",
        "use the wording",
        "use the following wording",
        "structure the section",
        "structure the chapter",
        "include a section",
        "include a paragraph",
        "draft in the same style",
        "gold text",
        "gold section",
        "gold patent",
        "claim 1",
        "claim 2",
        "claim 3",
        "numbered claim",
        "dependent claim",
        "dependent claims",
        "<soh>",
        "<eoh>",
    ]

    for i, req in enumerate(requirements, start=1):
        req = str(req).strip()
        req_lower = req.lower()

        if word_count(req) < 4:
            errors.append(f"Requirement {i} is too short.")

        if sentence_count(req) > 1:
            errors.append(f"Requirement {i} contains more than one sentence.")

        if any(p in req_lower for p in forbidden_phrases):
            errors.append(f"Requirement {i} contains a forbidden formal-writing instruction or marker.")

    overlap, phrase = has_long_exact_overlap(req_text, gold_text, n=7)

    if overlap:
        errors.append(f"Long exact overlap with gold text detected: '{phrase}'.")

    return len(errors) == 0, errors


def get_word_limit_for_row(row: pd.Series, cleaned_gold_text: str) -> int:
    if "length_words_raw" in row and not pd.isna(row["length_words_raw"]):
        return round_up_to_50(row["length_words_raw"])

    return round_up_to_50(word_count(cleaned_gold_text))
