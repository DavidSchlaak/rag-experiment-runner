"""Parse one tender concept into the chapter-block JSON used for the RAG pipeline.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError as exc:
    raise SystemExit("Missing dependency. Install it with: pip install pymupdf") from exc



# CONFIGURATION -- adapt for each concept

# Relative paths, independently of PyCharm's working folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "tender_dataset" / "concepts_and_RFPs_raw"
OUTPUT_DIR = PROJECT_ROOT / "tender_dataset" / "parsed_concepts"

# Specify ONE PDF filename from INPUT_DIR.
# The Aalen filename below is illustrative, it can be replaced with any other raw concept.
INPUT_FILENAME = "Konzept_Aalen.pdf"
TENDER_ID = "aalen_urbane_datenplattform_2022"
THEME = "Datenplattform"

# Set this to the first page containing the actual concept, AFTER any contents
# pages. Otherwise headings in the table of contents may be matched by mistake.
START_PAGE = 1  # PDF pages are numbered from 1, not 0
END_PAGE = None  # Optional inclusive last page

# Replace this list for EACH new tender.
# Names become the JSON's canonical 'chapter' values and structure-path labels.
# The list below reproduces all 45 chapters of the attached Aalen example.
CHAPTERS = [
    "0 Einleitung",
    "1 Leistungsangebot",
    "2 Leistungsbaustein 1: Urbane Datenplattform",
    "2.1 Plattform Infrastruktur",
    "2.1.1 Basis- und Plattform-Infrastruktur",
    "2.1.2 Skalierung und Performance der Plattform",
    "2.1.3 Monitoring",
    "2.2 IoT Hub und Datendrehscheibe",
    "2.3 Datenverarbeitung und Speicherung der Daten",
    "2.4 Open und private Data Portal",
    "2.4.1 Open Data",
    "2.4.2 Private Data",
    "2.5 Übersicht der vorhandenen Plattform Module",
    "3 Leistungsbaustein 2: LoRaWAN Sensorik",
    "4 Leistungsbaustein 3: Anbindung externer Datenquellen",
    "5 Leistungsbaustein 4: Verarbeitung und Speicherung der Sensordaten",
    "5.1 Datendrehscheibe - Decoder - Weitergabe",
    "5.2 Datenspeicherung",
    "5.3 Datenverarbeitung und -verknüpfung",
    "6 Leistungsbaustein 5: Visualisierung und Automatisierung",
    "6.1 Visualisierung der Testanwendungen im Smarten Cockpit",
    "6.2 Automatisierung und Initiierung bedarfsgerechter Aktionen",
    "6.3 Rechtemanagement für das Cockpit",
    "7 Leistungsbaustein 6: Betrieb der Plattform",
    "8 Leistungsbaustein 7: Begleitender Support und Projektmanagement",
    "8.1 Projektmanagement",
    "8.1.1 Übergreifende Projektsteuerung",
    "8.1.1.1 Mitwirkungspflichten",
    "8.1.2 Stakeholder-Management",
    "8.1.3 Projektorganisation & Rollen",
    "8.2 Support Konzept",
    "8.3 Sicherstellungen von Synergien zum Projektverlauf des „Showcases\" der",
    "8.4 Schulungskonzept & Schulung Mitarbeiter & Bereitstellung Dokumentation",
    "9 Zusatzbaustein 2: Synergien durch das Aufsetzen der Plattform im inter",
    "9.1 Preisminderung – Erklärungen",
    "9.2 Onboardingprozess von neuen Mandanten",
    "10 Referenz - Implementierung einer Mobilitätsplattform",
    "11 Referenz - Smart City App für die Stadt Northeim",
    "12 Referenz - LoRaWAN-Netzaufbau & Hauswächter App",
    "13 Referenz - Umweltdashboard",
    "14 Referenz - LoRaWAN Netzaufbau",
    "15 Vorstellung der Firmen und Teammitglieder",
    "15.1 Smart City Solutions GmbH",
    "15.2 NAS Smart Platforms GmbH",
    "15.3 KI-P GmbH",
]

# Optional overrides for PDF headings that are wrapped, truncated, or visually
# different from the canonical names. Patterns match the COMPLETE combined
# heading (up to MAX_HEADING_LINES), case-insensitively. Use raw regex strings.
# The Aalen JSON happens to contain two truncated canonical chapter titles.
CUSTOM_HEADING_REGEX = {
    "8.3 Sicherstellungen von Synergien zum Projektverlauf des „Showcases\" der":
        r"8\s*\.\s*3\.?\s+Sicherstellungen\s+von\s+Synergien\b.*",
    "9 Zusatzbaustein 2: Synergien durch das Aufsetzen der Plattform im inter":
        r"9\.?\s+Zusatzbaustein\s+2:\s+Synergien\s+durch\s+das\s+Aufsetzen\b.*",
}

# Optional full titles for structure_path when 'chapter' is intentionally short.
# In the Aalen example, these two JSON chapter labels were cut off in the source
# experiment although their hierarchy labels retained the complete titles.
STRUCTURE_TITLE_OVERRIDES = {
    "8.3": "Sicherstellungen von Synergien zum Projektverlauf des „Showcases\" der Stadt Aalen und ggfs. Abstimmung im interkommunalen Projektteam",
    "9": "Zusatzbaustein 2: Synergien durch das Aufsetzen der Plattform im interkommunalen Zusammenschluss zwischen Aalen und Heidenheim",
}

# A numbered chapter automatically inherits its parent.
# Example: {"Anhang: Rollenübersicht": "8"}
EXPLICIT_PARENTS = {}

MAX_HEADING_LINES = 3
FOOTER_REGEX = [
    r"^Seite\s+\d+(?:\s+(?:von|/)\s*\d+)?$",
    r"^Page\s+\d+(?:\s+(?:of|/)\s*\d+)?$",
    r"^\d+$",  # isolated page numbers
]



# PARSER -- normally no changes needed below this line
NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")
NUMERIC_CHAPTER = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.+)$")
LIST_MARKER = re.compile(r"^(?:[-•–▪§]|\d+[.)]|[a-zA-Z][)])(?:\s+|$)")
TOC_DOTS = re.compile(r"\.{3,}\s*\d+\s*$")


@dataclass(frozen=True)
class Line:
    page: int
    text: str


@dataclass(frozen=True)
class HeadingHit:
    chapter: str
    start: int
    span: int
    page: int


def clean_line(text: str) -> str:
    """Normalize PDF whitespace without destroying spaces between words."""
    text = text.replace("\u00ad", "").replace("\u00a0", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_match(text: str) -> str:
    text = clean_line(text)
    for old in ("–", "—", "−", "‑"):
        text = text.replace(old, "-")
    for old in ("„", "“", "”", "«", "»"):
        text = text.replace(old, '"')
    return text.replace("’", "'")


def read_pdf_lines(path: Path) -> list[Line]:
    filters = [re.compile(pattern, re.IGNORECASE) for pattern in FOOTER_REGEX]
    lines: list[Line] = []

    with fitz.open(path) as pdf:
        last_page = len(pdf) if END_PAGE is None else min(END_PAGE, len(pdf))
        if not 1 <= START_PAGE <= last_page:
            raise ValueError(f"Invalid START_PAGE={START_PAGE}; PDF has {len(pdf)} pages")

        for page_index in range(START_PAGE - 1, last_page):
            page_number = page_index + 1
            text = pdf[page_index].get_text("text", sort=True) or ""
            if not text.strip():
                print(f"WARNING: page {page_number} has no extractable text (OCR may be needed).")

            for raw_line in text.splitlines():
                line = clean_line(raw_line)
                if line and any(pattern.fullmatch(line) for pattern in filters):
                    continue
                if TOC_DOTS.search(line):
                    continue
                lines.append(Line(page_number, line))
            lines.append(Line(page_number, ""))  # separate page paragraphs

    return lines


def default_heading_regex(chapter: str) -> str:
    """Accept '2.1 Title' and '2.1. Title', including spaces around dots."""
    chapter = normalize_for_match(chapter)
    match = NUMERIC_CHAPTER.fullmatch(chapter)
    if match:
        number, title = match.groups()
        numeric_regex = r"\s*\.\s*".join(re.escape(part) for part in number.split("."))
        prefix = numeric_regex + r"\.?\s*"
    else:
        prefix, title = "", chapter

    words = title.split()
    title_regex = r"\s+".join(re.escape(word) for word in words)
    return rf"{prefix}{title_regex}\s*"


def compiled_heading_regexes() -> dict[str, re.Pattern[str]]:
    patterns = {}
    for chapter in CHAPTERS:
        source = CUSTOM_HEADING_REGEX.get(chapter, default_heading_regex(chapter))
        patterns[chapter] = re.compile(source, re.IGNORECASE)
    return patterns


def find_headings(lines: list[Line]) -> tuple[list[HeadingHit], list[str]]:
    """Find manual headings sequentially so body text/list items cannot reorder them."""
    patterns = compiled_heading_regexes()
    hits: list[HeadingHit] = []
    missing: list[str] = []
    cursor = 0

    for chapter in CHAPTERS:
        found = None
        for i in range(cursor, len(lines)):
            if not lines[i].text:
                continue

            combined = []
            for span in range(1, MAX_HEADING_LINES + 1):
                j = i + span - 1
                if j >= len(lines) or not lines[j].text or lines[j].page != lines[i].page:
                    break
                combined.append(lines[j].text)
                candidate = normalize_for_match(" ".join(combined))
                if patterns[chapter].fullmatch(candidate):
                    found = HeadingHit(chapter, i, span, lines[i].page)
                    break
            if found:
                break

        if found:
            hits.append(found)
            cursor = found.start + found.span
        else:
            missing.append(chapter)

    return hits, missing


def clean_body(lines: list[str]) -> str:
    """Join PDF line wrapping, preserving blank-line and list separation."""
    paragraphs: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            paragraphs.append(" ".join(current).strip())
            current.clear()

    for line in lines:
        line = clean_line(line)
        if not line:
            flush()
        elif LIST_MARKER.match(line):
            flush()
            paragraphs.append(line)
        elif current and current[-1].endswith("-") and line[0].islower():
            # Merge soft line-break hyphenation: 'Daten-' + 'plattform'.
            current[-1] = current[-1][:-1] + line
        else:
            current.append(line)

    flush()
    return "\n\n".join(paragraphs)


def build_structure_paths() -> dict[str, str]:
    """Use manually defined chapter titles for every level of the hierarchy."""
    by_number = {}
    for chapter in CHAPTERS:
        match = NUMERIC_CHAPTER.fullmatch(chapter)
        if match:
            number, title = match.groups()
            if number in by_number:
                raise ValueError(f"Duplicate chapter number: {number}")
            by_number[number] = STRUCTURE_TITLE_OVERRIDES.get(number, title)

    paths = {}
    for chapter in CHAPTERS:
        number_match = NUMERIC_CHAPTER.fullmatch(chapter)
        if number_match:
            number = number_match.group(1)
            parts = number.split(".")
            ancestors = [".".join(parts[:i]) for i in range(1, len(parts) + 1)]
            unknown = [ancestor for ancestor in ancestors if ancestor not in by_number]
            if unknown:
                raise ValueError(f"Missing parent chapter(s) for {chapter}: {unknown}")
            lineage = [f"{ancestor} {by_number[ancestor]}" for ancestor in ancestors]
        else:
            lineage = []
            parent = EXPLICIT_PARENTS.get(chapter)
            if parent:
                parent_chapter = next(
                    (name for name in CHAPTERS if NUMERIC_CHAPTER.fullmatch(name)
                     and NUMERIC_CHAPTER.fullmatch(name).group(1) == parent),
                    None,
                )
                if parent_chapter is None:
                    raise ValueError(f"Invalid explicit parent {parent!r} for {chapter!r}")
                if parent_chapter not in paths:
                    raise ValueError(f"Parent {parent_chapter!r} must precede {chapter!r}")
                lineage.extend(paths[parent_chapter].split(" > ")[1:])
            lineage.append(chapter)

        paths[chapter] = " > ".join(["CONCEPT Konzept", *lineage])
    return paths


def build_blocks(lines: list[Line], hits: list[HeadingHit]) -> list[dict]:
    paths = build_structure_paths()
    blocks = []

    for index, hit in enumerate(hits):
        next_start = hits[index + 1].start if index + 1 < len(hits) else len(lines)
        heading_text = " ".join(
            line.text for line in lines[hit.start:hit.start + hit.span]
        )
        body = clean_body([line.text for line in lines[hit.start + hit.span:next_start]])
        text = heading_text if not body else f"{heading_text}\n\n{body}"
        block_id = "a-" + str(uuid.uuid5(NAMESPACE, f"{TENDER_ID}:concept:{hit.chapter}:{index}"))

        blocks.append({
            "tender_id": TENDER_ID,
            "theme": THEME,
            "chapter": hit.chapter,
            "block_id": block_id,
            "text": text,
            "length_tokens": len(text.split()),  # Same whitespace approximation as sample JSON
            "structure_path": paths[hit.chapter],
        })

    return blocks


def main() -> None:
    if not INPUT_FILENAME or Path(INPUT_FILENAME).name != INPUT_FILENAME:
        raise ValueError("INPUT_FILENAME must be a single filename inside INPUT_DIR")
    if Path(INPUT_FILENAME).suffix.lower() != ".pdf":
        raise ValueError("This parser expects a PDF. Convert Word documents to PDF first.")
    if not CHAPTERS or len(CHAPTERS) != len(set(CHAPTERS)):
        raise ValueError("CHAPTERS must contain unique chapter names in document order")

    input_path = INPUT_DIR / INPUT_FILENAME
    if not input_path.is_file():
        raise FileNotFoundError(f"Input PDF not found: {input_path}")

    lines = read_pdf_lines(input_path)
    hits, missing = find_headings(lines)

    print(f"Matched {len(hits)}/{len(CHAPTERS)} chapters in {INPUT_FILENAME}.")
    if missing:
        print("Unmatched chapters (check START_PAGE, titles and CUSTOM_HEADING_REGEX):")
        for chapter in missing:
            print(f"  - {chapter}")
        raise ValueError("Parsing incomplete. No JSON written; correct chapter detection first.")

    blocks = build_blocks(lines, hits)
    for block in blocks:
        print(f"  {block['chapter']} -> {block['length_tokens']} words | {block['structure_path']}")
    short = [b["chapter"] for b in blocks if b["length_tokens"] <= 6]
    if short:
        print(f"NOTE: {len(short)} heading-only/very short blocks. Review whether this is expected.")

    data = {"blocks": blocks, "links": []}  # Exactly the sample JSON schema; no LV linking.
    output_path = OUTPUT_DIR / f"dataset_{TENDER_ID}.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(blocks)} blocks to: {output_path}")


if __name__ == "__main__":
    main()
