"""Create parent and child JSONL files and their embeddings for the datasets.
"""

from pathlib import Path

# --- Parent–child chunking and dense embeddings ---
# Each dataset keeps its original filenames, metadata, and record order. Child chunks are built from parent blocks, and embedding rows match the JSONL records.
#


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

import json
import re
import uuid
from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Resolve all dataset paths from the project root.
DATA_ROOT = PROJECT_ROOT
TENDER_DIR = DATA_ROOT / "tender_dataset"
LEDGAR_DIR = DATA_ROOT / "ledgar_dataset"
HUPD_DIR = DATA_ROOT / "hupd_dataset"
INPUT_PATTERN = "dataset_*.json"
EMBEDDING_MODEL = "BAAI/bge-m3"

PARENT_TOKEN_THRESHOLD = 192
TARGET_CHILD_TOKENS = 128
MIN_CHILD_TOKENS = 64
MAX_CHILD_TOKENS = 256

bge_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

# --- 2. Shared chunking ---
#
# Token counts exclude special tokens. Parents of at most 192 tokens remain one child; longer parents use sentence-based chunks targeting 128 tokens. Sentences remain intact. A final residual below 64 tokens is merged into the previous chunk. Both long sentences and this residual merge can exceed 256 tokens; the limit is not a strict cap.

def count_bge_tokens(text: str) -> int:
    return len(bge_tokenizer.encode(str(text or ''), add_special_tokens=False))

def split_sentences(text: str) -> List[str]:
    text = re.sub('\\s+', ' ', str(text or '')).strip()
    if not text:
        return []
    sentences = re.split('(?<=[.!?])\\s+(?=[A-ZÄÖÜ0-9])', text)
    return [s.strip() for s in sentences if s.strip()]

def create_child_chunks(text: str) -> List[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []
    chunks = []
    current = []
    current_tokens = 0
    # Accumulate whole sentences using their individual token counts.
    for sent in sentences:
        sent_tokens = count_bge_tokens(sent)
        if not current and sent_tokens > MAX_CHILD_TOKENS:
            chunks.append(sent)
            continue
        candidate_tokens = current_tokens + sent_tokens
        if current and candidate_tokens > MAX_CHILD_TOKENS:
            chunks.append(' '.join(current).strip())
            current = [sent]
            current_tokens = sent_tokens
            continue
        current.append(sent)
        current_tokens = candidate_tokens
        if current_tokens >= TARGET_CHILD_TOKENS:
            chunks.append(' '.join(current).strip())
            current = []
            current_tokens = 0
    # Preserve the original residual merge, even if it exceeds the maximum.
    if current:
        residual = ' '.join(current).strip()
        if chunks and count_bge_tokens(residual) < MIN_CHILD_TOKENS:
            chunks[-1] = (chunks[-1] + ' ' + residual).strip()
        else:
            chunks.append(residual)
    return chunks

# --- 3. Record loading and retrieval text ---
#
# Tender text includes theme, chapter and structure path. LEDGAR and HUPD text includes theme and chapter.

def save_jsonl(records, path):
    """Write records in their existing order."""
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path, skip_blank_lines=True):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if not skip_blank_lines or line.strip()]

def is_concept_block(block: Dict) -> bool:
    path = (block.get('structure_path') or '').upper()
    return path.startswith('CONCEPT')

def load_tender_blocks(base_dir) -> Iterable[Dict]:
    for path in base_dir.rglob(INPUT_PATTERN):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            for block in data:
                block['_source'] = str(path)
                yield block
        elif isinstance(data, dict) and 'blocks' in data:
            for block in data['blocks']:
                block['_source'] = str(path)
                yield block

def build_retrieval_text(record, text, include_path=False):
    """Match the original metadata prefix exactly."""
    prefix = f"Theme: {record.get('theme', '')}\nChapter: {record.get('chapter', '')}\n"
    if include_path:
        prefix += f"Path: {record.get('structure_path', '')}\n"
    return (prefix + "\n" + text).strip()


def embed_texts(texts, cast_float32=True):
    """Encode one ordered collection with the original model settings."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32") if cast_float32 else embeddings

# --- 4. Dataset-specific records ---

def derive_tender_children(blocks):
    """Keep concept blocks and retain tender metadata in both record types."""
    children = []
    parents = []
    for block in blocks:
        if not is_concept_block(block):
            continue
        parent_id = str(block.get('block_id') or uuid.uuid4()).strip()
        text = str(block.get('text', '')).strip()
        if not text:
            continue
        parent_tokens_bge_m3 = count_bge_tokens(text)
        parents.append({
            'parent_id': parent_id,
            'tender_id': block.get('tender_id'),
            'theme': block.get('theme'),
            'chapter': block.get('chapter'),
            'structure_path': block.get('structure_path'),
            'text': text,
            'length_tokens_bge_m3': parent_tokens_bge_m3,
        })
        if parent_tokens_bge_m3 > PARENT_TOKEN_THRESHOLD:
            child_texts = create_child_chunks(text)
        else:
            child_texts = [text]
        for i, child_text in enumerate(child_texts):
            children.append({
                'child_id': f'{parent_id}_c{i}',
                'parent_id': parent_id,
                'tender_id': block.get('tender_id'),
                'theme': block.get('theme'),
                'chapter': block.get('chapter'),
                'structure_path': block.get('structure_path'),
                'child_index': i,
                'child_text': child_text,
                'length_tokens_bge_m3': count_bge_tokens(child_text),
            })
    return (children, parents)

def derive_dataset_children(blocks: List[Dict]):
    """Build LEDGAR or HUPD children from their prepared parent JSONL records."""
    children = []
    for block in blocks:
        parent_id = str(block.get('parent_id') or block.get('block_id', '')).strip()
        text = str(block.get('text', '')).strip()
        if not parent_id:
            continue
        if not text:
            continue
        parent_tokens_bge_m3 = count_bge_tokens(text)
        if parent_tokens_bge_m3 > PARENT_TOKEN_THRESHOLD:
            child_texts = create_child_chunks(text)
        else:
            child_texts = [text]
        for i, child_text in enumerate(child_texts):
            retrieval_text = build_retrieval_text(block, child_text)
            child_record = {
                'child_id': f'{parent_id}_c{i}',
                'parent_id': parent_id,
                'theme': block.get('theme'),
                'chapter': block.get('chapter'),
                'child_index': i,
                'child_text': child_text,
                'retrieval_text': retrieval_text,
                'length_tokens_bge_m3': count_bge_tokens(child_text),
            }
            children.append(child_record)
    return children

# --- 5. Embedding and export ---
#
# Each encode call retains batch size 32, normalized embeddings and NumPy output.

def print_token_stats(label, records):
    lengths = [record["length_tokens_bge_m3"] for record in records]
    print(label)
    print("Min:", min(lengths), "Max:", max(lengths),
          "Mean:", round(sum(lengths) / len(lengths), 2))


def run_child_pipeline(base_dir, dataset):
    """Build and save one dataset without changing record order."""
    output_dir = base_dir / "rag_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    is_tender = dataset == "tender"
    prefix = "" if is_tender else dataset + "_"

    if is_tender:
        blocks = list(load_tender_blocks(base_dir))
        children, parents = derive_tender_children(blocks)
    else:
        parent_jsonl = output_dir / f"{prefix}parents.jsonl"
        if not parent_jsonl.exists():
            raise FileNotFoundError(
                f"Parent JSONL not found: {parent_jsonl}. Run dataset preparation first."
            )
        blocks = load_jsonl(parent_jsonl)
        children = derive_dataset_children(blocks)
        parents = blocks
        if not parents:
            raise ValueError("No parent records created. Check input JSON and block_id/text fields.")
        if not children:
            raise ValueError("No child records created. Check input JSON and text fields.")
        print_token_stats("Parent token stats BGE-M3:", parents)

    print(dataset.upper(), "loaded blocks:", len(blocks))
    print("Parent blocks:", len(parents), "Child chunks:", len(children))
    if children:
        print_token_stats("Child token stats BGE-M3:", children)

    texts = ([build_retrieval_text(child, child.get("child_text", ""), include_path=True)
              for child in children] if is_tender
             else [child["retrieval_text"] for child in children])
    embeddings = embed_texts(texts)

    # Tender creates both records from parsed_concepts. LEDGAR and HUPD
    # already receive their parent JSONL from dataset preparation.
    if is_tender:
        save_jsonl(children, output_dir / "children.jsonl")
        save_jsonl(parents, output_dir / "parents.jsonl")
    else:
        save_jsonl(children, output_dir / f"{prefix}children.jsonl")
    np.save(output_dir / f"{prefix}child_embeddings.npy", embeddings)
    print("Artifacts saved to:", output_dir, "Embedding matrix:", embeddings.shape)
    if embeddings.shape[0] != len(children):
        raise ValueError(f"Embedding count mismatch: {embeddings.shape[0]} embeddings for {len(children)} children.")


def run_parent_pipeline(base_dir, dataset):
    """Embed complete parents in the same order as the saved JSONL."""
    output_dir = base_dir / "rag_artifacts"
    is_tender = dataset == "tender"
    prefix = "" if is_tender else dataset + "_"
    parents = load_jsonl(output_dir / f"{prefix}parents.jsonl", skip_blank_lines=not is_tender)
    print("Parents loaded:", len(parents))
    if not is_tender and not parents:
        raise ValueError(f"No parents loaded. Check {prefix}parents.jsonl.")
    texts = [build_retrieval_text(parent, str(parent.get("text", "")), include_path=is_tender)
             for parent in parents]
    # Tender parents retain the encoder's dtype; the other arrays use float32.
    embeddings = embed_texts(texts, cast_float32=not is_tender)
    output_path = output_dir / f"{prefix}parent_embeddings.npy"
    np.save(output_path, embeddings)
    print("Saved:", output_path, "Embedding matrix:", embeddings.shape)
    if not is_tender and embeddings.shape[0] != len(parents):
        raise ValueError(f"Embedding count mismatch: {embeddings.shape[0]} embeddings for {len(parents)} parents.")

# --- 7. Tender artifacts ---

run_child_pipeline(TENDER_DIR, "tender")
run_parent_pipeline(TENDER_DIR, "tender")

# --- Tender diagnostics ---
#
# Preview helpers are defined before use. Norms are read from the actual child and parent files; the original parent check reused the child array. 

def inspect_tender_artifacts():
    """Inspect the saved child and parent arrays separately."""
    output_dir = TENDER_DIR / "rag_artifacts"
    parents = load_jsonl(output_dir / "parents.jsonl")
    for parent in parents[:3]:
        print("=" * 80)
        print(build_retrieval_text(parent, str(parent.get("text", "")), include_path=True)[:1500])
    for filename in ("child_embeddings.npy", "parent_embeddings.npy"):
        embeddings = np.load(output_dir / filename)
        norms = np.linalg.norm(embeddings, axis=1)
        print(filename, "Shape:", embeddings.shape)
        print("Norm min:", norms.min(), "Max:", norms.max(), "Mean:", norms.mean())

def check_tender_ids():
    json_files = list(TENDER_DIR.rglob(INPUT_PATTERN))
    print('Number of dataset JSON files found:', len(json_files))
    concept_block_ids = defaultdict(list)

    def extract_blocks(data):
        """
    Handles JSON files that are either:
    - a list of blocks
    - a dict containing a list under common keys
    """
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ['blocks', 'data', 'items', 'documents']:
                if key in data and isinstance(data[key], list):
                    return data[key]
        return []
    for json_path in json_files:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        blocks = extract_blocks(data)
        for i, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            block_id = block.get('block_id')
            if isinstance(block_id, str) and block_id.startswith('a-'):
                concept_block_ids[block_id].append({
                    'file': str(json_path),
                    'index': i,
                    'tender_id': block.get('tender_id'),
                    'theme': block.get('theme'),
                    'chapter': block.get('chapter'),
                })
    duplicates = {block_id: occurrences for block_id, occurrences in concept_block_ids.items() if len(occurrences) > 1}
    print('\nConcept block ID check')
    print('----------------------')
    print('Total concept blocks:', sum((len(v) for v in concept_block_ids.values())))
    print('Unique concept block IDs:', len(concept_block_ids))
    print('Duplicate concept block IDs:', len(duplicates))
    if not duplicates:
        print('\n All concept block_ids are unique.')
    else:
        print('\n Duplicate concept block_ids found:\n')
        for block_id, occurrences in duplicates.items():
            print(f'Duplicate block_id: {block_id}')
            for occ in occurrences:
                print(f"  File: {occ['file']}")
                print(f"  Index: {occ['index']}")
                print(f"  Tender ID: {occ['tender_id']}")
                print(f"  Theme: {occ['theme']}")
                print(f"  Chapter: {occ['chapter']}")
                print()

inspect_tender_artifacts()
check_tender_ids()

# --- 8. LEDGAR artifacts ---

run_child_pipeline(LEDGAR_DIR, "ledgar")
run_parent_pipeline(LEDGAR_DIR, "ledgar")

# --- 9. HUPD artifacts ---

run_child_pipeline(HUPD_DIR, "hupd")
run_parent_pipeline(HUPD_DIR, "hupd")

