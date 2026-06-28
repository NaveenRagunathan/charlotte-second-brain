"""
Split podcast transcripts into ~400-token semantic chunks.
- Uses paragraph breaks where available
- Falls back to sentence-boundary splitting for dense text
"""
import os
import glob
import re

TRANSCRIPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "podcast-transcript")
CHUNK_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "podcast-chunks")
os.makedirs(CHUNK_DIR, exist_ok=True)

TARGET_TOKENS = 400


def token_count(text):
    return len(text.split())


def split_sentences(text):
    """Split text into sentences on period-space-capital or em-dash boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"])|(?<= — )', text)
    return [s.strip() for s in sentences if s.strip() and token_count(s.strip()) > 5]


def split_paragraphs(text):
    """Split by double newlines first, then further split long paragraphs."""
    paras = re.split(r'\n\n+', text)
    result = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if token_count(p) > TARGET_TOKENS * 2:
            sentences = split_sentences(p)
            result.extend(sentences)
        else:
            result.append(p)
    return result


def chunk_items(items, target=TARGET_TOKENS):
    """Merge items (paragraphs/sentences) into chunks of roughly target tokens."""
    chunks = []
    current = []
    current_tokens = 0

    for item in items:
        item_tokens = token_count(item)

        if not current:
            current.append(item)
            current_tokens = item_tokens
        elif current_tokens + item_tokens <= target * 1.5:
            current.append(item)
            current_tokens += item_tokens
        else:
            chunks.append(" ".join(current))
            current = [item]
            current_tokens = item_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks


def process_file(filepath):
    with open(filepath) as f:
        text = f.read().strip()

    base = os.path.splitext(os.path.basename(filepath))[0]
    items = split_paragraphs(text)
    chunks = chunk_items(items)
    paths = []
    for i, chunk in enumerate(chunks, 1):
        filename = f"{base}_chunk_{i:02d}.md"
        outpath = os.path.join(CHUNK_DIR, filename)
        with open(outpath, "w") as f:
            f.write(chunk)
        paths.append((filename, token_count(chunk)))
    return paths


if __name__ == "__main__":
    old_chunks = glob.glob(os.path.join(CHUNK_DIR, "*.md"))
    for f in old_chunks:
        os.remove(f)

    files = sorted(glob.glob(os.path.join(TRANSCRIPT_DIR, "*.md")))
    print(f"Found {len(files)} transcripts to chunk\n")
    total_before = 0
    total_after = 0
    for f in files:
        with open(f) as fh:
            total_before += token_count(fh.read())
        chunks = process_file(f)
        total_chunk_tokens = sum(t for _, t in chunks)
        total_after += total_chunk_tokens
        print(f"  {os.path.basename(f)} → {len(chunks)} chunks")
        for name, tokens in chunks:
            pct = tokens / TARGET_TOKENS * 100
            bar = "█" * int(pct / 5)
            print(f"    {name} ({tokens} tok, {pct:.0f}%) {bar}")

    chunk_count = len(glob.glob(os.path.join(CHUNK_DIR, "*.md")))
    print(f"\nBefore: {total_before} tokens in {len(files)} files")
    print(f"After:  {total_after} tokens in {chunk_count} chunks")
    print(f"Done — written to {CHUNK_DIR}/")
