# === chunking.py ===
import re

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text_by_length(text: str, chunk_size=800, overlap=100) -> list:
    """
    Chunk text into overlapping pieces.
    """
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks

def prepare_chunks(all_texts: dict) -> list:
    """
    Returns list of dicts: [{"doc": filename, "content": chunk}, ...]
    """
    all_chunks = []
    for filename, text in all_texts.items():
        text = clean_text(text)
        chunks = chunk_text_by_length(text)
        for c in chunks:
            all_chunks.append({"doc": filename, "content": c})
    return all_chunks
