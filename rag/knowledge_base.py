
import os
import glob

from config import KNOWLEDGE_BASE_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_documents(kb_dir: str = KNOWLEDGE_BASE_DIR):
    docs = []
    for path in glob.glob(os.path.join(kb_dir, "*.txt")) + glob.glob(os.path.join(kb_dir, "*.md")):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        docs.append({"source": os.path.basename(path), "text": text})
    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


def build_chunks(kb_dir: str = KNOWLEDGE_BASE_DIR):
    """Returns list of {"source": filename, "text": chunk_text}."""
    all_chunks = []
    for doc in load_documents(kb_dir):
        for chunk in chunk_text(doc["text"]):
            if chunk.strip():
                all_chunks.append({"source": doc["source"], "text": chunk.strip()})
    return all_chunks
