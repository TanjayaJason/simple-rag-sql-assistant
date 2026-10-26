import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
import chromadb
import ollama
from pypdf import PdfReader
from docx import Document

# -----------------------------
# CONFIG
# -----------------------------
EMBED_MODEL = "qwen3-embedding:0.6b"
DOCS_FOLDER = Path(__file__).resolve().parents[1] / "data" / "docs"
COLLECTION_NAME = "docs"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
FILENAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,123}\.(txt|pdf|docx)$", re.IGNORECASE)
WINDOWS_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}

@lru_cache(maxsize=1)
def get_client():
    return chromadb.PersistentClient(path=str(DOCS_FOLDER.parent / "chroma_db"))


def document_path(filename: str) -> Path:
    if (not isinstance(filename, str) or len(filename) > 128
            or not FILENAME_PATTERN.fullmatch(filename)
            or filename.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES):
        raise ValueError("Use a plain TXT, PDF, or DOCX filename (max 128 characters)")
    return DOCS_FOLDER / filename

# -----------------------------
# CHUNKING FUNCTION
# -----------------------------
def chunk_text(text, chunk_size=50, overlap=5):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

# -----------------------------
# TEXT EXTRACTION
# -----------------------------
def extract_text(filepath):
    suffix = str(filepath).lower()

    if suffix.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    elif suffix.endswith(".pdf"):
        reader = PdfReader(filepath)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        return text

    elif suffix.endswith(".docx"):
        doc = Document(filepath)
        text = ""

        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"

        return text

    return ""

# -----------------------------
# INDEX ONE FILE
# -----------------------------
def index_file(filepath, filename):
    collection = get_client().get_or_create_collection(name=COLLECTION_NAME)

    text = extract_text(filepath)

    if not text.strip():
        return 0

    chunks = chunk_text(text)

    all_chunks = []
    all_embeddings = []
    all_ids = []
    all_metadatas = []

    for i, chunk in enumerate(chunks):
        embedding = ollama.embed(
            model=EMBED_MODEL,
            input=chunk + "<|endoftext|>"
        )["embeddings"][0]

        all_chunks.append(chunk)
        all_embeddings.append(embedding)
        all_ids.append(f"{filename}_chunk_{i}")
        all_metadatas.append({
            "source": filename,
            "chunk": i
        })

    collection.add(
        documents=all_chunks,
        embeddings=all_embeddings,
        ids=all_ids,
        metadatas=all_metadatas
    )

    return len(chunks)

# -----------------------------
# REINDEX ALL FILES
# -----------------------------
def reindex_documents():
    try:
        get_client().delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    total_chunks = 0

    DOCS_FOLDER.mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(DOCS_FOLDER):
        filepath = DOCS_FOLDER / filename

        if filepath.is_file():
            try:
                document_path(filename)
            except ValueError:
                continue
            total_chunks += index_file(str(filepath), filename)

    return total_chunks

# -----------------------------
# DELETE FILE
# -----------------------------
def delete_file(filename: str) -> int:
    filepath = document_path(filename)
    collection = get_client().get_or_create_collection(name=COLLECTION_NAME)

    # Get all chunk IDs belonging to this file
    results = collection.get(where={"source": filename})

    if not results["ids"]:
        return 0

    collection.delete(ids=results["ids"])

    # Delete physical file from disk too
    if os.path.exists(filepath):
        os.remove(filepath)

    return len(results["ids"])


def list_documents() -> list[dict]:
    collection = get_client().get_or_create_collection(name=COLLECTION_NAME)
    counts = Counter(
        metadata["source"]
        for metadata in collection.get(include=["metadatas"])["metadatas"]
        if metadata and metadata.get("source")
    )
    documents = []
    for filename, count in sorted(counts.items()):
        try:
            filepath = document_path(filename)
        except ValueError:
            continue
        documents.append({
            "doc_id": filename,
            "chunks_indexed": count,
            "size_bytes": filepath.stat().st_size if filepath.is_file() else None,
        })
    return documents
