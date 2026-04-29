import os
import chromadb
import ollama
from pypdf import PdfReader
from docx import Document

# -----------------------------
# CONFIG
# -----------------------------
EMBED_MODEL = "mxbai-embed-large"
DOCS_FOLDER = "./docs"
COLLECTION_NAME = "docs"

client = chromadb.PersistentClient(path="./chroma_db")

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

    if filepath.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    elif filepath.endswith(".pdf"):
        reader = PdfReader(filepath)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        return text

    elif filepath.endswith(".docx"):
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
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

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
            input=chunk
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
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    total_chunks = 0

    for filename in os.listdir(DOCS_FOLDER):
        filepath = os.path.join(DOCS_FOLDER, filename)

        if filename.endswith((".txt", ".pdf", ".docx")):
            total_chunks += index_file(filepath, filename)

    return total_chunks

# -----------------------------
# DELETE FILE
# -----------------------------
def delete_file(filename: str) -> int:
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Get all chunk IDs belonging to this file
    results = collection.get(where={"source": filename})

    if not results["ids"]:
        return 0

    collection.delete(ids=results["ids"])

    # Delete physical file from disk too
    filepath = os.path.join(DOCS_FOLDER, filename)
    if os.path.exists(filepath):
        os.remove(filepath)

    return len(results["ids"])