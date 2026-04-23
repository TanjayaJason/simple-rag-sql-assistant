from fastapi import FastAPI
from pydantic import BaseModel
from chroma import reindex_documents
import chromadb
import ollama

# -----------------------------
# CONFIG
# -----------------------------
EMBED_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3.2:3b"
DOCS_FOLDER = "./docs"
COLLECTION_NAME = "docs"

app = FastAPI()

# -----------------------------
# CHROMA SETUP
# -----------------------------
client = chromadb.PersistentClient(path="./chroma_db")
# collection = client.get_or_create_collection(name=COLLECTION_NAME)

# -----------------------------
# MODEL
# -----------------------------
class QuestionRequest(BaseModel):
    question: str

# -----------------------------
# Root Endpoint
# -----------------------------
@app.get("/")
def root():
    return {"message": "Hello There!"}

# -----------------------------
# Index Endpoint
# -----------------------------
@app.post("/reindex")
def reindex():
    total_chunks = reindex_documents()

    return {
        "message": "Reindex complete",
        "chunks_indexed": total_chunks
    }

# -----------------------------
# Ask Endpoint
# -----------------------------
@app.post("/ask")
def ask_question(request: QuestionRequest):

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    query_embedding = ollama.embed(
        model=EMBED_MODEL,
        input=request.question
    )["embeddings"][0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    context = "\n".join(results["documents"][0])

    prompt = f"""
Use the following context to answer the user's question.
If the answer is not in the context, say "I don't know."
Do not infer, assume, or guess.

Context:
{context}

Question:
{request.question}

Answer:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response["message"]["content"]

    return {
        "question": request.question,
        "answer": answer
    }