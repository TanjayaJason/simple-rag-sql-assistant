from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from chroma import reindex_documents, index_file
from vanna_setup import vn
import os
import chromadb
import ollama
import logging

# -----------------------------
# CONFIG
# -----------------------------
EMBED_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3.2:3b"
DOCS_FOLDER = "./docs"
COLLECTION_NAME = "docs"
logger = logging.getLogger(__name__)
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

class TrainRequest(BaseModel):
    question: str
    sql: str

# -----------------------------
# Root Endpoint
# -----------------------------
@app.get("/")
def root():
    return {"message": "Hello There!"}

# -----------------------------
# Upload Endpoint
# -----------------------------
@app.post("/upload")
def upload_document(file: UploadFile = File(...)):

    if not file.filename.endswith((".txt", ".pdf", ".docx")):
        return {"error": "Only TXT, PDF, DOCX supported"}

    filepath = os.path.join(DOCS_FOLDER, file.filename)

    with open(filepath, "wb") as f:
        f.write(file.file.read())

    chunks_indexed = index_file(filepath, file.filename)

    return {
        "message": f"{file.filename} uploaded and indexed",
        "chunks_indexed": chunks_indexed
    }

# -----------------------------
# Reindex Endpoint
# -----------------------------
@app.post("/reindex")
def reindex():
    total_chunks = reindex_documents()

    return {
        "message": "Reindex complete",
        "chunks_indexed": total_chunks
    }

# -----------------------------
# Train Endpoint (Text2SQL)
# -----------------------------
@app.post("/train")
def train(request: TrainRequest):
    try:
        vn.train(question=request.question, sql=request.sql)
        return {
            "message": "Training successful",
            "question": request.question,
            "sql": request.sql
        }
    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# INTENT CLASSIFIER
# -----------------------------

SQL_KEYWORDS = [
    "how many", "total", "revenue", "count", "sum", "average",
    "top", "highest", "lowest", "most", "least", "list all",
    "show all", "which course", "which student", "enrolled",
    "price", "per course", "per student", "last month"
]

RAG_KEYWORDS = [
    # General
    "what is", "explain", "describe", "how does", "how do",
    "what are", "tell me about", "guide", "documentation",
    "definition", "difference between", "compare",
    # Your specific topics
    "python", "rag", "retrieval", "llm", "language model",
    "chromadb", "vector", "embedding", "vanna", "vanna ai",
    "fastapi", "postgresql", "postgres", "endpoint", "api",
    "chunk", "index", "reindex", "collection", "similarity"
]
def keyword_classify(question: str):
    q = question.lower()

    sql_score = sum(1 for kw in SQL_KEYWORDS if kw in q)
    rag_score = sum(1 for kw in RAG_KEYWORDS if kw in q)

    if sql_score > rag_score:
        return "SQL", "keyword"
    elif rag_score > sql_score:
        return "RAG", "keyword"
    else:
        return None, "ambiguous"  # fallback to LLM

def llm_classify(question: str):
    prompt = f"""
You are an intent classifier for a hybrid AI system that has two tools:

1. SQL: Use this for questions that require querying a database about:
   - Students (count, list, registration)
   - Courses (price, category, title, revenue)
   - Enrollments (purchases, counts, dates)
   - Aggregations: total, average, highest, lowest, how many, top N

2. RAG: Use this for questions that require searching technical documentation about:
   - Python, FastAPI, PostgreSQL
   - RAG, LLM, Language Models
   - ChromaDB, vector stores, embeddings
   - Vanna AI, Text2SQL
   - Concepts, explanations, how-to guides

Classify the following question as either SQL or RAG.

Reply ONLY in this exact format:
INTENT: <SQL or RAG>
CONFIDENCE: <high, medium, or low>
REASON: <one sentence explanation>

Question: {question}
"""
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    content = response["message"]["content"]

    intent = "RAG"
    confidence = "low"
    reason = "Could not determine intent"

    for line in content.splitlines():
        if line.startswith("INTENT:"):
            intent = line.split(":")[1].strip()
        elif line.startswith("CONFIDENCE:"):
            confidence = line.split(":")[1].strip()
        elif line.startswith("REASON:"):
            reason = line.split(":")[1].strip()

    return intent, confidence, reason

def classify_intent(question: str):
    intent, method = keyword_classify(question)

    if method == "ambiguous":
        intent, confidence, reason = llm_classify(question)
        return intent, f"llm ({confidence})", reason
    
    return intent, method, f"Matched by keyword heuristic"

# -----------------------------
# ORCHESTRATOR
# -----------------------------

def run_rag(question: str):
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    query_embedding = ollama.embed(
        model=EMBED_MODEL,
        input=question
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
{question}

Answer:
"""
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

def run_sql(question: str):
    sql = vn.generate_sql(question=question)
    df = vn.run_sql(sql=sql)
    data_as_text = df.to_string(index=False)

    prompt = f"""
You are a data analyst. Use the following database query result to answer the user's question.
If the result is empty, say "No data found."

Query Result:
{data_as_text}

Question:
{question}

Answer:
"""
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"], sql, df.to_dict(orient="records")

# -----------------------------
# Updated /ask Endpoint
# -----------------------------

@app.post("/ask")
def ask_question(request: QuestionRequest):
    intent, method, reason = classify_intent(request.question)

    if intent == "SQL":
        try:
            answer, sql, result = run_sql(request.question)
            return {
                "question": request.question,
                "tool_used": "SQL",
                "classifier": method,
                "confidence_note": reason,
                "sql": sql,
                "result": result,
                "answer": answer
            }
        except Exception as e:
            logger.error(f"SQL failed: {e}")
            return {"error": str(e), "tool_used": "SQL", "classifier": method}

    else:  # RAG
        answer = run_rag(request.question)
        return {
            "question": request.question,
            "tool_used": "RAG",
            "classifier": method,
            "confidence_note": reason,
            "answer": answer
        }