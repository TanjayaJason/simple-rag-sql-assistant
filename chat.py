import os
import chromadb
import ollama
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from chroma import reindex_documents, index_file, delete_file
from vanna_setup import vn

# -----------------------------
# CONFIG
# -----------------------------
load_dotenv()  # <-- moved to top

EMBED_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3.2:3b"
DOCS_FOLDER = "./docs"
COLLECTION_NAME = "docs"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

# -----------------------------
# CHROMA AND DATABASE SETUP
# -----------------------------
client = chromadb.PersistentClient(path="./chroma_db")

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )

# -----------------------------
# MODELS
# -----------------------------
class QuestionRequest(BaseModel):
    question: str

class TrainRequest(BaseModel):
    question: str
    sql: str

# -----------------------------
# HISTORY HELPERS
# -----------------------------
def save_history(question: str, answer: str, tool_used: str, sql_generated: str = None):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO conversation_history (question, answer, tool_used, sql_generated)
            VALUES (%s, %s, %s, %s)
            """,
            (question, answer, tool_used, sql_generated)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to save history: {e}")

def get_history(limit: int = 10):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT question, answer, tool_used, sql_generated, created_at
            FROM conversation_history
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        return []

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
    # Specific topics
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
        return None, "ambiguous"

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

    return intent, method, "Matched by keyword heuristic"

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
        n_results=3,
        include=["documents", "metadatas"]
    )

    context = "\n".join(results["documents"][0])
    sources = list(set(
        meta["source"] for meta in results["metadatas"][0]
    ))

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

    return response["message"]["content"], sources

def run_sql(question: str):
    sql = vn.generate_sql(question=question)
    df = vn.run_sql(sql=sql)
    data_as_text = df.to_string(index=False)

    prompt = f"""
You are a data analyst. Answer the question directly and concisely based on the query result below.
Do not restate the table. Do not explain your reasoning. Just give the answer in 1-2 sentences.
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
# ENDPOINTS
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith((".txt", ".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Only TXT, PDF, DOCX supported")

    filepath = os.path.join(DOCS_FOLDER, file.filename)
    try:
        with open(filepath, "wb") as f:
            f.write(file.file.read())
        chunks_indexed = index_file(filepath, file.filename)
        return {
            "message": f"{file.filename} uploaded and indexed",
            "doc_id": file.filename,
            "chunks_indexed": chunks_indexed
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/reindex")
def reindex():
    try:
        total_chunks = reindex_documents()
        return {
            "message": "Reindex complete",
            "chunks_indexed": total_chunks
        }
    except Exception as e:
        logger.error(f"Reindex failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reindex failed: {str(e)}")

@app.post("/train")
def train(request: TrainRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if not request.sql.strip():
        raise HTTPException(status_code=400, detail="SQL cannot be empty")
    try:
        vn.train(question=request.question, sql=request.sql)
        return {
            "message": "Training successful",
            "question": request.question,
            "sql": request.sql
        }
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/ask")
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    intent, method, reason = classify_intent(request.question)

    if intent == "SQL":
        try:
            answer, sql, result = run_sql(request.question)
            save_history(request.question, answer, "SQL", sql)
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
            raise HTTPException(status_code=500, detail=f"SQL execution failed: {str(e)}")
    else:
        try:
            answer, sources = run_rag(request.question)
            save_history(request.question, answer, "RAG")
            return {
                "question": request.question,
                "tool_used": "RAG",
                "classifier": method,
                "confidence_note": reason,
                "sources": sources,
                "answer": answer
            }
        except Exception as e:
            logger.error(f"RAG failed: {e}")
            raise HTTPException(status_code=500, detail=f"RAG execution failed: {str(e)}")

@app.get("/history")
def history(limit: int = 10):
    if limit <= 0:
        raise HTTPException(status_code=400, detail="Limit must be greater than 0")
    if limit > 100:
        raise HTTPException(status_code=400, detail="Limit cannot exceed 100")
    try:
        records = get_history(limit=limit)
        return {
            "count": len(records),
            "history": records
        }
    except Exception as e:
        logger.error(f"History fetch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

@app.delete("/docs/{doc_id}", status_code=200)
def delete_document(doc_id: str):
    try:
        chunks_deleted = delete_file(doc_id)
        if chunks_deleted == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{doc_id}' not found in store"
            )
        return {
            "message": f"{doc_id} successfully deleted",
            "chunks_deleted": chunks_deleted
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")