import os
import chromadb
import ollama
import logging
import psycopg2
from openai import OpenAI
from vanna_setup import vn
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# -----------------------------
# CONFIG
# -----------------------------
load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = "mxbai-embed-large"
LLM_MODEL = "gpt-4o-mini"
DOCS_FOLDER = "./docs"
COLLECTION_NAME = "docs"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    "price", "per course", "per student", "last month",
    "tell me about the course", "about the course", "course details"
]

RAG_KEYWORDS = [
    "what is", "explain", "describe", "how does", "how do",
    "what are", "tell me about", "guide", "documentation",
    "definition", "difference between", "compare",
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
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0].message.content

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
        n_results=5,
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
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content, sources

def run_sql(question: str):
    sql = vn.generate_sql(question=question)
    if not sql.strip().upper().startswith("SELECT"):
        raise ValueError(f"Vanna could not generate SQL: {sql}")

    df = vn.run_sql(sql=sql)
    if df.empty:
        return "No data found.", sql, []
    
    data_as_text = df.to_string(index=False)

    prompt = f"""
You are a data analyst. Answer the question directly and concisely based on the query result below.
Your answer must reflect ALL rows in the result, not just a summary total.
Do not restate the table. Do not explain your reasoning.
If the result is empty, say "No data found."

Query Result:
{data_as_text}

Question:
{question}

Answer:
"""
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content, sql, df.to_dict(orient="records")