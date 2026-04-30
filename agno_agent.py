from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool
import ollama
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from vanna_setup import vn
import os

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = "mxbai-embed-large"
LLM_MODEL = "gpt-4o-mini"
COLLECTION_NAME = "docs"

chroma_client = chromadb.PersistentClient(path="./chroma_db")

# -----------------------------
# TOOLS
# -----------------------------
@tool
def rag_tool(question: str) -> str:
    """
    Search technical documentation to answer questions about:
    Python, FastAPI, PostgreSQL, RAG, LLM, Language Models,
    ChromaDB, vector stores, embeddings, Vanna AI, Text2SQL,
    concepts, explanations, and how-to guides.
    """
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

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

    answer = response.choices[0].message.content
    return f"SOURCES: {', '.join(sources)}\nANSWER: {answer}"

@tool
def sql_tool(question: str) -> str:
    """
    Query the database to answer questions about:
    students (count, list, registration),
    courses (price, category, title, revenue),
    enrollments (purchases, counts, dates),
    aggregations like total, average, highest, lowest, how many, top N.
    """
    sql = vn.generate_sql(question=question)

    if not sql.strip().upper().startswith("SELECT"):
        return f"ERROR: Could not generate valid SQL: {sql}"

    df = vn.run_sql(sql=sql)
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

    answer = response.choices[0].message.content
    return f"SQL: {sql}\nANSWER: {answer}"

# -----------------------------
# AGENT
# -----------------------------
agent = Agent(
    name="RAG SQL Agent",
    model=OpenAIChat(id=LLM_MODEL),
    tools=[rag_tool, sql_tool],
    instructions="""
You are a hybrid AI assistant with two tools:
1. rag_tool — for questions about technical documentation (Python, RAG, LLM, ChromaDB, Vanna AI, FastAPI, PostgreSQL concepts)
2. sql_tool — for questions about database data (students, courses, enrollments, revenue, counts)

Always pick the most appropriate tool based on the question.
Never answer from your own knowledge — always use a tool.
If the first tool returns "I don't know" or fails, try the other tool before giving up.

After answering, always end your response with exactly these three lines:
TOOL_USED: <rag_tool or sql_tool>
CONFIDENCE: <high, medium, or low>
REASON: <one sentence why you chose this tool>
""",
    markdown=False
)