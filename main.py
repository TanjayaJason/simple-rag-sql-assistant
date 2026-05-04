import os
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from chroma import reindex_documents, index_file, delete_file
from chat import (
    save_history, get_history, logger
)
from schema import QuestionRequest, TrainRequest
from vanna_setup import vn
from langchain_agent import agent_executor

# -----------------------------
# CONFIG
# -----------------------------
app = FastAPI()

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

    filepath = os.path.join("./docs", file.filename)
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
    start = time.time()

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = agent_executor.invoke({"messages": [{"role": "user", "content": request.question}]})
        content = result["messages"][-1].content

        # Extract tool outputs from intermediate messages
        sql = None
        sources = []

        for msg in result["messages"]:
            if hasattr(msg, "content") and isinstance(msg.content, str):
                if msg.content.startswith("SQL:"):
                    # Extract SQL from sql_tool output
                    for line in msg.content.splitlines():
                        if line.startswith("SQL:"):
                            sql = line.replace("SQL:", "").strip()
                if msg.content.startswith("SOURCES:"):
                    # Extract sources from rag_tool output
                    for line in msg.content.splitlines():
                        if line.startswith("SOURCES:"):
                            sources = [s.strip() for s in line.replace("SOURCES:", "").split(",")]

        # Parse tool_used, confidence, reason from final answer
        tool_used = "RAG"
        confidence = "low"
        reason = "Could not determine"

        for line in content.splitlines():
            if line.startswith("TOOL_USED:"):
                tool_used = "SQL" if "sql_tool" in line.lower() else "RAG"
            elif line.startswith("CONFIDENCE:"):
                confidence = line.split(":")[1].strip()
            elif line.startswith("REASON:"):
                reason = line.split(":")[1].strip()

        # Clean answer
        answer = "\n".join(
            line for line in content.splitlines()
            if not line.startswith(("TOOL_USED:", "CONFIDENCE:", "REASON:"))
        ).strip()

        elapsed = round(time.time() - start, 2)
        logger.info(f"Request completed in {elapsed}s — tool: {tool_used}")
        save_history(request.question, answer, tool_used)

        return {
            "question": request.question,
            "tool_used": tool_used,
            "confidence": confidence,
            "reason": reason,
            "sql": sql,
            "sources": sources,
            "answer": answer,
            "response_time_seconds": elapsed
        }
    except Exception as e:
        logger.error(f"Agent failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

@app.get("/history")
def history(limit: int = 10):
    if limit <= 0:
        raise HTTPException(status_code=400, detail="Limit must be greater than 0")
    if limit > 10:
        raise HTTPException(status_code=400, detail="Limit cannot exceed 10")
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