import os
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from chroma import reindex_documents, index_file, delete_file
from chat import (
    classify_intent, run_rag, run_sql,
    save_history, get_history, logger
)
from schema import QuestionRequest, TrainRequest
from vanna_setup import vn

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

    intent, method, reason = classify_intent(request.question)

    if intent == "SQL":
        try:
            answer, sql, result = run_sql(request.question)
            save_history(request.question, answer, "SQL", sql)
            response = {
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
            response = {
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

    elapsed = round(time.time() - start, 2)
    logger.info(f"Request completed in {elapsed}s — tool: {intent}")
    response["response_time_seconds"] = elapsed

    return response

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