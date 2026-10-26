import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from app.chroma import (
    MAX_UPLOAD_BYTES, document_path, reindex_documents, index_file,
    delete_file, list_documents,
)
from app.chat import (
    classify_intent, run_rag, run_sql,
    save_history, get_history, logger
)
from app.schema import QuestionRequest, TrainRequest
from app.sql_guard import validate_read_query
from app.vanna_setup import get_vanna

# -----------------------------
# CONFIG
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    logger.info("Server starting; external services initialize on demand")
    
    yield  # server is running, handling requests
    
    # shutdown
    logger.info("👋 Server shutting down...")

app = FastAPI(
    title="Agentic RAG + Text2SQL Backend",
    description="A hybrid RAG + Text2SQL system",
    version="1.0.0",
    lifespan=lifespan,
)

# -----------------------------
# ENDPOINTS
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    try:
        filepath = document_path(file.filename or "")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Document exceeds the 10 MB limit")
    if not contents:
        raise HTTPException(status_code=400, detail="Document is empty")

    filepath.parent.mkdir(parents=True, exist_ok=True)
    try:
        with filepath.open("xb") as output:
            output.write(contents)
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail="Document already exists") from e

    try:
        chunks_indexed = index_file(str(filepath), file.filename)
        if chunks_indexed == 0:
            raise HTTPException(status_code=400, detail="Document contains no extractable text")
        return {
            "message": f"{file.filename} uploaded and indexed",
            "doc_id": file.filename,
            "chunks_indexed": chunks_indexed
        }
    except Exception as e:
        filepath.unlink(missing_ok=True)
        if isinstance(e, HTTPException):
            raise
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Upload failed") from e

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
        validate_read_query(request.sql)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        get_vanna().train(question=request.question, sql=request.sql)
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


@app.get("/documents")
def documents():
    try:
        return {"documents": list_documents()}
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to list documents") from e

@app.delete("/docs/{doc_id}", status_code=200)
def delete_document(doc_id: str):
    try:
        document_path(doc_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
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
