# frontend/api.py
import httpx

BASE_URL = "http://localhost:8000"

def ask_question(question: str) -> dict:
    try:
        response = httpx.post(
            f"{BASE_URL}/ask",
            json={"question": question},
            timeout=60.0
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def upload_document(file_bytes: bytes, filename: str) -> dict:
    try:
        response = httpx.post(
            f"{BASE_URL}/upload",
            files={"file": (filename, file_bytes)},
            timeout=30.0
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_history(limit: int = 10) -> dict:
    try:
        response = httpx.get(
            f"{BASE_URL}/history",
            params={"limit": limit},
            timeout=10.0
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def train(question: str, sql: str) -> dict:
    try:
        response = httpx.post(
            f"{BASE_URL}/train",
            json={"question": question, "sql": sql},
            timeout=30.0
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_health() -> dict:
    try:
        response = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        return response.json()
    except Exception as e:
        return {"error": str(e)}
    
def reindex() -> dict:
    try:
        response = httpx.post(f"{BASE_URL}/reindex", timeout=60.0)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def delete_document(doc_id: str) -> dict:
    try:
        response = httpx.delete(
            f"{BASE_URL}/docs/{doc_id}",
            timeout=10.0
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}