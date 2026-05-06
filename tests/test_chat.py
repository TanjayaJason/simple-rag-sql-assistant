from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# -----------------------------
# 1. TEST ROOT
# -----------------------------
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.0.0"}

# -----------------------------
# 2. TEST ASK - RAG ROUTING
# -----------------------------
def test_ask_routes_to_rag():
    response = client.post("/ask", json={"question": "What is ChromaDB?"})
    assert response.status_code == 200
    data = response.json()
    assert data["tool_used"] == "RAG"
    assert "answer" in data
    assert "question" in data
    assert "confidence" in data
    assert "reason" in data
    assert "sources" in data
    assert data["sql"] is None

# -----------------------------
# 3. TEST ASK - SQL ROUTING
# -----------------------------
def test_ask_routes_to_sql():
    response = client.post("/ask", json={"question": "How many students are registered?"})
    assert response.status_code == 200
    data = response.json()
    assert data["tool_used"] == "SQL"
    assert "answer" in data
    assert "question" in data
    assert "confidence" in data
    assert "reason" in data
    assert "sql" in data
    assert data["sql"] is not None
    assert data["sources"] == []

# -----------------------------
# 4. TEST ASK - EMPTY QUESTION
# -----------------------------
def test_ask_empty_question():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty"

# -----------------------------
# 5. TEST ASK - MISSING FIELD
# -----------------------------
def test_ask_missing_field():
    response = client.post("/ask", json={})
    assert response.status_code == 422  # FastAPI auto handles this

# -----------------------------
# 6. TEST TRAIN - SUCCESS
# -----------------------------
def test_train_success():
    response = client.post("/train", json={
        "question": "How many courses are available?",
        "sql": "SELECT COUNT(*) FROM courses;"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Training successful"
    assert "question" in data
    assert "sql" in data

# -----------------------------
# 7. TEST TRAIN - EMPTY QUESTION
# -----------------------------
def test_train_empty_question():
    response = client.post("/train", json={
        "question": "",
        "sql": "SELECT * FROM courses;"
    })
    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty"

# -----------------------------
# 8. TEST TRAIN - EMPTY SQL
# -----------------------------
def test_train_empty_sql():
    response = client.post("/train", json={
        "question": "Show all courses",
        "sql": ""
    })
    assert response.status_code == 400
    assert response.json()["detail"] == "SQL cannot be empty"

# -----------------------------
# 9. TEST HISTORY
# -----------------------------
def test_history():
    response = client.get("/history")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "history" in data
    assert isinstance(data["history"], list)

# -----------------------------
# 10. TEST HISTORY - INVALID LIMIT
# -----------------------------
def test_history_invalid_limit():
    response = client.get("/history?limit=0")
    assert response.status_code == 400
    assert response.json()["detail"] == "Limit must be greater than 0"

# -----------------------------
# 11. TEST HISTORY - LIMIT EXCEEDED
# -----------------------------
def test_history_limit_exceeded():
    response = client.get("/history?limit=11")
    assert response.status_code == 400
    assert response.json()["detail"] == "Limit cannot exceed 10"

# -----------------------------
# 12. TEST UPLOAD - WRONG FILE TYPE
# -----------------------------
def test_upload_wrong_file_type():
    response = client.post(
        "/upload",
        files={"file": ("test.csv", b"some,csv,content", "text/csv")}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Only TXT, PDF, DOCX supported"

# -----------------------------
# 13. TEST RESPONSE TIME EXISTS
# -----------------------------
def test_response_time_in_ask():
    response = client.post("/ask", json={"question": "How many students are registered?"})
    assert response.status_code == 200
    data = response.json()
    assert "response_time_seconds" in data
    assert isinstance(data["response_time_seconds"], float)