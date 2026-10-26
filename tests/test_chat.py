"""API behavior without OpenAI, Ollama, or PostgreSQL connections."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.main as main
import app.chroma as chroma


@pytest.fixture
def client():
    return TestClient(main.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "version": "1.0.0"}


@pytest.mark.parametrize(
    "intent,question,runner_result,expected",
    [
        ("SQL", "How many students?", ("Eight", "SELECT count(*) FROM students", [{"count": 8}]), "sql"),
        ("RAG", "What is RAG?", ("Retrieval augmented generation", ["rag.txt"]), "sources"),
    ],
)
def test_ask_routes_without_live_services(client, monkeypatch, intent, question, runner_result, expected):
    monkeypatch.setattr(main, "classify_intent", lambda _: (intent, "test", "mocked"))
    monkeypatch.setattr(main, "run_sql" if intent == "SQL" else "run_rag", lambda _: runner_result)
    monkeypatch.setattr(main, "save_history", lambda *args: None)

    response = client.post("/ask", json={"question": question})

    assert response.status_code == 200
    assert response.json()["tool_used"] == intent
    assert expected in response.json()
    assert "response_time_seconds" in response.json()


def test_ask_rejects_empty_and_missing_questions(client):
    assert client.post("/ask", json={"question": ""}).status_code == 400
    assert client.post("/ask", json={}).status_code == 422


def test_training_and_history_without_live_services(client, monkeypatch):
    trained = []
    monkeypatch.setattr(main, "get_vanna", lambda: SimpleNamespace(train=lambda **kw: trained.append(kw)))
    monkeypatch.setattr(main, "get_history", lambda limit: [{"question": "sample"}])

    response = client.post("/train", json={"question": "How many?", "sql": "SELECT 1"})
    assert response.status_code == 200
    assert trained == [{"question": "How many?", "sql": "SELECT 1"}]
    assert client.post("/train", json={"question": "", "sql": "SELECT 1"}).status_code == 400
    assert client.post("/train", json={"question": "Delete?", "sql": "DELETE FROM students"}).status_code == 400
    assert client.get("/history?limit=1").json()["count"] == 1
    assert client.get("/history?limit=0").status_code == 400
    assert client.get("/history?limit=11").status_code == 400


def test_upload_validates_name_size_and_duplicates(client, monkeypatch, tmp_path):
    monkeypatch.setattr(chroma, "DOCS_FOLDER", tmp_path)
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 4)
    monkeypatch.setattr(main, "index_file", lambda *args: 1)

    def upload(filename, contents):
        return client.post("/upload", files={"file": (filename, contents)})

    assert upload("../bad.txt", b"ok").status_code == 400
    assert upload("bad\\name.txt", b"ok").status_code == 400
    assert upload("CON.txt", b"ok").status_code == 400
    assert upload("bad.csv", b"ok").status_code == 400
    assert upload("empty.txt", b"").status_code == 400
    assert upload("large.txt", b"12345").status_code == 413
    assert upload("safe.txt", b"text").status_code == 200
    assert (tmp_path / "safe.txt").read_bytes() == b"text"
    assert upload("safe.txt", b"text").status_code == 409


def test_upload_removes_unindexed_file(client, monkeypatch, tmp_path):
    monkeypatch.setattr(chroma, "DOCS_FOLDER", tmp_path)
    monkeypatch.setattr(main, "index_file", lambda *args: 0)
    response = client.post("/upload", files={"file": ("empty.txt", b" ")})
    assert response.status_code == 400
    assert not (tmp_path / "empty.txt").exists()


def test_document_listing_and_delete(client, monkeypatch):
    monkeypatch.setattr(main, "list_documents", lambda: [{"doc_id": "rag.txt", "chunks_indexed": 2, "size_bytes": 10}])
    monkeypatch.setattr(main, "delete_file", lambda name: 2 if name == "rag.txt" else 0)
    assert client.get("/documents").json()["documents"][0]["doc_id"] == "rag.txt"
    assert client.delete("/docs/rag.txt").status_code == 200
    assert client.delete("/docs/missing.txt").status_code == 404
    assert client.delete("/docs/bad.csv").status_code == 400
