from types import SimpleNamespace

import app.chroma as chroma


def test_document_listing_counts_chunks_and_reports_missing_files(monkeypatch, tmp_path):
    (tmp_path / "rag.txt").write_text("RAG")
    metadata = [
        {"source": "rag.txt"}, {"source": "rag.txt"},
        {"source": "missing.pdf"}, {"source": "../unsafe.txt"},
    ]
    collection = SimpleNamespace(get=lambda **kwargs: {"metadatas": metadata})
    monkeypatch.setattr(chroma, "DOCS_FOLDER", tmp_path)
    monkeypatch.setattr(chroma, "get_client", lambda: SimpleNamespace(get_or_create_collection=lambda **kwargs: collection))

    assert chroma.list_documents() == [
        {"doc_id": "missing.pdf", "chunks_indexed": 1, "size_bytes": None},
        {"doc_id": "rag.txt", "chunks_indexed": 2, "size_bytes": 3},
    ]
