# Agentic RAG + Text2SQL Backend
BHP-AI-TRAIN-001 | PT Berca Hardayaperkasa

A FastAPI backend that answers natural language questions by routing to either a SQL database (via Vanna AI) or a document knowledge base (via ChromaDB RAG).

---

## Stack
- **FastAPI** — REST API framework
- **Vanna AI** — Text2SQL generation
- **ChromaDB** — Vector store for RAG and Vanna
- **PostgreSQL** — Business database
- **Ollama** — Local LLM and embedding model

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Pull Ollama models
Make sure you have Ollama installed first: https://ollama.com/download

Then pull the required models:
```bash
ollama pull llama3.2:3b
ollama pull mxbai-embed-large
```

### 3. Configure environment variables
Create a `.env` file in the project root:
```
DB_HOST=localhost
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_PORT=5432
```

### 4. Set up the database
Create all required tables by running the SQL in the **Database Setup** section below.

### 5. Train Vanna
```bash
python vanna_train.py
```

### 6. Run the server
```bash
uvicorn chat:app --reload
```

Swagger UI available at: http://localhost:8000/docs

---

## Database Setup

Run the following SQL on your PostgreSQL database to create all required tables:

### Business Tables
```sql
CREATE TABLE courses (
    id SERIAL PRIMARY KEY,
    title VARCHAR(100),
    category VARCHAR(50),
    price NUMERIC(10,2)
);

CREATE TABLE students (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100)
);

CREATE TABLE enrollments (
    id SERIAL PRIMARY KEY,
    student_id INT REFERENCES students(id),
    course_id INT REFERENCES courses(id),
    purchase_date DATE
);
```

### Conversation History Table
```sql
CREATE TABLE conversation_history (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    tool_used VARCHAR(10) NOT NULL,
    sql_generated TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Service liveness check |
| POST | /ask | Ask a question (auto-routes to SQL or RAG) |
| POST | /train | Add new Vanna training data |
| POST | /docs/upload | Upload document to RAG store |
| POST | /reindex | Reindex all documents |
| GET | /history | Get recent Q&A history |
| DELETE | /docs/{doc_id} | Remove document from store |

---

## Sample curl Commands

### Ask a question
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which course has the highest price?"}'
```

### Upload a document
```bash
curl -X POST http://localhost:8000/docs/upload \
  -F "file=@./docs/chromadb.txt"
```

### Add training data
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"question": "How many students?", "sql": "SELECT COUNT(*) FROM students;"}'
```

### Get history
```bash
curl http://localhost:8000/history?limit=5
```

### Delete a document
```bash
curl -X DELETE http://localhost:8000/docs/chromadb.txt
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| DB_HOST | PostgreSQL host |
| DB_NAME | PostgreSQL database name |
| DB_USER | PostgreSQL username |
| DB_PASSWORD | PostgreSQL password |
| DB_PORT | PostgreSQL port (default: 5432) |

---

## Project Structure
```
project/
├── chat.py          # Main FastAPI app
├── chroma.py        # RAG indexing and retrieval
├── vanna_setup.py   # Vanna + PostgreSQL setup
├── vanna_train.py   # One-time Vanna training script
├── test_chat.py     # Pytest tests
├── requirements.txt
├── .env
├── docs/            # Uploaded documents
├── chroma_db/       # RAG vector store
└── vanna_chroma/    # Vanna vector store
```

---

## Running Tests
```bash
pytest test_chat.py -v
```