# Agentic RAG + Text2SQL Backend
A FastAPI backend that answers natural language questions by routing to either a SQL database (via Vanna AI) or a document knowledge base (via ChromaDB RAG).

---

## Stack
- **FastAPI** — REST API framework
- **Vanna AI** — Text2SQL generation
- **ChromaDB** — Vector store for RAG and Vanna
- **PostgreSQL** — Business database
- **OpenAI GPT-4o-mini** — LLM for SQL generation and RAG answers
- **Ollama** — Local embedding model only

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Pull Ollama models
Make sure you have Ollama installed first: https://ollama.com/download

Then pull the required embedding model:
```bash
ollama pull qwen3-embedding:0.6b
```

### 3. Configure environment variables
Create a `.env` file in the project root:
```
DB_HOST=localhost
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_PORT=5432
OPENAI_API_KEY=sk-...
```

### 4. Set up the database
Create all required tables by running the SQL in the **Database Setup** section below.

### 5. Train Vanna
```bash
python scripts/vanna_train.py
```

### 6. Run the server
```bash
uvicorn app.main:app --reload
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

### Sample Data
```sql
-- Courses
INSERT INTO courses (title, category, price) VALUES
('Python Fundamentals', 'Programming', 100.00),
('FastAPI Backend', 'Backend', 120.00),
('RAG Basics', 'AI', 150.00),
('ChromaDB Essentials', 'AI', 130.00),
('PostgreSQL Mastery', 'Database', 110.00),
('LLM Engineering', 'AI', 200.00),
('Docker for Developers', 'DevOps', 90.00),
('REST API Design', 'Backend', 115.00),
('Vector Databases', 'AI', 175.00),
('Data Engineering with Python', 'Programming', 160.00);

-- Students
INSERT INTO students (name, email) VALUES
('Alice', 'alice@email.com'),
('Bob', 'bob@email.com'),
('Charlie', 'charlie@email.com'),
('Diana', 'diana@email.com'),
('Evan', 'evan@email.com'),
('Fiona', 'fiona@email.com'),
('George', 'george@email.com'),
('Hannah', 'hannah@email.com');

-- Enrollments
INSERT INTO enrollments (student_id, course_id, purchase_date) VALUES
(1, 1, '2026-04-01'), (1, 3, '2026-04-02'), (1, 6, '2026-04-10'),
(2, 1, '2026-04-03'), (2, 4, '2026-04-03'), (2, 9, '2026-04-15'),
(3, 3, '2026-04-04'), (3, 4, '2026-04-04'), (3, 7, '2026-04-20'),
(4, 2, '2026-04-05'), (4, 5, '2026-04-05'), (4, 8, '2026-04-18'),
(5, 6, '2026-04-06'), (5, 9, '2026-04-06'), (5, 10, '2026-04-22'),
(6, 1, '2026-04-07'), (6, 3, '2026-04-07'),
(7, 2, '2026-04-08'), (7, 5, '2026-04-08'), (7, 7, '2026-04-25'),
(8, 4, '2026-04-09'), (8, 6, '2026-04-09'), (8, 10, '2026-04-28');
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
| POST | /ask | Ask a question (auto-routes to SQL or RAG via LangChain agent) |
| POST | /train | Add new Vanna training data |
| POST | /upload | Upload document to RAG store |
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
curl -X POST http://localhost:8000/upload \
  -F "file=@./data/docs/chromadb.txt"
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
| OPENAI_API_KEY | Your OpenAI API key |

---

## Project Structure
```
project/
├── app/                    # Main application code
│   ├── __init__.py
│   ├── main.py             # FastAPI endpoints
│   ├── chat.py             # Business logic and history helpers
│   ├── chroma.py           # RAG indexing and retrieval
│   ├── schema.py           # Pydantic models
│   ├── vanna_setup.py      # Vanna + PostgreSQL setup
│
├── scripts/
│   └── vanna_train.py      # One-time Vanna training script
│
├── data/
│   ├── docs/               # Uploaded documents
│   ├── chroma_db/          # RAG vector store
│   └── vanna_chroma/       # Vanna vector store
│
├── tests/
│   └── test_chat.py        # Pytest tests
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Running Tests
```bash
pytest tests/test_chat.py -v
```