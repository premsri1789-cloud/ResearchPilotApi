# Research Pilot API 🔍

A FastAPI-based **Corrective RAG (CRAG)** backend that lets users upload research documents and chat with them using an LLM. When the answer isn't found in the uploaded document, the agent automatically falls back to a real-time web search via Tavily — all scoped by `user_id` and `session_id` for multi-user isolation.

---

## ✨ Features

- **PDF Ingestion with Multimodal Support** — Converts PDFs to Markdown using PyMuPDF4LLM, extracts embedded images/tables, and generates visual summaries using a Groq vision-capable model before storing chunks in the vector store.
- **Local Vector Storage** — Uses Qdrant in local persistence mode with BAAI/bge-small-en-v1.5 (384-dim) embeddings via FastEmbed — no cloud vector DB required.
- **Multi-user / Multi-session Isolation** — Every ingested chunk is tagged with `user_id` and `session_id`, and retrieval is filtered by both fields so users only see their own documents.
- **LangGraph CRAG Pipeline** — A 5-node agentic graph handles query rewriting → retrieval → relevancy grading → conditional web search → generation.
- **Tavily Web Search Fallback** — If retrieved documents are graded as irrelevant, the agent searches the web in real time and uses those results to answer.
- **Chat History Context** — Last 3 conversation turns (6 messages) are stored in SQLite and used to rewrite follow-up questions into standalone queries.
- **CORS Ready** — Pre-configured for Angular dev server at `localhost:4200`.
- **Dockerized** — Includes a production-ready Dockerfile targeting port 8080 (Cloud Run compatible).

---

## 🏗️ Architecture

```
User Request
     │
     ▼
FastAPI  (/api/upload  |  /api/chat)
     │
     ▼
LangGraph Agent
  ┌──────────────────────────────────────┐
  │  1. Query Rewriter                   │  ← rewrites follow-ups using chat history
  │  2. Retrieve (Qdrant, user+session)  │  ← filtered vector search
  │  3. Grade Documents                  │  ← LLM relevancy check (JSON score)
  │  4a. Generate (if relevant)          │  ← answer from context
  │  4b. Web Search → Generate (if not)  │  ← Tavily fallback
  └──────────────────────────────────────┘
```

### Project Structure

```
ResearchPilotApi/
├── main.py                     # FastAPI app, /api/upload and /api/chat endpoints
├── Dockerfile
├── Requirements.txt
├── .example.env
└── app/
    ├── core/
    │   ├── vector_db.py        # Qdrant local client
    │   └── memory_db.py        # SQLite chat history store
    └── graph/
        ├── state.py            # AgentState TypedDict
        ├── Ingestion.py        # PDF → Markdown → Embed → Qdrant pipeline
        └── Langgraph_engine.py # 5-node CRAG LangGraph
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- [Groq API Key](https://console.groq.com/)
- [Tavily API Key](https://app.tavily.com/)

### 1. Clone & Install

```bash
git clone https://github.com/your-username/ResearchPilotApi.git
cd ResearchPilotApi
pip install -r Requirements.txt
```

### 2. Configure Environment

```bash
cp .example.env .env
```

Edit `.env` and fill in your keys:

```env
GROQ_API_KEY="your-groq-api-key"
TAVILY_API_KEY="your-tavily-api-key"
```

> The remaining keys in `.example.env` (`QDRANT_URL`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`) are optional and not required for local mode.

### 3. Run

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

---

## 📡 API Reference

### `POST /api/upload`

Ingest a PDF document into the vector store for a specific user and session.

**Request** — `multipart/form-data`

| Field        | Type   | Description                        |
|--------------|--------|------------------------------------|
| `file`       | File   | PDF file to ingest                 |
| `session_id` | string | Unique identifier for the session  |
| `user_id`    | string | Unique identifier for the user     |

**Response**

```json
{ "message": "Document ingested successfully inside local vector store." }
```

---

### `POST /api/chat`

Send a question and get an answer grounded in the uploaded document or web search.

**Request Body** — `application/json`

```json
{
  "session_id": "session-abc",
  "user_id": "user-123",
  "question": "What is the main finding of the paper?",
  "history": [
    { "role": "user", "content": "What is this paper about?" },
    { "role": "assistant", "content": "This paper is about..." }
  ]
}
```

**Response**

```json
{ "answer": "The main finding is..." }
```

---

## 🧠 How the CRAG Pipeline Works

1. **Query Rewriter** — If there's conversation history, the raw question is rewritten into a self-contained query using the LLM, so retrieval doesn't depend on prior context being implicit.

2. **Retrieve** — The rewritten query is embedded using `BAAI/bge-small-en-v1.5` and used to search Qdrant, filtered strictly by `user_id` + `session_id`. Top 3 chunks are returned.

3. **Grade Documents** — The LLM scores the retrieved chunks against the query (`yes` / `no`). If irrelevant, the documents list is cleared.

4. **Conditional Routing**:
   - Documents present → **Generate**
   - Documents empty → **Web Search** → **Generate**

5. **Generate** — The final answer is generated by `llama-3.3-70b-versatile` on Groq using the combined context (document chunks and/or web results).

---

## 🐳 Docker

### Build & Run

```bash
docker build -t research-pilot-api .
docker run -p 8080:8080 --env-file .env research-pilot-api
```

The container exposes port `8080` and is compatible with Google Cloud Run out of the box.

---

## 🛠️ Tech Stack

| Layer           | Technology                              |
|-----------------|------------------------------------------|
| API Framework   | FastAPI + Uvicorn                        |
| LLM             | Groq (`llama-3.3-70b-versatile`)         |
| Embeddings      | FastEmbed (`BAAI/bge-small-en-v1.5`)     |
| Vector Store    | Qdrant (local persistence)               |
| Agent Framework | LangGraph                                |
| PDF Parsing     | PyMuPDF4LLM                              |
| Web Search      | Tavily                                   |
| Chat Memory     | SQLite                                   |
| Containerization| Docker                                   |

---

