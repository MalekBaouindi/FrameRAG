# FrameworkRAG

RAG + Graph RAG + multi-agent system for comparing RAG/LLM frameworks (LangChain, LlamaIndex, Haystack). Built with Qdrant, Neo4j, Groq, and LangGraph.

## Architecture

```
User → Streamlit UI → FastAPI → Qdrant (hybrid search)
                         ├──→ Neo4j (graph queries)
                         └──→ LangGraph agents
```

- **`/query`** — Hybrid search (dense + sparse via SPLADE) across indexed docs, optional reranking, LLM answer via Groq
- **`/query/graph`** — Entity-relationship search via Neo4j (components, concepts, integrations)
- **`/query/agents`** — Multi-agent orchestration (supervisor routes to vector/graph/comparison specialists)

## Live Demo

👉 **[https://framerag-frontend.onrender.com](https://framerag-frontend.onrender.com)**

## Quick Start

### Prerequisites
- Python 3.12
- Docker Desktop (for Qdrant + Neo4j)
- Groq API key ([free at console.groq.com](https://console.groq.com))

### 1. Start services
```powershell
docker compose up -d qdrant neo4j
```

### 2. Set API key
```powershell
$env:GROQ_API_KEY = 'gsk_your_key_here'
```

### 3. Start API server
```powershell
cd backend
.venv\Scripts\python -m uvicorn app.main:app --reload --port 8000
```

### 4. Start frontend (optional)
```powershell
.venv\Scripts\streamlit run frontend/app.py
```

### 5. Run ingestion (one-time)
```powershell
.venv\Scripts\python ingestion/pipeline.py langchain
.venv\Scripts\python ingestion/pipeline.py llamaindex
.venv\Scripts\python ingestion/pipeline.py haystack
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/query` | POST | Hybrid search (dense + sparse) + LLM answer |
| `/query/graph` | POST | Neo4j entity relationship search |
| `/query/agents` | POST | LangGraph multi-agent orchestration |

### Example
```powershell
$body = @{query="What vector stores does LangChain support?"; top_k=5} | ConvertTo-Json
Invoke-RestMethod http://localhost:8000/query -Method Post -Body $body -ContentType "application/json"
```

## Data

| Framework | Chunks | Qdrant Collection |
|-----------|--------|-------------------|
| LangChain | 12,167 | `langchain-latest` |
| LlamaIndex | 575 | `llamaindex-latest` |
| Haystack | 754 | `haystack-latest` |

### Neo4j Graph
- 11,168 chunk nodes
- 190 components
- 142 integrations
- 67 concepts
- 3 framework nodes

## Evaluation
- **Faithfulness**: 0.366 ± 0.267 (RAGAS, 10 questions, llama-3.1-8b-instant as judge)

## Project Structure
```
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── core/config.py       # Settings
│   │   ├── models/schemas.py    # Pydantic models
│   │   ├── routes/
│   │   │   ├── health.py
│   │   │   └── query.py         # /query, /graph, /agents
│   │   └── services/
│   │       ├── qdrant_service.py
│   │       ├── neo4j_service.py
│   │       └── agent_service.py # LangGraph agents
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   └── app.py                   # Streamlit UI
├── ingestion/
│   ├── scraper.py               # Playwright BFS scraper
│   ├── chunker.py               # HTML→Markdown→split
│   ├── pipeline.py              # scrape→chunk→embed→index
│   └── entity_extractor.py      # LLM extraction → Neo4j
├── evaluation/
│   └── evaluate.py              # RAGAS evaluation
├── data/                        # Scraped pages, chunks
├── docker-compose.yml
└── README.md
```
