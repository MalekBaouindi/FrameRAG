# FrameworkRAG — Checkpoint

## What's done

### Infrastructure
- [x] Docker Desktop installed
- [x] `docker-compose.yml` with Qdrant + Neo4j + FastAPI backend
- [x] All Python dependencies installed
- [x] All containers running (qdrant, neo4j, backend)
- [x] `/health` endpoint responds OK

### Evaluation
- [x] RAGAS faithfulness baseline: **0.366 ± 0.267** (10-questions, `llama-3.1-8b-instant` as judge)
- [ ] Answer relevancy blocked by Groq free tier (`n > 1` not supported)
- [ ] Full eval with ground truth + context precision when API quota allows

### Ingestion pipeline
- [x] Scraper — Playwright BFS capped at 50 pages, uses `deque` + `queued` set for O(1) BFS and dedup
- [x] Chunker — HTML → Markdown → header split → recursive split, SHA256 chunk IDs
- [x] Embedding generation + Qdrant indexing — `fastembed` (`BAAI/bge-small-en-v1.5`, 384-dim), batch upsert (200/batch)
- [x] Ingestion runs successful: **13,496 total chunks across 3 Qdrant collections** (langchain: 12,167 / llamaindex: 575 / haystack: 754)
- [x] Entity extraction + Neo4j population — 11,168 chunks, 190 components, 142 integrations, 67 concepts, 3 frameworks

### Query API
- [x] `POST /query` — hybrid search (dense + sparse via SPLADE), re-ranking (cross-encoder), LLM generation via Groq
- [x] `POST /query/graph` — Graph RAG (entity relationships via Neo4j)
- [x] `POST /query/agents` — LangGraph multi-agent (supervisor routes to vector/graph/comparison agents)

### Supported frameworks
- [x] LangChain — 12,167 chunks indexed
- [x] LlamaIndex — 575 chunks indexed
- [x] Haystack — 754 chunks indexed

### Cleanup
- [x] Removed Playwright + Chromium from backend Dockerfile (was unused, added 500MB+)
- [x] Removed Playwright/BeautifulSoup/lxml from backend requirements
- [x] Neo4j service: lazy driver init (no crash if Neo4j unavailable at startup)
- [x] Neo4j driver closed on app shutdown
- [x] Hardcoded passwords removed from config defaults (use `.env`)
- [x] Scraper: `deque` for O(1) BFS, `queued` set to prevent URL duplicates
- [x] Chunker: SHA256 instead of MD5 for chunk IDs
```
RAG Project/
├── docker-compose.yml
├── CHECKPOINT.md
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── __init__.py
│       ├── main.py              # FastAPI app with /health route
│       ├── core/
│       │   ├── __init__.py
│       │   └── config.py        # Settings via pydantic-settings
│       ├── models/
│       │   ├── __init__.py
│       │   └── schemas.py       # Chunk, Question, Answer, Source
│       ├── routes/
│       │   ├── __init__.py
│       │   ├── health.py        # GET /health
│       │   └── query.py         # POST /query, POST /query/graph (stubs)
│       └── services/
│           ├── __init__.py
│           ├── qdrant_service.py  # QdrantClient wrapper
│           └── neo4j_service.py   # Neo4j GraphDatabase wrapper
├── ingestion/
│   ├── __init__.py
│   ├── scraper.py              # Playwright scraper (capped at 50 pages)
│   ├── chunker.py              # HTML→Markdown→Header splitting→Recursive split
│   ├── pipeline.py             # Orchestration (scrape → chunk → embed → index)
│   └── requirements.txt
└── data/
    ├── raw/langchain/pages.json      # 49 scraped pages
    ├── chunks/langchain/chunks.json  # 1332 chunks
    ├── qdrant/                       # Docker volume
    └── neo4j/                        # Docker volume
```

## Quick start

```powershell
# Start services
docker compose up -d

# Run ingestion (--skip-scrape to reuse existing scraped pages)
docker compose exec backend python ingestion/pipeline.py langchain --skip-scrape

# Test API
Invoke-RestMethod http://localhost:8000/health
```

## Next milestones

- [x] Week 1-2: Embeddings + Qdrant indexing in `pipeline.py`
- [x] Week 3-4: RAG query endpoint (retrieve from Qdrant + LLM generation via Groq)
- [x] Week 5-6: LlamaIndex & Haystack indexed, hybrid search (dense + sparse via SPLADE)
- [x] Week 7-8: Neo4j graph population + Graph RAG endpoint
- [x] Week 9-10: LangGraph multi-agent orchestration (supervisor + vector/graph/comparison agents)
- [x] Week 11: RAGAS evaluation (faithfulness baseline: 0.366)
- [ ] Week 12: Docker polish + README + frontend
