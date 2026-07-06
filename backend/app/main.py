from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routes import health, query
from app.services.neo4j_service import neo4j_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _ = neo4j_service.driver
        app.state.neo4j = neo4j_service
        print("Neo4j connected")
    except Exception as e:
        app.state.neo4j = None
        print(f"Neo4j unavailable: {e}")
    yield
    neo4j_service.close()
    app.state.neo4j = None


app = FastAPI(
    title="FrameworkRAG",
    description="RAG + Graph RAG + multi-agent system for comparing RAG/LLM frameworks",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(query.router)
