from fastapi import APIRouter, HTTPException, Request
from openai import OpenAI
from app.models.schemas import Question, Answer, Source
from app.services.qdrant_service import qdrant_service
from app.services.agent_service import run_agent
from app.core.config import settings
from app.services.embedder_service import get_embedder, get_sparse_embedder, get_reranker

router = APIRouter()

SYSTEM_PROMPT = (
    "You are a RAG assistant specialized in LLM frameworks (LangChain, LlamaIndex, Haystack, etc.). "
    "Answer the user's question based ONLY on the provided documentation chunks below. "
    "If the chunks don't contain enough information, say you don't know. "
    "Cite the source URL and section for each claim you make. "
    "When comparing frameworks, point out similarities and differences."
)


def embed_query(text: str, request: Request) -> tuple[list, tuple | None]:
    embedder = get_embedder()
    sparse_embedder = get_sparse_embedder()

    dense = list(embedder.embed([text]))[0].tolist()

    if sparse_embedder:
        sp = list(sparse_embedder.embed([text]))[0]
        indices = sp.indices if hasattr(sp, "indices") else sp[0]
        values = sp.values if hasattr(sp, "values") else sp[1]
        sparse = (indices, values)
    else:
        sparse = None

    return dense, sparse


def maybe_rerank(query: str, results: list, reranker, top_k: int) -> list:
    if reranker is None:
        return results[:top_k]
    pairs = [(query, r.payload["content"]) for r in results]
    scores = list(reranker.rerank(pairs))
    scored = list(zip(results, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in scored[:top_k]]


@router.post("/query", response_model=Answer)
async def query(question: Question, request: Request):
    top_k = question.top_k or settings.top_k
    dense_vec, sparse_vec = embed_query(question.query, request)

    if sparse_vec:
        results = qdrant_service.hybrid_search(
            collection_name=settings.qdrant_collection,
            dense_vector=dense_vec,
            sparse_vector=sparse_vec,
            limit=top_k,
        )
    else:
        results = qdrant_service.search(
            collection_name=settings.qdrant_collection,
            query_vector=dense_vec,
            limit=top_k,
        )

    if not results:
        return Answer(answer="No relevant documents found.", sources=[], query=question.query)

    reranker = get_reranker()
    results = maybe_rerank(question.query, results, reranker, top_k)

    context_parts = []
    sources = []
    for r in results:
        meta = r.payload["metadata"]
        section = meta.get("header2") or meta.get("header1") or meta.get("header3") or ""
        context_parts.append(
            f"[Source: {meta['url']} | Section: {section}]\n{r.payload['content']}"
        )
        sources.append(Source(
            url=meta["url"],
            section=section,
            content=r.payload["content"][:300],
            score=0.0,
        ))

    context = "\n\n".join(context_parts)

    client = OpenAI(
        base_url=settings.groq_base_url,
        api_key=settings.groq_api_key,
    )

    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question.query}"},
        ],
        temperature=0.3,
    )

    answer_text = response.choices[0].message.content
    return Answer(answer=answer_text, sources=sources, query=question.query)


@router.post("/query/graph", response_model=Answer)
async def query_graph(question: Question, request: Request):
    query_text = question.query
    top_k = question.top_k or settings.top_k

    graph = request.app.state.neo4j
    if not graph or not graph._driver:
        return Answer(answer="Graph database (Neo4j) is not available. Please start Neo4j and restart the server.", sources=[], query=query_text)

    terms = [t.strip().lower() for t in query_text.split() if len(t.strip()) > 3]
    if not terms:
        terms = [query_text.lower()]

    results = graph.query_graph(terms, limit=top_k * 3)

    if not results:
        return Answer(answer="No graph data found for your question. Try the /query endpoint instead.", sources=[], query=query_text)

    context_parts = []
    sources = []
    seen_urls = set()

    for r in results:
        entity = r.get("entity", "")
        etype = r.get("type", "")
        desc = r.get("description", "")
        relations = r.get("relations", [])
        chunks = r.get("chunks", [])

        context_parts.append(f"[Entity: {entity} ({etype})]")
        if desc:
            context_parts.append(f"  Description: {desc}")
        if relations:
            rel_strs = []
            for rel in relations:
                if rel.get("name"):
                    rel_strs.append(f"  -> {rel['rel_type']} -> {rel['name']} ({rel['type']})")
            if rel_strs:
                context_parts.extend(rel_strs)

        for chunk_text in chunks:
            if chunk_text and len(chunk_text) > 20:
                context_parts.append(f"  Chunk: {chunk_text[:300]}")

        if chunks:
            source = Source(url=r.get("url", "") or "graph", section=entity, content=desc[:300] if desc else entity, score=0.0)
            if source.url not in seen_urls:
                sources.append(source)
                seen_urls.add(source.url)

    context = "\n".join(context_parts)

    client = OpenAI(
        base_url=settings.groq_base_url,
        api_key=settings.groq_api_key,
    )

    graph_prompt = (
        "You are a RAG assistant specialized in LLM frameworks (LangChain, LlamaIndex, Haystack, etc.). "
        "Answer the user's question based on the graph knowledge below, which shows entities, "
        "their relationships, and document chunks. "
        "Explain how different components, concepts, and integrations relate to each other. "
        "If the information is insufficient, say so."
    )

    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": graph_prompt},
            {"role": "user", "content": f"Graph Knowledge:\n{context}\n\nQuestion: {query_text}"},
        ],
        temperature=0.3,
    )

    answer_text = response.choices[0].message.content
    return Answer(answer=answer_text, sources=sources, query=query_text)


@router.post("/query/agents", response_model=Answer)
async def query_agents(question: Question, request: Request):
    embedder = get_embedder()
    sparse_embedder = get_sparse_embedder()

    result = run_agent(question.query, embedder, sparse_embedder)

    last_msg = result["messages"][-1] if result.get("messages") else {}
    answer_text = last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg)

    sources = []
    for r in result.get("vector_results", [])[:5]:
        meta = r.payload.get("metadata", {})
        sources.append(Source(
            url=meta.get("url", ""),
            section=meta.get("header2", ""),
            content=r.payload.get("content", "")[:300],
            score=r.score or 0.0,
        ))

    return Answer(answer=answer_text, sources=sources, query=question.query)
