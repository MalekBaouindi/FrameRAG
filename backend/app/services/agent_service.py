"""
LangGraph multi-agent orchestration for RAG framework comparison.

Agents:
  - supervisor: routes questions to the right specialist
  - vector_agent: searches Qdrant (hybrid dense+sparse) for relevant docs
  - graph_agent: queries Neo4j for entity relationships
  - comparison_agent: compares frameworks by searching all collections
  - generate: produces final answer from collected context
"""
from typing import TypedDict, Literal, Annotated, Sequence
from operator import add
import json

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from openai import OpenAI

from app.core.config import settings
from app.services.qdrant_service import qdrant_service
from app.services.neo4j_service import neo4j_service


class AgentState(TypedDict):
    query: str
    messages: Annotated[Sequence[dict], add]
    vector_results: list
    graph_results: list
    comparison_results: list
    next_agent: str


def classify_question(query: str, client: OpenAI) -> str:
    prompt = f"""Classify this RAG framework question into ONE category:
- "vector": asks about specific components, APIs, integrations, or how-to (requires docs search)
- "graph": asks about relationships, comparisons of entities, or what X is connected to
- "comparison": explicitly compares two or more frameworks (LangChain vs LlamaIndex, etc.)
- "general": anything else

Question: {query}
Category:"""
    resp = client.chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=20,
    )
    return resp.choices[0].message.content.strip().lower()


def supervisor_node(state: AgentState, client: OpenAI) -> dict:
    category = classify_question(state["query"], client)
    routing = {
        "graph": "graph_agent",
        "comparison": "comparison_agent",
        "vector": "vector_agent",
        "general": "vector_agent",
    }
    next_agent = routing.get(category, "vector_agent")
    return {"next_agent": next_agent, "messages": [{"role": "system", "content": f"Routing to {next_agent}"}]}


def vector_agent_node(state: AgentState, embedder, sparse_embedder) -> dict:
    query = state["query"]
    dense = list(embedder.embed([query]))[0].tolist()
    sp = list(sparse_embedder.embed([query]))[0]
    indices = sp.indices if hasattr(sp, "indices") else sp[0]
    values = sp.values if hasattr(sp, "values") else sp[1]

    collections = ["langchain-latest", "llamaindex-latest", "haystack-latest"]
    all_results = []
    for coll in collections:
        try:
            results = qdrant_service.hybrid_search(
                collection_name=coll,
                dense_vector=dense,
                sparse_vector=(indices, values),
                limit=5,
            )
            for r in results:
                r.payload["collection"] = coll
            all_results.extend(results)
        except Exception:
            pass

    all_results.sort(key=lambda r: r.score or 0, reverse=True)
    all_results = all_results[:10]

    return {
        "vector_results": all_results,
        "next_agent": "generate",
        "messages": [{"role": "system", "content": f"Vector search: found {len(all_results)} chunks"}],
    }


def graph_agent_node(state: AgentState) -> dict:
    query = state["query"]
    terms = [t.strip().lower() for t in query.split() if len(t.strip()) > 3]
    if not terms:
        terms = [query.lower()]

    try:
        results = neo4j_service.query_graph(terms, limit=10)
    except Exception:
        results = []

    return {
        "graph_results": results,
        "next_agent": "generate",
        "messages": [{"role": "system", "content": f"Graph search: found {len(results)} entities"}],
    }


def comparison_agent_node(state: AgentState, embedder, sparse_embedder) -> dict:
    query = state["query"]
    dense = list(embedder.embed([query]))[0].tolist()
    sp = list(sparse_embedder.embed([query]))[0]
    indices = sp.indices if hasattr(sp, "indices") else sp[0]
    values = sp.values if hasattr(sp, "values") else sp[1]

    collections = ["langchain-latest", "llamaindex-latest", "haystack-latest"]
    per_framework = {}
    for coll in collections:
        try:
            results = qdrant_service.hybrid_search(
                collection_name=coll,
                dense_vector=dense,
                sparse_vector=(indices, values),
                limit=5,
            )
            per_framework[coll] = results
        except Exception:
            per_framework[coll] = []

    return {
        "comparison_results": per_framework,
        "next_agent": "generate",
        "messages": [{"role": "system", "content": f"Comparison: queried {len(collections)} frameworks"}],
    }


def generate_node(state: AgentState, client: OpenAI) -> dict:
    context_parts = []

    if state.get("vector_results"):
        context_parts.append("=== Documentation Chunks ===")
        for r in state["vector_results"][:8]:
            meta = r.payload.get("metadata", {})
            coll = r.payload.get("collection", "unknown")
            section = meta.get("header2") or meta.get("header1") or ""
            context_parts.append(
                f"[{coll}] {meta.get('url', '')} | Section: {section}\n{r.payload.get('content', '')[:500]}"
            )

    if state.get("graph_results"):
        context_parts.append("\n=== Knowledge Graph Entities ===")
        for r in state["graph_results"]:
            entity = r.get("entity", "")
            etype = r.get("type", "")
            desc = r.get("description", "")
            relations = r.get("relations", [])
            context_parts.append(f"Entity: {entity} ({etype})")
            if desc:
                context_parts.append(f"  Description: {desc}")
            for rel in relations[:3]:
                if rel.get("name"):
                    context_parts.append(f"  -> {rel['rel_type']} -> {rel['name']} ({rel['type']})")

    if state.get("comparison_results"):
        context_parts.append("\n=== Cross-Framework Comparison ===")
        for coll, results in state["comparison_results"].items():
            fw_name = coll.replace("-latest", "")
            context_parts.append(f"\n--- {fw_name} ---")
            for r in results[:3]:
                meta = r.payload.get("metadata", {})
                context_parts.append(f"  {meta.get('url', '')}: {r.payload.get('content', '')[:200]}")

    context = "\n".join(context_parts)
    query = state["query"]

    prompt = (
        "You are a RAG assistant specialized in LLM frameworks (LangChain, LlamaIndex, Haystack). "
        "Synthesize the context below and answer the user's question. "
        "Cite specific sources and compare frameworks when relevant. "
        "If information is insufficient, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )

    resp = client.chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.3,
    )

    return {
        "messages": [{"role": "assistant", "content": resp.choices[0].message.content}],
        "next_agent": END,
    }


def should_continue(state: AgentState) -> str:
    return state.get("next_agent", "generate")


def build_agent(embedder, sparse_embedder):
    client = OpenAI(base_url=settings.groq_base_url, api_key=settings.groq_api_key)

    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", lambda s: supervisor_node(s, client))
    workflow.add_node("vector_agent", lambda s: vector_agent_node(s, embedder, sparse_embedder))
    workflow.add_node("graph_agent", lambda s: graph_agent_node(s))
    workflow.add_node("comparison_agent", lambda s: comparison_agent_node(s, embedder, sparse_embedder))
    workflow.add_node("generate", lambda s: generate_node(s, client))

    workflow.set_entry_point("supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "vector_agent": "vector_agent",
            "graph_agent": "graph_agent",
            "comparison_agent": "comparison_agent",
            "generate": "generate",
        },
    )

    for agent in ["vector_agent", "graph_agent", "comparison_agent"]:
        workflow.add_edge(agent, "generate")

    workflow.add_edge("generate", END)

    return workflow.compile()


def run_agent(query: str, embedder, sparse_embedder) -> dict:
    app = build_agent(embedder, sparse_embedder)
    initial = AgentState(
        query=query,
        messages=[{"role": "user", "content": query}],
        vector_results=[],
        graph_results=[],
        comparison_results=[],
        next_agent="supervisor",
    )
    result = app.invoke(initial)
    return result
