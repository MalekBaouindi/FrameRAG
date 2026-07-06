"""
Entity extraction and Neo4j population.

Strategy:
1. Group chunks by source URL
2. Reconstruct full page content
3. Extract entities via Groq LLM (batched per page)
4. Create nodes & relationships in Neo4j
"""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from openai import OpenAI
from neo4j import GraphDatabase


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY not set")
    sys.exit(1)

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "ragdev123")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

EXTRACTION_PROMPT = """You are analyzing a RAG framework documentation page. Extract entities and relationships.

Return a JSON object with this exact structure:
{
  "entities": {
    "components": [{"name": "...", "description": "..."}],
    "concepts": [{"name": "...", "description": "..."}],
    "integrations": [{"name": "...", "description": "..."}]
  },
  "relationships": [
    {"source": "<component_name>", "target": "<concept_or_integration_name>", "type": "IMPLEMENTS|INTEGRATES_WITH|RELATED_TO"}
  ]
}

Guidelines:
- Components: classes, modules, functions, APIs (e.g., VectorStore, LLMChain, DocumentLoader)
- Concepts: abstract ideas, patterns, techniques (e.g., RAG, Embedding, Chunking, Retrieval)
- Integrations: third-party tools, services, platforms (e.g., OpenAI, Pinecone, Chroma, Weaviate)
- Only include entities explicitly mentioned or clearly described in the text
- Relationships: connect components to concepts they implement, or components to integrations
- Be concise with descriptions (1 sentence max)

Return ONLY valid JSON, no markdown, no explanation.
"""


def group_chunks_by_page():
    frameworks = ["langchain", "llamaindex", "haystack"]
    pages = defaultdict(list)

    for fw in frameworks:
        path = Path("data/chunks") / fw / "chunks.json"
        if not path.exists():
            print(f"  Skipping {fw}: no chunks file")
            continue
        with open(path, encoding="utf-8") as f:
            chunks = json.load(f)
        for c in chunks:
            url = c.get("metadata", {}).get("url", "unknown")
            pages[(fw, url)].append(c)
        print(f"  {fw}: {len(chunks)} chunks, {len(set(c['metadata']['url'] for c in chunks))} unique pages")

    return pages


def reconstruct_page(chunks: list[dict]) -> str:
    sorted_chunks = sorted(chunks, key=lambda c: c.get("id", ""))
    parts = []
    for c in sorted_chunks:
        meta = c.get("metadata", {})
        parts.append(f"[Section: {meta.get('header2', '')}]")
        parts.append(c.get("content", ""))
    return "\n\n".join(parts)


def extract_entities(text: str, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": f"Extract entities from:\n\n{text[:8000]}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"    LLM call failed (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def create_constraints(driver):
    with driver.session() as session:
        for label in ["Component", "Concept", "Integration"]:
            session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.name IS UNIQUE")
        session.run("CREATE INDEX IF NOT EXISTS FOR (n:Chunk) ON (n.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (n:Framework) ON (n.name)")


def populate_graph(driver, framework: str, url: str, page_text: str, chunks: list[dict]):
    with driver.session() as session:
        # Create Framework node
        session.run(
            "MERGE (f:Framework {name: $name, version: $version})",
            name=framework, version="latest",
        )

        # Create Chunk nodes linked to Framework
        for c in chunks:
            session.run(
                """
                MERGE (ch:Chunk {id: $id})
                SET ch.content = $content, ch.url = $url, ch.title = $title
                WITH ch
                MATCH (f:Framework {name: $framework})
                MERGE (ch)-[:BELONGS_TO]->(f)
                """,
                id=c["id"], content=c["content"][:500],
                url=c["metadata"]["url"], title=c["metadata"].get("title", ""),
                framework=framework,
            )

        # Extract entities via LLM
        result = extract_entities(page_text)
        if not result:
            return

        entities = result.get("entities", {})
        relationships = result.get("relationships", [])

        # Create Component nodes
        for comp in entities.get("components", []):
            session.run(
                "MERGE (n:Component {name: $name}) SET n.description = $desc",
                name=comp["name"], desc=comp.get("description", ""),
            )

        # Create Concept nodes
        for concept in entities.get("concepts", []):
            session.run(
                "MERGE (n:Concept {name: $name}) SET n.description = $desc",
                name=concept["name"], desc=concept.get("description", ""),
            )

        # Create Integration nodes
        for integ in entities.get("integrations", []):
            session.run(
                "MERGE (n:Integration {name: $name}) SET n.description = $desc",
                name=integ["name"], desc=integ.get("description", ""),
            )

        # Link Chunks to entities they mention
        all_entity_names = [e["name"] for e in entities.get("components", [])] \
            + [e["name"] for e in entities.get("concepts", [])] \
            + [e["name"] for e in entities.get("integrations", [])]

        for c in chunks:
            for ent_name in all_entity_names:
                session.run(
                    """
                    MATCH (ch:Chunk {id: $chunk_id})
                    MATCH (e {name: $name})
                    MERGE (ch)-[:MENTIONS]->(e)
                    """,
                    chunk_id=c["id"], name=ent_name,
                )

        # Create inter-entity relationships
        for rel in relationships:
            rel_type = rel.get("type", "RELATED_TO")
            session.run(
                """
                MATCH (s {name: $source})
                MATCH (t {name: $target})
                CALL apoc.create.relationship(s, $rel_type, {}, t) YIELD rel
                RETURN rel
                """,
                source=rel["source"], target=rel["target"], rel_type=rel_type,
            )


def main():
    pages = group_chunks_by_page()
    total_pages = sum(len(urls) for urls in pages.values())
    print(f"Total pages to process: {total_pages}")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    create_constraints(driver)

    processed = 0
    errors = 0
    for (framework, url), chunks in sorted(pages.items()):
        page_text = reconstruct_page(chunks)
        try:
            populate_graph(driver, framework, url, page_text, chunks)
            processed += 1
            print(f"  [{processed}/{total_pages}] {framework}: {url[:80]}")
        except Exception as e:
            errors += 1
            print(f"  ERROR [{processed + errors}/{total_pages}] {framework}: {url[:60]} → {e}")

        # Rate limit: Groq free tier ~30 req/min
        if processed % 10 == 0:
            print(f"  Rate limit pause...")
            time.sleep(2)

    driver.close()
    print(f"\nDone: {processed} pages processed, {errors} errors")


if __name__ == "__main__":
    main()
