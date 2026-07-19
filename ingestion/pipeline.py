"""
Orchestrate the full ingestion pipeline:
1. Scrape docs from a framework
2. Chunk them semantically
3. Generate dense + sparse embeddings
4. Index into Qdrant (dense + sparse vectors)
5. Extract entities into Neo4j
"""
import sys
import argparse
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import numpy as np
from fastembed import TextEmbedding, SparseTextEmbedding
from scraper import scrape_docs
from chunker import process_framework
from app.services.qdrant_service import qdrant_service


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "prithivida/Splade_PP_en_v1"


def generate_dense(chunks: list[dict], batch_size: int = 8) -> list[np.ndarray]:
    model = TextEmbedding(model_name=EMBEDDING_MODEL)
    texts = [chunk["content"] for chunk in chunks]
    all_embeddings = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = list(model.embed(batch))
        all_embeddings.extend(batch_embeddings)
        print(f"  Dense embedding: {min(i + batch_size, total)}/{total}", flush=True)
    return all_embeddings


def generate_sparse(chunks: list[dict], batch_size: int = 8) -> list[tuple[np.ndarray, np.ndarray]]:
    model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    texts = [chunk["content"] for chunk in chunks]
    all_sparse = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        batch_sparse = list(model.embed(batch))
        all_sparse.extend(batch_sparse)
        print(f"  Sparse embedding: {min(i + batch_size, total)}/{total}", flush=True)
    return all_sparse


def index_to_qdrant(collection_name: str, chunks: list[dict], dense: list, sparse: list | None = None, reindex: bool = False):
    if reindex:
        try:
            qdrant_service.delete_collection(collection_name)
        except Exception:
            pass
    vector_size = len(dense[0])
    qdrant_service.ensure_collection(collection_name, vector_size=vector_size)
    qdrant_service.add_chunks(collection_name, chunks, dense, sparse)
    print(f"  Indexed {len(chunks)} points into Qdrant collection '{collection_name}'")


def run_pipeline(framework: str, version: str = "latest", skip_scrape: bool = False, reindex: bool = False):
    if not skip_scrape:
        print(f"Step 1: Scraping {framework} docs...")
        scrape_docs(framework)

    print(f"Step 2: Chunking {framework} docs...")
    chunks = process_framework(framework, version=version)

    collection_name = f"{framework}-{version}"
    print(f"Step 3: Generating dense embeddings...")
    dense = generate_dense(chunks)

    print(f"Step 4: Generating sparse embeddings...")
    sparse = generate_sparse(chunks)

    print(f"Step 5: Indexing into Qdrant (dense + sparse)...")
    index_to_qdrant(collection_name, chunks, dense, sparse, reindex=reindex)

    print(f"Step 6: Extract entities & populate Neo4j...")
    # TODO: implement entity extraction + Neo4j ingestion

    print(f"Pipeline complete for {framework}: {len(chunks)} chunks indexed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("framework", nargs="?", default="langchain")
    parser.add_argument("--version", default="latest")
    parser.add_argument("--skip-scrape", action="store_true")
    parser.add_argument("--reindex", action="store_true", help="Drop and recreate Qdrant collection")
    args = parser.parse_args()
    run_pipeline(args.framework, args.version, args.skip_scrape, args.reindex)
