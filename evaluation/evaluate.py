"""
RAGAS evaluation for FrameworkRAG.

Measures:
- Faithfulness: is the answer grounded in the retrieved context?
- Answer Relevancy: how relevant is the answer to the question?
- Context Precision: are the retrieved contexts relevant?

Usage:
  $env:GROQ_API_KEY='...'; python evaluation/evaluate.py
"""
import sys, os, json, time
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "")
if not os.environ["GROQ_API_KEY"]:
    print("ERROR: GROQ_API_KEY not set")
    sys.exit(1)

os.environ["NEO4J_PASSWORD"] = "ragdev123"

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from openai import OpenAI
from fastembed import TextEmbedding, SparseTextEmbedding

from app.core.config import settings
from app.services.qdrant_service import qdrant_service

TEST_SET = [
    {"question": "What is a VectorStore in LangChain and how do I use it?"},
    {"question": "How does LlamaIndex handle document indexing?"},
    {"question": "What retrieval strategies does Haystack support?"},
    {"question": "How do I set up a RAG pipeline in LangChain?"},
    {"question": "Compare the embedding models supported by LangChain and LlamaIndex"},
    {"question": "What is the difference between a retriever and a vector store?"},
    {"question": "How do I integrate OpenAI embeddings in LangChain?"},
    {"question": "What chunking strategies does LlamaIndex recommend?"},
    {"question": "How does Haystack's pipeline architecture work?"},
    {"question": "What is the Query Pipeline in LangChain?"},
]


def retrieve_context(query: str, embedder, sparse_embedder, top_k: int = 5) -> list[str]:
    dense = list(embedder.embed([query]))[0].tolist()
    sp = list(sparse_embedder.embed([query]))[0]
    indices = sp.indices if hasattr(sp, "indices") else sp[0]
    values = sp.values if hasattr(sp, "values") else sp[1]

    # Search default collection first
    try:
        results = qdrant_service.hybrid_search(
            collection_name=settings.qdrant_collection,
            dense_vector=dense,
            sparse_vector=(indices, values),
            limit=top_k,
        )
        return [r.payload.get("content", "") for r in results]
    except Exception as e:
        print(f"    Qdrant error: {e}")
        return []


def generate_answer(query: str, contexts: list[str], client: OpenAI) -> str:
    if not contexts:
        return "No relevant documents found in the knowledge base."
    context_text = "\n\n".join(f"Doc {i+1}: {c[:800]}" for i, c in enumerate(contexts))
    prompt = (
        "Answer the question based ONLY on the provided documents. "
        "If insufficient information, say so.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {query}"
    )
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500,
            )
            return resp.choices[0].message.content
        except Exception as e:
            wait = 60 * (attempt + 1)
            print(f"    LLM error (attempt {attempt+1}), retrying in {wait}s: {e}")
            time.sleep(wait)
    return "Error generating answer."


def main():
    print("[1/4] Loading embedders...")
    embedder = TextEmbedding(model_name=settings.embedding_model)
    sparse_embedder = SparseTextEmbedding(model_name=settings.sparse_model)
    client = OpenAI(base_url=settings.groq_base_url, api_key=settings.groq_api_key)
    print("  OK")

    print(f"[2/4] Processing {len(TEST_SET)} test questions...")
    questions = []
    answers = []
    contexts = []
    for i, item in enumerate(TEST_SET):
        q = item["question"]
        print(f"  [{i+1}/{len(TEST_SET)}] {q[:60]}...")
        ctx = retrieve_context(q, embedder, sparse_embedder)
        ans = generate_answer(q, ctx, client)
        questions.append(q)
        answers.append(ans)
        contexts.append(ctx)
        print(f"    contexts: {len(ctx)} chunks, answer: {len(ans)} chars")

    print("[3/4] Building dataset...")
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    dataset = Dataset.from_dict(data)

    print("[4/4] Running RAGAS evaluation...")
    print("  (this calls Groq API as judge — may take ~2 min)")

    from langchain_openai import ChatOpenAI
    from langchain_core.embeddings import Embeddings

    os.environ["OPENAI_API_KEY"] = settings.groq_api_key

    class FastEmbedLangChain(Embeddings):
        def __init__(self, model):
            self.model = model
        def embed_query(self, text):
            return list(self.model.embed([text]))[0].tolist()
        def embed_documents(self, texts):
            return [list(self.model.embed([t]))[0].tolist() for t in texts]

    fast_embed_model = TextEmbedding(model_name=settings.embedding_model)
    ragas_embedder = FastEmbedLangChain(fast_embed_model)

    judge = ChatOpenAI(
        model="llama-3.1-8b-instant",
        openai_api_key=settings.groq_api_key,
        openai_api_base=settings.groq_base_url,
        temperature=0,
    )

    score = evaluate(
        dataset,
        metrics=[faithfulness],
        llm=judge,
        embeddings=ragas_embedder,
    )

    print("\n" + "=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    try:
        df = score.to_pandas()
        for col in df.columns:
            if col not in ("contexts", "question", "answer"):
                vals = df[col].dropna()
                if len(vals) > 0:
                    numeric_vals = pd.to_numeric(vals, errors="coerce").dropna()
                    if len(numeric_vals) > 0:
                        print(f"  {col}: {numeric_vals.mean():.3f} +/- {numeric_vals.std():.3f}")
                    else:
                        print(f"  {col}: all non-numeric (check API config)")
    except Exception as e:
        print(f"  Could not format results: {e}")
        print(f"  Raw scores: {score}")

    # Save detailed results
    results_path = Path("evaluation/results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    detailed = {
        "scores": str(score),
        "questions": questions,
        "answers": answers,
        "contexts": contexts,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nDetailed results saved to {results_path}")


if __name__ == "__main__":
    main()
