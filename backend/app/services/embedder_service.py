from app.core.config import settings


_embedder = None
_sparse_embedder = None
_reranker = None


def get_embedder():
    global _embedder
    if _embedder is None:
        print("Loading dense embedding model (first request)...")
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(model_name=settings.embedding_model)
        print("Dense model ready")
    return _embedder


def get_sparse_embedder():
    if not settings.enable_sparse:
        return None
    global _sparse_embedder
    if _sparse_embedder is None:
        print("Loading sparse embedding model (first request)...")
        from fastembed import SparseTextEmbedding
        _sparse_embedder = SparseTextEmbedding(model_name=settings.sparse_model)
        print("Sparse model ready")
    return _sparse_embedder


def get_reranker():
    if not settings.enable_reranker:
        return None
    global _reranker
    if _reranker is None:
        try:
            print("Loading reranker model (first request)...")
            from fastembed import Rerank
            _reranker = Rerank(model_name=settings.rerank_model)
            print("Reranker ready")
        except ImportError:
            _reranker = None
            print("Rerank model not available")
    return _reranker
