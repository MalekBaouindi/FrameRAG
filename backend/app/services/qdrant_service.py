from qdrant_client import QdrantClient, models
from app.core.config import settings


BATCH_SIZE = 200


class QdrantService:
    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            https=settings.qdrant_https,
            timeout=120,
        )

    def ensure_collection(self, collection_name: str, vector_size: int = 384):
        collections = self.client.get_collections().collections
        if not any(c.name == collection_name for c in collections):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(full_scan_threshold=10000)
                    )
                },
            )

    def delete_collection(self, collection_name: str):
        self.client.delete_collection(collection_name)
        print(f"  Deleted collection '{collection_name}'", flush=True)

    def add_chunks(self, collection_name: str, chunks: list, dense_embeddings: list, sparse_embeddings: list | None = None):
        total = len(chunks)
        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            batch = chunks[start:end]
            batch_dense = dense_embeddings[start:end]
            batch_sparse = sparse_embeddings[start:end] if sparse_embeddings else None

            points = []
            for i, chunk in enumerate(batch):
                vector = {"": batch_dense[i]}
                if batch_sparse:
                    sp = batch_sparse[i]
                    indices = sp.indices if hasattr(sp, "indices") else sp[0]
                    values = sp.values if hasattr(sp, "values") else sp[1]
                    vector["sparse"] = models.SparseVector(
                        indices=indices.tolist() if hasattr(indices, "tolist") else indices,
                        values=values.tolist() if hasattr(values, "tolist") else values,
                    )
                points.append(models.PointStruct(
                    id=chunk["id"],
                    vector=vector,
                    payload=chunk,
                ))

            self.client.upsert(
                collection_name=collection_name,
                points=points,
            )
            print(f"  Indexed {end}/{total} points into Qdrant", flush=True)

    def search(
        self,
        collection_name: str,
        query_vector: list,
        limit: int = 5,
        query_filter: dict | None = None,
    ):
        return self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
        ).points

    def hybrid_search(
        self,
        collection_name: str,
        dense_vector: list,
        sparse_vector: tuple | None = None,
        limit: int = 5,
        query_filter: dict | None = None,
    ):
        prefetch = [models.Prefetch(query=dense_vector, limit=limit * 2)]
        if sparse_vector:
            indices, values = sparse_vector
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(
                        indices=indices.tolist() if hasattr(indices, "tolist") else indices,
                        values=values.tolist() if hasattr(values, "tolist") else values,
                    ),
                    using="sparse",
                    limit=limit * 2,
                )
            )

        kwargs = {}
        if query_filter:
            kwargs["query_filter"] = query_filter
        results = self.client.query_points(
            collection_name=collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            **kwargs,
        ).points
        return results


qdrant_service = QdrantService()
