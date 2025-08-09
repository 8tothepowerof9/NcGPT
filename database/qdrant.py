import uuid
from typing import List, Tuple
from qdrant_client import QdrantClient, models
from qdrant_client.models import QueryResponse
from app.config import QDRANT_API_KEY, QDRANT_URL

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

class QdrantWorker():
    
    @staticmethod
    def create_collection(
        collection_name: str
    ) -> None:
        
        if not client.collection_exists(collection_name=collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense-embedding": models.VectorParams(
                        size=1024,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse-embedding": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                }
            )

    @staticmethod
    def add_vectors(
        collection_name: str,
        dense_vectors: List[List[float]],
        texts: List[str],
        sparse_vectors: Tuple[List[int], List[float]]
    ) -> None:
        
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense-embedding": dense_vector,
                    "sparse-embedding": models.SparseVector(indices=indices, values=values)},
                payload={"text": text}
            )
            for dense_vector, (indices, values), text in zip(dense_vectors, sparse_vectors, texts)
        ]
        
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        
    @staticmethod
    def retrieve(
        collection_name: str,
        query: List[float],
        query_indices: List[int],
        query_values: List[float]
    ) -> List[QueryResponse]:
        
        search_result = client.query_batch_points(
            collection_name=collection_name,
            requests=[
                models.QueryRequest(
                    query=query,
                    using="dense-embedding",
                    limit=50, 
                    with_payload=True
                ),
                models.QueryRequest(
                    query=models.SparseVector(
                        indices=query_indices,
                        values=query_values,
                    ),
                    limit=50, 
                    with_payload=True,
                    using="sparse-embedding"
                ),
            ]
        )
        
        return search_result
    
qdrant_worker = QdrantWorker()