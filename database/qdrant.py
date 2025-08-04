import os
import uuid
from typing import List
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import QueryResponse

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

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
                    "text-embedding": models.VectorParams(
                        size=3072,
                        distance=models.Distance.COSINE
                    )
                },
            )

    @staticmethod
    def add_vectors(
        collection_name: str,
        dense_vectors: List[List[float]],
        texts: List[str]
    ) -> None:
        
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={"text-embedding": dense_vector},
                payload={"text": text}
            )
            for dense_vector, text in zip(dense_vectors, texts)
        ]
        
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        
    @staticmethod
    def retrieve(
        collection_name: str,
        query: List[float]
    ) -> QueryResponse:
        
        search_result = client.query_points(
            collection_name=collection_name,
            query=query,
            with_payload=True,
            limit=3,
            using="text-embedding"
        ).points
        
        return search_result
    
qdrant_worker = QdrantWorker()