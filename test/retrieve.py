import asyncio
from app.utils import embedder
from database import qdrant_worker
from app.utils.helper import helper

if __name__ == "__main__":
    COLLECTION_NAME = "test"
    prompt = "how many cores does dspy have?"
    query = asyncio.run(embedder.compute_dense_vector(text=prompt))
    query_indices, query_values = asyncio.run(embedder.compute_sparse_vector(text=prompt))
    
    points = qdrant_worker.retrieve(
        collection_name=COLLECTION_NAME, 
        query=query,
        query_indices=query_indices,
        query_values=query_values
    )
    
    results = helper.reciprocal_rank_fusion(points=points, payload=["text"], n_points=5)
    
    print()
    for result in results:   
        print(result)
        print("\n")
