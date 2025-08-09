import asyncio
from app.utils import embedder
from database import qdrant_worker
from app.utils.helper import helper

async def compute_query_vectors(prompt):
    dense_task = embedder.compute_dense_vector(text=prompt)
    sparse_task = embedder.compute_sparse_vector(text=prompt)
    dense_vector, (sparse_indices, sparse_values) = await asyncio.gather(
        dense_task, sparse_task
    )
    return dense_vector, sparse_indices, sparse_values

if __name__ == "__main__":
    COLLECTION_NAME = "dspy"
    prompt = """How can I use dspy.Signature?"""

    query, query_indices, query_values = asyncio.run(embedder.hybrid_embed_query(text=prompt))

    # Retrieve and rank results
    points = qdrant_worker.retrieve(
        collection_name=COLLECTION_NAME, 
        query=query,
        query_indices=query_indices,
        query_values=query_values
    )
    
    results = helper.reciprocal_rank_fusion(points=points, payload=["text"], n_points=3)
    
    print()
    for result in results:   
        print(result)
        print("\n")
