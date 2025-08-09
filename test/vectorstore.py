from app.services.vectorstore import QdrantCollectionConfig, VectorStore


def main():
    # Initialize vector store
    vs = VectorStore()

    # Define collection config
    config = QdrantCollectionConfig(
        name="test_collection",
        dense_dim=8,  # Use small dim for testing
    )

    # Register and create collection
    vs.register_collection(config, ensure=True)
    print("Collections on server:", vs.list_collections())
    print("Registered collections:", vs.list_registered())

    # Prepare dummy data
    dense_vectors = [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    ]
    sparse_vectors = [
        ([0, 2, 4], [1.0, 0.5, 0.2]),
        ([1, 3, 5], [0.9, 0.4, 0.1]),
    ]
    payloads = [
        {"text": "first point"},
        {"text": "second point"},
    ]

    # Upsert points
    vs.upsert_points(
        collection_name="test_collection",
        dense_vectors=dense_vectors,
        sparse_vectors=sparse_vectors,
        payloads=payloads,
    )
    print("Count after upsert:", vs.count("test_collection"))

    # Hybrid search
    query_dense = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    query_sparse_idx = [0, 2, 4]
    query_sparse_val = [1.0, 0.5, 0.2]
    response = vs.hybrid_search(
        collection_name="test_collection",
        dense_query=query_dense,
        sparse_idx=query_sparse_idx,
        sparse_val=query_sparse_val,
        k=2,
    )

    # Clean up
    vs.delete_collection("test_collection")
    print("Deleted test_collection.")


if __name__ == "__main__":
    main()
