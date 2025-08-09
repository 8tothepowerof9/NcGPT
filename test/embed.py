from app.services import Embedder

if __name__ == "__main__":
    embedder = Embedder()

    try:
        print(embedder.get_vector_size())
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Make sure model is downloaded and ollama is running if using Ollama.")

    # Test compute vectors
    text = "This is a test sentence for embedding."

    embedder.compute_dense_vector(text)
    embedder.compute_sparse_vector(text)
    embedder.compute_hybrid_vector(text)

    print("Embedding operations completed successfully.")
