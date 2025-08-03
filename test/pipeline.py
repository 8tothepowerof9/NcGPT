from app.utils import compute_dense_vector
from database import create_collection, add_vectors

if __name__ == "__main__":
    COLLECTION_NAME = "NcGPT"

    chunks = [
        "Chunk 1: Introduction to topic / setup",
        "Chunk 2: Core explanation or body content",
        "Chunk 3: Supporting examples or data",
        "Chunk 4: Analysis or interpretation",
        "Chunk 5: Summary or conclusion"
    ]
    
    create_collection(collection_name=COLLECTION_NAME)

    dense_vectors = [compute_dense_vector(text=chunk) for chunk in chunks]
    texts = [chunk for chunk in chunks]
        
    add_vectors(
        collection_name=COLLECTION_NAME,
        dense_vectors=dense_vectors,
        texts=texts
    )