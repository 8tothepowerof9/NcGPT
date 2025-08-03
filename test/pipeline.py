from app.utils import compute_dense_vector
from database import create_collection, add_vectors

if __name__ == "__main__":
    COLLECTION_NAME = "NcGPT"

    chunks = [
        "Chunk 1: Introduction to topic / setup",
        "Chunk 2: Core explanation or body content",
        "Chunk 3: Supporting examples or data",
        "Chunk 4: Analysis or interpretation",
        "Chunk 5: Summary or conclusion",
        "My name is Duy",
        "Im 20 years old",
        "Im an intern at FPTSoftware",
        "Today is Sunday",
        "The weather is 24 degree Celsius today"
    ]
    
    create_collection(collection_name=COLLECTION_NAME)

    dense_vectors = [compute_dense_vector(text=chunk) for chunk in chunks]
    texts = [chunk for chunk in chunks]
        
    add_vectors(
        collection_name=COLLECTION_NAME,
        dense_vectors=dense_vectors,
        texts=texts
    )