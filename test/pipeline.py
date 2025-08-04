from database import qdrant_worker
from app.utils.helper import helper
from app.utils import compute_dense_vector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.sitemap import SitemapLoader

if __name__ == "__main__":
    COLLECTION_NAME = "dspy"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200
    )
    
    sitemap_loader = SitemapLoader(
        web_path="https://dspy.ai/sitemap.xml",
        parsing_function=helper.compact_text
    )
    
    qdrant_worker.create_collection(collection_name=COLLECTION_NAME)

    docs = sitemap_loader.load()

    chunks = []
    for doc in docs:
        split = text_splitter.split_text(doc.page_content)
        chunks.extend(split)
        if len(chunks) >= 2:
            chunks = chunks[:2]
            break
    
    dense_vectors = [
        compute_dense_vector(text=chunk)
        for chunk in chunks
    ]

    qdrant_worker.add_vectors(
        collection_name=COLLECTION_NAME,
        dense_vectors=dense_vectors,
        texts=chunks
    )
