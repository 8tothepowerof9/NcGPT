import asyncio
from app.utils import embedder
from more_itertools import chunked
from database import qdrant_worker
from app.utils.helper import helper
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders.sitemap import SitemapLoader

if __name__ == "__main__":
    COLLECTION_NAME = "test"
    BATCH_SIZE = 100
    
    sitemap_loader = SitemapLoader(
        web_path="https://dspy.ai/sitemap.xml",
        parsing_function=helper.compact_text
    )
    
    qdrant_worker.create_collection(collection_name=COLLECTION_NAME)

    docs = sitemap_loader.load()
    
    texts = [doc.page_content for doc in docs[:5] if doc.page_content.strip()]
    
    lc_semantic_chunker = SemanticChunker(
        embeddings=OllamaEmbeddings(model="nomic-embed-text"),
        min_chunk_size=1000
    )
    
    lc_semantic_chunks = lc_semantic_chunker.create_documents(texts)
    
    # with open("demofile.txt", "w", encoding="utf-8") as f:
    #     f.write("\n\n".join(doc.page_content for doc in docs))
    
    # split_docs = text_splitter.split_documents(docs)
    
    chunks = [doc.page_content for doc in lc_semantic_chunks]
    
    total_chunks = len(chunks)
    print(f"Total chunks to process: {total_chunks}")

    # Process in batches to save memory
    for batch_idx, chunk_batch in enumerate(chunked(chunks, BATCH_SIZE)):
        print(f"Processing batch {batch_idx + 1} of {(total_chunks + BATCH_SIZE - 1) // BATCH_SIZE}")
        
        async def process_chunk_batch(chunk_batch):
            dense_vectors = []
            sparse_vectors = []

            async def process_chunk(i, chunk):
                print(f"  Processing chunk {i+1}/{len(chunk_batch)} in batch {batch_idx + 1}")
                dense = await embedder.compute_dense_vector(text=chunk)
                indices, values = await embedder.compute_sparse_vector(text=chunk)
                return dense, (indices, values)

            tasks = [process_chunk(i, chunk) for i, chunk in enumerate(chunk_batch)]
            results = await asyncio.gather(*tasks)

            for dense, sparse in results:
                dense_vectors.append(dense)
                sparse_vectors.append(sparse)

            return dense_vectors, sparse_vectors

        # Run async processing for the batch
        dense_vectors, sparse_vectors = asyncio.run(process_chunk_batch(chunk_batch))
        
        # Upload immediately and free memory
        print(f"  Uploading batch {batch_idx + 1}...")
        qdrant_worker.add_vectors(
            collection_name=COLLECTION_NAME,
            dense_vectors=dense_vectors,
            texts=chunk_batch,
            sparse_vectors=sparse_vectors
        )
        
        # Clear vectors from memory
        dense_vectors.clear()