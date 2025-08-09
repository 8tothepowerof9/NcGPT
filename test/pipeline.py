import asyncio
from app.utils import embedder
from more_itertools import chunked
from database import qdrant_worker
from app.utils.helper import helper
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.text import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

if __name__ == "__main__":
    COLLECTION_NAME = "unsloth"
    BATCH_SIZE = 100
    
    loader = DirectoryLoader(
        path=r"D:\NcGPT\test\{}".format(COLLECTION_NAME),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
        show_progress=True,
    )
    
    qdrant_worker.create_collection(collection_name=COLLECTION_NAME)

    docs = loader.load()
    
    texts = [doc.page_content for doc in docs if doc.page_content.strip()]
    
    # Solution 1: Configure OllamaEmbeddings with specific parameters
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large",
    )
    
    try:
        # Try SemanticChunker first with fixed configuration
        lc_semantic_chunker = SemanticChunker(
            embeddings=embeddings,
            # Add these parameters to control batching
            min_chunk_size=150,       # Initial split size
            breakpoint_threshold_amount=90,
        )
        
        # Process texts one by one instead of in batches
        lc_semantic_chunks = []
        for i, text in enumerate(texts):
            print(f"Processing document {i+1}/{len(texts)} with SemanticChunker")
            try:
                chunks = lc_semantic_chunker.create_documents([text])
                lc_semantic_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing document {i+1} with SemanticChunker: {e}")
                # Fallback to character splitting for problematic documents
                char_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )
                fallback_chunks = char_splitter.create_documents([text])
                lc_semantic_chunks.extend(fallback_chunks)
                
    except Exception as e:
        print(f"SemanticChunker failed completely: {e}")
        print("Falling back to RecursiveCharacterTextSplitter")
        
        # Solution 3: Complete fallback to RecursiveCharacterTextSplitter
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        lc_semantic_chunks = char_splitter.create_documents(texts)
    
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
                try:
                    dense = await embedder.compute_dense_vector(text=chunk)
                    indices, values = await embedder.compute_sparse_vector(text=chunk)
                    return dense, (indices, values)
                except Exception as e:
                    print(f"    Error processing chunk {i+1}: {e}")
                    # Return empty vectors for failed chunks
                    return None, (None, None)

            tasks = [process_chunk(i, chunk) for i, chunk in enumerate(chunk_batch)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    print(f"    Task failed with exception: {result}")
                    continue
                dense, sparse = result
                if dense is not None and sparse[0] is not None:
                    dense_vectors.append(dense)
                    sparse_vectors.append(sparse)

            return dense_vectors, sparse_vectors

        # Run async processing for the batch
        try:
            dense_vectors, sparse_vectors = asyncio.run(process_chunk_batch(chunk_batch))
            
            # Only upload if we have valid vectors
            if dense_vectors and sparse_vectors:
                print(f"  Uploading batch {batch_idx + 1} with {len(dense_vectors)} valid chunks...")
                # Filter chunk_batch to match the number of valid vectors
                valid_chunks = chunk_batch[:len(dense_vectors)]
                qdrant_worker.add_vectors(
                    collection_name=COLLECTION_NAME,
                    dense_vectors=dense_vectors,
                    texts=valid_chunks,
                    sparse_vectors=sparse_vectors
                )
            else:
                print(f"  Skipping batch {batch_idx + 1} - no valid vectors generated")
                
        except Exception as e:
            print(f"  Error processing batch {batch_idx + 1}: {e}")
            continue
        
        # Clear vectors from memory
        if 'dense_vectors' in locals():
            dense_vectors.clear()
        if 'sparse_vectors' in locals():
            sparse_vectors.clear()

    print("Processing complete!")