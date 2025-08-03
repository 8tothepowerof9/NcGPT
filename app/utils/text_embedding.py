import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def compute_dense_vector(text: str = None) -> List[float]:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        # With the `text-embedding-3` class
        # of models, you can specify the size
        # of the embeddings you want returned.
        # dimensions=3072
    )
    return embeddings.embed_query(text=text)
