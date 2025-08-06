import os
import torch
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from transformers import AutoModelForMaskedLM, AutoTokenizer

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class Embedder:
    
    @staticmethod
    async def compute_dense_vector(text: str) -> List[float]:
        embeddings = OllamaEmbeddings(
                    model="mxbai-embed-large",
                )
        return embeddings.embed_query(text=text)

    @staticmethod
    async def compute_sparse_vector(text: str) -> Tuple[List[int], List[float]]:
        tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
        embedder = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil")
        tokens = tokenizer(text,
                    max_length=512,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt"
                )
        output = embedder(**tokens)
        logits, attention_mask = output.logits, tokens.attention_mask
        relu_log = torch.log(1 + torch.relu(logits))
        weighted_log = relu_log * attention_mask.unsqueeze(-1)
        max_val, _ = torch.max(weighted_log, dim=1)
        vec = max_val.squeeze()

        indices = torch.nonzero(vec, as_tuple=True)[0].tolist()

        if isinstance(indices, int):
                indices = [indices]

        values = vec[indices].tolist() if indices else []

        return indices, values
    
embedder = Embedder()