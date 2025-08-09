import torch
from typing import Optional, Union, Tuple, List
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from transformers import AutoModelForMaskedLM, AutoTokenizer
from ..models import EmbedProvider
from ..core.config import LLM_PROVIDER, API_KEY, EMBEDDING_MODEL_NAME


class Embedder:
    """
    Embedder class to handle embedding operations using different providers.
    This class is a wrapper around LangChain's embedding clients for OpenAI and Ollama.
    """

    def __init__(
        self,
        provider: Union[EmbedProvider, str] = EmbedProvider(LLM_PROVIDER),
        api_key: Optional[str] = API_KEY,
        **kwargs,
    ):
        """
        Initializes the Embedder with the specified provider and API key.

        Args:
            provider (EmbedProvider, optional): Provider to the embedding model. Defaults to EmbedProvider(LLM_PROVIDER).
            api_key (Optional[str], optional): api key to the embedding model provider. If using local provider, no api key is needed. Defaults to OPENAI_API_KEY.
            kwargs: Additional keyword arguments to pass to the embedding client.

        Raises:
            ValueError: If the provider is not supported.
        """
        if isinstance(provider, str):
            try:
                provider = EmbedProvider(provider)
            except ValueError:
                raise ValueError(f"Unsupported embedding provider: {provider}")

        self.provider = provider
        self.client: Optional[Embeddings] = None

        if provider == EmbedProvider.OPENAI:
            self.client = OpenAIEmbeddings(
                model=EMBEDDING_MODEL_NAME,
                api_key=api_key,
                **kwargs,
            )
        elif provider == EmbedProvider.OLLAMA:
            self.client = OllamaEmbeddings(
                model=EMBEDDING_MODEL_NAME,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def get_vector_size(self) -> int:
        """
        Returns the size of the embedding vector.

        This method retrieves a sample embedding and returns its length, which represents the size of the embedding vector.

        Returns:
            int: The size of the embedding vector.
        """
        sample_embedding = self.client.embed_query(text="Embed Query Example")
        return len(sample_embedding)

    def get_client(self) -> Embeddings:
        """
        Returns the embedding client.

        This method provides access to the embedding client instance, which can be used to perform embedding operations.

        Returns:
            Embeddings: The embedding client instance.
        """
        return self.client

    def compute_sparse_vector(
        self,
        text: str,
        tokenizer_name: str = "naver/splade-cocondenser-ensembledistil",
        embedder_name: str = "naver/splade-cocondenser-ensembledistil",
        max_length: int = 512,
        truncation: bool = True,
        padding: str = "longest",
        **tokenizer_kwargs,
    ) -> Tuple[List[int], List[float]]:
        """
        Computes a sparse vector for the given text using a masked language model.

        Args:
            text (str): The input text to embed.
            tokenizer_name (str): Name of the tokenizer model. Defaults to naver/splade-cocondenser-ensembledistil.
            embedder_name (str): Name of the embedder model. Defaults to naver/splade-cocondenser-ensembledistil.
            max_length (int, optional): Maximum sequence length. Defaults to 512.
            truncation (bool, optional): Whether to truncate sequences. Defaults to True.
            padding (str, optional): Padding strategy. Defaults to "longest".
            **tokenizer_kwargs: Additional keyword arguments for the tokenizer.

        Returns:
            Tuple[List[int], List[float]]: Indices and values of the sparse vector.
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        embedder = AutoModelForMaskedLM.from_pretrained(embedder_name)

        tokens = tokenizer(
            text,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_tensors="pt",
            **tokenizer_kwargs,
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

    def compute_dense_vector(self, text: str) -> List[float]:
        """
        Computes a dense vector for the given text using the embedding client.

        Args:
            text (str): The input text to embed.

        Returns:
            List[float]: The dense vector representation of the input text.
        """
        return self.client.embed_query(text=text)

    def compute_hybrid_vector(
        self,
        text: str,
        tokenizer_name: str = "naver/splade-cocondenser-ensembledistil",
        embedder_name: str = "naver/splade-cocondenser-ensembledistil",
        max_length: int = 512,
        truncation: bool = True,
        padding: str = "longest",
        **tokenizer_kwargs,
    ) -> Tuple[List[float], List[int], List[float]]:
        """
        Computes both dense and sparse vectors for the given text.

        Args:
            text (str): The input text to embed.
            tokenizer_name (str, optional): Name of the tokenizer model. Defaults to "naver/splade-cocondenser-ensembledistil".
            embedder_name (str, optional): Name of the embedder model. Defaults to "naver/splade-cocondenser-ensembledistil".
            max_length (int, optional): Maximum sequence length.
            truncation (bool, optional): Whether to truncate sequences.
            padding (str, optional): Padding strategy.
            **tokenizer_kwargs: Additional keyword arguments for the tokenizer.

        Returns:
            Tuple[List[float], List[int], List[float]]: Dense vector, sparse indices, sparse values.
        """
        dense_vector = self.compute_dense_vector(text=text)
        sparse_indices, sparse_values = self.compute_sparse_vector(
            text=text,
            tokenizer_name=tokenizer_name,
            embedder_name=embedder_name,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            **tokenizer_kwargs,
        )
        return dense_vector, sparse_indices, sparse_values
