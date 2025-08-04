from typing import Optional, Union
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from ..models import EmbedProvider
from ..core.config import LLM_PROVIDER, OPENAI_API_KEY, MODEL_NAME


class Embedder:
    """
    Embedder class to handle embedding operations using different providers.
    This class is a wrapper around LangChain's embedding clients for OpenAI and Ollama.
    """

    def __init__(
        self,
        provider: Union[EmbedProvider, str] = EmbedProvider(LLM_PROVIDER),
        api_key: Optional[str] = OPENAI_API_KEY,
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
                model="text-embedding-3-large",
                api_key=api_key,
                **kwargs,
            )
        elif provider == EmbedProvider.OLLAMA:
            self.client = OllamaEmbeddings(
                model=MODEL_NAME,
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
        sample_embedding = self.client.embed_query(text="Dit Me Duy")
        return len(sample_embedding)

    def get_client(self) -> Embeddings:
        """
        Returns the embedding client.

        This method provides access to the embedding client instance, which can be used to perform embedding operations.

        Returns:
            Embeddings: The embedding client instance.
        """
        return self.client
