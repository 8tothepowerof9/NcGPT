from typing import Optional, Union
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from ..core.config import LLM_PROVIDER, OPENAI_API_KEY, MODEL_NAME
from ..models import Provider


class LLM:
    """
    A service class for interacting with different LLM providers.
    This is a wrapper around LangChain's chat models to provide a unified interface
    for different LLM providers like OpenAI, Ollama, and VLLM.
    """

    def __init__(
        self,
        provider: Union[Provider, str] = Provider(LLM_PROVIDER),
        model: str = MODEL_NAME,
        api_key: Optional[str] = OPENAI_API_KEY,
        **kwargs,
    ):
        """
        Initializes the LLM service with the specified provider, model, and API key.

        Args:
            provider (Provider, optional): LLM Provider. Defaults to LLM_PROVIDER.
            model (str, optional): The name of the model to use. Defaults to MODEL_NAME.
            api_key (Optional[str], optional): The api key of the provider, if use local providers, no api key is needed. Defaults to OPENAI_API_KEY.
            kwargs: Additional keyword arguments to pass to the chat model client.

        Raises:
            NotImplementedError: If the provider is vllm, which is not implemented yet.
            ValueError: If the provider is not supported.
        """
        if isinstance(provider, str):
            try:
                provider = Provider(provider)
            except ValueError:
                raise ValueError(f"Unsupported provider: {provider}")

        self.provider = provider
        self.client: Optional[BaseChatModel] = None

        if provider == Provider.OPENAI:
            self.client = ChatOpenAI(
                model=model,
                api_key=api_key,
                **kwargs,
            )
        elif provider == Provider.OLLAMA:
            self.client = ChatOllama(
                model=model,
                **kwargs,
            )
        elif provider == Provider.VLLM:
            raise NotImplementedError("vllm provider is not implemented yet.")
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def get_client(self) -> BaseChatModel:
        """
        Returns the chat model client.

        Returns:
            BaseChatModel: The chat model client instance.
        """
        return self.client
