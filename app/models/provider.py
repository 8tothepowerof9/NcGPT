from enum import Enum


class Provider(Enum):
    """
    Enum for different LLM provider types.
    """

    OPENAI = "openai"
    OLLAMA = "ollama"
    VLLM = "vllm"


class EmbedProvider(Enum):
    """
    Enum for different embedding provider types.
    """

    OPENAI = "openai"
    OLLAMA = "ollama"
