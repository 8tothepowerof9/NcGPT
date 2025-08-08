from enum import Enum
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
from contextlib import contextmanager


class Provider(Enum):
    """
    Enum for different LLM provider types.
    """

    OPENAI = "openai"
    OLLAMA = "ollama_chat"


@dataclass(frozen=True)
class LLMConfig:
    provider: Provider
    model: str
    api_key: Optional[str] = None  # OpenAI etc. Not needed for local Ollama.
    api_base: Optional[str] = None  # e.g. "http://localhost:11434" for Ollama
    extra: Dict[str, Any] = field(default_factory=dict)  # temperature, max_tokens, etc.


class EmbedProvider(Enum):
    """
    Enum for different embedding provider types.
    """

    OPENAI = "openai"
    OLLAMA = "ollama"
