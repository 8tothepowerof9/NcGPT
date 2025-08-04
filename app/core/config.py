import os
from dotenv import load_dotenv

load_dotenv()

# Database
QDRANT_URL = os.getenv("QDRANT_URL", None)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)


# LLM Provider
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
EMBEDDER_PROVIDER = os.getenv("EMBEDDER_PROVIDER", "openai")

if LLM_PROVIDER not in ["openai", "ollama", "vllm"]:
    raise ValueError("LLM_PROVIDER must be one of 'openai', 'ollama', or 'vllm'.")

if EMBEDDER_PROVIDER not in ["openai", "ollama"]:
    raise ValueError("EMBEDDER_PROVIDER must be one of 'openai' or 'ollama'")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY must be set when using OpenAI as the LLM provider."
    )

MODEL_NAME = os.getenv("MODEL_NAME", None)

if not MODEL_NAME:
    raise ValueError("MODEL_NAME must be set.")

if EMBEDDER_PROVIDER == "openai" and not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY must be set when using OpenAI as the embedder provider."
    )

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", None)

if not EMBEDDING_MODEL_NAME:
    raise ValueError("EMBEDDING_MODEL_NAME must be set.")
