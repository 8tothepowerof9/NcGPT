import os
from dotenv import load_dotenv

load_dotenv()

# Database
QDRANT_URL = os.getenv("QDRANT_URL", None)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)


# LLM Provider
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # Use openai or vllm
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY must be set when using OpenAI as the LLM provider."
    )
