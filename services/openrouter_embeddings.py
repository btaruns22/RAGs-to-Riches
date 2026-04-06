"""OpenRouter embedding client used for local Chroma retrieval."""
import os
from typing import List

import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def get_embedding_model() -> str:
    return os.getenv("OPENROUTER_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def embed_texts(texts: List[str], model: str | None = None) -> List[List[float]]:
    """Embed a batch of texts using the OpenRouter embeddings API."""
    api_key = require_env("OPENROUTER_API_KEY")
    selected_model = model or get_embedding_model()

    response = requests.post(
        OPENROUTER_EMBEDDINGS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": selected_model,
            "input": texts,
        },
        timeout=60,
    )
    response.raise_for_status()

    payload = response.json()
    return [item["embedding"] for item in payload["data"]]
