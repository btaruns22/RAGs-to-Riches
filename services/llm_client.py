"""Shared LLM client configuration for OpenAI or OpenRouter-compatible usage."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

if TYPE_CHECKING:
    from openai import OpenAI


def build_llm_client() -> "OpenAI":
    """Build an OpenAI-compatible client, preferring OpenRouter when configured."""
    from openai import OpenAI

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_api_key:
        return OpenAI(
            api_key=openrouter_api_key,
            base_url=OPENROUTER_API_BASE,
        )

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        return OpenAI(api_key=openai_api_key)

    raise ValueError(
        "Missing LLM API key. Set OPENROUTER_API_KEY for OpenRouter or OPENAI_API_KEY for OpenAI."
    )
