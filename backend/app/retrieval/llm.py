"""LLM client abstraction.

Provides :func:`get_llm` which returns a LangChain ``BaseChatModel``
configured from application settings. Supports Ollama (dev) and
OpenAI-compatible APIs (production / DeepSeek).
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Return a cached LLM instance based on ``settings.llm_provider``.

    Supported providers:

    * ``ollama`` — uses ``langchain_ollama.ChatOllama``
    * ``openai``  — uses ``langchain_openai.ChatOpenAI``
    * ``deepseek`` — uses ``langchain_openai.ChatOpenAI`` with DeepSeek base URL

    Returns:
        A LangChain chat model ready for ``ainvoke`` / ``astream``.
    """
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
        )
        logger.info(
            "LLM initialised: Ollama (%s) at %s",
            settings.llm_model,
            settings.ollama_base_url,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
            temperature=settings.llm_temperature,
        )
        logger.info("LLM initialised: OpenAI (%s)", settings.llm_model)

    elif provider == "deepseek":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_api_base,
            temperature=settings.llm_temperature,
        )
        logger.info("LLM initialised: DeepSeek (%s)", settings.llm_model)

    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            "Expected one of: ollama, openai, deepseek."
        )

    return llm
