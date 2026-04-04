"""LLM client abstraction.

Provides :func:`get_llm` which returns a LangChain ``BaseChatModel``
configured from application settings. Supports Ollama (dev) and
OpenAI-compatible APIs (production / DeepSeek).

When ``settings.llm_disable_thinking`` is *True*, a thin wrapper
automatically appends ``/nothink`` to the last human message before
each LLM call.  This disables Qwen3's built-in chain-of-thought mode
so that the entire token budget is spent on the ``content`` field
rather than ``reasoning_content``.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage

from app.config import settings

logger = logging.getLogger(__name__)


# ── Thin wrapper that injects /nothink ──────────────────────────────────────

class _NoThinkWrapper(BaseChatModel):
    """Proxy that appends ``/nothink`` to every request.

    Delegates all real work to the wrapped *inner* model.
    """

    inner: BaseChatModel

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return f"nothink-{self.inner._llm_type}"

    @staticmethod
    def _inject(messages: list[BaseMessage]) -> list[BaseMessage]:
        """Return a copy with /nothink appended to the last human message."""
        out: list[BaseMessage] = []
        injected = False
        for msg in reversed(messages):
            if not injected and isinstance(msg, HumanMessage):
                text = msg.content if isinstance(msg.content, str) else str(msg.content)
                if not text.rstrip().endswith("/nothink"):
                    msg = HumanMessage(content=text.rstrip() + "\n/nothink")
                injected = True
            out.append(msg)
        out.reverse()

        # If input was a plain string converted to messages list, handle edge case
        if not injected and out:
            last = out[-1]
            text = last.content if isinstance(last.content, str) else str(last.content)
            out[-1] = type(last)(content=text.rstrip() + "\n/nothink")
        return out

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any) -> Any:
        return self.inner._generate(self._inject(messages), stop=stop, **kwargs)

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any) -> Any:
        return await self.inner._agenerate(self._inject(messages), stop=stop, **kwargs)


# ── Factory ─────────────────────────────────────────────────────────────────

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

    if settings.llm_disable_thinking:
        logger.info("LLM thinking mode disabled — wrapping with /nothink injector")
        llm = _NoThinkWrapper(inner=llm)

    return llm
