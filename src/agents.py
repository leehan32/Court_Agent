"""LLM prompt factories and reusable chains for the legal assistant."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Tuple

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

load_dotenv()

__all__ = (
    "build_chat_model",
    "llm",
    "summarize_chain",
    "rag_response_chain",
    "draft_chain",
    "simulation_chain",
    "format_references_for_prompt",
)


def build_chat_model() -> BaseChatModel:
    """Return a chat model instance based on runtime configuration."""

    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    if provider == "openai" or (not provider and os.getenv("OPENAI_API_KEY")):
        from langchain_openai import ChatOpenAI

        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model_name, temperature=temperature)

    # Default to Ollama-compatible local models to satisfy on-prem requirements.
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "llama3")
    try:
        from langchain_community.chat_models import ChatOllama
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "ChatOllama is not available. Install langchain-community with Ollama support "
            "or set LLM_PROVIDER=openai with a valid OPENAI_API_KEY."
        ) from exc

    return ChatOllama(model=model_name, temperature=temperature, base_url=base_url)


@lru_cache(maxsize=1)
def _cached_llm() -> BaseChatModel:
    return build_chat_model()


llm: BaseChatModel = _cached_llm()

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

summarize_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a bilingual Korean legal analyst. Read the provided document text and "
            "return a JSON object with two keys: 'summary' containing a concise Korean summary "
            "and 'issues' containing a list (array) of the top legal issues extracted from the text.",
        ),
        ("human", "{document_text}"),
    ]
)

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are assisting a litigation lawyer. Given a matter description and retrieved legal "
            "references, describe how each citation supports the case. Respond in JSON with the key "
            "'analysis' containing a list of objects with 'source' and 'reasoning'.",
        ),
        (
            "human",
            "Matter summary: {summary}\n\nIssues: {issues}\n\nRetrieved references:\n{references}",
        ),
    ]
)

draft_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You draft formal Korean legal briefs. Using the supplied summary, issues, and annotated "
            "references, produce a structured draft with sections for 사실관계, 쟁점 및 법률근거, and 결론. "
            "Cite each reference inline using [출처 n] notation where n is the index of the reference.",
        ),
        (
            "human",
            "Summary: {summary}\nIssues: {issues}\nReference analysis: {reference_analysis}",
        ),
    ]
)

simulation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You play the role of opposing counsel preparing counter arguments. Review the provided draft "
            "and identify at least two potential rebuttals and unfavourable precedents. Return JSON with keys "
            "'counter_arguments' (list of strings) and 'risky_precedents' (list of strings).",
        ),
        ("human", "Draft to challenge:\n{draft_text}"),
    ]
)


# ---------------------------------------------------------------------------
# Assembled chains
# ---------------------------------------------------------------------------

summarize_chain: Runnable = summarize_prompt | llm
rag_response_chain: Runnable = rag_prompt | llm
draft_chain: Runnable = draft_prompt | llm
simulation_chain: Runnable = simulation_prompt | llm


def format_references_for_prompt(references: Tuple[str, ...]) -> str:
    """Utility to format reference snippets for prompt injection."""

    lines = []
    for idx, ref in enumerate(references, start=1):
        lines.append(f"[{idx}] {ref}")
    return "\n".join(lines)
