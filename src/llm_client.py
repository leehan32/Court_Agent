"""Standalone helper functions for ad-hoc RAG prompts in the demo app."""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .agents import build_chat_model


RAG_PROMPT_TEMPLATE = """
당신은 변호사를 보조하는 전문 법률 AI 어시스턴트입니다.
제시된 [참고 자료]를 바탕으로 변호사의 [질문]에 대해 전문적이고 간결하게 답변하세요.
답변은 반드시 [참고 자료]에 근거해야 하며, 자료에 없는 내용은 추측하지 마세요.

[참고 자료]
{context}

[질문]
{query}

[답변]
"""


def get_rag_answer(query, context_chunks, provider: str | None = None):
    """검색된 RAG 청크와 쿼리를 바탕으로 LLM 답변을 생성합니다.

    Parameters
    ----------
    query:
        사용자의 질문.
    context_chunks:
        RAG 검색으로 수집한 문맥 청크 목록.
    provider:
        선택적으로 사용할 LLM 제공자. ``None``이면 환경변수 설정을 따릅니다.
    """

    context_str = "\n\n---\n\n".join(
        f"출처: {chunk['source']}\n내용: {chunk['text']}" for chunk in context_chunks
    )

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    chain = prompt | build_chat_model(provider_override=provider) | StrOutputParser()

    response = chain.invoke({
        "context": context_str,
        "query": query,
    })
    return response, context_str