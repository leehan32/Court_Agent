"""LangGraph nodes implementing the assistant pipeline."""
from __future__ import annotations

import json
from contextlib import closing
from typing import List

from langchain_core.messages import BaseMessage

from .agents import (
    draft_chain,
    format_references_for_prompt,
    rag_response_chain,
    summarize_chain,
    simulation_chain,
)
from .db_utils import get_db_connection, set_rls_user
from .file_processor import ingest_document
from .state import AssistantState, RetrievedChunk
from .vector_db import (
    search_private_chunks,
    search_public_precedents,
    search_public_statutes,
)


def _unpack_response(response: BaseMessage | str) -> str:
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        return str(response.content)
    return str(response)


def ingest_node(state: AssistantState) -> AssistantState:
    """Parse the uploaded file and persist it to the database."""

    required_fields = ["firm_id", "user_id", "file_name", "file_bytes"]
    for field in required_fields:
        if field not in state:
            raise ValueError(f"ingest_node requires '{field}' in the state")

    parsed, stored = ingest_document(
        firm_id=state["firm_id"],
        user_id=state["user_id"],
        file_bytes=state["file_bytes"],
        file_name=state["file_name"],
        mime_type=state.get("mime_type"),
        enable_ocr=state.get("enable_ocr", False),
    )

    state["raw_text"] = parsed.text
    state["doc_id"] = stored.doc_id
    state["revision_id"] = stored.revision_id
    state["chunk_ids"] = stored.chunk_ids
    state["pii_flag"] = stored.pii_flag
    return state


def summarize_node(state: AssistantState) -> AssistantState:
    """Produce an executive summary and extract legal issues."""

    document_text = state.get("raw_text")
    if not document_text:
        raise ValueError("summarize_node requires 'raw_text' in the state")

    response = summarize_chain.invoke({"document_text": document_text})
    payload = _unpack_response(response)

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        issues: List[str] = [line.strip("-• ") for line in payload.splitlines() if line.strip()]
        state["summary"] = issues[0] if issues else payload
        state["issues"] = issues[1:] if len(issues) > 1 else issues
        return state

    state["summary"] = data.get("summary", "")
    raw_issues = data.get("issues", [])
    if isinstance(raw_issues, list):
        state["issues"] = [str(item) for item in raw_issues]
    elif isinstance(raw_issues, str):
        state["issues"] = [part.strip() for part in raw_issues.split("\n") if part.strip()]
    else:
        state["issues"] = []
    return state


def rag_chain_node(state: AssistantState) -> AssistantState:
    """Retrieve private and public precedents for the matter."""

    summary = state.get("summary")
    issues = state.get("issues", [])
    if not summary:
        raise ValueError("rag_chain_node requires 'summary' in the state")

    query = f"{summary}\n\n" + "\n".join(issues)
    references: List[RetrievedChunk] = []

    with closing(get_db_connection()) as conn:
        set_rls_user(conn, state["firm_id"])
        references.extend(search_private_chunks(conn, query, top_k=5))
        references.extend(search_public_statutes(conn, query, top_k=3))
        references.extend(search_public_precedents(conn, query, top_k=3))

    state["rag_results"] = references

    reference_strings = tuple(f"{item['source']} => {item['text']}" for item in references)
    if reference_strings:
        response = rag_response_chain.invoke(
            {
                "summary": summary,
                "issues": "\n".join(issues),
                "references": format_references_for_prompt(reference_strings),
            }
        )
        payload = _unpack_response(response)
        try:
            data = json.loads(payload)
            analysis_items = data.get("analysis", [])
            if isinstance(analysis_items, list):
                formatted = []
                for item in analysis_items:
                    source = item.get("source", "출처 미상") if isinstance(item, dict) else str(item)
                    reasoning = item.get("reasoning", "") if isinstance(item, dict) else ""
                    formatted.append(f"{source}: {reasoning}".strip())
                state["reference_analysis"] = "\n".join(formatted)
            else:
                state["reference_analysis"] = payload
        except json.JSONDecodeError:
            state["reference_analysis"] = payload
    else:
        state["reference_analysis"] = "관련 근거를 찾지 못했습니다."

    return state


def drafting_node(state: AssistantState) -> AssistantState:
    """Generate a first draft using retrieved authorities."""

    summary = state.get("summary", "")
    issues = state.get("issues", [])
    reference_analysis = state.get("reference_analysis", "")

    response = draft_chain.invoke(
        {
            "summary": summary,
            "issues": "\n".join(issues),
            "reference_analysis": reference_analysis,
        }
    )
    state["draft_text"] = _unpack_response(response)
    return state


def simulation_node(state: AssistantState) -> AssistantState:
    """Simulate counter arguments and risky precedents."""

    draft_text = state.get("draft_text")
    if not draft_text:
        raise ValueError("simulation_node requires 'draft_text' in the state")

    response = simulation_chain.invoke({"draft_text": draft_text})
    payload = _unpack_response(response)
    try:
        data = json.loads(payload)
        counter = data.get("counter_arguments", [])
        precedents = data.get("risky_precedents", [])
        lines = ["예상 반론:"]
        lines.extend(f"- {item}" for item in counter)
        lines.append("\n불리할 수 있는 판례:")
        lines.extend(f"- {item}" for item in precedents)
        state["simulation_report"] = "\n".join(lines)
    except json.JSONDecodeError:
        state["simulation_report"] = payload
    return state


def feedback_node(state: AssistantState) -> AssistantState:
    """Persist user feedback to the feedback table if provided."""

    feedback = state.get("user_feedback")
    if not feedback:
        state["feedback_saved"] = False
        return state

    doc_id = state.get("doc_id")
    if not doc_id:
        raise ValueError("feedback_node requires 'doc_id' when user_feedback is provided")

    with closing(get_db_connection()) as conn:
        set_rls_user(conn, state["firm_id"])
        chunk_id = state.get("chunk_ids", [None])[0]
        label = state.get("feedback_label", "revise")
        reason = state.get("feedback_reason")
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO feedback (firm_id, user_id, doc_id, chunk_id, reason, revised_text, label)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    state["firm_id"],
                    state["user_id"],
                    doc_id,
                    chunk_id,
                    reason,
                    feedback,
                    label,
                ),
            )
        conn.commit()
    state["feedback_saved"] = True
    return state
