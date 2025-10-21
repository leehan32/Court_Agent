"""Application state definitions for the B2B legal assistant."""
from __future__ import annotations

from typing import List, Optional, TypedDict


class RetrievedChunk(TypedDict, total=False):
    """Representation of a retrieved text chunk that can be surfaced to the LLM."""

    text: str
    source: str
    chunk_id: Optional[int]


class AssistantState(TypedDict, total=False):
    """Workflow state shared between LangGraph nodes."""

    # Session context
    firm_id: int
    user_id: int
    file_name: str
    mime_type: Optional[str]
    file_bytes: bytes
    enable_ocr: bool

    # Ingestion output
    raw_text: str
    doc_id: Optional[int]
    revision_id: Optional[int]
    chunk_ids: List[int]
    pii_flag: bool

    # Analytical artefacts
    summary: str
    issues: List[str]
    rag_results: List[RetrievedChunk]
    reference_analysis: str
    draft_text: str
    simulation_report: str

    # Optional feedback payload coming from the UI layer
    user_feedback: Optional[str]
    feedback_reason: Optional[str]
    feedback_label: Optional[str]
    feedback_saved: bool

    # Export hooks
    export_path: Optional[str]
