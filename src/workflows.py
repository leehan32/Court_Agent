"""High level helpers that orchestrate ingestion and drafting workflows."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .db_utils import ProcessingJobLogger
from .file_processor import ParsedDocument, StoredDocument, ingest_document
from .nodes import drafting_node, rag_chain_node, simulation_node, summarize_node
from .state import AssistantState


def fetch_document_overview(conn, doc_id: int) -> Dict[str, object]:
    """Return basic metadata for a document or raise if it is missing."""

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT doc_id, title, page_count, pii_flag
              FROM document
             WHERE doc_id = %s
            """,
            (doc_id,),
        )
        row = cur.fetchone()
    if not row:
        raise LookupError("문서를 찾을 수 없습니다.")
    return {
        "doc_id": row[0],
        "title": row[1],
        "page_count": row[2],
        "pii_flag": row[3],
    }


def fetch_document_chunks(conn, doc_id: int) -> List[Dict[str, object]]:
    """Return stored chunks for a document ordered by their position."""

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_id, position, text
              FROM doc_chunk
             WHERE doc_id = %s
             ORDER BY position
            """,
            (doc_id,),
        )
        rows = cur.fetchall()

    return [
        {
            "chunk_id": row[0],
            "position": row[1],
            "text": row[2],
        }
        for row in rows
    ]


def fetch_embedding_overview(conn, doc_id: int) -> List[Dict[str, object]]:
    """Return aggregated embedding information per model for a document."""

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.model_name, COUNT(*), MIN(e.created_at), MAX(e.created_at)
              FROM doc_chunk_embedding e
              JOIN doc_chunk c ON c.chunk_id = e.chunk_id
             WHERE c.doc_id = %s
             GROUP BY e.model_name
             ORDER BY e.model_name
            """,
            (doc_id,),
        )
        rows = cur.fetchall()

    return [
        {
            "model_name": row[0],
            "chunk_count": row[1],
            "first_indexed_at": row[2].isoformat() if row[2] else None,
            "last_indexed_at": row[3].isoformat() if row[3] else None,
        }
        for row in rows
    ]


def _run_generation_pipeline(state: AssistantState, include_simulation: bool = True) -> AssistantState:
    """Run the summarize → rag → draft → simulate sequence on a state."""

    state = summarize_node(state)
    state = rag_chain_node(state)
    state = drafting_node(state)
    if include_simulation:
        state = simulation_node(state)
    return state


def run_pipeline_for_document(
    conn,
    doc_id: int,
    firm_id: int,
    user_id: int,
    *,
    include_simulation: bool = True,
    tracker: Optional[ProcessingJobLogger] = None,
) -> Tuple[AssistantState, Dict[str, object], List[Dict[str, object]], Optional[int]]:
    """Execute the drafting pipeline for a stored document."""

    overview = fetch_document_overview(conn, doc_id)
    chunks = fetch_document_chunks(conn, doc_id)
    if not chunks:
        raise ValueError("문서 청크가 존재하지 않습니다. 파싱 과정을 확인하세요.")

    state: AssistantState = {
        "firm_id": firm_id,
        "user_id": user_id,
        "doc_id": doc_id,
        "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
        "raw_text": "\n\n".join(chunk["text"] for chunk in chunks),
    }

    draft_job_id: Optional[int] = None
    if tracker:
        draft_job_id = tracker.start_step(
            "draft",
            doc_id=doc_id,
            detail={
                "include_simulation": include_simulation,
                "chunk_count": len(chunks),
            },
        )
    try:
        state = _run_generation_pipeline(state, include_simulation=include_simulation)
    except Exception as exc:
        if tracker and draft_job_id is not None:
            tracker.fail_step(draft_job_id, detail={"error": str(exc)})
        raise

    if tracker and draft_job_id is not None:
        tracker.succeed_step(
            draft_job_id,
            detail={
                "include_simulation": include_simulation,
                "chunk_count": len(chunks),
                "summary_length": len(state.get("summary") or ""),
            },
            doc_id=doc_id,
        )

    return state, overview, chunks, draft_job_id


def ingest_and_run_pipeline(
    conn,
    *,
    firm_id: int,
    user_id: int,
    file_bytes: bytes,
    file_name: str,
    mime_type: Optional[str] = None,
    enable_ocr: bool = False,
    include_simulation: bool = True,
    tracker: Optional[ProcessingJobLogger] = None,
) -> Dict[str, object]:
    """Ingest a document and immediately run the drafting pipeline."""

    workflow_job_id: Optional[int] = None
    if tracker:
        workflow_job_id = tracker.start_step(
            "workflow",
            detail={
                "file_name": file_name,
                "include_simulation": include_simulation,
            },
        )

    try:
        parsed, stored = ingest_document(
            firm_id=firm_id,
            user_id=user_id,
            file_bytes=file_bytes,
            file_name=file_name,
            mime_type=mime_type,
            enable_ocr=enable_ocr,
            conn=conn,
            tracker=tracker,
        )
        state, overview, chunks, draft_job_id = run_pipeline_for_document(
            conn,
            stored.doc_id,
            firm_id,
            user_id,
            include_simulation=include_simulation,
            tracker=tracker,
        )
    except Exception as exc:
        if tracker and workflow_job_id is not None:
            tracker.fail_step(workflow_job_id, detail={"error": str(exc)})
        raise

    if tracker and workflow_job_id is not None:
        tracker.succeed_step(
            workflow_job_id,
            detail={
                "include_simulation": include_simulation,
                "chunk_count": len(chunks),
                "summary_length": len(state.get("summary") or ""),
            },
            doc_id=stored.doc_id,
        )

    return {
        "parsed": parsed,
        "stored": stored,
        "state": state,
        "overview": overview,
        "chunks": chunks,
        "workflow_job_id": workflow_job_id,
        "draft_job_id": draft_job_id,
    }
