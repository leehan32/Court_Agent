"""FastAPI router exposing the MVP workflow endpoints."""
from __future__ import annotations

import io
from contextlib import closing
from typing import Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .db_utils import ProcessingJobLogger, get_db_connection, set_rls_user
from .file_processor import ingest_document
from .workflows import (
    fetch_document_chunks,
    fetch_document_overview,
    fetch_embedding_overview,
    ingest_and_run_pipeline,
    run_pipeline_for_document,
)

try:  # Optional dependency used in /export/docx
    import docx  # type: ignore
except ImportError:  # pragma: no cover - optional dependency guard
    docx = None


class RequestContext(BaseModel):
    firm_id: int
    user_id: int


async def get_request_context(request: Request) -> RequestContext:
    """Extract firm/user identifiers from request headers."""

    firm_header = request.headers.get("X-Firm-Id")
    user_header = request.headers.get("X-User-Id")
    if firm_header is None or user_header is None:
        raise HTTPException(status_code=401, detail="요청 헤더에 X-Firm-Id / X-User-Id 가 필요합니다.")
    try:
        firm_id = int(firm_header)
        user_id = int(user_header)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=401, detail="헤더 값이 올바른 정수가 아닙니다.") from exc
    return RequestContext(firm_id=firm_id, user_id=user_id)


app = FastAPI(title="Court Agent API", version="0.1.0")


class BackgroundWorkflowTracker(ProcessingJobLogger):
    """Tracker that reuses an existing workflow job entry."""

    def __init__(
        self,
        conn,
        firm_id: int,
        user_id: int,
        *,
        workflow_job_id: Optional[int] = None,
    ) -> None:
        super().__init__(conn, firm_id, user_id)
        self._workflow_job_id = workflow_job_id

    def start_step(
        self,
        step: str,
        *,
        doc_id: Optional[int] = None,
        status: str = "running",
        detail: Optional[dict] = None,
    ) -> int:
        if step == "workflow" and self._workflow_job_id is not None:
            job_id = self._workflow_job_id
            self._update_job(job_id, status, detail=detail, doc_id=doc_id)
            if job_id not in self._started_jobs:
                self._started_jobs.append(job_id)
            self._workflow_job_id = None
            return job_id
        return super().start_step(
            step,
            doc_id=doc_id,
            status=status,
            detail=detail,
        )


def _run_workflow_pipeline_task(
    *,
    workflow_job_id: int,
    upload_job_id: int,
    firm_id: int,
    user_id: int,
    file_bytes: bytes,
    file_name: str,
    mime_type: Optional[str],
    enable_ocr: bool,
    include_simulation: bool,
) -> None:
    conn = None
    tracker: Optional[BackgroundWorkflowTracker] = None
    try:
        conn = get_db_connection()
        set_rls_user(conn, firm_id)
        tracker = BackgroundWorkflowTracker(
            conn,
            firm_id,
            user_id,
            workflow_job_id=workflow_job_id,
        )
        try:
            result = ingest_and_run_pipeline(
                conn,
                firm_id=firm_id,
                user_id=user_id,
                file_bytes=file_bytes,
                file_name=file_name,
                mime_type=mime_type,
                enable_ocr=enable_ocr,
                include_simulation=include_simulation,
                tracker=tracker,
            )
        except Exception as exc:
            tracker.fail_step(upload_job_id, detail={"error": str(exc)})
            tracker.cancel_pending()
            return

        stored = result["stored"]
        tracker.succeed_step(
            upload_job_id,
            detail={"bytes": len(file_bytes)},
            doc_id=stored.doc_id,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        if tracker is not None:
            tracker.fail_step(upload_job_id, detail={"error": str(exc)})
            tracker.cancel_pending()
        else:
            try:
                fallback_conn = get_db_connection()
                try:
                    set_rls_user(fallback_conn, firm_id)
                    fallback_tracker = ProcessingJobLogger(
                        fallback_conn, firm_id, user_id
                    )
                    fallback_tracker.fail_step(
                        upload_job_id, detail={"error": str(exc)}
                    )
                    fallback_tracker.fail_step(
                        workflow_job_id, detail={"error": str(exc)}
                    )
                finally:
                    fallback_conn.close()
            except Exception:  # pragma: no cover - best effort logging
                pass
    finally:
        if conn is not None:
            conn.close()


def _open_connection(context: RequestContext):
    conn = get_db_connection()
    set_rls_user(conn, context.firm_id)
    return conn
@app.post("/upload")
async def upload_document(
    context: RequestContext = Depends(get_request_context),
    file: UploadFile = File(...),
    enable_ocr: bool = Form(False),
):
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="업로드된 파일이 비어 있습니다.")

    conn = _open_connection(context)
    tracker = ProcessingJobLogger(conn, context.firm_id, context.user_id)
    upload_job_id = tracker.start_step(
        "upload",
        detail={
            "file_name": file.filename,
            "mime_type": file.content_type,
            "size": len(file_bytes),
        },
    )

    try:
        parsed, stored = ingest_document(
            context.firm_id,
            context.user_id,
            file_bytes,
            file.filename or "uploaded",  # pragma: no cover - UploadFile always has name
            mime_type=file.content_type,
            enable_ocr=enable_ocr,
            conn=conn,
            tracker=tracker,
        )
        tracker.succeed_step(
            upload_job_id,
            detail={"bytes": len(file_bytes)},
            doc_id=stored.doc_id,
        )
        jobs = tracker.list_jobs_for_doc(stored.doc_id)
        return {
            "doc_id": stored.doc_id,
            "revision_id": stored.revision_id,
            "chunk_count": len(stored.chunk_ids),
            "pii_flag": stored.pii_flag,
            "jobs": jobs,
        }
    except ValueError as exc:
        tracker.fail_step(upload_job_id, detail={"error": str(exc)})
        tracker.cancel_pending()
        raise HTTPException(status_code=400, detail=str(exc))
    except NotImplementedError as exc:
        tracker.fail_step(upload_job_id, detail={"error": str(exc)})
        tracker.cancel_pending()
        raise HTTPException(status_code=501, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive guard
        tracker.fail_step(upload_job_id, detail={"error": str(exc)})
        tracker.cancel_pending()
        raise HTTPException(status_code=500, detail="문서를 처리하는 중 오류가 발생했습니다.") from exc
    finally:
        conn.close()


@app.post("/workflow")
async def upload_and_generate_workflow(
    background_tasks: BackgroundTasks,
    context: RequestContext = Depends(get_request_context),
    file: UploadFile = File(...),
    enable_ocr: bool = Form(False),
    include_simulation: bool = Form(True),
):
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="업로드된 파일이 비어 있습니다.")

    conn = _open_connection(context)
    tracker = ProcessingJobLogger(conn, context.firm_id, context.user_id)
    upload_job_id = tracker.start_step(
        "upload",
        detail={
            "file_name": file.filename,
            "mime_type": file.content_type,
            "size": len(file_bytes),
        },
    )

    workflow_job_id = tracker.start_step(
        "workflow",
        status="queued",
        detail={
            "file_name": file.filename or "uploaded",
            "include_simulation": include_simulation,
        },
    )

    conn.close()

    background_tasks.add_task(
        _run_workflow_pipeline_task,
        workflow_job_id=workflow_job_id,
        upload_job_id=upload_job_id,
        firm_id=context.firm_id,
        user_id=context.user_id,
        file_bytes=file_bytes,
        file_name=file.filename or "uploaded",
        mime_type=file.content_type,
        enable_ocr=enable_ocr,
        include_simulation=include_simulation,
    )

    return JSONResponse(status_code=202, content={"job_id": workflow_job_id})


@app.get("/parse/{doc_id}")
def get_parse_result(doc_id: int, context: RequestContext = Depends(get_request_context)):
    with closing(_open_connection(context)) as conn:
        try:
            overview = fetch_document_overview(conn, doc_id)
        except LookupError:
            raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
        chunks = fetch_document_chunks(conn, doc_id)
    return {
        "doc_id": overview["doc_id"],
        "title": overview["title"],
        "page_count": overview["page_count"],
        "pii_flag": overview["pii_flag"],
        "chunk_count": len(chunks),
        "chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "position": chunk["position"],
                "preview": chunk["text"][:200],
            }
            for chunk in chunks
        ],
    }


@app.get("/embed/{doc_id}")
def get_embedding_status(doc_id: int, context: RequestContext = Depends(get_request_context)):
    with closing(_open_connection(context)) as conn:
        try:
            overview = fetch_document_overview(conn, doc_id)
        except LookupError:
            raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
        embeddings = fetch_embedding_overview(conn, doc_id)
    return {
        "doc_id": overview["doc_id"],
        "title": overview["title"],
        "pii_flag": overview["pii_flag"],
        "embeddings": embeddings,
    }


class DraftRequest(BaseModel):
    doc_id: int
    include_simulation: bool = True


@app.post("/draft")
def generate_draft(payload: DraftRequest, context: RequestContext = Depends(get_request_context)):
    with closing(_open_connection(context)) as conn:
        tracker = ProcessingJobLogger(conn, context.firm_id, context.user_id)
        try:
            state, overview, chunks, draft_job_id = run_pipeline_for_document(
                conn,
                payload.doc_id,
                context.firm_id,
                context.user_id,
                include_simulation=payload.include_simulation,
                tracker=tracker,
            )
        except LookupError:
            tracker.cancel_pending()
            raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
        except ValueError as exc:
            tracker.cancel_pending()
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:  # pragma: no cover - defensive guard
            tracker.cancel_pending()
            raise HTTPException(status_code=500, detail="문서 초안을 생성하는 중 오류가 발생했습니다.") from exc

        jobs = tracker.list_jobs_for_doc(payload.doc_id)

    return {
        "doc_id": overview["doc_id"],
        "title": overview["title"],
        "summary": state.get("summary"),
        "issues": state.get("issues", []),
        "rag_results": state.get("rag_results", []),
        "draft_text": state.get("draft_text"),
        "simulation_report": state.get("simulation_report"),
        "draft_job_id": draft_job_id,
        "jobs": jobs,
    }


class FeedbackRequest(BaseModel):
    doc_id: int
    feedback: str
    label: str = "revise"
    reason: Optional[str] = None
    chunk_id: Optional[int] = None


@app.post("/feedback")
def submit_feedback(payload: FeedbackRequest, context: RequestContext = Depends(get_request_context)):
    with closing(_open_connection(context)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO feedback (firm_id, user_id, doc_id, chunk_id, reason, revised_text, label)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING fb_id
                """,
                (
                    context.firm_id,
                    context.user_id,
                    payload.doc_id,
                    payload.chunk_id,
                    payload.reason,
                    payload.feedback,
                    payload.label,
                ),
            )
            fb_id = cur.fetchone()[0]
        conn.commit()
    return JSONResponse(status_code=201, content={"feedback_id": fb_id})


class ExportDocxRequest(BaseModel):
    draft_text: str
    file_name: Optional[str] = None


@app.post("/export/docx")
def export_docx(payload: ExportDocxRequest, context: RequestContext = Depends(get_request_context)):
    if docx is None:  # pragma: no cover - optional dependency guard
        raise HTTPException(status_code=500, detail="python-docx 패키지가 설치되어 있지 않습니다.")

    document = docx.Document()
    for paragraph in payload.draft_text.splitlines():
        document.add_paragraph(paragraph)

    buffer = io.BytesIO()
    document.save(buffer)
    buffer.seek(0)
    filename = payload.file_name or "draft.docx"
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename=\"{filename}\""},
    )


@app.get("/jobs/{job_id}")
def get_job(job_id: int, context: RequestContext = Depends(get_request_context)):
    with closing(_open_connection(context)) as conn:
        logger = ProcessingJobLogger(conn, context.firm_id, context.user_id)
        job = logger.fetch_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    return job

