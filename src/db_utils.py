"""Database helpers and ingestion job logging utilities."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import psycopg2


def get_db_connection():
    """Create a new PostgreSQL connection using environment variables."""

    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        database=os.getenv("POSTGRES_DB", "legal_db"),
        user=os.getenv("POSTGRES_USER", "user"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
        port=os.getenv("POSTGRES_PORT", 5432),
    )


def set_rls_user(conn, firm_id: int) -> None:
    """Ensure the session is tagged with the firm_id for RLS policies."""

    with conn.cursor() as cur:
        cur.execute("SELECT set_config('app.firm_id', %s, false)", (str(firm_id),))
    conn.commit()


class ProcessingJobLogger:
    """Convenience wrapper around the processing_job table."""

    def __init__(self, conn, firm_id: int, user_id: Optional[int] = None) -> None:
        self.conn = conn
        self.firm_id = firm_id
        self.user_id = user_id
        self._started_jobs: List[int] = []

    # ------------------------------------------------------------------
    def _serialise_detail(self, detail: Optional[Dict[str, Any]]) -> Optional[str]:
        payload: Dict[str, Any] = {}
        if self.user_id is not None:
            payload["requested_by"] = self.user_id
        if detail:
            payload.update(detail)
        return json.dumps(payload) if payload else None

    def _update_job(
        self,
        job_id: int,
        status: str,
        *,
        detail: Optional[Dict[str, Any]] = None,
        doc_id: Optional[int] = None,
    ) -> None:
        with self.conn.cursor() as cur:
            params: List[Any] = [status]
            query = [
                "UPDATE processing_job",
                "   SET status = %s,",
                "       updated_at = now()",
            ]
            if detail is not None:
                query.append(",       detail = %s")
                params.append(self._serialise_detail(detail))
            if doc_id is not None:
                query.append(",       doc_id = %s")
                params.append(doc_id)
            query.append(" WHERE job_id = %s")
            params.append(job_id)
            cur.execute("\n".join(query), tuple(params))
        self.conn.commit()

    # ------------------------------------------------------------------
    def start_step(
        self,
        step: str,
        *,
        doc_id: Optional[int] = None,
        status: str = "running",
        detail: Optional[Dict[str, Any]] = None,
    ) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO processing_job (firm_id, doc_id, step, status, detail)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING job_id
                """,
                (
                    self.firm_id,
                    doc_id,
                    step,
                    status,
                    self._serialise_detail(detail),
                ),
            )
            job_id = cur.fetchone()[0]
        self.conn.commit()
        self._started_jobs.append(job_id)
        return job_id

    def succeed_step(
        self,
        job_id: int,
        *,
        detail: Optional[Dict[str, Any]] = None,
        doc_id: Optional[int] = None,
    ) -> None:
        self._update_job(job_id, "succeeded", detail=detail, doc_id=doc_id)
        if job_id in self._started_jobs:
            self._started_jobs.remove(job_id)

    def fail_step(
        self,
        job_id: int,
        *,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._update_job(job_id, "failed", detail=detail)
        if job_id in self._started_jobs:
            self._started_jobs.remove(job_id)

    def list_jobs_for_doc(self, doc_id: int) -> List[Dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT job_id, step, status, detail, created_at, updated_at
                  FROM processing_job
                 WHERE doc_id = %s
                 ORDER BY created_at
                """,
                (doc_id,),
            )
            rows = cur.fetchall()

        jobs: List[Dict[str, Any]] = []
        for row in rows:
            detail = row[3]
            if isinstance(detail, str):
                try:
                    detail_payload = json.loads(detail)
                except json.JSONDecodeError:
                    detail_payload = detail
            else:
                detail_payload = detail
            jobs.append(
                {
                    "job_id": row[0],
                    "step": row[1],
                    "status": row[2],
                    "detail": detail_payload,
                    "created_at": row[4].isoformat() if row[4] else None,
                    "updated_at": row[5].isoformat() if row[5] else None,
                }
            )
        return jobs

    def fetch_job(self, job_id: int) -> Optional[Dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT job_id, doc_id, step, status, detail, created_at, updated_at
                  FROM processing_job
                 WHERE job_id = %s
                """,
                (job_id,),
            )
            row = cur.fetchone()

        if not row:
            return None

        detail = row[4]
        if isinstance(detail, str):
            try:
                detail = json.loads(detail)
            except json.JSONDecodeError:
                pass
        return {
            "job_id": row[0],
            "doc_id": row[1],
            "step": row[2],
            "status": row[3],
            "detail": detail,
            "created_at": row[5].isoformat() if row[5] else None,
            "updated_at": row[6].isoformat() if row[6] else None,
        }

    def cancel_pending(self) -> None:
        """Mark every started job as failed to avoid ghost entries."""

        for job_id in list(self._started_jobs):
            self.fail_step(job_id, detail={"error": "작업이 중단되었습니다."})
        self._started_jobs.clear()

