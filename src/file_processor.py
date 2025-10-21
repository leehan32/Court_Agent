"""File ingestion helpers for the B2B legal assistant."""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from pgvector.utils import Vector

try:  # Optional dependency for DOCX parsing
    import docx  # type: ignore
except ImportError:  # pragma: no cover - optional dependency guard
    docx = None

try:  # Optional dependency for HWP parsing
    import pyhwp  # type: ignore
except ImportError:  # pragma: no cover - optional dependency guard
    pyhwp = None

from .db_utils import ProcessingJobLogger, get_db_connection, set_rls_user

# Module-level logger reserved for future debugging hooks (not used directly yet).
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

_embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
    chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
    length_function=len,
)


@dataclass
class ParsedChunk:
    text: str
    position: int
    page_from: Optional[int] = None
    page_to: Optional[int] = None
    heading: Optional[str] = None


@dataclass
class ParsedDocument:
    text: str
    file_ext: str
    mime_type: Optional[str]
    page_count: int
    byte_size: int
    sha256: str
    pii_flag: bool
    chunks: List[ParsedChunk]
    ocr_used: bool = False
    ocr_engine: Optional[str] = None


@dataclass
class StoredDocument:
    doc_id: int
    revision_id: int
    chunk_ids: List[int]
    pii_flag: bool


HANGUL_ID_PATTERN = re.compile(r"\b\d{6}-?[1-4]\d{6}\b")
PHONE_PATTERN = re.compile(r"\b01[0-9]-?\d{3,4}-?\d{4}\b")
ACCOUNT_PATTERN = re.compile(r"\b\d{2,4}-\d{2,4}-\d{2,6}\b")


def _detect_pii(text: str) -> bool:
    return bool(
        HANGUL_ID_PATTERN.search(text)
        or PHONE_PATTERN.search(text)
        or ACCOUNT_PATTERN.search(text)
    )


def _extract_pdf_text(file_bytes: bytes) -> Tuple[str, int]:
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        pages = [page.get_text("text") for page in doc]
        text = "\n".join(page.strip() for page in pages if page.strip())
        return text, doc.page_count


def _extract_docx_text(file_bytes: bytes) -> Tuple[str, int]:
    if docx is None:  # pragma: no cover - optional dependency guard
        raise RuntimeError("python-docx가 설치되어 있지 않습니다. requirements.txt를 확인하세요.")

    document = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [para.text.strip() for para in document.paragraphs if para.text.strip()]
    return "\n".join(paragraphs), max(1, len(paragraphs) // 40)


def _extract_hwp_text(file_bytes: bytes) -> Tuple[str, int]:  # pragma: no cover - hwp files are rare in CI
    if pyhwp is None:
        raise RuntimeError("pyhwp 라이브러리가 설치되어 있지 않습니다. HWP 파일 처리를 위해 설치해주세요.")

    # pyhwp는 파일 경로를 선호하기 때문에 임시 파일을 사용합니다.
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".hwp") as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        doc = pyhwp.HWPDocument(tmp.name)  # type: ignore[attr-defined]
        sections = []
        for section in doc.bodytext.section_list:  # type: ignore[attr-defined]
            for paragraph in section.paragraph_list:  # type: ignore[attr-defined]
                text = "".join(text.text for text in paragraph.text)  # type: ignore[attr-defined]
                if text.strip():
                    sections.append(text.strip())
        return "\n".join(sections), max(1, len(sections) // 40)


def parse_document(
    file_bytes: bytes,
    file_name: str,
    mime_type: Optional[str] = None,
    enable_ocr: bool = False,
) -> ParsedDocument:
    """Parse an uploaded document and return structured text chunks."""

    file_ext = file_name.split(".")[-1].lower()
    if file_ext == "pdf":
        text, page_count = _extract_pdf_text(file_bytes)
    elif file_ext == "docx":
        text, page_count = _extract_docx_text(file_bytes)
    elif file_ext == "hwp":
        text, page_count = _extract_hwp_text(file_bytes)
    elif file_ext == "txt":
        text = file_bytes.decode("utf-8", errors="ignore")
        page_count = max(1, len(text) // 2000)
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {file_ext}")

    if not text.strip():
        if enable_ocr:
            raise NotImplementedError("OCR 엔진 연동이 아직 구현되지 않았습니다.")
        raise ValueError("문서에서 텍스트를 추출할 수 없습니다. OCR 설정을 확인하세요.")

    chunks: List[ParsedChunk] = []
    for idx, chunk_text in enumerate(_text_splitter.split_text(text)):
        chunks.append(ParsedChunk(text=chunk_text, position=idx))

    sha256 = hashlib.sha256(file_bytes).hexdigest()
    pii_flag = _detect_pii(text)

    return ParsedDocument(
        text=text,
        file_ext=file_ext,
        mime_type=mime_type,
        page_count=page_count,
        byte_size=len(file_bytes),
        sha256=sha256,
        pii_flag=pii_flag,
        chunks=chunks,
    )


def store_document(
    conn,
    firm_id: int,
    user_id: int,
    title: str,
    parsed: ParsedDocument,
    *,
    tracker: Optional[ProcessingJobLogger] = None,
) -> StoredDocument:
    """Persist the parsed document and its embeddings into PostgreSQL."""

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO document (firm_id, user_id, title, source_type, mime_type, file_ext, sha256, bytes, page_count, pii_flag)
            VALUES (%s, %s, %s, 'upload', %s, %s, %s, %s, %s, %s)
            RETURNING doc_id
            """,
            (
                firm_id,
                user_id,
                title,
                parsed.mime_type,
                parsed.file_ext,
                parsed.sha256,
                parsed.byte_size,
                parsed.page_count,
                parsed.pii_flag,
            ),
        )
        doc_id = cur.fetchone()[0]

        cur.execute(
            """
            INSERT INTO document_revision (doc_id, revision_no, ocr_used, ocr_engine, parse_log)
            VALUES (%s, 1, %s, %s, %s)
            RETURNING rev_id
            """,
            (
                doc_id,
                parsed.ocr_used,
                parsed.ocr_engine,
                json.dumps({"chunk_count": len(parsed.chunks)}),
            ),
        )
        rev_id = cur.fetchone()[0]

        chunk_ids: List[int] = []
        chunk_job_id = None
        if tracker:
            chunk_job_id = tracker.start_step(
                "chunk",
                doc_id=doc_id,
                detail={"chunk_count": len(parsed.chunks)},
            )
        try:
            for chunk in parsed.chunks:
                cur.execute(
                    """
                    INSERT INTO doc_chunk (doc_id, rev_id, position, page_from, page_to, heading, text)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING chunk_id
                    """,
                    (
                        doc_id,
                        rev_id,
                        chunk.position,
                        chunk.page_from,
                        chunk.page_to,
                        chunk.heading,
                        chunk.text,
                    ),
                )
                chunk_id = cur.fetchone()[0]
                chunk_ids.append(chunk_id)
        except Exception as exc:
            if tracker and chunk_job_id is not None:
                tracker.fail_step(
                    chunk_job_id,
                    detail={"error": str(exc)},
                )
            raise

        if tracker and chunk_job_id is not None:
            tracker.succeed_step(
                chunk_job_id,
                detail={"chunk_count": len(chunk_ids)},
                doc_id=doc_id,
            )

        embeddings = _embeddings.embed_documents([chunk.text for chunk in parsed.chunks])
        embed_job_id = None
        if tracker:
            embed_job_id = tracker.start_step(
                "embed",
                doc_id=doc_id,
                detail={"model": EMBEDDING_MODEL, "chunk_count": len(chunk_ids)},
            )
        try:
            for chunk_id, embedding_vector in zip(chunk_ids, embeddings):
                cur.execute(
                    """
                    INSERT INTO doc_chunk_embedding (chunk_id, model_name, dim, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        chunk_id,
                        EMBEDDING_MODEL,
                        EMBEDDING_DIM,
                        Vector(embedding_vector),
                    ),
                )
        except Exception as exc:
            if tracker and embed_job_id is not None:
                tracker.fail_step(embed_job_id, detail={"error": str(exc)})
            raise

        if tracker and embed_job_id is not None:
            tracker.succeed_step(
                embed_job_id,
                detail={"model": EMBEDDING_MODEL, "chunk_count": len(chunk_ids)},
                doc_id=doc_id,
            )

        cur.execute(
            """
            INSERT INTO private_case_meta (doc_id, firm_id, user_id, case_type, tags)
            VALUES (%s, %s, %s, '미분류', ARRAY['자동업로드'])
            ON CONFLICT (doc_id) DO NOTHING
            """,
            (doc_id, firm_id, user_id),
        )

    conn.commit()
    return StoredDocument(doc_id=doc_id, revision_id=rev_id, chunk_ids=chunk_ids, pii_flag=parsed.pii_flag)


def ingest_document(
    firm_id: int,
    user_id: int,
    file_bytes: bytes,
    file_name: str,
    mime_type: Optional[str] = None,
    enable_ocr: bool = False,
    conn=None,
    *,
    tracker: Optional[ProcessingJobLogger] = None,
) -> Tuple[ParsedDocument, StoredDocument]:
    """High level helper used by LangGraph nodes to ingest a document."""

    parse_job_id = None
    if tracker:
        parse_job_id = tracker.start_step(
            "parse",
            detail={"file_name": file_name, "enable_ocr": enable_ocr},
        )
    try:
        parsed = parse_document(
            file_bytes=file_bytes,
            file_name=file_name,
            mime_type=mime_type,
            enable_ocr=enable_ocr,
        )
    except Exception as exc:
        if tracker and parse_job_id is not None:
            tracker.fail_step(parse_job_id, detail={"error": str(exc)})
        raise
    if tracker and parse_job_id is not None:
        tracker.succeed_step(
            parse_job_id,
            detail={
                "file_name": file_name,
                "page_count": parsed.page_count,
                "chunk_count": len(parsed.chunks),
                "pii_flag": parsed.pii_flag,
            },
        )
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        set_rls_user(conn, firm_id)
        stored = store_document(
            conn,
            firm_id,
            user_id,
            title=file_name,
            parsed=parsed,
            tracker=tracker,
        )
    finally:
        if close_conn:
            conn.close()

    return parsed, stored
