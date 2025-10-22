"""OCR 기반으로 공용 판례 데이터를 데이터베이스에 적재하는 도구."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings

from src.db_utils import get_db_connection
from src.file_processor import EMBEDDING_DIM, EMBEDDING_MODEL
from src.naver_ocr import NaverOCRError, NaverOCRClient


LOGGER = logging.getLogger("precedent_ingest")
_EMBEDDER: Optional[SentenceTransformerEmbeddings] = None


@dataclass
class PrecedentSection:
    heading: Optional[str]
    text: str
    order: int


_KEYWORD_HEADINGS = (
    "판시사항",
    "판결요지",
    "판결요약",
    "주문",
    "이유",
    "요지",
    "사실관계",
    "판단",
    "쟁점",
    "결론",
    "참고판례",
)


def _load_sidecar_metadata(file_path: Path) -> Dict[str, str]:
    sidecar = file_path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        with sidecar.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                return {str(key): str(value) for key, value in data.items() if value is not None}
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("%s 메타데이터 파일을 읽는 중 오류: %s", sidecar.name, exc)
    return {}


def _parse_filename_metadata(stem: str) -> Dict[str, str]:
    tokens = [token.strip() for token in re.split(r"[_\-]+", stem) if token.strip()]
    court = None
    case_no = None
    title = None
    case_date = None

    for token in tokens:
        if court is None and token.endswith("법원"):
            court = token
            continue
        if case_no is None and re.match(r"\d{4}[가-힣A-Za-z]{1,3}\d+", token):
            case_no = token
            continue
        if case_date is None and re.match(r"\d{4}[./-]\d{1,2}[./-]\d{1,2}", token):
            case_date = token
            continue

    if tokens:
        candidate = tokens[-1]
        if candidate not in {court, case_no, case_date}:
            title = candidate

    metadata: Dict[str, str] = {}
    if court:
        metadata["court"] = court
    if case_no:
        metadata["case_no"] = case_no
    if title:
        metadata["title"] = title
    if case_date:
        metadata["date"] = case_date
    return metadata


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _looks_like_heading(line: str) -> bool:
    condensed = line.replace(" ", "")
    if len(condensed) <= 4 and condensed.endswith(":"):
        return True
    if any(keyword in condensed for keyword in _KEYWORD_HEADINGS):
        return True
    if re.match(r"^제?\d+\.?$", condensed):
        return True
    if re.match(r"^[IVX]+\.", condensed):
        return True
    return False


def _parse_precedent_text(text: str) -> List[PrecedentSection]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sections: List[PrecedentSection] = []
    heading: Optional[str] = None
    body: List[str] = []

    def flush() -> None:
        nonlocal heading, body
        if not body and not heading:
            return
        content = "\n".join(body).strip()
        if not content and heading:
            content = heading
        sections.append(
            PrecedentSection(
                heading=heading,
                text=content,
                order=len(sections),
            )
        )
        heading = None
        body = []

    for line in lines:
        if _looks_like_heading(line):
            flush()
            heading = line
            continue
        body.append(line)

    flush()

    if not sections and lines:
        sections.append(PrecedentSection(heading=None, text="\n".join(lines), order=0))

    return sections


def _embed_texts(texts: Sequence[str]) -> List[List[float]]:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    return _EMBEDDER.embed_documents(list(texts))


def _store_precedent(
    *,
    conn,
    metadata: Dict[str, str],
    sections: Sequence[PrecedentSection],
    source_path: Path,
) -> None:
    if not sections:
        LOGGER.warning("%s: OCR 결과에서 본문을 추출할 수 없어 건너뜁니다.", source_path.name)
        return

    embeddings = _embed_texts([section.text for section in sections])

    case_no = metadata.get("case_no")
    court = metadata.get("court")
    case_date = _parse_date(metadata.get("date"))
    title = metadata.get("title") or source_path.stem
    summary = metadata.get("summary")

    with conn.cursor() as cur:
        case_id = None
        if case_no:
            cur.execute(
                """
                SELECT case_id
                  FROM precedent
                 WHERE case_no = %s AND (court = %s OR (court IS NULL AND %s IS NULL))
                """,
                (case_no, court, court),
            )
            row = cur.fetchone()
            if row:
                case_id = row[0]
                cur.execute(
                    """
                    UPDATE precedent
                       SET court = %s,
                           date = %s,
                           title = %s,
                           summary = %s,
                           meta = %s
                     WHERE case_id = %s
                    """,
                    (
                        court,
                        case_date,
                        title,
                        summary,
                        json.dumps({"source_file": source_path.name, **metadata}),
                        case_id,
                    ),
                )
                cur.execute("DELETE FROM precedent_section WHERE case_id = %s", (case_id,))

        if case_id is None:
            cur.execute(
                """
                INSERT INTO precedent (court, case_no, date, title, summary, meta)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING case_id
                """,
                (
                    court,
                    case_no,
                    case_date,
                    title,
                    summary,
                    json.dumps({"source_file": source_path.name, **metadata}),
                ),
            )
            case_id = cur.fetchone()[0]

        for section, embedding in zip(sections, embeddings):
            cur.execute(
                """
                INSERT INTO precedent_section (case_id, position, heading, text, meta)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING section_id
                """,
                (
                    case_id,
                    section.order,
                    section.heading,
                    section.text,
                    json.dumps({"order": section.order}),
                ),
            )
            section_id = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO precedent_section_embedding (section_id, model_name, dim, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    section_id,
                    EMBEDDING_MODEL,
                    EMBEDDING_DIM,
                    np.asarray(embedding, dtype=np.float32),
                ),
            )

    conn.commit()
    LOGGER.info("%s: %d개의 섹션을 저장했습니다.", source_path.name, len(sections))


def _iter_input_files(directory: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(directory.glob(pattern))


def ingest_directory(
    *,
    directory: Path,
    pattern: str,
    client: NaverOCRClient,
    default_metadata: Dict[str, str],
) -> None:
    conn = get_db_connection()
    try:
        for file_path in _iter_input_files(directory, pattern):
            LOGGER.info("%s OCR 처리 시작", file_path.name)
            try:
                extracted_text = client.infer_file(str(file_path))
            except (ValueError, NaverOCRError) as exc:
                LOGGER.error("%s OCR 실패: %s", file_path.name, exc)
                continue

            if not extracted_text.strip():
                LOGGER.warning("%s OCR 결과가 비어 있어 건너뜁니다.", file_path.name)
                continue

            metadata = dict(default_metadata)
            metadata.update(_parse_filename_metadata(file_path.stem))
            metadata.update(_load_sidecar_metadata(file_path))
            sections = _parse_precedent_text(extracted_text)
            _store_precedent(conn=conn, metadata=metadata, sections=sections, source_path=file_path)
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("기준법예제"),
        help="OCR을 적용할 판례 파일이 위치한 디렉터리",
    )
    parser.add_argument(
        "--pattern",
        default="*.pdf",
        help="처리할 파일 패턴 (glob)",
    )
    parser.add_argument(
        "--default-court",
        default=os.getenv("DEFAULT_PRECEDENT_COURT"),
        help="메타데이터에 사용할 기본 법원명",
    )
    parser.add_argument(
        "--default-summary",
        default=os.getenv("DEFAULT_PRECEDENT_SUMMARY"),
        help="요약문 기본값",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="로깅 레벨",
    )
    parser.add_argument(
        "--invoke-url",
        default=os.getenv("NAVER_OCR_INVOKE_URL"),
        help="Naver OCR 호출 URL (환경변수 우선)",
    )
    parser.add_argument(
        "--secret-key",
        default=os.getenv("NAVER_OCR_SECRET_KEY"),
        help="Naver OCR 시크릿 키",
    )
    parser.add_argument(
        "--api-key-id",
        default=os.getenv("NAVER_OCR_API_KEY_ID"),
        help="Naver API Gateway Key ID",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("NAVER_OCR_API_KEY"),
        help="Naver API Gateway Key",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    client = NaverOCRClient(
        invoke_url=args.invoke_url,
        secret_key=args.secret_key,
        api_key_id=args.api_key_id,
        api_key=args.api_key,
    )

    default_metadata = {
        key: value
        for key, value in {
            "court": args.default_court,
            "summary": args.default_summary,
        }.items()
        if value
    }

    ingest_directory(
        directory=args.input_dir,
        pattern=args.pattern,
        client=client,
        default_metadata=default_metadata,
    )


if __name__ == "__main__":
    main()
