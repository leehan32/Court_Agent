"""OCR 기반으로 전용(개인) 판례 문서를 적재하는 도구."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from dotenv import load_dotenv

from src.db_utils import get_db_connection, set_rls_user
from src.file_processor import build_parsed_document_from_text, store_document
from src.naver_ocr import NaverOCRError, NaverOCRClient


LOGGER = logging.getLogger("private_precedent_ingest")
DEFAULT_OCR_ENGINE = "naver_clova"


def _load_sidecar_metadata(file_path: Path) -> Dict[str, object]:
    sidecar = file_path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        with sidecar.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                return data
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("%s 메타데이터 파일을 읽는 중 오류: %s", sidecar.name, exc)
    return {}


def _normalise_tags(value: Optional[object]) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value).strip()]


def _iter_input_files(directory: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(directory.glob(pattern))


def _update_private_meta(conn, doc_id: int, metadata: Dict[str, object]) -> None:
    case_type = metadata.get("case_type") or metadata.get("category")
    tags = _normalise_tags(metadata.get("tags"))
    opponent = metadata.get("opponent")
    notes = metadata.get("notes")
    court = metadata.get("court")
    case_no = metadata.get("case_no")

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE private_case_meta
               SET case_type = COALESCE(%s, case_type),
                   court = COALESCE(%s, court),
                   case_no = COALESCE(%s, case_no),
                   opponent = COALESCE(%s, opponent),
                   tags = CASE WHEN %s IS NULL THEN tags ELSE %s END,
                   notes = COALESCE(%s, notes)
             WHERE doc_id = %s
            """,
            (
                case_type,
                court,
                case_no,
                opponent,
                tags,
                tags,
                notes,
                doc_id,
            ),
        )
    conn.commit()


def ingest_directory(
    *,
    directory: Path,
    pattern: str,
    client: NaverOCRClient,
    firm_id: int,
    user_id: int,
    defaults: Dict[str, object],
    ocr_engine: str,
) -> None:
    conn = get_db_connection()
    try:
        set_rls_user(conn, firm_id)
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

            metadata: Dict[str, object] = dict(defaults)
            metadata.update(_load_sidecar_metadata(file_path))

            title = str(metadata.get("title") or file_path.stem)
            try:
                parsed = build_parsed_document_from_text(
                    extracted_text,
                    file_ext="ocr.txt",
                    mime_type="text/plain",
                    ocr_used=True,
                    ocr_engine=ocr_engine,
                )
            except ValueError as exc:
                LOGGER.warning("%s ParsedDocument 생성 실패: %s", file_path.name, exc)
                continue

            stored = store_document(
                conn,
                firm_id,
                user_id,
                title=title,
                parsed=parsed,
                source_type="precedent",
            )

            _update_private_meta(conn, stored.doc_id, metadata)
            LOGGER.info("%s: doc_id=%d, chunk=%d개 저장", file_path.name, stored.doc_id, len(stored.chunk_ids))
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firm-id", type=int, required=True, help="저장할 회사(firm) ID")
    parser.add_argument("--user-id", type=int, required=True, help="업로드를 수행하는 사용자 ID")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("기준법예제"),
        help="OCR을 적용할 파일이 위치한 디렉터리",
    )
    parser.add_argument("--pattern", default="*.pdf", help="처리할 파일 패턴 (glob)")
    parser.add_argument("--default-case-type", default=os.getenv("DEFAULT_PRIVATE_CASE_TYPE"))
    parser.add_argument("--default-court", default=os.getenv("DEFAULT_PRIVATE_COURT"))
    parser.add_argument("--default-opponent", default=os.getenv("DEFAULT_PRIVATE_OPPONENT"))
    parser.add_argument(
        "--default-tags",
        default=os.getenv("DEFAULT_PRIVATE_TAGS"),
        help="콤마로 구분된 태그 기본값",
    )
    parser.add_argument("--default-notes", default=os.getenv("DEFAULT_PRIVATE_NOTES"))
    parser.add_argument("--ocr-engine", default=DEFAULT_OCR_ENGINE, help="저장할 OCR 엔진 이름")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    parser.add_argument("--invoke-url", default=os.getenv("NAVER_OCR_INVOKE_URL"))
    parser.add_argument("--secret-key", default=os.getenv("NAVER_OCR_SECRET_KEY"))
    parser.add_argument("--api-key-id", default=os.getenv("NAVER_OCR_API_KEY_ID"))
    parser.add_argument("--api-key", default=os.getenv("NAVER_OCR_API_KEY"))
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

    defaults: Dict[str, object] = {
        "case_type": args.default_case_type,
        "court": args.default_court,
        "opponent": args.default_opponent,
        "tags": args.default_tags,
        "notes": args.default_notes,
    }
    defaults = {key: value for key, value in defaults.items() if value}

    if "tags" in defaults:
        defaults["tags"] = _normalise_tags(defaults["tags"])

    ingest_directory(
        directory=args.input_dir,
        pattern=args.pattern,
        client=client,
        firm_id=args.firm_id,
        user_id=args.user_id,
        defaults=defaults,
        ocr_engine=args.ocr_engine,
    )


if __name__ == "__main__":
    main()
