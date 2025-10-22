"""OCR 기반으로 법령 원문을 데이터베이스에 적재하는 도구."""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings

from src.db_utils import get_db_connection
from src.file_processor import EMBEDDING_DIM, EMBEDDING_MODEL
from src.naver_ocr import NaverOCRError, NaverOCRClient


LOGGER = logging.getLogger("statute_ingest")
_EMBEDDER: Optional[SentenceTransformerEmbeddings] = None


@dataclass
class StatuteArticle:
    number: str
    heading: Optional[str]
    text: str
    order: int


def _split_title(stem: str) -> tuple[str, Optional[str]]:
    stem = stem.strip()
    if not stem:
        return stem, None

    # 파일명이 "민법CIVIL ACT" 형태인 경우 한글/영문을 분리한다.
    korean_chars = []
    latin_chars = []
    encountered_latin = False
    for char in stem:
        if not encountered_latin and ("A" <= char <= "Z" or "a" <= char <= "z"):
            encountered_latin = True
        if encountered_latin:
            latin_chars.append(char)
        else:
            korean_chars.append(char)

    ko = "".join(korean_chars).strip()
    en = "".join(latin_chars).strip() or None
    return ko or stem, en


def _parse_statute_text(text: str) -> List[StatuteArticle]:
    import re

    cleaned_lines = [line.strip() for line in text.splitlines() if line.strip()]
    normalized_text = "\n".join(cleaned_lines)

    article_pattern = re.compile(r"(?=제\s*\d+[^\s]*\s*조)")
    matches = list(article_pattern.finditer(normalized_text))

    articles: List[StatuteArticle] = []

    if not matches:
        if normalized_text:
            articles.append(
                StatuteArticle(number="전문", heading=None, text=normalized_text, order=0)
            )
        return articles

    preface = normalized_text[: matches[0].start()].strip()
    order_counter = 0
    if preface:
        articles.append(StatuteArticle(number="전문", heading=None, text=preface, order=order_counter))
        order_counter += 1

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized_text)
        raw_article = normalized_text[start:end].strip()
        if not raw_article:
            continue

        lines = [line.strip() for line in raw_article.splitlines() if line.strip()]
        if not lines:
            continue

        header_line = lines[0]
        header_match = re.match(r"^(제\s*\d+[^\s]*\s*조)\s*(\([^)]*\))?\s*(.*)$", header_line)

        if header_match:
            number = header_match.group(1).replace(" ", "")
            heading = (header_match.group(2) or "").strip() or None
            rest_of_line = (header_match.group(3) or "").strip()
            body_lines = []
            if rest_of_line:
                body_lines.append(rest_of_line)
            body_lines.extend(lines[1:])
        else:
            number = "전문"
            heading = None
            body_lines = lines

        body_text = "\n".join(body_lines).strip()
        articles.append(
            StatuteArticle(
                number=number,
                heading=heading,
                text=body_text or header_line,
                order=order_counter,
            )
        )
        order_counter += 1

    return articles


def _embed_texts(texts: Sequence[str]) -> List[List[float]]:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    return _EMBEDDER.embed_documents(list(texts))


def _store_statute(
    *,
    conn,
    title_ko: str,
    title_en: Optional[str],
    version_label: str,
    articles: Sequence[StatuteArticle],
    effective_date: Optional[date],
    promulgation_date: Optional[date],
) -> None:
    if not articles:
        LOGGER.warning("%s: OCR 결과에서 조문을 추출할 수 없어 건너뜁니다.", title_ko)
        return

    embeddings = _embed_texts([article.text for article in articles])

    with conn.cursor() as cur:
        cur.execute("SELECT statute_id FROM statute WHERE title_ko = %s", (title_ko,))
        row = cur.fetchone()
        if row:
            statute_id = row[0]
            cur.execute("DELETE FROM statute_article WHERE statute_id = %s", (statute_id,))
        else:
            cur.execute(
                """
                INSERT INTO statute (law_code, title_ko, title_en)
                VALUES (%s, %s, %s)
                RETURNING statute_id
                """,
                (None, title_ko, title_en),
            )
            statute_id = cur.fetchone()[0]

        article_ids: List[int] = []
        for article, embedding in zip(articles, embeddings):
            cur.execute(
                """
                INSERT INTO statute_article (
                    statute_id,
                    version,
                    effective_date,
                    promulgation_date,
                    number,
                    heading,
                    text,
                    meta
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING article_id
                """,
                (
                    statute_id,
                    version_label,
                    effective_date,
                    promulgation_date,
                    article.number,
                    article.heading,
                    article.text,
                    json.dumps({"order": article.order}),
                ),
            )
            article_id = cur.fetchone()[0]
            article_ids.append(article_id)
            cur.execute(
                """
                INSERT INTO statute_article_embedding (article_id, model_name, dim, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    article_id,
                    EMBEDDING_MODEL,
                    EMBEDDING_DIM,
                    np.asarray(embedding, dtype=np.float32),
                ),
            )

    conn.commit()
    LOGGER.info("%s: %d개의 조문을 저장했습니다.", title_ko, len(article_ids))


def _iter_input_files(directory: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(directory.glob(pattern))


def ingest_directory(
    *,
    directory: Path,
    pattern: str,
    client: NaverOCRClient,
    version_label: str,
    effective_date: Optional[date],
    promulgation_date: Optional[date],
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

            title_ko, title_en = _split_title(file_path.stem)
            articles = _parse_statute_text(extracted_text)
            _store_statute(
                conn=conn,
                title_ko=title_ko,
                title_en=title_en,
                version_label=version_label,
                articles=articles,
                effective_date=effective_date,
                promulgation_date=promulgation_date,
            )
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("기준법예제"),
        help="OCR을 적용할 법령 파일이 위치한 디렉터리",
    )
    parser.add_argument(
        "--pattern",
        default="*.pdf",
        help="처리할 파일 패턴 (glob)",
    )
    parser.add_argument(
        "--version-label",
        default=date.today().isoformat(),
        help="statute_article.version 값으로 사용할 라벨",
    )
    parser.add_argument(
        "--effective-date",
        type=lambda value: date.fromisoformat(value),
        default=None,
        help="시행일 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--promulgation-date",
        type=lambda value: date.fromisoformat(value),
        default=None,
        help="공포일 (YYYY-MM-DD)",
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

    ingest_directory(
        directory=args.input_dir,
        pattern=args.pattern,
        client=client,
        version_label=args.version_label,
        effective_date=args.effective_date,
        promulgation_date=args.promulgation_date,
    )


if __name__ == "__main__":
    main()

