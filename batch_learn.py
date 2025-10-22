"""Utility script to bulk ingest public precedent data into PostgreSQL."""
from __future__ import annotations

import json
import sys
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import numpy as np
from rich import print
from rich.progress import track

from src.db_utils import get_db_connection
from src.file_processor import EMBEDDING_DIM, EMBEDDING_MODEL
from langchain_community.embeddings import SentenceTransformerEmbeddings


embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)


def _load_cases(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _parse_date(value: str | None):
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def ingest_precedents(cases: Iterable[dict]) -> None:
    with closing(get_db_connection()) as conn:
        with conn.cursor() as cur:
            for case in track(list(cases), description="Ingesting precedents"):
                cur.execute(
                    """
                    INSERT INTO precedent (court, case_no, date, title, summary, meta)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING case_id
                    """,
                    (
                        case.get("court"),
                        case.get("case_no"),
                        _parse_date(case.get("date")),
                        case.get("title"),
                        case.get("summary"),
                        json.dumps(case.get("meta", {})),
                    ),
                )
                case_id = cur.fetchone()[0]

                sections = case.get("sections") or []
                section_texts = [section.get("text", "") for section in sections]
                vectors = embeddings.embed_documents(section_texts) if section_texts else []

                for idx, section in enumerate(sections):
                    text = section.get("text", "")
                    if not text.strip():
                        continue
                    cur.execute(
                        """
                        INSERT INTO precedent_section (case_id, position, heading, text, meta)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING section_id
                        """,
                        (
                            case_id,
                            idx,
                            section.get("heading"),
                            text,
                            json.dumps(section.get("meta", {})),
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
                            np.asarray(vectors[idx], dtype=np.float32),
                        ),
                    )
        conn.commit()


def main() -> None:
    if len(sys.argv) != 2:
        print("[red]Usage: python batch_learn.py <precedents.jsonl>[/red]")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"[red]파일을 찾을 수 없습니다: {path}[/red]")
        sys.exit(1)

    cases = _load_cases(path)
    if not cases:
        print("[yellow]처리할 사건이 없습니다.[/yellow]")
        return

    print(f"[cyan]총 {len(cases)}건의 판례를 적재합니다.[/cyan]")
    ingest_precedents(cases)
    print("[green]판례 적재가 완료되었습니다.[/green]")


if __name__ == "__main__":
    main()
