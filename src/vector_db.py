"""Vector search helpers wrapping pgvector queries."""
from __future__ import annotations

from typing import List

from langchain_community.embeddings import SentenceTransformerEmbeddings

from .file_processor import EMBEDDING_DIM, EMBEDDING_MODEL

embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)


def _format_results(rows, source_prefix: str) -> List[dict]:
    formatted = []
    for text, source_label, identifier in rows:
        formatted.append(
            {
                "text": text,
                "source": f"{source_prefix}: {source_label}",
                "chunk_id": identifier,
            }
        )
    return formatted


def search_private_chunks(conn, query: str, top_k: int = 3) -> List[dict]:
    query_vector = embeddings.embed_query(query)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT c.text, d.title, c.chunk_id
            FROM doc_chunk_embedding e
            JOIN doc_chunk c ON e.chunk_id = c.chunk_id
            JOIN document d ON c.doc_id = d.doc_id
            WHERE e.model_name = %s
            ORDER BY e.embedding <=> %s::VECTOR({EMBEDDING_DIM})
            LIMIT %s
            """,
            (EMBEDDING_MODEL, query_vector, top_k),
        )
        rows = cur.fetchall()
    return _format_results(rows, "전용문서")


def search_public_statutes(conn, query: str, top_k: int = 2) -> List[dict]:
    query_vector = embeddings.embed_query(query)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT sa.text, CONCAT(s.title_ko, ' ', COALESCE(sa.number, '')), sa.article_id
            FROM statute_article_embedding e
            JOIN statute_article sa ON e.article_id = sa.article_id
            JOIN statute s ON sa.statute_id = s.statute_id
            WHERE e.model_name = %s
            ORDER BY e.embedding <=> %s::VECTOR({EMBEDDING_DIM})
            LIMIT %s
            """,
            (EMBEDDING_MODEL, query_vector, top_k),
        )
        rows = cur.fetchall()
    return _format_results(rows, "법령")


def search_public_precedents(conn, query: str, top_k: int = 2) -> List[dict]:
    query_vector = embeddings.embed_query(query)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT ps.text, CONCAT(p.court, ' ', COALESCE(p.case_no, '')), ps.section_id
            FROM precedent_section_embedding e
            JOIN precedent_section ps ON e.section_id = ps.section_id
            JOIN precedent p ON ps.case_id = p.case_id
            WHERE e.model_name = %s
            ORDER BY e.embedding <=> %s::VECTOR({EMBEDDING_DIM})
            LIMIT %s
            """,
            (EMBEDDING_MODEL, query_vector, top_k),
        )
        rows = cur.fetchall()
    return _format_results(rows, "판례")
