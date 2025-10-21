# 파일명: src/vector_db.py (대대적 수정)
from langchain_community.embeddings import SentenceTransformerEmbeddings
from .file_processor import EMBEDDING_MODEL, EMBEDDING_DIM

# 임베딩 모델 인스턴스 (file_processor와 동일해야 함)
embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

def search_private_chunks(conn, query, top_k=3):
    """
    현재 세션(firm_id)의 '전용 DB'에서 문서 청크를 검색합니다.
    RLS가 적용되므로 firm_id를 명시적으로 WHERE절에 넣을 필요가 없습니다.
    """
    query_vector = embeddings.embed_query(query)
    
    with conn.cursor() as cur:
        # RLS가 활성화된 doc_chunk_embedding 테이블을 쿼리합니다.
        cur.execute(
            f"""
            SELECT c.text, d.title
            FROM doc_chunk_embedding e
            JOIN doc_chunk c ON e.chunk_id = c.chunk_id
            JOIN document d ON c.doc_id = d.doc_id
            WHERE e.model_name = %s
            ORDER BY e.embedding <=> %s::VECTOR({EMBEDDING_DIM})
            LIMIT %s
            """,
            (EMBEDDING_MODEL, query_vector, top_k)
        )
        results = cur.fetchall()
        return [{"text": row[0], "source": f"전용문서: {row[1]}"} for row in results]

def search_public_statutes(conn, query, top_k=2):
    """'공용 법령 DB'에서 관련 조항을 검색합니다."""
    query_vector = embeddings.embed_query(query)
    
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT sa.text, s.title_ko, sa.number
            FROM statute_article_embedding e
            JOIN statute_article sa ON e.article_id = sa.article_id
            JOIN statute s ON sa.statute_id = s.statute_id
            WHERE e.model_name = %s
            ORDER BY e.embedding <=> %s::VECTOR({EMBEDDING_DIM})
            LIMIT %s
            """,
            (EMBEDDING_MODEL, query_vector, top_k)
        )
        results = cur.fetchall()
        return [{"text": row[0], "source": f"법령: {row[1]} {row[2]}"} for row in results]

# TODO: search_public_precedents (판례 검색) 함수도 위와 유사하게 구현
# (init.sql의 precedent_section_embedding 테이블을 쿼리)