from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain.docstore.document import Document
import os

# 사용할 임베딩 모델 설정 (이전과 동일)
model_name = "jhgan/ko-sbert-nli"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# PostgreSQL 연결 정보
# docker-compose.yml에 설정한 값과 동일해야 합니다.
postgres_host = os.getenv("POSTGRES_HOST", "localhost")
postgres_port = os.getenv("POSTGRES_PORT", os.getenv("PGPORT", "5433"))
postgres_db = os.getenv("POSTGRES_DB", "vectordb")
postgres_user = os.getenv("POSTGRES_USER", "user")
postgres_password = os.getenv("POSTGRES_PASSWORD", "password")

connection_string = (
    f"postgresql+psycopg2://{postgres_user}:{postgres_password}"
    f"@{postgres_host}:{postgres_port}/{postgres_db}"
)
collection_name = "agent_court_cases"

# PGVector 스토어 객체 생성
# 이 객체를 통해 DB에 접속하고 데이터를 관리합니다.
vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection_string,
    use_jsonb=True,
)


def _is_existing_collection_error(error: Exception) -> bool:
    message = str(error).lower()
    return "already exists" in message or "duplicate" in message


try:
    vector_store.create_collection()
except Exception as e:
    if not _is_existing_collection_error(e):
        raise


def ensure_collection():
    try:
        vector_store.create_collection()
    except Exception as e:
        if _is_existing_collection_error(e):
            return
        raise

def add_case_to_db(case_summary: str, verdict: str, plaintiff_lesson: str, defendant_lesson: str):
    """
    재판이 끝난 사건의 요약과 결과를 PostgreSQL DB에 추가합니다.
    """
    ensure_collection()
    doc = Document(
        page_content=case_summary,
        metadata={
            "verdict": verdict,
            "plaintiff_lesson": plaintiff_lesson,
            "defendant_lesson": defendant_lesson
        }
    )
    vector_store.add_documents([doc])
    print(f"✅ PostgreSQL 벡터 DB에 '{case_summary[:20]}...' 사건이 저장되었습니다.")

def search_similar_cases(query: str, k: int = 2):
    """
    현재 사건과 유사한 과거 사건을 PostgreSQL DB에서 검색합니다.
    """
    try:
        results = vector_store.similarity_search_with_score(query, k=k)
        
        if not results:
            return "유사한 과거 사건을 찾지 못했습니다."
        
        formatted_results = []
        for doc, score in results:
            similarity = (1 - score) * 100
            formatted_results.append(
                f"유사도 {similarity:.2f}% - 사건 요약: {doc.page_content}\n"
                f"  - 최종 판결: {doc.metadata['verdict']}\n"
                f"  - 원고측 교훈: {doc.metadata['plaintiff_lesson']}\n"
                f"  - 피고측 교훈: {doc.metadata['defendant_lesson']}"
            )
        
        return "\n\n".join(formatted_results)
    except Exception as e:
        # DB에 테이블이 아직 없거나 비어있을 때 예외가 발생할 수 있습니다.
        print(f"벡터 DB 검색 중 오류 발생: {e}")
        return "아직 검색할 과거 사건 데이터가 없습니다."
