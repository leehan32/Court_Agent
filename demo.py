# 파일명: demo.py (프로젝트 최상위 폴더에 생성)
import os

import streamlit as st
from dotenv import load_dotenv
from src.db_utils import get_db_connection, set_rls_user
from src.file_processor import EMBEDDING_DIM, ingest_document
from src.vector_db import (
    search_private_chunks,
    search_public_precedents,
    search_public_statutes,
)
from src.llm_client import get_rag_answer

# .env 파일 로드 (OPENAI_API_KEY, DB 접속 정보 등)
load_dotenv()

st.set_page_config(page_title="B2B 법률 AI 어시스턴트 (데모)", layout="wide")
st.title("⚖️ B2B 법률 AI 어시스턴트 (데모)")

# --- 1. 데모용 '펌(Firm)' 로그인 (RLS 시뮬레이션) ---
if 'firm_id' not in st.session_state:
    st.session_state.firm_id = None
    st.session_state.user_id = None
    st.session_state.conn = None

# 데모용 펌 선택
firm_list = {
    "A 로펌 (ID: 1)": {"firm_id": 1, "user_id": 1},
    "B 로펌 (ID: 2)": {"firm_id": 2, "user_id": 2},
}
selected_firm = st.sidebar.selectbox(
    "데모용 펌(로펌)을 선택하세요:",
    list(firm_list.keys()),
    index=None,
    placeholder="로그인할 펌을 선택하세요"
)

provider_options = {
    "기본 설정 (환경변수 기준)": None,
    "OpenAI": "openai",
    "로컬 LLM (Ollama)": "ollama",
}

env_default_provider = os.getenv("LLM_PROVIDER", "").strip().lower() or None
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = env_default_provider

provider_labels = list(provider_options.keys())

try:
    default_index = provider_labels.index(
        next(
            label
            for label, value in provider_options.items()
            if value == st.session_state.llm_provider
        )
    )
except StopIteration:
    default_index = 0

selected_provider_label = st.sidebar.selectbox(
    "사용할 LLM 제공자:",
    provider_labels,
    index=default_index,
)
st.session_state.llm_provider = provider_options[selected_provider_label]

if selected_firm:
    firm_info = firm_list[selected_firm]
    if st.session_state.firm_id != firm_info["firm_id"]:
        st.session_state.firm_id = firm_info["firm_id"]
        st.session_state.user_id = firm_info["user_id"]
        
        # DB 연결 및 RLS 세션 설정
        try:
            if st.session_state.conn:
                st.session_state.conn.close()
            st.session_state.conn = get_db_connection()
            set_rls_user(st.session_state.conn, st.session_state.firm_id)
            st.sidebar.success(f"{selected_firm}으로 로그인되었습니다. (RLS 활성)")
        except Exception as e:
            st.sidebar.error(f"DB 연결 실패: {e}")
            st.session_state.firm_id = None
            
else:
    st.session_state.firm_id = None
    if st.session_state.conn:
        st.session_state.conn.close()
        st.session_state.conn = None

if not st.session_state.firm_id:
    st.info("먼저 사이드바에서 데모용 펌(로펌)을 선택하여 로그인하세요.")
    st.stop()

# --- 2. 파일 업로드 ---
st.header("1. 전용 DB에 파일 업로드")
uploaded_file = st.file_uploader(
    "HWP, DOCX, PDF, TXT 파일을 업로드하세요. (OCR은 미지원)",
    type=["hwp", "docx", "pdf", "txt"]
)

if uploaded_file:
    if st.button(f"'{uploaded_file.name}' 처리 및 임베딩"):
        with st.spinner("파일을 파싱하고 임베딩하여 '전용 DB'에 저장 중입니다... (시간이 걸릴 수 있습니다)"):
            try:
                parsed, stored = ingest_document(
                    firm_id=st.session_state.firm_id,
                    user_id=st.session_state.user_id,
                    file_bytes=uploaded_file.getvalue(),
                    file_name=uploaded_file.name,
                    mime_type=uploaded_file.type,
                    conn=st.session_state.conn,
                )
                st.success(
                    f"파일 처리 완료! (문서 ID: {stored.doc_id}, 청크 수: {len(parsed.chunks)})"
                )
            except Exception as e:
                st.error(f"파일 처리 중 오류 발생: {e}")

# --- 3. RAG 기반 질의응답 ---
st.header("2. RAG 기반 질의응답")
st.info(
    f"업로드한 '전용 문서'와 '공용 법령/판례'를 모두 검색하여 답변합니다. (임베딩 차원: {EMBEDDING_DIM})"
)

query = st.text_input("질문:", placeholder="업로드한 문서의 내용을 요약해줘")

if query:
    with st.spinner("RAG 검색 및 LLM 답변 생성 중..."):
        try:
            # RAG 검색 (전용 DB + 공용 DB)
            private_results = search_private_chunks(st.session_state.conn, query, top_k=3)
            statute_results = search_public_statutes(st.session_state.conn, query, top_k=2)
            precedent_results = search_public_precedents(st.session_state.conn, query, top_k=2)

            context_chunks = private_results + statute_results + precedent_results
            
            if not context_chunks:
                st.warning("관련된 참고 자료를 찾지 못했습니다. LLM이 부정확하게 답변할 수 있습니다.")
            
            # LLM 답변 생성
            answer, context_str = get_rag_answer(
                query,
                context_chunks,
                provider=st.session_state.llm_provider,
            )
            
            st.subheader("AI 답변")
            st.markdown(answer)
            
            with st.expander("AI가 참고한 자료 (RAG Context)"):
                st.text(context_str)
                
        except Exception as e:
            st.error(f"답변 생성 중 오류 발생: {e}")

# --- 4. 전용 DB 문서 목록 (RLS 시연) ---
st.header(f"3. '{selected_firm}'의 전용 문서 목록 (RLS 시연)")
st.text("이 목록은 RLS 정책에 의해 현재 로그인한 펌의 문서만 보여줍니다.")
try:
    with st.session_state.conn.cursor() as cur:
        # RLS가 적용되므로 firm_id를 명시하지 않아도 자동으로 필터링됨
        cur.execute(
            """
            SELECT doc_id, title, file_ext, created_at
            FROM document
            WHERE source_type = 'upload'
            ORDER BY created_at DESC
            LIMIT 10
            """
        )
        docs = cur.fetchall()
        
        if not docs:
            st.info("아직 업로드한 문서가 없습니다.")
        else:
            st.dataframe(docs, column_config={
                "doc_id": "문서 ID",
                "title": "파일명",
                "file_ext": "타입",
                "created_at": "업로드 시간"
            })
except Exception as e:
    st.error(f"문서 목록 로딩 실패: {e}")