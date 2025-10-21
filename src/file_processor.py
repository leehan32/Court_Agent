# 파일명: src/file_processor.py (신규)
import io
import fitz  # PyMuPDF
import docx # python-docx
import hwp # pyhwp
import hashlib
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .db_utils import get_db_connection

# 사용할 임베딩 모델 (requirements.txt에 sentence-transformers 필요)
# all-MiniLM-L6-v2는 384 차원입니다.
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 
EMBEDDING_DIM = 384
# !!주의!!: init.sql의 모든 VECTOR(1024)를 VECTOR(384)로 수정해야 합니다.

embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )

def parse_file(file_content, file_name, mime_type):
    """파일 내용과 MIME 타입을 기반으로 텍스트를 추출합니다."""
    text = ""
    page_count = 0
    file_ext = file_name.split('.')[-1].lower()

    if file_ext == 'pdf':
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
            page_count = doc.page_count
    elif file_ext == 'docx':
        doc = docx.Document(io.BytesIO(file_content))
        text = "\n".join([para.text for para in doc.paragraphs])
        page_count = 1 # docx는 페이지 계산이 복잡하므로 데모에선 1로 통일
    elif file_ext == 'hwp':
        # pyhwp는 BytesIO를 직접 지원하지 않을 수 있습니다. 
        # 데모를 위해 임시 파일 저장이 필요할 수 있으나, BytesIO를 먼저 시도합니다.
        try:
            # pyhwp 사용 시 임시 파일 저장이 더 안정적일 수 있습니다.
            # with open("temp.hwp", "wb") as f:
            #     f.write(file_content)
            # hwp_doc = hwp.open("temp.hwp")
            # text = hwp_doc.get_text()
            # os.remove("temp.hwp")
            
            hwp_doc = hwp.open(io.BytesIO(file_content))
            text = hwp_doc.get_text()
            page_count = 1 # hwp도 페이지 계산이 복잡
        except Exception as e:
            print(f"HWP 파싱 오류 (pyhwp는 파일 경로/임시 파일 저장이 필요할 수 있음): {e}")
            text = "" # 오류 시 빈 텍스트 반환
            
    elif file_ext == 'txt':
        try:
            text = file_content.decode('utf-8')
        except UnicodeDecodeError:
            text = file_content.decode('euc-kr', errors='ignore') # 한글 깨짐 방지
        page_count = 1
        
    # TODO: 스캔된 PDF/이미지의 경우 Naver OCR API 호출 로직 추가

    return text, page_count, file_ext

def process_and_embed_file(conn, firm_id, user_id, uploaded_file):
    """
    Streamlit의 UploadedFile 객체를 받아 파싱, 청킹, 임베딩, DB 저장을 수행합니다.
    데모용으로 모든 작업을 동기식으로 처리합니다.
    """
    
    file_content = uploaded_file.getvalue()
    file_name = uploaded_file.name
    mime_type = uploaded_file.type
    
    # 1. 파일 파싱
    text, page_count, file_ext = parse_file(file_content, file_name, mime_type)
    if not text:
        raise ValueError("파일에서 텍스트를 추출할 수 없습니다. (HWP 파싱 오류 또는 빈 파일)")

    # 2. 문서(Document) 레코드 생성
    sha256 = hashlib.sha256(file_content).hexdigest()
    bytes_size = len(file_content)
    
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO document (firm_id, user_id, title, source_type, mime_type, file_ext, sha256, bytes, page_count)
            VALUES (%s, %s, %s, 'upload', %s, %s, %s, %s, %s)
            RETURNING doc_id
            """,
            (firm_id, user_id, file_name, mime_type, file_ext, sha256, bytes_size, page_count)
        )
        doc_id = cur.fetchone()[0]
        
        # 데모용: 간단한 리비전 생성
        cur.execute(
            """
            INSERT INTO document_revision (doc_id, revision_no, ocr_used)
            VALUES (%s, 1, false) RETURNING rev_id
            """,
            (doc_id,)
        )
        rev_id = cur.fetchone()[0]

        # 3. 청킹
        text_splitter = get_text_splitter()
        chunks = text_splitter.split_text(text)
        
        chunk_ids = []
        for i, chunk_text in enumerate(chunks):
            cur.execute(
                """
                INSERT INTO doc_chunk (doc_id, rev_id, position, text)
                VALUES (%s, %s, %s, %s)
                RETURNING chunk_id
                """,
                (doc_id, rev_id, i, chunk_text)
            )
            chunk_ids.append(cur.fetchone()[0])

        # 4. 임베딩
        chunk_embeddings = embeddings.embed_documents(chunks)
        
        for chunk_id, embedding_vector in zip(chunk_ids, chunk_embeddings):
            # 참고: pgvector는 list를 받습니다.
            cur.execute(
                """
                INSERT INTO doc_chunk_embedding (chunk_id, model_name, dim, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (chunk_id, EMBEDDING_MODEL, EMBEDDING_DIM, embedding_vector)
            )
            
        # 5. 개인화 메타데이터 생성 (데모용)
        cur.execute(
            """
            INSERT INTO private_case_meta (doc_id, firm_id, user_id, case_type, tags)
            VALUES (%s, %s, %s, '미분류', %s)
            """,
            (doc_id, firm_id, user_id, ['신규업로드'])
        )
            
    conn.commit()
    return doc_id, len(chunks)