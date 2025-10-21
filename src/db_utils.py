# 파일명: src/db_utils.py (신규)
import psycopg2
import os

def get_db_connection():
    """DB 연결을 생성합니다."""
    # .env 파일에서 DB 정보를 읽어옵니다. (기존 .env.example 참고)
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        database=os.getenv("POSTGRES_DB", "legal_db"),
        user=os.getenv("POSTGRES_USER", "user"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
        port=os.getenv("POSTGRES_PORT", 5432)
    )
    return conn

def set_rls_user(conn, firm_id):
    """
    현재 DB 세션에 RLS를 위한 firm_id를 설정합니다.
    이것이 B2B 보안의 핵심입니다.
    """
    with conn.cursor() as cur:
        # init.sql의 current_setting('app.firm_id', true)::BIGINT 와 일치해야 함
        cur.execute("SELECT set_config('app.firm_id', %s, false)", (str(firm_id),))
    conn.commit()