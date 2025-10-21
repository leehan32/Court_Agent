-- 확장
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

-- 조직/사용자
CREATE TABLE IF NOT EXISTS firm (
  firm_id        BIGSERIAL PRIMARY KEY,
  name           TEXT NOT NULL,
  created_at     TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS app_user (
  user_id        BIGSERIAL PRIMARY KEY,
  firm_id        BIGINT REFERENCES firm(firm_id),
  email          TEXT UNIQUE NOT NULL,
  name           TEXT,
  role           TEXT CHECK (role IN ('admin','lawyer','staff')) DEFAULT 'lawyer',
  created_at     TIMESTAMPTZ DEFAULT now()
);

-- 업로드 문서(원본 단위)
CREATE TABLE IF NOT EXISTS document (
  doc_id         BIGSERIAL PRIMARY KEY,
  firm_id        BIGINT REFERENCES firm(firm_id),
  user_id        BIGINT REFERENCES app_user(user_id),
  title          TEXT,
  source_type    TEXT CHECK (source_type IN ('upload','statute','precedent','note')) DEFAULT 'upload',
  mime_type      TEXT,
  file_ext       TEXT,
  sha256         TEXT,
  bytes          BIGINT,
  page_count     INT,
  pii_flag       BOOLEAN DEFAULT FALSE,
  created_at     TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_document_firm_created ON document(firm_id, created_at DESC);

-- 전처리 리비전
CREATE TABLE IF NOT EXISTS document_revision (
  rev_id         BIGSERIAL PRIMARY KEY,
  doc_id         BIGINT REFERENCES document(doc_id) ON DELETE CASCADE,
  revision_no    INT NOT NULL,
  ocr_used       BOOLEAN DEFAULT FALSE,
  ocr_engine     TEXT,
  parse_log      JSONB,
  created_at     TIMESTAMPTZ DEFAULT now(),
  UNIQUE(doc_id, revision_no)
);

-- 문단/조항 청크
CREATE TABLE IF NOT EXISTS doc_chunk (
  chunk_id       BIGSERIAL PRIMARY KEY,
  doc_id         BIGINT REFERENCES document(doc_id) ON DELETE CASCADE,
  rev_id         BIGINT REFERENCES document_revision(rev_id) ON DELETE CASCADE,
  position       INT,
  page_from      INT,
  page_to        INT,
  section_tag    TEXT,
  heading        TEXT,
  text           TEXT NOT NULL,
  meta           JSONB,
  created_at     TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_doc_chunk_doc_pos ON doc_chunk(doc_id, position);
CREATE INDEX IF NOT EXISTS idx_doc_chunk_fts ON doc_chunk USING GIN (to_tsvector('simple', text));

-- 임베딩(문서 청크)
CREATE TABLE IF NOT EXISTS doc_chunk_embedding (
  emb_id         BIGSERIAL PRIMARY KEY,
  chunk_id       BIGINT REFERENCES doc_chunk(chunk_id) ON DELETE CASCADE,
  model_name     TEXT NOT NULL,
  dim            INT NOT NULL,
  embedding      VECTOR(384), -- 1024에서 384로 수정!
  created_at     TIMESTAMPTZ DEFAULT now(),
  UNIQUE(chunk_id, model_name)
);
CREATE INDEX IF NOT EXISTS idx_doc_chunk_embedding_vec
  ON doc_chunk_embedding USING ivfflat (embedding vector_cosine_ops);

-- 법령(공용)
CREATE TABLE IF NOT EXISTS statute (
  statute_id     BIGSERIAL PRIMARY KEY,
  law_code       TEXT,
  title_ko       TEXT NOT NULL,
  title_en       TEXT,
  organ          TEXT,
  created_at     TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS statute_article (
  article_id     BIGSERIAL PRIMARY KEY,
  statute_id     BIGINT REFERENCES statute(statute_id) ON DELETE CASCADE,
  version        TEXT NOT NULL,
  effective_date DATE,
  promulgation_date DATE,
  number         TEXT,
  heading        TEXT,
  text           TEXT NOT NULL,
  meta           JSONB,
  created_at     TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_statute_article_version ON statute_article(statute_id, effective_date DESC);

CREATE TABLE IF NOT EXISTS statute_article_embedding (
  emb_id         BIGSERIAL PRIMARY KEY,
  article_id     BIGINT REFERENCES statute_article(article_id) ON DELETE CASCADE,
  model_name     TEXT NOT NULL,
  dim            INT NOT NULL,
  embedding      VECTOR(384), -- 1024에서 384로 수정!
  created_at     TIMESTAMPTZ DEFAULT now(),
  UNIQUE(article_id, model_name)
);
CREATE INDEX IF NOT EXISTS idx_statute_article_embedding_vec
  ON statute_article_embedding USING ivfflat (embedding vector_cosine_ops);

-- 판례(공용)
CREATE TABLE IF NOT EXISTS precedent (
  case_id        BIGSERIAL PRIMARY KEY,
  court          TEXT,
  case_no        TEXT,
  date           DATE,
  title          TEXT,
  summary        TEXT,
  meta           JSONB,
  created_at     TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS precedent_section (
  section_id     BIGSERIAL PRIMARY KEY,
  case_id        BIGINT REFERENCES precedent(case_id) ON DELETE CASCADE,
  position       INT,
  heading        TEXT,
  text           TEXT NOT NULL,
  meta           JSONB
);

CREATE TABLE IF NOT EXISTS precedent_section_embedding (
  emb_id         BIGSERIAL PRIMARY KEY,
  section_id     BIGINT REFERENCES precedent_section(section_id) ON DELETE CASCADE,
  model_name     TEXT NOT NULL,
  dim            INT NOT NULL,
  embedding      VECTOR(384), -- 1024에서 384로 수정!
  created_at     TIMESTAMPTZ DEFAULT now(),
  UNIQUE(section_id, model_name)
);
CREATE INDEX IF NOT EXISTS idx_precedent_section_embedding_vec
  ON precedent_section_embedding USING ivfflat (embedding vector_cosine_ops);

-- 전용 메타/피드백
CREATE TABLE IF NOT EXISTS private_case_meta (
  doc_id         BIGINT PRIMARY KEY REFERENCES document(doc_id) ON DELETE CASCADE,
  firm_id        BIGINT REFERENCES firm(firm_id),
  user_id        BIGINT REFERENCES app_user(user_id),
  case_type      TEXT,
  court          TEXT,
  case_no        TEXT,
  opponent       TEXT,
  tags           TEXT[],
  notes          TEXT,
  created_at     TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_private_case_meta_tags ON private_case_meta USING GIN (tags);

CREATE TABLE IF NOT EXISTS feedback (
  fb_id          BIGSERIAL PRIMARY KEY,
  firm_id        BIGINT REFERENCES firm(firm_id),
  user_id        BIGINT REFERENCES app_user(user_id),
  doc_id         BIGINT REFERENCES document(doc_id) ON DELETE CASCADE,
  chunk_id       BIGINT REFERENCES doc_chunk(chunk_id) ON DELETE SET NULL,
  reason         TEXT,
  revised_text   TEXT,
  label          TEXT CHECK (label IN ('like','dislike','revise')) DEFAULT 'revise',
  created_at     TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS user_style_hint (
  user_id        BIGINT REFERENCES app_user(user_id),
  key            TEXT,
  value          TEXT,
  updated_at     TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (user_id, key)
);

-- 처리 이력
CREATE TABLE IF NOT EXISTS processing_job (
  job_id         BIGSERIAL PRIMARY KEY,
  doc_id         BIGINT REFERENCES document(doc_id),
  step           TEXT CHECK (step IN ('upload','parse','ocr','chunk','embed','done')),
  status         TEXT CHECK (status IN ('queued','running','succeeded','failed')) DEFAULT 'queued',
  detail         JSONB,
  created_at     TIMESTAMPTZ DEFAULT now(),
  updated_at     TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ocr_run (
  ocr_id         BIGSERIAL PRIMARY KEY,
  rev_id         BIGINT REFERENCES document_revision(rev_id),
  engine         TEXT,
  lang           TEXT,
  pages          INT[],
  accuracy_hint  NUMERIC,
  created_at     TIMESTAMPTZ DEFAULT now()
);

-- RLS (firm 격리 예시)
ALTER TABLE document ENABLE ROW LEVEL SECURITY;
ALTER TABLE doc_chunk ENABLE ROW LEVEL SECURITY;
ALTER TABLE private_case_meta ENABLE ROW LEVEL SECURITY;
ALTER TABLE feedback ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS doc_firm_isolation ON document;
CREATE POLICY doc_firm_isolation ON document
  USING (firm_id = current_setting('app.firm_id', true)::BIGINT);

DROP POLICY IF EXISTS chunk_firm_isolation ON doc_chunk;
CREATE POLICY chunk_firm_isolation ON doc_chunk
  USING (
    doc_id IN (SELECT doc_id FROM document
               WHERE firm_id = current_setting('app.firm_id', true)::BIGINT)
  );

DROP POLICY IF EXISTS private_meta_firm_isolation ON private_case_meta;
CREATE POLICY private_meta_firm_isolation ON private_case_meta
  USING (firm_id = current_setting('app.firm_id', true)::BIGGINT);

DROP POLICY IF EXISTS fb_firm_isolation ON feedback;
CREATE POLICY fb_firm_isolation ON feedback
  USING (firm_id = current_setting('app.firm_id', true)::BIGINT);


-- 데모를 위한 초기 데이터
INSERT INTO firm (name) VALUES ('A 로펌'), ('B 로펌') ON CONFLICT DO NOTHING;
INSERT INTO app_user (firm_id, email, name, role) VALUES
(1, 'lawyer1@firm-a.com', '김변호사', 'lawyer'),
(2, 'lawyer2@firm-b.com', '이변호사', 'lawyer') ON CONFLICT DO NOTHING;