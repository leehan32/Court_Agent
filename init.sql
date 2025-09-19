-- PostgreSQL initialization script for the vector database
-- Ensures the pgvector extension and LangChain tables exist on startup.

-- Enable pgvector to store embedding vectors.
CREATE EXTENSION IF NOT EXISTS vector;

-- Collection table used by langchain_postgres to group embeddings.
CREATE TABLE IF NOT EXISTS langchain_pg_collection (
    uuid UUID PRIMARY KEY,
    name VARCHAR NOT NULL UNIQUE,
    cmetadata JSON
);

-- Embedding storage table matching langchain_postgres expectations.
CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
    id VARCHAR PRIMARY KEY,
    collection_id UUID REFERENCES langchain_pg_collection (uuid) ON DELETE CASCADE,
    embedding VECTOR,
    document VARCHAR,
    cmetadata JSONB
);

-- Index metadata for efficient filtering by JSON properties.
CREATE INDEX IF NOT EXISTS ix_cmetadata_gin
    ON langchain_pg_embedding USING GIN (cmetadata jsonb_path_ops);

-- Optional helper index to speed up joins on collection references.
CREATE INDEX IF NOT EXISTS ix_langchain_pg_embedding_collection_id
    ON langchain_pg_embedding (collection_id);

