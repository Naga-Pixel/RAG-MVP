-- FTS Shadow Index: chunks table for Postgres Full-Text Search
-- Phase A: Shadow index only (ingestion writes, retrieval unchanged)
--
-- Run this migration in Supabase SQL Editor or via psql.

-- Create chunks table
CREATE TABLE IF NOT EXISTS chunks (
    tenant_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    title TEXT,
    text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, chunk_id)
);

-- Create GIN index on tsvector for full-text search
-- Using expression index to avoid storing duplicate tsvector column.
-- NOTE: 'english' is a fixed language choice for Phase A.
-- Future phases may make this tenant-aware or configurable.
CREATE INDEX IF NOT EXISTS idx_chunks_fts
ON chunks
USING GIN (to_tsvector('english', text));

-- Index for tenant filtering (common query pattern)
CREATE INDEX IF NOT EXISTS idx_chunks_tenant
ON chunks (tenant_id);

-- Comment for documentation
COMMENT ON TABLE chunks IS 'Shadow FTS index of Qdrant chunks for hybrid retrieval (Phase A, writes only)';
