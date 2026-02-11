-- Add external_id column to internal.chunks table
-- Enables consistent deletion using source file ID (e.g., Google Drive file ID)
-- instead of doc_id (file name stem) which can be ambiguous.
--
-- Run this migration in Supabase SQL Editor or via psql.

-- Add external_id column (nullable for backwards compatibility with existing rows)
ALTER TABLE internal.chunks
ADD COLUMN IF NOT EXISTS external_id TEXT;

-- Create index on external_id for efficient deletion queries
CREATE INDEX IF NOT EXISTS idx_chunks_external_id
ON internal.chunks (tenant_id, external_id)
WHERE external_id IS NOT NULL;

-- Comment for documentation
COMMENT ON COLUMN internal.chunks.external_id IS 'Source file identifier (e.g., Google Drive file ID) for consistent deletion';
