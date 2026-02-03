-- Add folder_id and folder_name columns to internal.chunks table
-- For folder-based scoping in retrieval
--
-- Run this migration in Supabase SQL Editor or via psql.

-- Add folder_id column (nullable for backwards compatibility)
ALTER TABLE internal.chunks
ADD COLUMN IF NOT EXISTS folder_id TEXT;

-- Add folder_name column (nullable for backwards compatibility)
ALTER TABLE internal.chunks
ADD COLUMN IF NOT EXISTS folder_name TEXT;

-- Create index on folder_id for efficient filtering
CREATE INDEX IF NOT EXISTS idx_chunks_folder
ON internal.chunks (tenant_id, folder_id)
WHERE folder_id IS NOT NULL;

-- Comment for documentation
COMMENT ON COLUMN internal.chunks.folder_id IS 'Folder identifier for folder-based scoping';
COMMENT ON COLUMN internal.chunks.folder_name IS 'Human-readable folder name for display';
