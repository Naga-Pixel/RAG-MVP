-- Track synced Google Drive files for incremental sync
-- Stores modification time to skip unchanged files on re-sync
--
-- Run this migration in Supabase SQL Editor or via psql.

-- Create synced files tracking table
CREATE TABLE IF NOT EXISTS google_drive_synced_files (
    user_id TEXT NOT NULL,
    file_id TEXT NOT NULL,
    folder_id TEXT NOT NULL,
    file_name TEXT,
    modified_at TIMESTAMPTZ NOT NULL,
    synced_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, file_id)
);

-- Index for efficient folder-based queries
CREATE INDEX IF NOT EXISTS idx_drive_synced_files_folder
ON google_drive_synced_files (user_id, folder_id);

-- Comment for documentation
COMMENT ON TABLE google_drive_synced_files IS 'Tracks synced Drive files for incremental sync - skip unchanged files';
