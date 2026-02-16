-- Add is_folder column to google_drive_folders table
-- to support individual file selection in addition to folders
--
-- Run this migration in Supabase SQL Editor or via psql.

-- Add is_folder column (default true for backwards compatibility)
ALTER TABLE google_drive_folders
ADD COLUMN IF NOT EXISTS is_folder BOOLEAN DEFAULT true;

-- Add mime_type column for files (to help with sync filtering)
ALTER TABLE google_drive_folders
ADD COLUMN IF NOT EXISTS mime_type TEXT;

-- Comment for documentation
COMMENT ON COLUMN google_drive_folders.is_folder IS 'True for folders, false for individual files';
COMMENT ON COLUMN google_drive_folders.mime_type IS 'MIME type for individual files (null for folders)';
