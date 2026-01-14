-- Migration: Google Drive OAuth and Folder Sources
-- Run this in Supabase SQL Editor or against your Postgres database
-- psql $DATABASE_URL -f migrations/001_drive_sources.sql

-- Table: google_drive_tokens
-- Stores encrypted refresh tokens for users who have connected Google Drive
CREATE TABLE IF NOT EXISTS google_drive_tokens (
    user_id TEXT PRIMARY KEY,
    refresh_token_enc TEXT NOT NULL,
    scope TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Table: google_drive_folders
-- Stores the Drive folders each user has selected for syncing
CREATE TABLE IF NOT EXISTS google_drive_folders (
    user_id TEXT NOT NULL,
    folder_id TEXT NOT NULL,
    folder_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, folder_id)
);

-- Table: google_oauth_states
-- Temporary storage for OAuth state tokens (CSRF protection)
CREATE TABLE IF NOT EXISTS google_oauth_states (
    state TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL
);

-- Index for efficient cleanup of expired states
CREATE INDEX IF NOT EXISTS idx_oauth_states_expires
ON google_oauth_states(expires_at);

-- Index for user lookups on folders
CREATE INDEX IF NOT EXISTS idx_drive_folders_user
ON google_drive_folders(user_id);
