"""
Connectors module for data source integrations.

Each connector implements the BaseConnector interface to provide
a consistent way to fetch documents from various sources.

Available connectors:
- LocalFilesConnector: Local file system
- GoogleDriveConnector: Google Drive (requires service account)

Future connectors:
- GmailConnector: Email ingestion
- CalendarConnector: Meeting transcripts
- DatabaseConnector: SQL/NoSQL databases
"""
from connectors.base import (
    BaseConnector,
    Document,
    ConnectorResult,
    SourceType,
)
from connectors.local_files import LocalFilesConnector
from connectors.google_drive import GoogleDriveConnector
from connectors.state_store import StateStore, get_state_store

__all__ = [
    "BaseConnector",
    "Document",
    "ConnectorResult",
    "SourceType",
    "LocalFilesConnector",
    "GoogleDriveConnector",
    "StateStore",
    "get_state_store",
]
