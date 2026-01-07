"""
Base connector class for all data source connectors.

All connectors must implement this interface to ensure consistent
document handling across different data sources.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Iterator


class SourceType(str, Enum):
    """Supported data source types."""
    LOCAL_FILE = "local_file"
    GOOGLE_DRIVE = "google_drive"
    EMAIL = "email"
    CALENDAR = "calendar"
    DATABASE = "database"
    TRANSCRIPT = "transcript"
    API = "api"


@dataclass
class Document:
    """
    Standardized document representation from any connector.
    
    All connectors must return documents in this format.
    """
    # Required fields
    content: str
    source_type: SourceType
    source_id: str  # Unique identifier within the source (file path, email id, etc.)
    
    # Standard metadata
    title: str | None = None
    doc_id: str | None = None  # User-defined doc ID if found in content
    tenant_id: str | None = None
    
    # External ID for idempotent upserts (stable across syncs)
    # For Google Drive: file ID
    # For local files: relative path hash
    external_id: str | None = None
    
    # Source-specific metadata
    metadata: dict = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime | None = None
    modified_at: datetime | None = None
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Set doc_id and external_id to source_id if not provided."""
        if self.doc_id is None:
            self.doc_id = self.source_id
        if self.external_id is None:
            self.external_id = self.source_id


@dataclass
class ConnectorResult:
    """Result of a connector sync operation."""
    documents: list[Document]
    total_found: int
    total_processed: int
    errors: list[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        if self.total_found == 0:
            return 1.0
        return self.total_processed / self.total_found


class BaseConnector(ABC):
    """
    Abstract base class for all data source connectors.
    
    Implement this interface to add new data sources (Google Drive, Email, etc.)
    """
    
    def __init__(self, tenant_id: str = "default"):
        self.tenant_id = tenant_id
    
    @property
    @abstractmethod
    def source_type(self) -> SourceType:
        """Return the type of this connector."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this connector."""
        pass
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the data source.
        
        Returns:
            True if connection successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Clean up connection resources."""
        pass
    
    @abstractmethod
    def list_documents(self) -> list[dict]:
        """
        List available documents without fetching content.
        
        Returns:
            List of document metadata dicts with at least 'source_id' and 'title'.
        """
        pass
    
    @abstractmethod
    def fetch_document(self, source_id: str) -> Document | None:
        """
        Fetch a single document by its source ID.
        
        Args:
            source_id: Unique identifier for the document in this source.
            
        Returns:
            Document object or None if not found.
        """
        pass
    
    def fetch_all(self) -> Iterator[Document]:
        """
        Fetch all documents from this source.
        
        Yields:
            Document objects.
        """
        for doc_info in self.list_documents():
            doc = self.fetch_document(doc_info["source_id"])
            if doc:
                yield doc
    
    def sync(self) -> ConnectorResult:
        """
        Sync all documents from this source.
        
        Returns:
            ConnectorResult with all fetched documents and stats.
        """
        documents = []
        errors = []
        doc_list = self.list_documents()
        
        for doc_info in doc_list:
            try:
                doc = self.fetch_document(doc_info["source_id"])
                if doc:
                    doc.tenant_id = self.tenant_id
                    documents.append(doc)
            except Exception as e:
                errors.append(f"Error fetching {doc_info.get('source_id', 'unknown')}: {e}")
        
        return ConnectorResult(
            documents=documents,
            total_found=len(doc_list),
            total_processed=len(documents),
            errors=errors,
        )
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False
