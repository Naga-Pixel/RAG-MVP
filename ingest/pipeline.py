"""
Unified ingestion pipeline for all connectors.

This module provides a single entry point for ingesting documents
from any connector into Qdrant.

Features:
- Idempotent upserts (same file + same version = same point IDs)
- Automatic cleanup of old versions
- Connector state tracking
"""
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from app.config import settings
from app.logging_config import get_logger
from app.qdrant_client import client as qdrant_client
from connectors.base import BaseConnector, Document, SourceType
from connectors.state_store import get_state_store
from ingest.chunking import chunk_text
from ingest.embedder import get_embeddings

logger = get_logger(__name__)


def generate_point_id(
    tenant_id: str,
    source: str,
    external_id: str,
    updated_at: str,
    chunk_index: int,
) -> str:
    """
    Generate a deterministic UUID for a Qdrant point.
    
    Same inputs always produce same UUID, ensuring idempotent upserts.
    
    Args:
        tenant_id: Tenant identifier.
        source: Source type (e.g., 'google_drive').
        external_id: Stable document ID from source.
        updated_at: Document version marker (ISO timestamp).
        chunk_index: Chunk index within document.
    
    Returns:
        UUID string suitable for Qdrant point ID.
    """
    # Create deterministic string from all components
    key = f"{tenant_id}:{source}:{external_id}:{updated_at}:{chunk_index}"
    
    # Use UUID5 with DNS namespace for determinism
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


@dataclass
class IngestionStats:
    """Statistics from an ingestion run."""
    connector_name: str
    documents_found: int = 0
    documents_processed: int = 0
    chunks_created: int = 0
    chunks_cleaned: int = 0  # Old versions removed
    errors: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    
    @property
    def duration_seconds(self) -> float | None:
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def to_dict(self) -> dict:
        return {
            "connector": self.connector_name,
            "documents_found": self.documents_found,
            "documents_processed": self.documents_processed,
            "chunks_created": self.chunks_created,
            "chunks_cleaned": self.chunks_cleaned,
            "errors": self.errors,
            "duration_seconds": self.duration_seconds,
        }


class IngestionPipeline:
    """
    Unified ingestion pipeline for all data sources.
    
    Handles:
    - Document fetching via connectors
    - Chunking
    - Embedding generation
    - Idempotent Qdrant storage
    - Old version cleanup
    - Connector state management
    """
    
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._collection_initialized = False
        self._state_store = get_state_store()
    
    def ensure_collection(self, dimension: int, recreate: bool = False) -> None:
        """Ensure Qdrant collection exists with correct dimension."""
        collection_name = settings.qdrant_collection
        
        collections = qdrant_client.get_collections().collections
        names = [c.name for c in collections]
        exists = collection_name in names
        
        if recreate and exists:
            logger.info(f"Deleting existing collection '{collection_name}'...")
            qdrant_client.delete_collection(collection_name=collection_name)
            exists = False
        
        if not exists:
            logger.info(f"Creating collection '{collection_name}' with dimension {dimension}...")
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                ),
            )
        
        self._collection_initialized = True
    
    def _cleanup_old_versions(
        self,
        tenant_id: str,
        source: str,
        external_id: str,
        current_updated_at: str,
    ) -> int:
        """
        Delete old version points for a document.
        
        Removes points where (tenant_id, source, external_id) match
        but updated_at is different from current version.
        
        Returns:
            Number of points deleted.
        """
        try:
            # Build filter for old versions
            # Match tenant_id, source, external_id but NOT current updated_at
            result = qdrant_client.delete(
                collection_name=settings.qdrant_collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
                        FieldCondition(key="source_type", match=MatchValue(value=source)),
                        FieldCondition(key="external_id", match=MatchValue(value=external_id)),
                    ],
                    must_not=[
                        FieldCondition(key="updated_at", match=MatchValue(value=current_updated_at)),
                    ],
                ),
            )
            # Qdrant delete doesn't return count easily, so we estimate
            return 0  # Can't get exact count from Qdrant delete
        except Exception as e:
            logger.warning(f"Could not cleanup old versions: {e}")
            return 0
    
    def ingest_document(self, doc: Document, recreate: bool = False) -> int:
        """
        Ingest a single document into Qdrant with idempotent point IDs.
        
        Args:
            doc: Document object from any connector.
            recreate: Whether to recreate the collection.
        
        Returns:
            Number of chunks inserted.
        """
        logger.info(f"Processing: {doc.title} ({doc.source_type.value})")
        logger.debug(f"  Source: {doc.source_id}")
        logger.debug(f"  External ID: {doc.external_id}")
        logger.info(f"  Content length: {len(doc.content)} characters")
        
        tenant_id = doc.tenant_id or settings.default_tenant_id
        source = doc.source_type.value
        external_id = doc.external_id or doc.source_id
        updated_at = doc.modified_at.isoformat() if doc.modified_at else doc.fetched_at.isoformat()
        
        # Build metadata from document
        metadata = {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "tenant_id": tenant_id,
            "source_type": source,
            "source_id": doc.source_id,
            "external_id": external_id,
            "updated_at": updated_at,
            **doc.metadata,
        }
        
        if doc.fetched_at:
            metadata["fetched_at"] = doc.fetched_at.isoformat()
        
        # Create chunks
        chunks = chunk_text(
            text=doc.content,
            doc_id=doc.doc_id,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            metadata=metadata,
        )
        logger.info(f"  Created {len(chunks)} chunks")
        
        if not chunks:
            return 0
        
        # Generate embeddings
        texts = [c["text"] for c in chunks]
        logger.info(f"  Generating embeddings...")
        embeddings, dimension = get_embeddings(texts)
        logger.debug(f"  Got {len(embeddings)} embeddings (dimension: {dimension})")
        
        # Ensure collection exists
        if not self._collection_initialized:
            self.ensure_collection(dimension, recreate=recreate)
        
        # Create points with deterministic IDs
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = generate_point_id(
                tenant_id=tenant_id,
                source=source,
                external_id=external_id,
                updated_at=updated_at,
                chunk_index=i,
            )
            
            payload = {
                **chunk["metadata"],
                "text": chunk["text"],
                "chunk_index": i,
            }
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )
        
        # Cleanup old versions before upserting new ones
        if not recreate:
            self._cleanup_old_versions(tenant_id, source, external_id, updated_at)
        
        # Upsert to Qdrant (idempotent - same ID overwrites)
        logger.info(f"  Upserting {len(points)} points...")
        qdrant_client.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
        )
        logger.debug(f"  Done!")
        
        return len(points)
    
    def ingest_from_connector(
        self,
        connector: BaseConnector,
        recreate: bool = False,
    ) -> IngestionStats:
        """
        Ingest all documents from a connector.
        
        Args:
            connector: Any connector implementing BaseConnector.
            recreate: Whether to recreate the collection (only on first doc).
        
        Returns:
            IngestionStats with results.
        """
        stats = IngestionStats(connector_name=connector.name)
        source_key = connector.source_type.value
        
        logger.info("=" * 50)
        logger.info(f"Starting ingestion from: {connector.name}")
        logger.info(f"Tenant: {connector.tenant_id}")
        logger.info("=" * 50)
        
        # Load existing state
        state = self._state_store.get_state(connector.tenant_id, source_key) or {}
        last_sync = state.get("last_sync_at")
        if last_sync:
            logger.info(f"Last sync: {last_sync}")
        
        # Connect to source
        if not connector.connect():
            stats.errors.append(f"Failed to connect to {connector.name}")
            stats.completed_at = datetime.utcnow()
            return stats
        
        ingestion_success = False
        try:
            # Get document list
            doc_list = connector.list_documents()
            stats.documents_found = len(doc_list)
            logger.info(f"Found {stats.documents_found} documents")
            
            # Process each document
            for doc_info in doc_list:
                try:
                    doc = connector.fetch_document(doc_info["source_id"])
                    if doc:
                        chunks = self.ingest_document(
                            doc,
                            recreate=recreate and stats.documents_processed == 0,
                        )
                        stats.chunks_created += chunks
                        stats.documents_processed += 1
                except Exception as e:
                    error_msg = f"Error processing {doc_info.get('title', doc_info['source_id'])}: {e}"
                    logger.error(f"  {error_msg}")
                    stats.errors.append(error_msg)
            
            ingestion_success = True
        
        finally:
            connector.disconnect()
        
        stats.completed_at = datetime.utcnow()
        
        # Update state only on success
        if ingestion_success and stats.documents_processed > 0:
            self._state_store.update_state(
                tenant_id=connector.tenant_id,
                source=source_key,
                patch={
                    "last_sync_at": stats.completed_at.isoformat(),
                    "documents_synced": stats.documents_processed,
                    "chunks_created": stats.chunks_created,
                },
            )
        
        logger.info("=" * 50)
        logger.info("Ingestion complete!")
        logger.info(f"  Documents processed: {stats.documents_processed}/{stats.documents_found}")
        logger.info(f"  Chunks created: {stats.chunks_created}")
        if stats.errors:
            logger.warning(f"  Errors: {len(stats.errors)}")
        logger.info(f"  Duration: {stats.duration_seconds:.1f}s")
        logger.info("=" * 50)
        
        return stats
    
    def ingest_from_connectors(
        self,
        connectors: list[BaseConnector],
        recreate: bool = False,
    ) -> list[IngestionStats]:
        """
        Ingest from multiple connectors.
        
        Args:
            connectors: List of connectors to ingest from.
            recreate: Whether to recreate the collection.
        
        Returns:
            List of IngestionStats for each connector.
        """
        all_stats = []
        
        for i, connector in enumerate(connectors):
            # Only recreate on first connector
            stats = self.ingest_from_connector(
                connector,
                recreate=recreate and i == 0,
            )
            all_stats.append(stats)
        
        return all_stats


# Convenience function for simple use
def ingest(
    connector: BaseConnector,
    recreate: bool = False,
) -> IngestionStats:
    """
    Simple function to ingest documents from a connector.
    
    Args:
        connector: Any connector implementing BaseConnector.
        recreate: Whether to recreate the Qdrant collection.
    
    Returns:
        IngestionStats with results.
    """
    pipeline = IngestionPipeline()
    return pipeline.ingest_from_connector(connector, recreate=recreate)
