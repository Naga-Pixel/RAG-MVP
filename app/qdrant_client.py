"""
Qdrant client configuration and utilities.
"""
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import sentry_sdk

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
)


def get_qdrant_client() -> QdrantClient:
    """Return the Qdrant client singleton."""
    return client


def collection_exists() -> bool:
    """Check if the configured collection exists."""
    collections = client.get_collections().collections
    names = [c.name for c in collections]
    return settings.qdrant_collection in names


def ensure_collection(dimension: int = 1536) -> None:
    """
    Ensure the Qdrant collection exists.
    
    Args:
        dimension: Vector dimension (default 1536 for text-embedding-3-small).
    """
    if not collection_exists():
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE,
            ),
        )


def recreate_collection(dimension: int) -> None:
    """
    Recreate the collection with specified dimension.
    Deletes existing data.

    Args:
        dimension: Vector dimension for the new collection.
    """
    client.recreate_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(
            size=dimension,
            distance=Distance.COSINE,
        ),
    )


def upsert_points(
    points: list[PointStruct],
    tenant_id: str,
    doc_id_fallback: str | None = None,
    ingest_id: str | None = None,
) -> None:
    """
    Canonical upsert function: writes to Qdrant and optionally FTS shadow.

    All ingestion paths MUST use this function for Qdrant writes.
    This is the single location where FTS shadow writes occur.

    Args:
        points: List of PointStruct objects (all from same document).
        tenant_id: Tenant identifier.
        doc_id_fallback: Fallback doc_id if not derivable from payload.
        ingest_id: Optional unique ID for this ingest operation.
                   If None, one is generated.

    Doc ID consistency (Option A):
        doc_id is derived from payload to ensure Qdrant/Postgres match.
        If payloads have mixed/missing doc_id, we warn once and use fallback.

    Note:
        FTS shadow is fail-open: Postgres errors are logged but never
        interrupt the Qdrant upsert or raise to caller.
    """
    if ingest_id is None:
        ingest_id = uuid4().hex[:12]

    # Primary write: Qdrant (always happens first)
    client.upsert(collection_name=settings.qdrant_collection, points=points)

    # Shadow write: Postgres FTS (if enabled, fail-open)
    # NOTE: This affects WRITES ONLY, never retrieval or /ask.
    if settings.fts_shadow_enabled:
        # Initialize doc_id before derivation so it's in scope for exception handler
        doc_id = doc_id_fallback or "unknown"

        try:
            from app.fts_shadow import upsert_chunks_to_fts

            # Option A: derive doc_id from payload to ensure consistency
            payload_doc_ids = {
                (pt.payload or {}).get("doc_id")
                for pt in points
            }
            payload_doc_ids.discard(None)
            payload_doc_ids.discard("")

            if len(payload_doc_ids) == 1:
                doc_id = payload_doc_ids.pop()
            elif payload_doc_ids:
                # Mixed doc_ids in single batch - warn and use fallback
                logger.warning(
                    f"fts_shadow doc_id mismatch | tenant={tenant_id} | "
                    f"payload_doc_ids={payload_doc_ids} | fallback={doc_id_fallback} | ingest_id={ingest_id}"
                )
                # doc_id remains as fallback
            # else: no doc_id in payloads, doc_id remains as fallback

            # Extract title and folder info from first point (all chunks share same doc metadata)
            first_payload = (points[0].payload or {}) if points else {}
            title = first_payload.get("title")
            folder_id = first_payload.get("folder_id")
            folder_name = first_payload.get("folder_name")

            fts_chunks = []
            for pt in points:
                payload = pt.payload or {}
                fts_chunks.append({
                    "chunk_id": str(pt.id),
                    "text": payload.get("text", ""),
                    "title": title,
                })

            # Single FTS write per document (all chunks in one call)
            upsert_chunks_to_fts(
                fts_chunks,
                tenant_id,
                doc_id,
                ingest_id,
                folder_id=folder_id,
                folder_name=folder_name,
            )

        except Exception as e:
            # Fail-open: log, capture to Sentry, continue (never raise)
            logger.warning(
                f"fts_shadow_upsert failed | tenant={tenant_id} | doc_id={doc_id} | "
                f"chunks={len(points)} | err={type(e).__name__}: {e} | ingest_id={ingest_id}"
            )
            sentry_sdk.capture_exception(e)


def retrieve_points_by_ids(point_ids: list[str]) -> list:
    """
    Retrieve points from Qdrant by their IDs.

    Used by hybrid fusion to fetch keyword-only chunks that weren't
    in the vector search results.

    Args:
        point_ids: List of point UUIDs to retrieve.

    Returns:
        List of Qdrant Record objects with id, payload, and vector.
        Points not found are silently omitted from results.
    """
    if not point_ids:
        return []

    return client.retrieve(
        collection_name=settings.qdrant_collection,
        ids=point_ids,
        with_payload=True,
        with_vectors=False,  # We only need payload for context
    )


def upsert_chunks(chunks: list[dict], vectors: list[list[float]]) -> None:
    """
    Upsert chunks with their embeddings into Qdrant.

    Args:
        chunks: List of chunk dictionaries with chunk_id, text, and metadata.
        vectors: List of embedding vectors corresponding to each chunk.
    """
    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append(
            PointStruct(
                id=chunk["chunk_id"],
                vector=vector,
                payload={**chunk["metadata"], "text": chunk["text"]},
            )
        )

    # Extract tenant_id and doc_id from first chunk's metadata for fallback
    if chunks:
        metadata = chunks[0].get("metadata", {})
        tenant_id = metadata.get("tenant_id", settings.default_tenant_id)
        doc_id = metadata.get("doc_id", "unknown")
    else:
        tenant_id = settings.default_tenant_id
        doc_id = "unknown"

    upsert_points(points, tenant_id, doc_id_fallback=doc_id)
