"""
FTS Shadow Index module.

Provides shadow Postgres Full-Text Search indexing of chunks
during ingestion, behind a feature flag.

IMPORTANT: This module affects WRITES ONLY.
- It does NOT affect retrieval or /ask endpoints.
- It does NOT change any read path behavior.
- FTS_SHADOW_ENABLED=false means zero Postgres interaction here.

Phase A: Shadow insert only (fail-open, log on error).
Failure never interrupts Qdrant ingestion.
"""
import time

import psycopg2
from psycopg2.extras import execute_values
import sentry_sdk

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

# Batch size for Postgres inserts (internal batching, not logged per batch)
_PG_BATCH_SIZE = 250


def _get_fts_connection():
    """
    Get a Postgres connection for FTS shadow writes.
    Returns None if DATABASE_URL not configured.
    """
    if not settings.database_url:
        return None
    return psycopg2.connect(settings.database_url)


def upsert_chunks_to_fts(
    chunks: list[dict],
    tenant_id: str,
    doc_id: str,
    ingest_id: str,
) -> bool:
    """
    Upsert chunks to Postgres FTS shadow table (doc-scoped).

    All chunks must belong to the same document.

    Args:
        chunks: List of chunk dicts with keys:
            - chunk_id (str): Qdrant point UUID
            - text (str): Chunk text content
            - title (str|None): Document title (optional)
        tenant_id: Tenant identifier
        doc_id: Document identifier (all chunks share this)
        ingest_id: Unique identifier for this ingest operation

    Returns:
        True if successful, False on failure.

    Fail-open semantics:
        - Postgres failure never interrupts Qdrant ingest
        - On failure: emit one WARNING log, capture to Sentry, return False
        - Caller must not branch behavior on return value
    """
    if not settings.fts_shadow_enabled:
        return True

    if not chunks:
        return True

    chunk_count = len(chunks)
    start_ms = time.perf_counter()
    conn = None

    try:
        conn = _get_fts_connection()
        if conn is None:
            logger.warning(
                f"fts_shadow_upsert failed | tenant={tenant_id} | doc_id={doc_id} | "
                f"chunks={chunk_count} | err=DATABASE_URL not configured | ingest_id={ingest_id}"
            )
            return False

        cursor = conn.cursor()

        # Prepare all values: (tenant_id, chunk_id, doc_id, title, text)
        values = []
        for chunk in chunks:
            values.append((
                tenant_id,
                str(chunk.get("chunk_id", "")),
                doc_id,
                chunk.get("title"),
                str(chunk.get("text", "")),
            ))

        # Upsert in internal batches (no per-batch logging)
        for i in range(0, len(values), _PG_BATCH_SIZE):
            batch = values[i:i + _PG_BATCH_SIZE]
            execute_values(
                cursor,
                """
                INSERT INTO chunks (tenant_id, chunk_id, doc_id, title, text)
                VALUES %s
                ON CONFLICT (tenant_id, chunk_id) DO UPDATE SET
                    doc_id = EXCLUDED.doc_id,
                    title = EXCLUDED.title,
                    text = EXCLUDED.text
                """,
                batch,
                page_size=_PG_BATCH_SIZE,
            )

        conn.commit()
        elapsed_ms = int((time.perf_counter() - start_ms) * 1000)

        logger.info(
            f"fts_shadow_upsert ok | tenant={tenant_id} | doc_id={doc_id} | "
            f"chunks={chunk_count} | ms={elapsed_ms} | ingest_id={ingest_id}"
        )
        return True

    except Exception as e:
        elapsed_ms = int((time.perf_counter() - start_ms) * 1000)
        err_name = type(e).__name__
        err_msg = str(e).replace("\n", " ")[:200]

        logger.warning(
            f"fts_shadow_upsert failed | tenant={tenant_id} | doc_id={doc_id} | "
            f"chunks={chunk_count} | err={err_name}: {err_msg} | ingest_id={ingest_id}"
        )
        sentry_sdk.capture_exception(e)
        return False

    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
