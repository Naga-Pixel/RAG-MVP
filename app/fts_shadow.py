"""
FTS Shadow Index module.

Provides shadow Postgres Full-Text Search indexing and retrieval.

Phase A (FTS_SHADOW_ENABLED):
- Shadow writes during ingestion
- Fail-open, no retrieval behavior change

Phase B (KEYWORD_RETRIEVAL_LOGGING_ENABLED):
- Keyword retrieval for comparison logging only
- Does NOT affect /ask responses
- Does NOT change user-visible behavior
- Fail-open: Postgres errors are logged, vector-only continues
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


def _get_fts_table_fqtn() -> str:
    """Return fully-qualified table name for FTS shadow writes."""
    return f"{settings.fts_shadow_schema}.{settings.fts_shadow_table}"


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
    fqtn = _get_fts_table_fqtn()

    try:
        conn = _get_fts_connection()
        if conn is None:
            logger.warning(
                f"fts_shadow_upsert failed | table={fqtn} | tenant={tenant_id} | doc_id={doc_id} | "
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

        # Build upsert SQL with fully-qualified table name
        upsert_sql = f"""
            INSERT INTO {fqtn} (tenant_id, chunk_id, doc_id, title, text)
            VALUES %s
            ON CONFLICT (tenant_id, chunk_id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                title = EXCLUDED.title,
                text = EXCLUDED.text
        """

        # Upsert in internal batches (no per-batch logging)
        for i in range(0, len(values), _PG_BATCH_SIZE):
            batch = values[i:i + _PG_BATCH_SIZE]
            execute_values(cursor, upsert_sql, batch, page_size=_PG_BATCH_SIZE)

        conn.commit()
        elapsed_ms = int((time.perf_counter() - start_ms) * 1000)

        logger.info(
            f"fts_shadow_upsert ok | table={fqtn} | tenant={tenant_id} | doc_id={doc_id} | "
            f"chunks={chunk_count} | ms={elapsed_ms} | ingest_id={ingest_id}"
        )
        return True

    except Exception as e:
        elapsed_ms = int((time.perf_counter() - start_ms) * 1000)
        err_name = type(e).__name__
        err_msg = str(e).replace("\n", " ")[:200]

        logger.warning(
            f"fts_shadow_upsert failed | table={fqtn} | tenant={tenant_id} | doc_id={doc_id} | "
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


# =============================================================================
# Phase B: Keyword Retrieval (Logging Only)
# =============================================================================

def keyword_retrieve(
    query: str,
    tenant_id: str,
    limit: int = 10,
) -> list[dict]:
    """
    Retrieve chunks from Postgres FTS for comparison logging.

    Phase B: This function is for LOGGING ONLY.
    - Results are NOT used in /ask responses
    - Results are NOT merged with vector results
    - Failure returns empty list (fail-open)

    Args:
        query: User query string
        tenant_id: Tenant identifier for filtering
        limit: Maximum number of results (default 10)

    Returns:
        List of dicts with keys: chunk_id, doc_id, rank
        Empty list on error (fail-open).
    """
    if not settings.keyword_retrieval_logging_enabled:
        return []

    if not query or not query.strip():
        return []

    fqtn = _get_fts_table_fqtn()
    conn = None

    try:
        conn = _get_fts_connection()
        if conn is None:
            logger.warning(
                f"keyword_retrieve failed | table={fqtn} | tenant={tenant_id} | "
                f"err=DATABASE_URL not configured"
            )
            return []

        cursor = conn.cursor()

        # FTS query using websearch_to_tsquery for natural language parsing
        # Rank using ts_rank_cd (cover density ranking)
        # Match against combined title + text tsvector
        retrieve_sql = f"""
            SELECT
                chunk_id,
                doc_id,
                ts_rank_cd(
                    to_tsvector('english', coalesce(title, '') || ' ' || coalesce(text, '')),
                    websearch_to_tsquery('english', %s)
                ) AS rank
            FROM {fqtn}
            WHERE tenant_id = %s
              AND to_tsvector('english', coalesce(title, '') || ' ' || coalesce(text, ''))
                  @@ websearch_to_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
        """

        cursor.execute(retrieve_sql, (query, tenant_id, query, limit))
        rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append({
                "chunk_id": row[0],
                "doc_id": row[1],
                "rank": float(row[2]) if row[2] is not None else 0.0,
            })

        return results

    except Exception as e:
        err_name = type(e).__name__
        err_msg = str(e).replace("\n", " ")[:200]
        logger.warning(
            f"keyword_retrieve failed | table={fqtn} | tenant={tenant_id} | "
            f"err={err_name}: {err_msg}"
        )
        # Fail-open: return empty, do not raise
        return []

    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
