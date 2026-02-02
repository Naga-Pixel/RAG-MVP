"""
Keyword Retrieval module (Phase B).

Provides Postgres Full-Text Search retrieval for comparison logging.

IMPORTANT: This module is for LOGGING ONLY.
- Results are NOT used in /ask responses
- Results are NOT merged with vector results
- Does NOT change user-visible behavior
- Fail-open: Postgres errors return empty list

Feature flag: KEYWORD_RETRIEVAL_LOGGING_ENABLED
"""
import psycopg2

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


def _get_fts_table_fqtn() -> str:
    """Return fully-qualified table name for FTS queries."""
    return f"{settings.fts_shadow_schema}.{settings.fts_shadow_table}"


def _get_fts_connection():
    """
    Get a Postgres connection for FTS queries.
    Returns None if DATABASE_URL not configured.
    """
    if not settings.database_url:
        return None
    return psycopg2.connect(settings.database_url)


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
