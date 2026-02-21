"""
GDPR Compliance Module.

Provides user data deletion and export functionality for GDPR compliance:
- Right to be forgotten (Article 17): delete_all_user_data()
- Right to data portability (Article 20): export_user_data()

All operations are scoped to a single user (tenant_id) and cascade across
all storage systems: Qdrant, Postgres FTS, Google Drive tables.
"""
import json
from datetime import datetime
from typing import Any

import psycopg2
from psycopg2 import sql
import sentry_sdk

from app.config import settings
from app.logging_config import get_logger
from app.qdrant_client import client as qdrant_client

logger = get_logger(__name__)


def _get_db_connection():
    """Get a Postgres connection. Returns None if DATABASE_URL not configured."""
    if not settings.database_url:
        return None
    return psycopg2.connect(settings.database_url)


def delete_all_user_data(user_id: str) -> dict:
    """
    Delete ALL data for a user across all storage systems.

    This implements GDPR Article 17 (Right to Erasure / Right to be Forgotten).

    Deletes from:
    - Qdrant: All document chunks/vectors
    - Postgres FTS shadow: All indexed chunks
    - google_drive_tokens: OAuth refresh tokens
    - google_drive_folders: Connected folder records
    - google_drive_synced_files: Sync tracking records

    Args:
        user_id: The user's ID (tenant_id from Supabase JWT)

    Returns:
        Dict with deletion counts for each storage system

    Raises:
        Exception: If critical deletion fails (Qdrant)
    """
    results = {
        "user_id": user_id,
        "deleted_at": datetime.utcnow().isoformat(),
        "qdrant_points": 0,
        "fts_chunks": 0,
        "drive_tokens": 0,
        "drive_folders": 0,
        "drive_synced_files": 0,
        "errors": [],
    }

    # 1. Delete from Qdrant (critical - must succeed)
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        filter_conditions = Filter(
            must=[FieldCondition(key="tenant_id", match=MatchValue(value=user_id))]
        )

        # Count before delete
        count_result = qdrant_client.count(
            collection_name=settings.qdrant_collection,
            count_filter=filter_conditions,
            exact=False,
        )
        results["qdrant_points"] = count_result.count

        # Delete all user's points
        qdrant_client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=filter_conditions,
        )

        logger.info(f"gdpr_delete_qdrant ok | user={user_id} | points={results['qdrant_points']}")

    except Exception as e:
        logger.error(f"gdpr_delete_qdrant failed | user={user_id} | err={type(e).__name__}: {e}")
        sentry_sdk.capture_exception(e)
        results["errors"].append(f"Qdrant deletion failed: {str(e)}")
        raise  # Qdrant is critical, don't continue if it fails

    # 2. Delete from Postgres (FTS shadow + Drive tables)
    conn = None
    try:
        conn = _get_db_connection()
        if conn is None:
            logger.warning(f"gdpr_delete_postgres skipped | user={user_id} | err=DATABASE_URL not configured")
            results["errors"].append("Postgres not configured - skipped")
        else:
            cursor = conn.cursor()

            # Delete from FTS shadow table (using safe SQL composition)
            delete_fts_query = sql.SQL("DELETE FROM {} WHERE tenant_id = %s").format(
                settings.get_fts_table_sql()
            )
            cursor.execute(delete_fts_query, (user_id,))
            results["fts_chunks"] = cursor.rowcount
            logger.info(f"gdpr_delete_fts ok | user={user_id} | rows={results['fts_chunks']}")

            # Delete from google_drive_synced_files
            cursor.execute("DELETE FROM google_drive_synced_files WHERE user_id = %s", (user_id,))
            results["drive_synced_files"] = cursor.rowcount
            logger.info(f"gdpr_delete_synced_files ok | user={user_id} | rows={results['drive_synced_files']}")

            # Delete from google_drive_folders
            cursor.execute("DELETE FROM google_drive_folders WHERE user_id = %s", (user_id,))
            results["drive_folders"] = cursor.rowcount
            logger.info(f"gdpr_delete_folders ok | user={user_id} | rows={results['drive_folders']}")

            # Delete from google_drive_tokens
            cursor.execute("DELETE FROM google_drive_tokens WHERE user_id = %s", (user_id,))
            results["drive_tokens"] = cursor.rowcount
            logger.info(f"gdpr_delete_tokens ok | user={user_id} | rows={results['drive_tokens']}")

            conn.commit()

    except Exception as e:
        logger.error(f"gdpr_delete_postgres failed | user={user_id} | err={type(e).__name__}: {e}")
        sentry_sdk.capture_exception(e)
        results["errors"].append(f"Postgres deletion failed: {str(e)}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

    logger.info(f"gdpr_delete_complete | user={user_id} | results={results}")
    return results


def export_user_data(user_id: str) -> dict:
    """
    Export ALL data for a user in a portable format.

    This implements GDPR Article 20 (Right to Data Portability).

    Exports from:
    - Qdrant: All document chunks with metadata
    - Postgres: Drive connection info (metadata only, not tokens)

    Args:
        user_id: The user's ID (tenant_id from Supabase JWT)

    Returns:
        Dict containing all user data in a structured format
    """
    export = {
        "user_id": user_id,
        "exported_at": datetime.utcnow().isoformat(),
        "documents": [],
        "drive_folders": [],
        "drive_synced_files": [],
        "summary": {
            "total_documents": 0,
            "total_chunks": 0,
            "total_folders": 0,
        },
    }

    # 1. Export from Qdrant
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        filter_conditions = Filter(
            must=[FieldCondition(key="tenant_id", match=MatchValue(value=user_id))]
        )

        # Group chunks by document
        documents_map = {}  # doc_id -> {title, chunks, metadata}

        offset = None
        while True:
            result = qdrant_client.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=filter_conditions,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,  # Don't export vectors (not useful for user)
            )

            points, offset = result
            if not points:
                break

            for point in points:
                payload = point.payload or {}
                doc_id = payload.get("doc_id", "unknown")

                if doc_id not in documents_map:
                    documents_map[doc_id] = {
                        "doc_id": doc_id,
                        "title": payload.get("title"),
                        "source": payload.get("source"),
                        "source_file": payload.get("source_file"),
                        "folder_id": payload.get("folder_id"),
                        "folder_name": payload.get("folder_name"),
                        "chunks": [],
                    }

                documents_map[doc_id]["chunks"].append({
                    "chunk_id": str(point.id),
                    "text": payload.get("text", ""),
                })

            if offset is None:
                break

        export["documents"] = list(documents_map.values())
        export["summary"]["total_documents"] = len(documents_map)
        export["summary"]["total_chunks"] = sum(len(d["chunks"]) for d in documents_map.values())

        logger.info(f"gdpr_export_qdrant ok | user={user_id} | docs={export['summary']['total_documents']} | chunks={export['summary']['total_chunks']}")

    except Exception as e:
        logger.error(f"gdpr_export_qdrant failed | user={user_id} | err={type(e).__name__}: {e}")
        sentry_sdk.capture_exception(e)
        export["errors"] = export.get("errors", []) + [f"Qdrant export failed: {str(e)}"]

    # 2. Export from Postgres (metadata only)
    conn = None
    try:
        conn = _get_db_connection()
        if conn is None:
            logger.warning(f"gdpr_export_postgres skipped | user={user_id} | err=DATABASE_URL not configured")
        else:
            cursor = conn.cursor()

            # Export folder connections
            cursor.execute(
                "SELECT folder_id, folder_name FROM google_drive_folders WHERE user_id = %s",
                (user_id,)
            )
            export["drive_folders"] = [
                {"folder_id": row[0], "folder_name": row[1]}
                for row in cursor.fetchall()
            ]
            export["summary"]["total_folders"] = len(export["drive_folders"])

            # Export sync history (file metadata, not content)
            cursor.execute(
                """
                SELECT file_id, folder_id, file_name, modified_at, synced_at
                FROM google_drive_synced_files
                WHERE user_id = %s
                """,
                (user_id,)
            )
            export["drive_synced_files"] = [
                {
                    "file_id": row[0],
                    "folder_id": row[1],
                    "file_name": row[2],
                    "modified_at": row[3].isoformat() if row[3] else None,
                    "synced_at": row[4].isoformat() if row[4] else None,
                }
                for row in cursor.fetchall()
            ]

            logger.info(f"gdpr_export_postgres ok | user={user_id} | folders={len(export['drive_folders'])} | files={len(export['drive_synced_files'])}")

    except Exception as e:
        logger.error(f"gdpr_export_postgres failed | user={user_id} | err={type(e).__name__}: {e}")
        sentry_sdk.capture_exception(e)
        export["errors"] = export.get("errors", []) + [f"Postgres export failed: {str(e)}"]
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

    logger.info(f"gdpr_export_complete | user={user_id} | summary={export['summary']}")
    return export


def get_user_data_summary(user_id: str) -> dict:
    """
    Get a summary of data held for a user.

    This supports GDPR Article 15 (Right of Access) by providing
    a quick overview without full data export.

    Args:
        user_id: The user's ID (tenant_id from Supabase JWT)

    Returns:
        Dict with summary counts and metadata
    """
    summary = {
        "user_id": user_id,
        "generated_at": datetime.utcnow().isoformat(),
        "documents_count": 0,
        "chunks_count": 0,
        "folders_count": 0,
        "synced_files_count": 0,
        "google_drive_connected": False,
        "storage_estimate_bytes": 0,
    }

    # Count from Qdrant
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        filter_conditions = Filter(
            must=[FieldCondition(key="tenant_id", match=MatchValue(value=user_id))]
        )

        count_result = qdrant_client.count(
            collection_name=settings.qdrant_collection,
            count_filter=filter_conditions,
            exact=False,
        )
        summary["chunks_count"] = count_result.count
        # Rough estimate: ~500 bytes per chunk average
        summary["storage_estimate_bytes"] = count_result.count * 500

    except Exception as e:
        logger.error(f"gdpr_summary_qdrant failed | user={user_id} | err={type(e).__name__}: {e}")

    # Count from Postgres
    conn = None
    try:
        conn = _get_db_connection()
        if conn:
            cursor = conn.cursor()

            # Count distinct documents (using safe SQL composition)
            count_docs_query = sql.SQL(
                "SELECT COUNT(DISTINCT doc_id) FROM {} WHERE tenant_id = %s"
            ).format(settings.get_fts_table_sql())
            cursor.execute(count_docs_query, (user_id,))
            result = cursor.fetchone()
            summary["documents_count"] = result[0] if result else 0

            # Count folders
            cursor.execute(
                "SELECT COUNT(*) FROM google_drive_folders WHERE user_id = %s",
                (user_id,)
            )
            result = cursor.fetchone()
            summary["folders_count"] = result[0] if result else 0

            # Count synced files
            cursor.execute(
                "SELECT COUNT(*) FROM google_drive_synced_files WHERE user_id = %s",
                (user_id,)
            )
            result = cursor.fetchone()
            summary["synced_files_count"] = result[0] if result else 0

            # Check if Drive connected
            cursor.execute(
                "SELECT 1 FROM google_drive_tokens WHERE user_id = %s LIMIT 1",
                (user_id,)
            )
            summary["google_drive_connected"] = cursor.fetchone() is not None

    except Exception as e:
        logger.error(f"gdpr_summary_postgres failed | user={user_id} | err={type(e).__name__}: {e}")
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

    return summary
