"""
Scope Resolver module for folder/document scoping.

Detects scope from query text and resolves against folder names and doc titles.
Used when UI doesn't provide explicit scope but query implies one.

Examples of scope phrases:
- "in the GDPR folder..."
- "within the contracts folder..."
- "use only Diary of a CEO transcript..."
- "in the document X..."
- "from folder Y..."
"""
import re
from dataclasses import dataclass

import psycopg2

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ResolvedScope:
    """Result of scope resolution from query text."""
    folder_id: str | None = None
    folder_name: str | None = None
    doc_ids: list[str] | None = None
    doc_titles: list[str] | None = None
    confidence: float = 0.0
    reason: str = "none"
    source: str = "none"  # "ui" | "auto" | "none"


# Patterns to detect scope phrases in queries
# Order matters - more specific patterns first
SCOPE_PATTERNS = [
    # Folder patterns
    (r"in (?:the )?folder ['\"]?([^'\"]+?)['\"]?(?:\s|,|\.|$)", "folder"),
    (r"within (?:the )?folder ['\"]?([^'\"]+?)['\"]?(?:\s|,|\.|$)", "folder"),
    (r"from (?:the )?folder ['\"]?([^'\"]+?)['\"]?(?:\s|,|\.|$)", "folder"),
    (r"in (?:the )?['\"]?([^'\"]+?)['\"]? folder(?:\s|,|\.|$)", "folder"),
    # Document patterns - transcript/contract suffix (greedy capture)
    (r"(?:look |search )?in (?:the )?['\"]?(.+?)['\"]? transcript", "document"),
    (r"(?:look |search )?in (?:the )?['\"]?(.+?)['\"]? contract", "document"),
    (r"(?:look |search )?in (?:the )?['\"]?(.+?)['\"]? document", "document"),
    # Document patterns - prefix forms
    (r"(?:look |search )?in (?:the )?document ['\"]?([^'\"]+?)['\"]?(?:\s|,|\.|$)", "document"),
    (r"(?:look |search )?in (?:the )?transcript ['\"]?([^'\"]+?)['\"]?(?:\s|,|\.|$)", "document"),
    (r"use (?:only )?['\"]?(.+?)['\"]?(?:\s+transcript|\s+document)?(?:\s|,|\.|$)", "document"),
    (r"from (?:the )?document ['\"]?([^'\"]+?)['\"]?(?:\s|,|\.|$)", "document"),
    (r"within (?:the )?document ['\"]?([^'\"]+?)['\"]?(?:\s|,|\.|$)", "document"),
    (r"only (?:in |from )?(?:the )?['\"]?(.+?)['\"]?(?:\s+transcript|\s+document)", "document"),
]


def _extract_scope_phrase(query: str) -> tuple[str | None, str]:
    """
    Extract scope phrase from query text.

    Returns:
        Tuple of (extracted_name, scope_type) where scope_type is "folder" or "document".
        Returns (None, "none") if no scope phrase detected.
    """
    query_lower = query.lower()

    for pattern, scope_type in SCOPE_PATTERNS:
        match = re.search(pattern, query_lower, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            # Clean up common suffixes and prefixes
            extracted = re.sub(r'\s+(folder|document|transcript|contract)$', '', extracted, flags=re.IGNORECASE)
            extracted = re.sub(r'^(the|a)\s+', '', extracted, flags=re.IGNORECASE)
            extracted = extracted.strip()
            if extracted and len(extracted) >= 2:
                logger.debug(f"scope_extract | pattern={pattern[:30]}... | extracted={extracted}")
                return extracted, scope_type

    return None, "none"


def _fuzzy_match_score(needle: str, haystack: str) -> float:
    """
    Compute fuzzy match score between two strings.

    Returns score 0.0-1.0 where 1.0 is exact match.
    """
    needle_lower = needle.lower().strip()
    haystack_lower = haystack.lower().strip()

    # Exact match
    if needle_lower == haystack_lower:
        return 1.0

    # Contains match (needle in haystack)
    if needle_lower in haystack_lower:
        return 0.85 + (len(needle_lower) / len(haystack_lower)) * 0.1

    # Contains match (haystack in needle)
    if haystack_lower in needle_lower:
        return 0.75 + (len(haystack_lower) / len(needle_lower)) * 0.1

    # Word overlap - more generous scoring
    needle_words = set(needle_lower.split())
    haystack_words = set(haystack_lower.split())

    # Remove common stopwords for better matching
    stopwords = {'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for'}
    needle_words = needle_words - stopwords
    haystack_words = haystack_words - stopwords

    if needle_words and haystack_words:
        overlap = len(needle_words & haystack_words)
        # Score based on how much of the needle is covered
        needle_coverage = overlap / len(needle_words) if needle_words else 0
        if overlap >= 2:
            # Multiple word overlap is strong signal
            return 0.7 + needle_coverage * 0.25
        elif overlap == 1 and len(needle_words) <= 2:
            # Single word overlap OK for short queries
            return 0.65 + needle_coverage * 0.15

    return 0.0


def _match_folder(
    name: str,
    tenant_id: str,
) -> tuple[str | None, str | None, float]:
    """
    Match folder name against stored folders.

    Returns:
        Tuple of (folder_id, folder_name, confidence)
    """
    if not settings.database_url:
        return None, None, 0.0

    conn = None
    try:
        conn = psycopg2.connect(settings.database_url)
        cursor = conn.cursor()

        fqtn = f"{settings.fts_shadow_schema}.{settings.fts_shadow_table}"

        # Get distinct folder names for this tenant
        cursor.execute(f"""
            SELECT DISTINCT folder_id, folder_name
            FROM {fqtn}
            WHERE tenant_id = %s AND folder_id IS NOT NULL
        """, (tenant_id,))

        rows = cursor.fetchall()

        best_match = None
        best_score = 0.0

        for folder_id, folder_name in rows:
            if folder_name:
                score = _fuzzy_match_score(name, folder_name)
                if score > best_score:
                    best_score = score
                    best_match = (folder_id, folder_name)

        if best_match and best_score >= 0.6:
            return best_match[0], best_match[1], best_score

        return None, None, 0.0

    except Exception as e:
        logger.warning(f"scope_resolver folder match failed | err={type(e).__name__}: {e}")
        return None, None, 0.0
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def _match_document(
    name: str,
    tenant_id: str,
) -> tuple[list[str] | None, list[str] | None, float]:
    """
    Match document name/title against stored documents.

    Returns:
        Tuple of (doc_ids, doc_titles, confidence)
    """
    if not settings.database_url:
        return None, None, 0.0

    conn = None
    try:
        conn = psycopg2.connect(settings.database_url)
        cursor = conn.cursor()

        fqtn = f"{settings.fts_shadow_schema}.{settings.fts_shadow_table}"

        # Get distinct documents for this tenant
        cursor.execute(f"""
            SELECT DISTINCT doc_id, MAX(title) as title
            FROM {fqtn}
            WHERE tenant_id = %s
            GROUP BY doc_id
        """, (tenant_id,))

        rows = cursor.fetchall()

        matches = []

        for doc_id, title in rows:
            # Match against both doc_id and title
            score_by_id = _fuzzy_match_score(name, doc_id) if doc_id else 0.0
            score_by_title = _fuzzy_match_score(name, title) if title else 0.0
            best_score = max(score_by_id, score_by_title)

            if best_score >= 0.6:
                matches.append((doc_id, title, best_score))

        if matches:
            # Sort by score descending
            matches.sort(key=lambda x: x[2], reverse=True)

            # Take top match(es) with similar scores
            top_score = matches[0][2]
            similar_matches = [m for m in matches if m[2] >= top_score - 0.1]

            doc_ids = [m[0] for m in similar_matches]
            doc_titles = [m[1] for m in similar_matches if m[1]]

            return doc_ids, doc_titles, top_score

        return None, None, 0.0

    except Exception as e:
        logger.warning(f"scope_resolver doc match failed | err={type(e).__name__}: {e}")
        return None, None, 0.0
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def resolve_scope_from_query(
    query: str,
    tenant_id: str,
) -> ResolvedScope:
    """
    Detect and resolve scope from query text.

    Called when UI doesn't provide explicit scope. Parses query for
    scope phrases like "in folder X" or "use only document Y".

    Args:
        query: User query string
        tenant_id: Tenant identifier

    Returns:
        ResolvedScope with folder_id, doc_ids, confidence, and reason.
        High confidence (>=0.8) means auto-apply is recommended.
        Low confidence means log suggestion but don't auto-apply.
    """
    # Extract scope phrase from query
    extracted_name, scope_type = _extract_scope_phrase(query)

    if not extracted_name:
        return ResolvedScope(reason="no_scope_phrase", source="none")

    if scope_type == "folder":
        folder_id, folder_name, confidence = _match_folder(extracted_name, tenant_id)

        if folder_id and confidence >= 0.6:
            return ResolvedScope(
                folder_id=folder_id,
                folder_name=folder_name,
                confidence=confidence,
                reason=f"folder_match:{extracted_name}->{folder_name}",
                source="auto",
            )
        else:
            return ResolvedScope(
                confidence=0.0,
                reason=f"folder_no_match:{extracted_name}",
                source="none",
            )

    elif scope_type == "document":
        doc_ids, doc_titles, confidence = _match_document(extracted_name, tenant_id)

        if doc_ids and confidence >= 0.6:
            return ResolvedScope(
                doc_ids=doc_ids,
                doc_titles=doc_titles,
                confidence=confidence,
                reason=f"doc_match:{extracted_name}->{doc_titles[0] if doc_titles else doc_ids[0]}",
                source="auto",
            )
        else:
            return ResolvedScope(
                confidence=0.0,
                reason=f"doc_no_match:{extracted_name}",
                source="none",
            )

    return ResolvedScope(reason="unknown_scope_type", source="none")


def get_folders(tenant_id: str) -> list[dict]:
    """
    Get list of folders for a tenant.

    Returns:
        List of dicts with folder_id, folder_name, doc_count
    """
    if not settings.database_url:
        return []

    conn = None
    try:
        conn = psycopg2.connect(settings.database_url)
        cursor = conn.cursor()

        fqtn = f"{settings.fts_shadow_schema}.{settings.fts_shadow_table}"

        cursor.execute(f"""
            SELECT
                folder_id,
                MAX(folder_name) as folder_name,
                COUNT(DISTINCT doc_id) as doc_count
            FROM {fqtn}
            WHERE tenant_id = %s AND folder_id IS NOT NULL
            GROUP BY folder_id
            ORDER BY MAX(folder_name) NULLS LAST, folder_id
        """, (tenant_id,))

        rows = cursor.fetchall()

        return [
            {
                "folder_id": row[0],
                "folder_name": row[1],
                "doc_count": row[2],
            }
            for row in rows
        ]

    except Exception as e:
        logger.warning(f"get_folders failed | tenant={tenant_id} | err={type(e).__name__}: {e}")
        return []
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def get_documents_in_folder(
    tenant_id: str,
    folder_id: str | None = None,
) -> list[dict]:
    """
    Get documents, optionally filtered by folder.

    Args:
        tenant_id: Tenant identifier
        folder_id: Optional folder_id to filter by

    Returns:
        List of dicts with doc_id, title, chunk_count, folder_id, folder_name
    """
    if not settings.database_url:
        return []

    conn = None
    try:
        conn = psycopg2.connect(settings.database_url)
        cursor = conn.cursor()

        fqtn = f"{settings.fts_shadow_schema}.{settings.fts_shadow_table}"

        if folder_id:
            cursor.execute(f"""
                SELECT
                    doc_id,
                    MAX(title) as title,
                    COUNT(*) as chunk_count,
                    MAX(folder_id) as folder_id,
                    MAX(folder_name) as folder_name
                FROM {fqtn}
                WHERE tenant_id = %s AND folder_id = %s
                GROUP BY doc_id
                ORDER BY MAX(title) NULLS LAST, doc_id
            """, (tenant_id, folder_id))
        else:
            cursor.execute(f"""
                SELECT
                    doc_id,
                    MAX(title) as title,
                    COUNT(*) as chunk_count,
                    MAX(folder_id) as folder_id,
                    MAX(folder_name) as folder_name
                FROM {fqtn}
                WHERE tenant_id = %s
                GROUP BY doc_id
                ORDER BY MAX(title) NULLS LAST, doc_id
            """, (tenant_id,))

        rows = cursor.fetchall()

        return [
            {
                "doc_id": row[0],
                "title": row[1],
                "chunk_count": row[2],
                "folder_id": row[3],
                "folder_name": row[4],
            }
            for row in rows
        ]

    except Exception as e:
        logger.warning(f"get_documents_in_folder failed | tenant={tenant_id} | err={type(e).__name__}: {e}")
        return []
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
