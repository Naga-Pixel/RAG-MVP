"""
Hybrid Retrieval module (Phase C).

Provides Reciprocal Rank Fusion (RRF) for combining vector and keyword results.

IMPORTANT: This module is used ONLY when HYBRID_ENABLED=true.
- Fail-open: keyword errors fall back to vector-only
- Preserves existing reranker and citation behavior
- Does NOT change response schema

RRF Formula: score = 1 / (k + rank)
Where k is a constant (default 60) that dampens the effect of high rankings.
"""
from dataclasses import dataclass

from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class FusedChunk:
    """A chunk with RRF fusion score."""
    point_id: str
    text: str
    doc_id: str
    payload: dict
    rrf_score: float
    vec_rank: int | None  # 1-indexed rank in vector results, None if not present
    kw_rank: int | None   # 1-indexed rank in keyword results, None if not present


@dataclass
class FusionStats:
    """Statistics from RRF fusion for logging."""
    vec_count: int
    kw_count: int
    fused_count: int
    vec_only: int   # Chunks appearing only in vector results
    kw_only: int    # Chunks appearing only in keyword results
    overlap: int    # Chunks appearing in both


def rrf_fuse(
    vec_chunks: list[dict],
    kw_results: list[dict],
    rrf_k: int = 60,
    fused_k: int = 20,
) -> tuple[list[FusedChunk], FusionStats]:
    """
    Fuse vector and keyword results using Reciprocal Rank Fusion.

    RRF assigns each document a score based on its rank in each result set:
        score = sum(1 / (k + rank)) for each list containing the document

    Args:
        vec_chunks: Vector search results, list of dicts with keys:
            - point_id (str): Qdrant point UUID
            - text (str): Chunk text
            - doc_id (str): Document identifier
            - payload (dict): Full payload from Qdrant
        kw_results: Keyword search results, list of dicts with keys:
            - chunk_id (str): Matches point_id from vector results
            - doc_id (str): Document identifier
            - rank (float): FTS rank score (not used in RRF, position matters)
        rrf_k: RRF constant k (default 60)
        fused_k: Number of top fused results to return

    Returns:
        Tuple of (fused_chunks, stats):
        - fused_chunks: Top fused_k chunks sorted by RRF score descending
        - stats: FusionStats with counts for logging
    """
    # Build lookup for vector chunks by point_id
    vec_lookup: dict[str, dict] = {}
    for chunk in vec_chunks:
        point_id = chunk.get("point_id", "")
        if point_id:
            vec_lookup[point_id] = chunk

    # Build lookup for keyword results by chunk_id
    kw_lookup: dict[str, dict] = {}
    for result in kw_results:
        chunk_id = result.get("chunk_id", "")
        if chunk_id:
            kw_lookup[chunk_id] = result

    # Calculate RRF scores
    # Using 1-indexed ranks (rank 1 = top result)
    rrf_scores: dict[str, float] = {}
    vec_ranks: dict[str, int] = {}
    kw_ranks: dict[str, int] = {}

    # Add vector contributions
    for rank_idx, chunk in enumerate(vec_chunks, start=1):
        point_id = chunk.get("point_id", "")
        if point_id:
            rrf_scores[point_id] = rrf_scores.get(point_id, 0.0) + 1.0 / (rrf_k + rank_idx)
            vec_ranks[point_id] = rank_idx

    # Add keyword contributions
    for rank_idx, result in enumerate(kw_results, start=1):
        chunk_id = result.get("chunk_id", "")
        if chunk_id:
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank_idx)
            kw_ranks[chunk_id] = rank_idx

    # Build fused chunks with scores
    fused_chunks: list[FusedChunk] = []

    for point_id, score in rrf_scores.items():
        # Get chunk data from vector results (preferred) or construct minimal from keyword
        if point_id in vec_lookup:
            chunk = vec_lookup[point_id]
            fused_chunks.append(FusedChunk(
                point_id=point_id,
                text=chunk.get("text", ""),
                doc_id=chunk.get("doc_id", "unknown"),
                payload=chunk.get("payload", {}),
                rrf_score=score,
                vec_rank=vec_ranks.get(point_id),
                kw_rank=kw_ranks.get(point_id),
            ))
        elif point_id in kw_lookup:
            # Keyword-only result: we don't have full text, skip it
            # This is intentional: we need the full chunk text for context
            # Keyword-only hits would need a separate fetch from Qdrant
            # For Phase C, we only boost chunks that appear in vector results
            continue

    # Sort by RRF score descending
    fused_chunks.sort(key=lambda c: c.rrf_score, reverse=True)

    # Limit to fused_k
    fused_chunks = fused_chunks[:fused_k]

    # Calculate stats
    vec_ids = set(vec_lookup.keys())
    kw_ids = set(kw_lookup.keys())
    overlap = len(vec_ids & kw_ids)

    stats = FusionStats(
        vec_count=len(vec_chunks),
        kw_count=len(kw_results),
        fused_count=len(fused_chunks),
        vec_only=len(vec_ids - kw_ids),
        kw_only=len(kw_ids - vec_ids),
        overlap=overlap,
    )

    return fused_chunks, stats
