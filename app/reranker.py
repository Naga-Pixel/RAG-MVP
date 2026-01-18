"""
Reranker module for RAG retrieval.

Provides an interface for reranking retrieved chunks against a query.
Currently implements a stub that uses vector similarity scores.
Designed to be extended with real rerankers (Cohere, cross-encoder, etc.).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RankedChunk:
    """A chunk with its rerank score."""
    point_id: str
    text: str
    doc_id: str
    original_score: float
    rerank_score: float
    payload: dict


class Reranker(Protocol):
    """Protocol for reranker implementations."""

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int | None = None,
    ) -> list[RankedChunk]:
        """
        Rerank chunks against a query.

        Args:
            query: The user's query.
            chunks: List of chunk dicts with 'text', 'doc_id', 'score', 'payload'.
            top_k: Number of top results to return (None = return all).

        Returns:
            List of RankedChunk sorted by rerank_score descending.
        """
        ...


class VectorScoreReranker:
    """
    Stub reranker that uses the original vector similarity score.

    This is a passthrough that maintains the original ranking.
    Replace with a real reranker (Cohere, cross-encoder) for better results.
    """

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int | None = None,
    ) -> list[RankedChunk]:
        """Use original vector scores as rerank scores."""
        ranked = []
        for chunk in chunks:
            ranked.append(
                RankedChunk(
                    point_id=str(chunk.get("point_id", "")),
                    text=chunk.get("text", ""),
                    doc_id=chunk.get("doc_id", "unknown"),
                    original_score=chunk.get("score", 0.0),
                    rerank_score=chunk.get("score", 0.0),  # Use vector score as rerank score
                    payload=chunk.get("payload", {}),
                )
            )

        # Sort by rerank score (descending)
        ranked.sort(key=lambda x: x.rerank_score, reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked


# Future implementation example:
# class CohereReranker:
#     """Reranker using Cohere's rerank API."""
#
#     def __init__(self, api_key: str, model: str = "rerank-english-v2.0"):
#         self.client = cohere.Client(api_key)
#         self.model = model
#
#     def rerank(self, query: str, chunks: list[dict], top_k: int | None = None) -> list[RankedChunk]:
#         texts = [c["text"] for c in chunks]
#         response = self.client.rerank(query=query, documents=texts, model=self.model, top_n=top_k)
#         # Map response back to RankedChunk...


def get_reranker() -> Reranker:
    """
    Factory function to get the configured reranker.

    Currently returns the stub VectorScoreReranker.
    Extend this to return different rerankers based on config.
    """
    # TODO: Add config-based selection when real rerankers are added
    # e.g., if settings.rerank_provider == "cohere": return CohereReranker(...)
    return VectorScoreReranker()


def deduplicate_chunks(chunks: list[RankedChunk]) -> list[RankedChunk]:
    """
    Remove chunks with identical text content.

    Keeps the first occurrence (highest rerank score due to sorting).
    """
    seen_texts = set()
    deduped = []

    for chunk in chunks:
        # Normalize text for comparison
        text_key = chunk.text.strip().lower()
        if text_key not in seen_texts:
            seen_texts.add(text_key)
            deduped.append(chunk)

    return deduped


def apply_diversity(
    chunks: list[RankedChunk],
    max_per_doc: int = 3,
) -> list[RankedChunk]:
    """
    Limit chunks from the same document to ensure diversity.

    Args:
        chunks: List of RankedChunk, assumed sorted by rerank_score.
        max_per_doc: Maximum chunks to include from the same document.

    Returns:
        Filtered list respecting the per-document limit.
    """
    doc_counts: dict[str, int] = {}
    diverse = []

    for chunk in chunks:
        doc_id = chunk.doc_id
        current_count = doc_counts.get(doc_id, 0)

        if current_count < max_per_doc:
            diverse.append(chunk)
            doc_counts[doc_id] = current_count + 1

    return diverse


def rerank_and_filter(
    query: str,
    chunks: list[dict],
    rerank_k: int | None = None,
    final_k: int = 6,
    max_per_doc: int = 3,
    reranker: Reranker | None = None,
) -> list[RankedChunk]:
    """
    Full rerank pipeline: rerank → deduplicate → apply diversity → limit to final_k.

    Args:
        query: User's query.
        chunks: Raw chunks from vector search.
        rerank_k: Number of candidates to rerank (None = all).
        final_k: Number of final chunks to return.
        max_per_doc: Max chunks per document for diversity.
        reranker: Reranker instance (defaults to VectorScoreReranker).

    Returns:
        List of RankedChunk ready for context building.
    """
    if reranker is None:
        reranker = get_reranker()

    # Rerank
    ranked = reranker.rerank(query, chunks, top_k=rerank_k)
    logger.debug(f"Reranked {len(chunks)} chunks → {len(ranked)} candidates")

    # Deduplicate
    deduped = deduplicate_chunks(ranked)
    if len(deduped) < len(ranked):
        logger.debug(f"Deduplicated: {len(ranked)} → {len(deduped)} chunks")

    # Apply diversity
    diverse = apply_diversity(deduped, max_per_doc=max_per_doc)
    if len(diverse) < len(deduped):
        logger.debug(f"Diversity filter: {len(deduped)} → {len(diverse)} chunks")

    # Limit to final_k
    result = diverse[:final_k]
    logger.debug(f"Final selection: {len(result)} chunks")

    return result
