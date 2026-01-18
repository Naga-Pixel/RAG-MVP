"""
RAG service for query processing and answer generation.
"""
import uuid

from fastapi import HTTPException
from openai import OpenAI, RateLimitError
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.config import settings
from app.logging_config import get_logger
from app.qdrant_client import client as qdrant_client
from app.models import AskResponse, Source
from app.reranker import rerank_and_filter, RankedChunk

logger = get_logger(__name__)
client = OpenAI(api_key=settings.openai_api_key)


def retrieve(query: str, top_k: int | None = None, tenant_id: str | None = None):
    """
    Embed the query and retrieve top_k most similar chunks from Qdrant.
    Uses query_points (current qdrant-client API).
    """
    if top_k is None:
        top_k = settings.retrieval_limit
    if tenant_id is None:
        tenant_id = settings.default_tenant_id

    try:
        emb_resp = client.embeddings.create(
            model=settings.embedding_model,
            input=query,
        )
    except RateLimitError as e:
        raise HTTPException(
            status_code=503,
            detail="OpenAI rate limit / quota exceeded. Check billing or try again later.",
        ) from e

    embedding = emb_resp.data[0].embedding

    # Use proper Qdrant filter models for query_points
    query_filter = Filter(
        must=[
            FieldCondition(
                key="tenant_id",
                match=MatchValue(value=tenant_id),
            )
        ]
    )

    # query_points is the current recommended search API
    resp = qdrant_client.query_points(
        collection_name=settings.qdrant_collection,
        query=embedding,
        limit=top_k,
        query_filter=query_filter,
    )
    
    return resp.points


def build_context(points):
    """
    Build context string and sources list from Qdrant points.
    Used when reranking is disabled.
    """
    if not points:
        return "", []

    # Use final_k from settings (default 6)
    chosen = points[:settings.final_k]
    context_blocks: list[str] = []
    sources: list[Source] = []

    for p in chosen:
        payload = p.payload or {}
        text: str = payload.get("text", "")
        doc_id: str = payload.get("doc_id", "unknown")
        title = payload.get("title")

        snippet = text[:250] + "..." if len(text) > 250 else text

        context_blocks.append(f"[{doc_id}] {text}")
        sources.append(
            Source(
                doc_id=doc_id,
                title=title,
                snippet=snippet,
            )
        )

    context = "\n\n---\n\n".join(context_blocks)
    return context, sources


def build_context_from_ranked(ranked_chunks: list[RankedChunk]):
    """
    Build context string and sources list from reranked chunks.
    """
    if not ranked_chunks:
        return "", []

    context_blocks: list[str] = []
    sources: list[Source] = []

    for chunk in ranked_chunks:
        text = chunk.text
        doc_id = chunk.doc_id
        title = chunk.payload.get("title")

        snippet = text[:250] + "..." if len(text) > 250 else text

        context_blocks.append(f"[{doc_id}] {text}")
        sources.append(
            Source(
                doc_id=doc_id,
                title=title,
                snippet=snippet,
            )
        )

    context = "\n\n---\n\n".join(context_blocks)
    return context, sources


def answer_question(query: str, tenant_id: str | None = None) -> AskResponse:
    """
    Full RAG pipeline:
      1. Retrieve top chunks from Qdrant
      2. Optionally rerank, deduplicate, and apply diversity
      3. Build context
      4. Ask OpenAI to answer using ONLY that context
    """
    # Generate query ID for logging
    query_id = str(uuid.uuid4())[:8]

    # Determine retrieval count based on rerank settings
    if settings.rerank_enabled:
        retrieve_count = settings.retrieve_k
    else:
        # When rerank disabled, use retrieval_limit for backwards compatibility
        retrieve_count = settings.retrieval_limit

    logger.debug(
        f"[{query_id}] RAG query | rerank_enabled={settings.rerank_enabled} | "
        f"retrieve_k={retrieve_count} | final_k={settings.final_k}"
    )

    points = retrieve(query, top_k=retrieve_count, tenant_id=tenant_id)
    logger.debug(f"[{query_id}] Retrieved {len(points)} candidates from Qdrant")

    if not points:
        logger.debug(f"[{query_id}] No results found")
        return AskResponse(
            answer="No relevant information found in the indexed documents.",
            sources=[],
        )

    if settings.rerank_enabled:
        # Convert Qdrant points to chunk dicts for reranker
        chunks = []
        for p in points:
            payload = p.payload or {}
            chunks.append({
                "point_id": str(p.id),
                "text": payload.get("text", ""),
                "doc_id": payload.get("doc_id", "unknown"),
                "score": p.score if hasattr(p, "score") else 0.0,
                "payload": payload,
            })

        # Rerank, deduplicate, and apply diversity
        rerank_k = settings.rerank_k if settings.rerank_k else settings.retrieve_k
        ranked_chunks = rerank_and_filter(
            query=query,
            chunks=chunks,
            rerank_k=rerank_k,
            final_k=settings.final_k,
            max_per_doc=settings.max_chunks_per_doc,
        )

        # Log selected chunks
        selected_ids = [(c.point_id, c.doc_id) for c in ranked_chunks]
        logger.debug(f"[{query_id}] Final chunks: {selected_ids}")

        context, sources = build_context_from_ranked(ranked_chunks)
    else:
        # Original behavior without reranking
        context, sources = build_context(points)
        logger.debug(f"[{query_id}] Using top {len(sources)} chunks (no rerank)")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a retrieval-augmented assistant. Use ONLY the provided "
                "context to answer the user's question. If the answer is not in "
                "the context, say \"I don't know\" and do not hallucinate."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        },
    ]

    try:
        chat_resp = client.chat.completions.create(
            model=settings.chat_model,
            messages=messages,
            temperature=0.2,
        )
    except RateLimitError as e:
        raise HTTPException(
            status_code=503,
            detail="OpenAI rate limit / quota exceeded during answer generation.",
        ) from e

    answer = chat_resp.choices[0].message.content
    logger.debug(f"[{query_id}] Answer generated successfully")
    return AskResponse(answer=answer, sources=sources)
