"""
RAG service for query processing and answer generation.
"""
from fastapi import HTTPException
from openai import OpenAI, RateLimitError
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.config import settings
from app.qdrant_client import client as qdrant_client
from app.models import AskResponse, Source


client = OpenAI(api_key=settings.openai_api_key)


def retrieve(query: str, top_k: int | None = None):
    """
    Embed the query and retrieve top_k most similar chunks from Qdrant.
    Uses query_points (current qdrant-client API).
    """
    if top_k is None:
        top_k = settings.retrieval_limit

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
                match=MatchValue(value=settings.default_tenant_id),
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
    """
    if not points:
        return "", []

    # Simple MVP strategy: take up to 6 top points
    chosen = points[:6]
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


def answer_question(query: str) -> AskResponse:
    """
    Full RAG pipeline:
      1. Retrieve top chunks from Qdrant
      2. Build context
      3. Ask OpenAI to answer using ONLY that context
    """
    points = retrieve(query)

    if not points:
        return AskResponse(
            answer="No relevant information found in the indexed documents.",
            sources=[],
        )

    context, sources = build_context(points)

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
    return AskResponse(answer=answer, sources=sources)
