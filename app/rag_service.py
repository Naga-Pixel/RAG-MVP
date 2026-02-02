"""
RAG service for query processing and answer generation.
"""
import re
import uuid

from fastapi import HTTPException
from openai import OpenAI, RateLimitError
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.config import settings
from app.logging_config import get_logger
from app.qdrant_client import client as qdrant_client
from app.models import AskResponse, Source, Citation
from app.reranker import rerank_and_filter, RankedChunk, RerankerStats
from app.fts_shadow import keyword_retrieve

logger = get_logger(__name__)
client = OpenAI(api_key=settings.openai_api_key)

# Regex for extracting citations from answer text
# Matches: [doc_id], [doc_id:12], [doc_id#12]
CITATION_PATTERN = re.compile(r"\[(?P<doc>[A-Za-z0-9_\-]+)(?:(?:#|:)(?P<chunk>\d+))?\]")


def _log_keyword_comparison(
    query_id: str,
    query: str,
    tenant_id: str,
    vector_points: list,
) -> None:
    """
    Phase B: Log comparison between vector and keyword retrieval results.

    This function is for LOGGING ONLY:
    - Does NOT affect /ask responses
    - Does NOT merge results
    - Fail-open: errors are logged, execution continues

    Args:
        query_id: Request identifier for correlation
        query: User query string
        tenant_id: Tenant identifier
        vector_points: Points returned by vector retrieval
    """
    try:
        # Extract vector result summary (top 5)
        vec_top = []
        for p in vector_points[:5]:
            payload = p.payload or {}
            doc_id = payload.get("doc_id", "?")
            chunk_id = str(p.id)[:8] if p.id else "?"
            vec_top.append(f"{doc_id}:{chunk_id}")

        # Run keyword retrieval
        kw_results = keyword_retrieve(query=query, tenant_id=tenant_id, limit=10)

        # Extract keyword result summary (top 5)
        kw_top = []
        for r in kw_results[:5]:
            doc_id = r.get("doc_id", "?")
            chunk_id = str(r.get("chunk_id", "?"))[:8]
            kw_top.append(f"{doc_id}:{chunk_id}")

        # Calculate overlap (chunk_ids present in both)
        vec_chunk_ids = {str(p.id) for p in vector_points if p.id}
        kw_chunk_ids = {r.get("chunk_id") for r in kw_results if r.get("chunk_id")}
        overlap_count = len(vec_chunk_ids & kw_chunk_ids)

        # Truncate query for logging (max 80 chars)
        query_log = query[:80] + "..." if len(query) > 80 else query
        query_log = query_log.replace("\n", " ").replace("|", "/")

        # Single structured log line
        logger.info(
            f"kw_compare | rid={query_id} | query=\"{query_log}\" | "
            f"vec_count={len(vector_points)} | vec_top=[{', '.join(vec_top)}] | "
            f"kw_count={len(kw_results)} | kw_top=[{', '.join(kw_top)}] | "
            f"overlap={overlap_count}"
        )

    except Exception as e:
        # Fail-open: log warning, do not affect main flow
        logger.warning(
            f"kw_compare failed | rid={query_id} | err={type(e).__name__}: {e}"
        )


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


def extract_citations(answer: str) -> list[Citation]:
    """
    Extract citations from answer text.

    Supported formats:
    - [doc_id]
    - [doc_id:12]
    - [doc_id#12]

    Returns:
        List of Citation objects with doc_id and optional chunk_id.
    """
    citations: list[Citation] = []
    seen = set()  # Deduplicate

    for match in CITATION_PATTERN.finditer(answer):
        doc_id = match.group("doc")
        chunk_str = match.group("chunk")
        chunk_id = int(chunk_str) if chunk_str else None

        # Create tuple for deduplication
        citation_key = (doc_id, chunk_id)
        if citation_key not in seen:
            seen.add(citation_key)
            citations.append(Citation(doc_id=doc_id, chunk_id=chunk_id))

    return citations


def filter_cited_sources(sources: list[Source], citations: list[Citation]) -> list[Source]:
    """
    Filter retrieved sources to only those that were cited in the answer.

    Rules:
    - If citation is doc-level (chunk_id is None), include all chunks from that doc.
    - If citation includes chunk_id, include only matching chunk(s).
    - Preserve original order from sources list.
    - Deduplicate by (doc_id, chunk_id).

    Args:
        sources: All retrieved sources
        citations: Parsed citations from answer text

    Returns:
        Filtered list of sources that were actually cited.
    """
    if not citations:
        return []

    # Build lookup sets for efficient filtering
    doc_level_citations = {c.doc_id for c in citations if c.chunk_id is None}
    chunk_level_citations = {(c.doc_id, c.chunk_id) for c in citations if c.chunk_id is not None}

    cited_sources = []
    seen = set()  # Deduplicate by (doc_id, chunk_id)

    for source in sources:
        # Check if this source should be included
        include = False

        # Doc-level match: citation is [doc_id] without chunk_id
        if source.doc_id in doc_level_citations:
            include = True

        # Chunk-level match: citation is [doc_id:chunk_id] or [doc_id#chunk_id]
        if source.chunk_id is not None:
            try:
                chunk_num = int(source.chunk_id)
                if (source.doc_id, chunk_num) in chunk_level_citations:
                    include = True
            except (ValueError, TypeError):
                # chunk_id is not a valid integer, skip chunk-level matching
                pass

        # Add if matched and not already seen
        if include:
            dedup_key = (source.doc_id, source.chunk_id)
            if dedup_key not in seen:
                seen.add(dedup_key)
                cited_sources.append(source)

    return cited_sources


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
        chunk_id = str(p.id) if p.id else None

        snippet = text[:250] + "..." if len(text) > 250 else text

        context_blocks.append(f"[{doc_id}] {text}")
        sources.append(
            Source(
                doc_id=doc_id,
                title=title,
                snippet=snippet,
                chunk_id=chunk_id,
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
        chunk_id = chunk.point_id

        snippet = text[:250] + "..." if len(text) > 250 else text

        context_blocks.append(f"[{doc_id}] {text}")
        sources.append(
            Source(
                doc_id=doc_id,
                title=title,
                snippet=snippet,
                chunk_id=chunk_id,
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

    points = retrieve(query, top_k=retrieve_count, tenant_id=tenant_id)

    # Phase B: Keyword retrieval comparison logging (does NOT affect responses)
    if settings.keyword_retrieval_logging_enabled:
        _log_keyword_comparison(
            query_id=query_id,
            query=query,
            tenant_id=tenant_id or settings.default_tenant_id,
            vector_points=points,
        )

    if not points:
        logger.debug(
            f"[{query_id}] RAG retrieval | "
            f"rerank_enabled={settings.rerank_enabled} | "
            f"retrieved=0 | final=0 | selected=[]"
        )
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
        ranked_chunks, stats = rerank_and_filter(
            query=query,
            chunks=chunks,
            rerank_k=rerank_k,
            final_k=settings.final_k,
            max_per_doc=settings.max_chunks_per_doc,
        )

        # Build selected list: (doc_id, chunk_id)
        selected = [(c.doc_id, c.point_id) for c in ranked_chunks]

        # ONE structured debug log per request
        logger.debug(
            f"[{query_id}] RAG retrieval | "
            f"rerank_enabled=True | "
            f"retrieved={stats.input_count} | "
            f"after_dedup={stats.after_dedup} | "
            f"after_diversity={stats.after_diversity} | "
            f"final={stats.final_count} | "
            f"selected={selected}"
        )

        context, sources = build_context_from_ranked(ranked_chunks)
    else:
        # Original behavior without reranking
        chosen = points[:settings.final_k]
        selected = [
            (p.payload.get("doc_id", "unknown"), str(p.id))
            for p in chosen
            if p.payload
        ]

        # ONE structured debug log per request
        logger.debug(
            f"[{query_id}] RAG retrieval | "
            f"rerank_enabled=False | "
            f"retrieved={len(points)} | "
            f"final={len(chosen)} | "
            f"selected={selected}"
        )

        context, sources = build_context(points)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a retrieval-augmented assistant with strict grounding requirements.\n\n"
                "You MUST follow these rules exactly:\n\n"
                "1. Use ONLY the provided document context to answer the user's question.\n"
                "   Do NOT use general knowledge, assumptions, or prior training.\n\n"
                "2. Every factual claim in your answer MUST be directly supported by at least one\n"
                "   source from the provided context.\n\n"
                "3. Cite sources at the end of each factual sentence using the exact source\n"
                "   identifier as provided in the context (do not invent identifiers).\n\n"
                "4. If the answer is NOT explicitly found in the provided documents, respond with\n"
                "   exactly:\n"
                '   "Not found in the documents."\n\n'
                "5. Do NOT infer, extrapolate, guess, or fill in missing details.\n"
                "   If something is implied but not stated, treat it as NOT found.\n\n"
                "6. If the question is ambiguous or underspecified, ask a clarifying question\n"
                "   instead of answering.\n\n"
                "7. Do NOT hallucinate information. If you cannot support a claim with a source,\n"
                "   you must not include it.\n\n"
                "Notes:\n"
                "- Only factual claims require citations.\n"
                '- Clarifying questions and the response "Not found in the documents." do NOT\n'
                "  require citations.\n"
                "- Keep answers concise and focused on the user's question."
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

    # Extract citations from answer text
    citations = extract_citations(answer)

    # Filter sources to only those cited
    sources_cited = filter_cited_sources(sources, citations)

    # Log citation tracking
    logger.debug(
        f"[{query_id}] Citation tracking | "
        f"retrieved={len(sources)} | "
        f"citations={len(citations)} | "
        f"cited={len(sources_cited)}"
    )

    # Warn if citations exist but no sources matched
    if citations and not sources_cited:
        citation_labels = []
        for c in citations:
            if c.chunk_id:
                citation_labels.append(f"[{c.doc_id}:{c.chunk_id}]")
            else:
                citation_labels.append(f"[{c.doc_id}]")

        logger.warning(
            "[%s] Citations found but no matching sources | citations=%s",
            query_id,
            citation_labels,
        )


    return AskResponse(
        answer=answer,
        sources=sources,  # Unchanged for backwards compatibility
        citations=citations,
        sources_cited=sources_cited,
        sources_retrieved=sources,
    )
