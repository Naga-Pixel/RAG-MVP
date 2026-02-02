"""
Document-level Derived Summaries module (Phase D.1).

Generates explicit factual statements from implicit document content at ingest time.

IMPORTANT: This is an INGEST-TIME ONLY feature.
- Does NOT change retrieval logic
- Does NOT change answer prompts
- Does NOT loosen inference rules
- Summaries are indexed as normal chunks and fully citeable

Supported document types:
- Meeting transcripts: decisions, action items, open issues, risks
- Narrative documents: key facts, outcomes, developments

Feature flag: DOCUMENT_SUMMARY_ENABLED
"""
import json
import re
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI, RateLimitError
import sentry_sdk

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

DocumentType = Literal["meeting", "narrative", "unknown"]
SummaryCategory = Literal["decision", "action", "open_issue", "risk", "narrative_fact"]


@dataclass
class SummaryItem:
    """A single derived summary statement."""
    scope: str  # Always "document" for Phase D.1
    type: str  # Always "derived_summary"
    document_type: DocumentType
    category: SummaryCategory
    text: str


# Meeting transcript detection patterns
MEETING_PATTERNS = [
    r"\b(attendees?|participants?)\s*:",
    r"\b(agenda|minutes|meeting)\s+(notes?|summary)",
    r"\[(speaker|name|\d{1,2}:\d{2})\]",
    r"\b(action items?|next steps?|follow.?ups?)\s*:",
    r"^\s*-?\s*\w+:\s+",  # Speaker: dialogue pattern
    r"\b(discussed|decided|agreed|proposed|raised)\b",
]

# Narrative document detection patterns
NARRATIVE_PATTERNS = [
    r"\bchapter\s+\d+",
    r"\bpart\s+(one|two|three|i{1,3}|iv|v)",
    r"\b(narrator|author|protagonist)\b",
    r"\b(memoir|autobiography|biography)\b",
    r"\b(years? later|in \d{4}|grew up)\b",
]


def detect_document_type(content: str) -> DocumentType:
    """
    Heuristically detect document type from content.

    Args:
        content: Full document text.

    Returns:
        "meeting", "narrative", or "unknown"
    """
    content_lower = content.lower()
    content_sample = content_lower[:5000]  # Check first 5000 chars

    # Count pattern matches
    meeting_score = sum(
        1 for pattern in MEETING_PATTERNS
        if re.search(pattern, content_sample, re.IGNORECASE | re.MULTILINE)
    )

    narrative_score = sum(
        1 for pattern in NARRATIVE_PATTERNS
        if re.search(pattern, content_sample, re.IGNORECASE)
    )

    # Require at least 2 matches for confidence
    if meeting_score >= 2 and meeting_score > narrative_score:
        return "meeting"
    elif narrative_score >= 2 and narrative_score > meeting_score:
        return "narrative"

    return "unknown"


def _build_summary_prompt(content: str, document_type: DocumentType) -> str:
    """Build the summary generation prompt based on document type."""

    base_instructions = """You are generating a derived summary for a document ingested into a retrieval system.

The system does not allow inference at query time.
Convert implicit information in the document into explicit, factual statements.

Rules:
- Do not speculate or invent facts.
- Only include statements strongly supported by the document.
- Phrase each statement so it can directly answer a user question.
- Avoid prose; output short declarative statements.
- If no explicit outcomes exist, state that clearly."""

    if document_type == "meeting":
        type_instructions = """
This is a MEETING TRANSCRIPT. Generate 3-10 short factual statements covering:
- Decisions made (category: "decision")
- Commitments / action items - who agreed to do what (category: "action")
- Open or unresolved issues (category: "open_issue")
- Risks or concerns raised (category: "risk")

If something was discussed but not decided, state that explicitly as an open_issue."""

    elif document_type == "narrative":
        type_instructions = """
This is a NARRATIVE DOCUMENT (book, memoir, long-form text). Generate 2-6 explicit factual statements that:
- Capture key narrative facts or outcomes (category: "narrative_fact")
- Make implicit developments explicit
- Can directly answer user questions

Examples of good statements:
- "The author meets his father in person over the course of the book."
- "The narrative concludes without a full reconciliation."

Do not summarize themes or style. Focus on what happens, not interpretation."""

    else:
        type_instructions = """
This document's type is unclear. Generate 2-5 key factual statements that:
- Capture the most important explicit facts
- Can directly answer user questions
Use category: "narrative_fact" for all statements."""

    output_format = """

Output Format (strict JSON array):
[
  {
    "scope": "document",
    "type": "derived_summary",
    "document_type": "%s",
    "category": "<decision|action|open_issue|risk|narrative_fact>",
    "text": "<explicit factual statement>"
  }
]

Return ONLY the JSON array, no other text.""" % document_type

    # Truncate content to avoid token limits (roughly 12000 chars ~= 3000 tokens)
    max_content_chars = 12000
    if len(content) > max_content_chars:
        content = content[:max_content_chars] + "\n\n[... document truncated for summary generation ...]"

    return f"""{base_instructions}
{type_instructions}
{output_format}

Document content:
---
{content}
---"""


def _parse_summary_response(response_text: str, document_type: DocumentType) -> list[SummaryItem]:
    """Parse LLM response into SummaryItem objects."""
    items = []

    # Try to extract JSON array from response
    try:
        # Find JSON array in response
        json_match = re.search(r'\[[\s\S]*\]', response_text)
        if not json_match:
            logger.warning("doc_summary parse failed | no JSON array found")
            return []

        json_text = json_match.group(0)
        parsed = json.loads(json_text)

        if not isinstance(parsed, list):
            logger.warning("doc_summary parse failed | response is not a list")
            return []

        valid_categories = {"decision", "action", "open_issue", "risk", "narrative_fact"}

        for item in parsed:
            if not isinstance(item, dict):
                continue

            text = item.get("text", "").strip()
            category = item.get("category", "narrative_fact")

            # Validate
            if not text:
                continue
            if category not in valid_categories:
                category = "narrative_fact"

            items.append(SummaryItem(
                scope="document",
                type="derived_summary",
                document_type=document_type,
                category=category,
                text=text,
            ))

        return items

    except json.JSONDecodeError as e:
        logger.warning(f"doc_summary parse failed | JSON decode error: {e}")
        return []


def generate_document_summaries(
    content: str,
    doc_id: str,
    doc_title: str | None = None,
    document_type: DocumentType | None = None,
) -> list[SummaryItem]:
    """
    Generate document-level derived summaries.

    Fail-open: Returns empty list on any error.

    Args:
        content: Full document text.
        doc_id: Document identifier for logging.
        doc_title: Optional document title for logging.
        document_type: Override detected document type.

    Returns:
        List of SummaryItem objects, or empty list on error.
    """
    if not settings.document_summary_enabled:
        return []

    if not content or len(content.strip()) < 100:
        logger.debug(f"doc_summary skipped | doc_id={doc_id} | reason=content_too_short")
        return []

    # Detect or use provided document type
    if document_type is None:
        document_type = detect_document_type(content)

    if document_type == "unknown":
        logger.debug(f"doc_summary skipped | doc_id={doc_id} | reason=unknown_type")
        return []

    title_log = doc_title[:50] if doc_title else doc_id

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        model = settings.document_summary_model or settings.chat_model

        prompt = _build_summary_prompt(content, document_type)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=settings.document_summary_max_tokens,
        )

        response_text = response.choices[0].message.content or ""
        items = _parse_summary_response(response_text, document_type)

        logger.info(
            f"doc_summary ok | doc_id={doc_id} | title={title_log} | "
            f"type={document_type} | items={len(items)}"
        )

        return items

    except RateLimitError as e:
        logger.warning(
            f"doc_summary failed | doc_id={doc_id} | "
            f"err=RateLimitError | msg={e}"
        )
        return []

    except Exception as e:
        logger.warning(
            f"doc_summary failed | doc_id={doc_id} | "
            f"err={type(e).__name__}: {e}"
        )
        sentry_sdk.capture_exception(e)
        return []
