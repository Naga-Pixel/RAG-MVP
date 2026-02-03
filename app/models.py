from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str
    folder_id: str | None = None  # Optional: restrict search to specific folder
    doc_ids: list[str] | None = None  # Optional: restrict search to specific document(s)


class Source(BaseModel):
    doc_id: str
    title: str | None = None
    snippet: str
    chunk_id: str | None = None  # Qdrant point ID
    folder_id: str | None = None
    folder_name: str | None = None


class Citation(BaseModel):
    doc_id: str
    chunk_id: str | None = None


class ScopeMetadata(BaseModel):
    """Metadata about scope resolution for transparency."""
    folder_id: str | None = None
    folder_name: str | None = None
    doc_ids: list[str] | None = None
    source: str = "none"  # "ui" | "auto" | "none"
    confidence: float = 0.0
    reason: str = ""


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]  # Retrieved sources (unchanged for backwards compatibility)
    citations: list[Citation] = Field(default_factory=list)  # Parsed from answer text
    sources_cited: list[Source] = Field(default_factory=list)  # Subset of sources that were actually cited
    sources_retrieved: list[Source] = Field(default_factory=list)  # All retrieved sources
    scope: ScopeMetadata | None = None  # Scope metadata for transparency

