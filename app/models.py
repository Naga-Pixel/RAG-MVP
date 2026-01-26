from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    query: str

class Source(BaseModel):
    doc_id: str
    title: str | None = None
    snippet: str
    chunk_id: str | None = None  # Qdrant point ID

class Citation(BaseModel):
    doc_id: str
    chunk_id: str | None = None

class AskResponse(BaseModel):
    answer: str
    sources: list[Source]  # Retrieved sources (unchanged for backwards compatibility)
    citations: list[Citation] = Field(default_factory=list)  # Parsed from answer text
    sources_cited: list[Source] = Field(default_factory=list)  # Subset of sources that were actually cited
    sources_retrieved: list[Source] = Field(default_factory=list)  # All retrieved sources

