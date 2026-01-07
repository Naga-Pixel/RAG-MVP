from pydantic import BaseModel

class AskRequest(BaseModel):
    query: str

class Source(BaseModel):
    doc_id: str
    title: str | None = None
    snippet: str

class AskResponse(BaseModel):
    answer: str
    sources: list[Source]

