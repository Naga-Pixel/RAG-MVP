"""
Text chunking utilities for the ingestion pipeline.
"""
import hashlib
import uuid
from typing import TypedDict


class Chunk(TypedDict):
    chunk_id: str
    text: str
    metadata: dict


def generate_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    """Generate a deterministic UUID chunk ID based on document ID and content."""
    content = f"{doc_id}:{chunk_index}:{text[:100]}"  # Only use first 100 chars for speed
    content_hash = hashlib.md5(content.encode()).hexdigest()
    return str(uuid.UUID(content_hash))


def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
    metadata: dict | None = None,
) -> list[Chunk]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: The text to chunk.
        doc_id: Document identifier for generating chunk IDs.
        chunk_size: Target size of each chunk in characters.
        chunk_overlap: Number of overlapping characters between chunks.
        metadata: Additional metadata to attach to each chunk.
    
    Returns:
        List of Chunk dictionaries with chunk_id, text, and metadata.
    """
    if metadata is None:
        metadata = {}
    
    # Clean and normalize text
    text = text.strip()
    if not text:
        return []
    
    # Ensure overlap is less than chunk_size
    chunk_overlap = min(chunk_overlap, chunk_size // 2)
    
    # For very short texts, return as single chunk
    if len(text) <= chunk_size:
        return [
            Chunk(
                chunk_id=generate_chunk_id(doc_id, 0, text),
                text=text,
                metadata={**metadata, "doc_id": doc_id, "chunk_index": 0},
            )
        ]
    
    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # Try to break at word boundary if not at the end
        if end < len(text):
            # Look for last space in the chunk
            space_pos = text.rfind(' ', start + chunk_size // 2, end)
            if space_pos > start:
                end = space_pos
        
        chunk_text_content = text[start:end].strip()
        
        if chunk_text_content:
            chunks.append(
                Chunk(
                    chunk_id=generate_chunk_id(doc_id, chunk_index, chunk_text_content),
                    text=chunk_text_content,
                    metadata={**metadata, "doc_id": doc_id, "chunk_index": chunk_index},
                )
            )
            chunk_index += 1
        
        # Move forward, ensuring progress
        new_start = end - chunk_overlap
        if new_start <= start:
            new_start = start + chunk_size // 2  # Force progress
        start = new_start
        
        # Safety check
        if chunk_index > 10000:
            print(f"Warning: Too many chunks, stopping at {chunk_index}")
            break
    
    return chunks
