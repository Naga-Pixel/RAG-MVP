"""
Qdrant client configuration and utilities.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.config import settings

client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
)


def get_qdrant_client() -> QdrantClient:
    """Return the Qdrant client singleton."""
    return client


def collection_exists() -> bool:
    """Check if the configured collection exists."""
    collections = client.get_collections().collections
    names = [c.name for c in collections]
    return settings.qdrant_collection in names


def ensure_collection(dimension: int = 1536) -> None:
    """
    Ensure the Qdrant collection exists.
    
    Args:
        dimension: Vector dimension (default 1536 for text-embedding-3-small).
    """
    if not collection_exists():
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE,
            ),
        )


def recreate_collection(dimension: int) -> None:
    """
    Recreate the collection with specified dimension.
    Deletes existing data.
    
    Args:
        dimension: Vector dimension for the new collection.
    """
    client.recreate_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(
            size=dimension,
            distance=Distance.COSINE,
        ),
    )


def upsert_chunks(chunks: list[dict], vectors: list[list[float]]) -> None:
    """
    Upsert chunks with their embeddings into Qdrant.
    
    Args:
        chunks: List of chunk dictionaries with chunk_id, text, and metadata.
        vectors: List of embedding vectors corresponding to each chunk.
    """
    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append(
            PointStruct(
                id=chunk["chunk_id"],
                vector=vector,
                payload={**chunk["metadata"], "text": chunk["text"]},
            )
        )
    client.upsert(collection_name=settings.qdrant_collection, points=points)
