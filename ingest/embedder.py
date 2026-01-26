"""
Embedding utilities for the ingestion pipeline.
Uses OpenAI's embedding API with batching for large documents.
"""
import time
from openai import OpenAI, RateLimitError

from app.config import settings


def get_embeddings(texts: list[str], batch_size: int = 100) -> tuple[list[list[float]], int]:
    """
    Generate embeddings for a list of texts using OpenAI's API.
    Batches requests to avoid rate limits and memory issues.
    
    Args:
        texts: List of text strings to embed.
        batch_size: Number of texts to embed per API call.
    
    Returns:
        Tuple of (embeddings list, embedding dimension).
    """
    if not texts:
        return [], 0
    
    client = OpenAI(api_key=settings.openai_api_key)
    all_embeddings = []
    dimension = 0
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        retries = 0
        max_retries = 3
        
        while retries < max_retries:
            try:
                response = client.embeddings.create(
                    model=settings.embedding_model,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                if dimension == 0 and batch_embeddings:
                    dimension = len(batch_embeddings[0])
                
                break  # Success, exit retry loop
                
            except RateLimitError:
                retries += 1
                if retries < max_retries:
                    wait_time = 2 ** retries  # Exponential backoff: 2, 4, 8 seconds
                    print(f"    Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
    
    return all_embeddings, dimension


def get_single_embedding(text: str) -> tuple[list[float], int]:
    """
    Generate embedding for a single text.
    
    Args:
        text: Text string to embed.
    
    Returns:
        Tuple of (embedding vector, embedding dimension).
    """
    embeddings, dim = get_embeddings([text])
    return embeddings[0] if embeddings else [], dim
