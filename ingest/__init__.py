"""
Ingestion module for b_rag.

Components:
- loaders: File format parsers (PDF, DOCX, XLSX, etc.)
- chunking: Text chunking utilities
- embedder: OpenAI embedding generation
- pipeline: Unified ingestion pipeline for all connectors

Usage:
    # Using CLI
    python -m ingest.ingest_cli --dir data/raw --recreate
    
    # Using Python
    from connectors import LocalFilesConnector
    from ingest.pipeline import ingest
    
    connector = LocalFilesConnector("data/raw")
    stats = ingest(connector, recreate=True)
"""
from ingest.loaders import load_document, iter_documents
from ingest.chunking import chunk_text, Chunk
from ingest.embedder import get_embeddings, get_single_embedding
from ingest.pipeline import IngestionPipeline, IngestionStats, ingest

__all__ = [
    # Loaders
    "load_document",
    "iter_documents",
    # Chunking
    "chunk_text",
    "Chunk",
    # Embeddings
    "get_embeddings",
    "get_single_embedding",
    # Pipeline
    "IngestionPipeline",
    "IngestionStats",
    "ingest",
]
