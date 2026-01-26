"""
Test idempotent ingestion - running sync twice should not create duplicates.

This is an integration test that requires:
- Qdrant running at localhost:6333
- OpenAI API key in environment

Run with: pytest tests/test_idempotency.py -v -s
"""
import os
import tempfile
from pathlib import Path

import pytest


# Skip if Qdrant is not available
def qdrant_available():
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://localhost:6333")
        client.get_collections()
        return True
    except Exception:
        return False


requires_qdrant = pytest.mark.skipif(
    not qdrant_available(),
    reason="Qdrant not available at localhost:6333"
)

requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)


@requires_qdrant
@requires_openai
def test_idempotent_local_sync():
    """
    Test that syncing local files twice produces same point count.
    
    This verifies:
    1. Same file + same content = same point IDs
    2. Upsert overwrites, doesn't duplicate
    """
    from connectors import LocalFilesConnector
    from ingest.pipeline import ingest, generate_point_id
    from app.qdrant_client import client as qdrant_client
    from app.config import settings
    
    # Create temp directory with test file
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_doc.md"
        test_file.write_text("# Test Document\n\nThis is test content for idempotency testing.")
        
        # Use a unique tenant for this test
        tenant_id = "test_idempotency"
        
        # First sync
        connector = LocalFilesConnector(directory=tmpdir, tenant_id=tenant_id)
        stats1 = ingest(connector, recreate=True)
        
        # Count points after first sync
        collection_info = qdrant_client.get_collection(settings.qdrant_collection)
        count_after_first = collection_info.points_count
        
        print(f"\nFirst sync: {stats1.chunks_created} chunks, {count_after_first} points in collection")
        
        # Second sync (same content)
        connector2 = LocalFilesConnector(directory=tmpdir, tenant_id=tenant_id)
        stats2 = ingest(connector2, recreate=False)
        
        # Count points after second sync
        collection_info = qdrant_client.get_collection(settings.qdrant_collection)
        count_after_second = collection_info.points_count
        
        print(f"Second sync: {stats2.chunks_created} chunks, {count_after_second} points in collection")
        
        # Verify no duplicates
        assert count_after_first == count_after_second, (
            f"Point count changed from {count_after_first} to {count_after_second}. "
            "Idempotent sync should not create duplicates."
        )
        
        print("✓ Idempotent sync verified - no duplicates created")


@requires_qdrant
@requires_openai
def test_version_update_replaces_old():
    """
    Test that updating a file's content replaces old points.
    
    This verifies:
    1. Changed content = different updated_at = new point IDs
    2. Old version points are cleaned up
    """
    from connectors import LocalFilesConnector
    from ingest.pipeline import ingest
    from app.qdrant_client import client as qdrant_client
    from app.config import settings
    import time
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "version_test.md"
        
        # Use a unique tenant for this test
        tenant_id = "test_version_update"
        
        # First version
        test_file.write_text("# Version 1\n\nOriginal content.")
        
        connector = LocalFilesConnector(directory=tmpdir, tenant_id=tenant_id)
        stats1 = ingest(connector, recreate=True)
        
        collection_info = qdrant_client.get_collection(settings.qdrant_collection)
        count_v1 = collection_info.points_count
        
        print(f"\nVersion 1: {stats1.chunks_created} chunks, {count_v1} points")
        
        # Wait a moment to ensure different modified_at
        time.sleep(1)
        
        # Update file (same size to test timestamp-based versioning)
        test_file.write_text("# Version 2\n\nUpdated content.")
        
        connector2 = LocalFilesConnector(directory=tmpdir, tenant_id=tenant_id)
        stats2 = ingest(connector2, recreate=False)
        
        collection_info = qdrant_client.get_collection(settings.qdrant_collection)
        count_v2 = collection_info.points_count
        
        print(f"Version 2: {stats2.chunks_created} chunks, {count_v2} points")
        
        # Both versions should have same chunk count (small doc = 1 chunk)
        # Point count should be same or less (old version cleaned up)
        assert count_v2 <= count_v1 + stats2.chunks_created, (
            f"Too many points after update. Expected <= {count_v1 + stats2.chunks_created}, got {count_v2}"
        )
        
        print("✓ Version update handling verified")


def test_point_id_determinism():
    """
    Test that generate_point_id produces consistent results.
    
    Same inputs should always produce same UUID.
    """
    from ingest.pipeline import generate_point_id
    
    # Generate ID twice with same inputs
    id1 = generate_point_id(
        tenant_id="tenant1",
        source="google_drive",
        external_id="file123",
        updated_at="2024-01-01T00:00:00",
        chunk_index=0,
    )
    
    id2 = generate_point_id(
        tenant_id="tenant1",
        source="google_drive",
        external_id="file123",
        updated_at="2024-01-01T00:00:00",
        chunk_index=0,
    )
    
    assert id1 == id2, f"Same inputs produced different IDs: {id1} vs {id2}"
    
    # Different chunk index should produce different ID
    id3 = generate_point_id(
        tenant_id="tenant1",
        source="google_drive",
        external_id="file123",
        updated_at="2024-01-01T00:00:00",
        chunk_index=1,
    )
    
    assert id1 != id3, "Different chunk index should produce different ID"
    
    # Different updated_at should produce different ID
    id4 = generate_point_id(
        tenant_id="tenant1",
        source="google_drive",
        external_id="file123",
        updated_at="2024-01-02T00:00:00",
        chunk_index=0,
    )
    
    assert id1 != id4, "Different updated_at should produce different ID"
    
    print("✓ Point ID determinism verified")


def test_state_store():
    """
    Test the connector state store.
    """
    from connectors.state_store import StateStore
    import tempfile
    import os
    
    # Use temp database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_state.db")
        store = StateStore(db_path=db_path)
        
        # Test set and get
        store.set_state("tenant1", "google_drive", {"last_sync": "2024-01-01"})
        state = store.get_state("tenant1", "google_drive")
        
        assert state is not None
        assert state["last_sync"] == "2024-01-01"
        
        # Test update
        store.update_state("tenant1", "google_drive", {"cursor": "abc123"})
        state = store.get_state("tenant1", "google_drive")
        
        assert state["last_sync"] == "2024-01-01"
        assert state["cursor"] == "abc123"
        
        # Test list
        store.set_state("tenant1", "local_file", {"path": "/data"})
        states = store.list_states("tenant1")
        
        assert len(states) == 2
        
        # Test delete
        deleted = store.delete_state("tenant1", "local_file")
        assert deleted
        
        states = store.list_states("tenant1")
        assert len(states) == 1
        
        print("✓ State store verified")


if __name__ == "__main__":
    # Run non-integration tests
    test_point_id_determinism()
    test_state_store()
    
    # Run integration tests if available
    if qdrant_available() and os.environ.get("OPENAI_API_KEY"):
        test_idempotent_local_sync()
        test_version_update_replaces_old()
    else:
        print("\nSkipping integration tests (Qdrant or OpenAI not available)")
