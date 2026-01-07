"""
SQLite-based state store for connector sync state.

Stores per-tenant, per-source state including:
- Last sync timestamp
- Connector-specific cursors
- Sync metadata
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from threading import Lock


class StateStore:
    """
    Simple SQLite state store for connector state.
    
    Thread-safe with connection pooling for FastAPI compatibility.
    """
    
    def __init__(self, db_path: str = "data/state.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection (thread-safe)."""
        conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,  # Required for FastAPI
        )
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS connector_state (
                        tenant_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        state_json TEXT NOT NULL DEFAULT '{}',
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (tenant_id, source)
                    )
                """)
                conn.commit()
            finally:
                conn.close()
    
    def get_state(self, tenant_id: str, source: str) -> dict | None:
        """
        Get state for a tenant/source combination.
        
        Args:
            tenant_id: Tenant identifier.
            source: Source identifier (e.g., 'google_drive', 'local_files').
        
        Returns:
            State dict or None if not found.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "SELECT state_json, updated_at FROM connector_state WHERE tenant_id = ? AND source = ?",
                    (tenant_id, source)
                )
                row = cursor.fetchone()
                if row:
                    state = json.loads(row["state_json"])
                    state["_updated_at"] = row["updated_at"]
                    return state
                return None
            finally:
                conn.close()
    
    def set_state(self, tenant_id: str, source: str, state: dict) -> None:
        """
        Set state for a tenant/source combination (overwrites existing).
        
        Args:
            tenant_id: Tenant identifier.
            source: Source identifier.
            state: State dict to store.
        """
        # Remove internal fields before saving
        state_to_save = {k: v for k, v in state.items() if not k.startswith("_")}
        
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO connector_state (tenant_id, source, state_json, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (tenant_id, source, json.dumps(state_to_save), datetime.utcnow().isoformat())
                )
                conn.commit()
            finally:
                conn.close()
    
    def update_state(self, tenant_id: str, source: str, patch: dict) -> None:
        """
        Update state for a tenant/source combination (merges with existing).
        
        Args:
            tenant_id: Tenant identifier.
            source: Source identifier.
            patch: Dict of fields to update/add.
        """
        current = self.get_state(tenant_id, source) or {}
        # Remove internal fields
        current = {k: v for k, v in current.items() if not k.startswith("_")}
        current.update(patch)
        self.set_state(tenant_id, source, current)
    
    def delete_state(self, tenant_id: str, source: str) -> bool:
        """
        Delete state for a tenant/source combination.
        
        Returns:
            True if state was deleted, False if not found.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "DELETE FROM connector_state WHERE tenant_id = ? AND source = ?",
                    (tenant_id, source)
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()
    
    def list_states(self, tenant_id: str | None = None) -> list[dict]:
        """
        List all states, optionally filtered by tenant.
        
        Args:
            tenant_id: Optional tenant filter.
        
        Returns:
            List of state records.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                if tenant_id:
                    cursor = conn.execute(
                        "SELECT tenant_id, source, state_json, updated_at FROM connector_state WHERE tenant_id = ?",
                        (tenant_id,)
                    )
                else:
                    cursor = conn.execute(
                        "SELECT tenant_id, source, state_json, updated_at FROM connector_state"
                    )
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        "tenant_id": row["tenant_id"],
                        "source": row["source"],
                        "state": json.loads(row["state_json"]),
                        "updated_at": row["updated_at"],
                    })
                return results
            finally:
                conn.close()


# Global singleton instance
_state_store: StateStore | None = None


def get_state_store() -> StateStore:
    """Get the global state store instance."""
    global _state_store
    if _state_store is None:
        _state_store = StateStore()
    return _state_store
