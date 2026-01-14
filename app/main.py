from pathlib import Path
from datetime import datetime
from functools import lru_cache
import secrets

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.models import AskRequest, AskResponse
from app.logging_config import setup_logging
from app.rag_service import answer_question
from app.config import settings
from app.google_drive_auth import router as drive_router


# ============== Security ==============

def verify_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")):
    """
    Verify API key for protected endpoints.
    If API_KEY is not configured, authentication is disabled (for development).
    """
    if not settings.api_key:
        # API key not configured - allow access (development mode)
        return True

    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide X-API-Key header.",
        )

    # Use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )

    return True


def validate_sync_directory(directory: Path) -> Path:
    """
    Validate that a directory is allowed for sync operations.
    Prevents path traversal attacks.
    """
    # Resolve to absolute path and normalize
    resolved = directory.resolve()

    # If allowed directories are configured, enforce them
    if settings.allowed_sync_directories:
        allowed = False
        for allowed_dir in settings.allowed_sync_directories:
            allowed_path = Path(allowed_dir).resolve()
            try:
                resolved.relative_to(allowed_path)
                allowed = True
                break
            except ValueError:
                continue

        if not allowed:
            raise HTTPException(
                status_code=403,
                detail=f"Directory not in allowed list. Allowed directories: {settings.allowed_sync_directories}",
            )

    # Always block sensitive system directories
    blocked_prefixes = ["/etc", "/root", "/var", "/usr", "/bin", "/sbin", "/proc", "/sys"]
    for prefix in blocked_prefixes:
        if str(resolved).startswith(prefix):
            raise HTTPException(
                status_code=403,
                detail="Access to system directories is not allowed.",
            )

    return resolved


def validate_credentials_path(credentials_path: Path) -> Path:
    """
    Validate that credentials file is within the allowed credentials directory.
    Prevents path traversal attacks.
    """
    resolved = credentials_path.resolve()

    # Credentials must be in the configured credentials directory
    creds_dir = Path(settings.credentials_directory).resolve()

    try:
        resolved.relative_to(creds_dir)
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail=f"Credentials must be located in: {creds_dir}",
        )

    return resolved

app = FastAPI(title="b_rag API")
setup_logging()

# Include Google Drive OAuth routes
app.include_router(drive_router)

# Serve static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ============== Request/Response Models ==============

class SyncLocalRequest(BaseModel):
    directory: str = "data/raw"
    tenant_id: str = "default"
    recreate: bool = False


class SyncDriveRequest(BaseModel):
    credentials_path: str
    folder_id: str | None = None
    tenant_id: str = "default"
    recreate: bool = False


class SyncResponse(BaseModel):
    status: str
    connector: str
    documents_found: int
    documents_processed: int
    chunks_created: int
    errors: list[str]
    duration_seconds: float | None
    last_sync_at: str | None = None


class SourceInfo(BaseModel):
    name: str
    type: str
    available: bool
    description: str


class ConnectorState(BaseModel):
    tenant_id: str
    source: str
    last_sync_at: str | None
    documents_synced: int | None
    chunks_created: int | None


# ============== Endpoints ==============

@app.get("/")
def serve_frontend():
    """Serve the main frontend page."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "b_rag API is running. POST to /ask to query documents."}


@app.get("/config/frontend")
def get_frontend_config():
    """
    Return frontend configuration.
    Only exposes non-sensitive values needed by the frontend.
    """
    return {
        "supabase_url": settings.supabase_url,
        "supabase_anon_key": settings.supabase_anon_key,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Answer a question using RAG."""
    return answer_question(request.query)


@app.get("/sources", response_model=list[SourceInfo])
def list_sources():
    """List available data source connectors."""
    sources = [
        SourceInfo(
            name="Local Files",
            type="local",
            available=True,
            description="Ingest files from local directory (PDF, DOCX, XLSX, TXT, MD)",
        ),
        SourceInfo(
            name="Google Drive",
            type="google_drive",
            available=True,
            description="Sync files from Google Drive folder (requires service account)",
        ),
    ]
    return sources


@app.get("/state", response_model=list[ConnectorState], dependencies=[Depends(verify_api_key)])
def list_connector_states(tenant_id: str | None = None):
    """List connector sync states. Requires API key authentication."""
    from connectors.state_store import get_state_store

    store = get_state_store()
    states = store.list_states(tenant_id=tenant_id)

    return [
        ConnectorState(
            tenant_id=s["tenant_id"],
            source=s["source"],
            last_sync_at=s["state"].get("last_sync_at"),
            documents_synced=s["state"].get("documents_synced"),
            chunks_created=s["state"].get("chunks_created"),
        )
        for s in states
    ]


@app.get("/state/{tenant_id}/{source}", response_model=ConnectorState, dependencies=[Depends(verify_api_key)])
def get_connector_state(tenant_id: str, source: str):
    """Get connector sync state for a specific tenant/source. Requires API key authentication."""
    from connectors.state_store import get_state_store

    store = get_state_store()
    state = store.get_state(tenant_id, source)

    if not state:
        raise HTTPException(status_code=404, detail="State not found")

    return ConnectorState(
        tenant_id=tenant_id,
        source=source,
        last_sync_at=state.get("last_sync_at"),
        documents_synced=state.get("documents_synced"),
        chunks_created=state.get("chunks_created"),
    )


@app.post("/sync/local", response_model=SyncResponse, dependencies=[Depends(verify_api_key)])
def sync_local(request: SyncLocalRequest):
    """
    Sync documents from local directory.
    Requires API key authentication.

    This is a synchronous operation - for large directories,
    consider using the CLI instead.

    Idempotent: re-syncing same files won't create duplicates.
    """
    directory = Path(request.directory)

    # Validate directory to prevent path traversal attacks (before any other operations)
    validated_directory = validate_sync_directory(directory)

    if not validated_directory.exists():
        raise HTTPException(status_code=404, detail="Directory not found")

    # Import after validation to ensure security checks run first
    from connectors import LocalFilesConnector
    from connectors.state_store import get_state_store
    from ingest.pipeline import ingest

    # Get previous state
    store = get_state_store()
    prev_state = store.get_state(request.tenant_id, "local_file")

    connector = LocalFilesConnector(
        directory=validated_directory,
        tenant_id=request.tenant_id,
    )

    stats = ingest(connector, recreate=request.recreate)

    # Get updated state
    new_state = store.get_state(request.tenant_id, "local_file")

    return SyncResponse(
        status="completed",
        connector=stats.connector_name,
        documents_found=stats.documents_found,
        documents_processed=stats.documents_processed,
        chunks_created=stats.chunks_created,
        errors=stats.errors,
        duration_seconds=stats.duration_seconds,
        last_sync_at=new_state.get("last_sync_at") if new_state else None,
    )


@app.post("/sync/google-drive", response_model=SyncResponse, dependencies=[Depends(verify_api_key)])
def sync_google_drive(request: SyncDriveRequest):
    """
    Sync documents from Google Drive.
    Requires API key authentication.

    Requires:
    - Service account credentials JSON file (must be in credentials directory)
    - Folder shared with service account email

    Idempotent: re-syncing same files won't create duplicates.
    """
    credentials_path = Path(request.credentials_path)

    # Validate credentials path to prevent path traversal attacks (before any other operations)
    validated_credentials = validate_credentials_path(credentials_path)

    if not validated_credentials.exists():
        raise HTTPException(
            status_code=404,
            detail="Credentials file not found",
        )

    # Import after validation to ensure security checks run first
    from connectors import GoogleDriveConnector
    from connectors.state_store import get_state_store
    from ingest.pipeline import ingest

    # Get previous state
    store = get_state_store()
    prev_state = store.get_state(request.tenant_id, "google_drive")

    connector = GoogleDriveConnector(
        credentials_path=str(validated_credentials),
        folder_id=request.folder_id,
        tenant_id=request.tenant_id,
    )

    stats = ingest(connector, recreate=request.recreate)

    if stats.documents_found == 0 and not stats.errors:
        raise HTTPException(
            status_code=400,
            detail="No documents found. Check folder ID and sharing permissions.",
        )

    # Get updated state
    new_state = store.get_state(request.tenant_id, "google_drive")

    return SyncResponse(
        status="completed",
        connector=stats.connector_name,
        documents_found=stats.documents_found,
        documents_processed=stats.documents_processed,
        chunks_created=stats.chunks_created,
        errors=stats.errors,
        duration_seconds=stats.duration_seconds,
        last_sync_at=new_state.get("last_sync_at") if new_state else None,
    )


@app.get("/health")
def health_check():
    """Health check endpoint. Returns minimal status information."""
    return {
        "status": "healthy",
    }
