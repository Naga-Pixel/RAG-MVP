from pathlib import Path
from datetime import datetime
from functools import lru_cache
import secrets
import time
import logging
import os
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.models import AskRequest, AskResponse
from app.logging_config import setup_logging, get_logger
from app.logging_utils import request_id_ctx, new_request_id
from app.rag_service import answer_question
from app.config import settings
from app.google_drive_auth import router as drive_router, verify_supabase_token
from app.qdrant_client import client as qdrant_client

logger = get_logger(__name__)


# ============== Rate Limiting ==============

limiter = Limiter(key_func=get_remote_address)


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


def _sentry_before_send(event, hint):
    req = event.get("request") or {}
    # Remove request body & cookies
    req.pop("data", None)
    req.pop("cookies", None)
    headers = req.get("headers") or {}
    headers.pop("authorization", None)
    req["headers"] = headers
    event["request"] = req
    return event

sentry_dsn = os.getenv("SENTRY_DSN")
if sentry_dsn:
    sentry_sdk.init(
        dsn=sentry_dsn,
        integrations=[FastApiIntegration()],
        environment=os.getenv("SENTRY_ENVIRONMENT", "dev"),
        send_default_pii=False,
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0") or "0"),
        before_send=_sentry_before_send,
    )


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or new_request_id()
    token = request_id_ctx.set(rid)

    start = time.time()
    try:
        response = await call_next(request)
        duration_ms = int((time.time() - start) * 1000)

        response.headers["X-Request-ID"] = rid

        logging.getLogger("app.request").info(
            "%s %s -> %s (%dms)",
            request.method,
            request.url.path,
            getattr(response, "status_code", "?"),
            duration_ms,
        )
        return response
    finally:
        request_id_ctx.reset(token)


# ============== Middleware ==============

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS - configure allowed origins
# In production, set CORS_ORIGINS env var to your domain(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

logger.info(f"CORS enabled for origins: {settings.cors_origins}")

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
@limiter.limit(settings.rate_limit_ask)
async def ask(request: Request, body: AskRequest, user: dict = Depends(verify_supabase_token)):
    """Answer a question using RAG. Rate limited to prevent API abuse."""
    return answer_question(
        body.query,
        tenant_id=user["user_id"],
        folder_id=body.folder_id,
        doc_ids=body.doc_ids,
    )


class FolderInfo(BaseModel):
    folder_id: str
    folder_name: str | None = None
    doc_count: int = 0


class DocumentInfo(BaseModel):
    doc_id: str
    title: str | None = None
    chunk_count: int = 0
    folder_id: str | None = None
    folder_name: str | None = None


@app.get("/folders", response_model=list[FolderInfo])
async def list_folders(user: dict = Depends(verify_supabase_token)):
    """
    List available folders for the authenticated user (tenant).
    Used by UI to populate folder selector for scoped queries.
    """
    from app.scope_resolver import get_folders

    tenant_id = user["user_id"]
    folders = get_folders(tenant_id)

    return [
        FolderInfo(
            folder_id=f["folder_id"],
            folder_name=f["folder_name"],
            doc_count=f["doc_count"],
        )
        for f in folders
    ]


@app.get("/documents", response_model=list[DocumentInfo])
async def list_documents(
    folder_id: str | None = None,
    user: dict = Depends(verify_supabase_token),
):
    """
    List available documents for the authenticated user (tenant).
    Optionally filter by folder_id.
    Used by UI to populate document selector for scoped queries.
    """
    import psycopg2

    tenant_id = user["user_id"]

    # Try Postgres FTS table first (faster for distinct queries)
    if settings.database_url:
        try:
            conn = psycopg2.connect(settings.database_url)
            cursor = conn.cursor()

            fqtn = f"{settings.fts_shadow_schema}.{settings.fts_shadow_table}"

            if folder_id:
                cursor.execute(f"""
                    SELECT doc_id, MAX(title) as title, COUNT(*) as chunk_count,
                           MAX(folder_id) as folder_id, MAX(folder_name) as folder_name
                    FROM {fqtn}
                    WHERE tenant_id = %s AND folder_id = %s
                    GROUP BY doc_id
                    ORDER BY MAX(title) NULLS LAST, doc_id
                """, (tenant_id, folder_id))
            else:
                cursor.execute(f"""
                    SELECT doc_id, MAX(title) as title, COUNT(*) as chunk_count,
                           MAX(folder_id) as folder_id, MAX(folder_name) as folder_name
                    FROM {fqtn}
                    WHERE tenant_id = %s
                    GROUP BY doc_id
                    ORDER BY MAX(title) NULLS LAST, doc_id
                """, (tenant_id,))

            rows = cursor.fetchall()
            conn.close()

            return [
                DocumentInfo(
                    doc_id=row[0],
                    title=row[1],
                    chunk_count=row[2],
                    folder_id=row[3],
                    folder_name=row[4],
                )
                for row in rows
            ]
        except Exception as e:
            logger.warning(f"Failed to list documents from Postgres: {e}")
            # Fall through to Qdrant

    # Fallback: Query Qdrant (slower but works without Postgres)
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Build filter conditions
        must_conditions = [FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
        if folder_id:
            must_conditions.append(FieldCondition(key="folder_id", match=MatchValue(value=folder_id)))

        # Scroll through all points for tenant to get unique doc_ids
        # This is less efficient but works as fallback
        doc_map = {}  # doc_id -> {title, count, folder_id, folder_name}

        offset = None
        while True:
            result = qdrant_client.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=Filter(must=must_conditions),
                limit=100,
                offset=offset,
                with_payload=["doc_id", "title", "folder_id", "folder_name"],
                with_vectors=False,
            )

            points, offset = result
            if not points:
                break

            for p in points:
                payload = p.payload or {}
                doc_id = payload.get("doc_id", "unknown")
                title = payload.get("title")
                f_id = payload.get("folder_id")
                f_name = payload.get("folder_name")

                if doc_id not in doc_map:
                    doc_map[doc_id] = {"title": title, "count": 0, "folder_id": f_id, "folder_name": f_name}
                doc_map[doc_id]["count"] += 1

            if offset is None:
                break

        return [
            DocumentInfo(
                doc_id=doc_id,
                title=info["title"],
                chunk_count=info["count"],
                folder_id=info["folder_id"],
                folder_name=info["folder_name"],
            )
            for doc_id, info in sorted(doc_map.items(), key=lambda x: (x[1]["title"] or "", x[0]))
        ]

    except Exception as e:
        logger.warning(f"Failed to list documents from Qdrant: {e}")
        return []


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
@limiter.limit(settings.rate_limit_sync)
def sync_local(request: Request, body: SyncLocalRequest):
    """
    Sync documents from local directory.
    Requires API key authentication.

    This is a synchronous operation - for large directories,
    consider using the CLI instead.

    Idempotent: re-syncing same files won't create duplicates.
    """
    directory = Path(body.directory)

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
    prev_state = store.get_state(body.tenant_id, "local_file")

    connector = LocalFilesConnector(
        directory=validated_directory,
        tenant_id=body.tenant_id,
    )

    stats = ingest(connector, recreate=body.recreate)

    # Get updated state
    new_state = store.get_state(body.tenant_id, "local_file")

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
@limiter.limit(settings.rate_limit_sync)
def sync_google_drive(request: Request, body: SyncDriveRequest):
    """
    Sync documents from Google Drive.
    Requires API key authentication. Rate limited.

    Requires:
    - Service account credentials JSON file (must be in credentials directory)
    - Folder shared with service account email

    Idempotent: re-syncing same files won't create duplicates.
    """
    credentials_path = Path(body.credentials_path)

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
    prev_state = store.get_state(body.tenant_id, "google_drive")

    connector = GoogleDriveConnector(
        credentials_path=str(validated_credentials),
        folder_id=body.folder_id,
        tenant_id=body.tenant_id,
    )

    stats = ingest(connector, recreate=body.recreate)

    if stats.documents_found == 0 and not stats.errors:
        raise HTTPException(
            status_code=400,
            detail="No documents found. Check folder ID and sharing permissions.",
        )

    # Get updated state
    new_state = store.get_state(body.tenant_id, "google_drive")

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


class UploadSyncResponse(BaseModel):
    status: str
    folder_name: str
    documents_found: int
    documents_processed: int
    chunks_created: int
    errors: list[str]
    duration_seconds: float | None


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".xlsx", ".rtf"}


@app.post("/sync/upload", response_model=UploadSyncResponse)
@limiter.limit(settings.rate_limit_sync)
async def sync_upload(
    request: Request,
    files: list[UploadFile] = File(...),
    folder_name: str = "Local folder",
    user: dict = Depends(verify_supabase_token),
):
    """
    Sync documents from uploaded files.
    Accepts multipart/form-data with multiple files.

    This endpoint processes files uploaded from the browser's folder picker.
    No filesystem paths are stored - only the folder name for display.
    """
    import hashlib
    from datetime import datetime

    from connectors.base import Document, SourceType
    from connectors.state_store import get_state_store
    from ingest.loaders import load_document_from_bytes
    from ingest.pipeline import IngestionPipeline

    tenant_id = user["user_id"]
    started_at = datetime.utcnow()
    errors = []
    documents = []

    # Generate a stable folder_id from tenant + folder_name
    folder_id = hashlib.sha256(f"{tenant_id}:{folder_name}".encode()).hexdigest()[:16]

    # Filter to supported files only
    supported_files = [
        f for f in files
        if Path(f.filename).suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    for upload_file in supported_files:
        try:
            # Read file content
            content_bytes = await upload_file.read()
            filename = upload_file.filename

            # Generate a stable ID from content hash (for idempotent upserts)
            content_hash = hashlib.sha256(content_bytes).hexdigest()[:16]

            # Load document content based on file type
            result = load_document_from_bytes(content_bytes, filename)

            # Handle xlsx which returns tuple
            if isinstance(result, tuple):
                text_content, sheet_metadata = result
                extra_metadata = {"sheets": sheet_metadata}
            else:
                text_content = result
                extra_metadata = {}

            if not text_content.strip():
                errors.append(f"Empty content: {filename}")
                continue

            # Create Document object with folder metadata
            doc = Document(
                content=text_content,
                source_type=SourceType.LOCAL_FILE,
                source_id=filename,
                title=Path(filename).stem,
                tenant_id=tenant_id,
                external_id=content_hash,
                metadata={
                    "original_filename": filename,
                    "folder_id": folder_id,
                    "folder_name": folder_name,
                    **extra_metadata,
                },
            )
            documents.append(doc)

        except ValueError as e:
            errors.append(f"Unsupported file: {filename}")
        except Exception as e:
            errors.append(f"Error processing {filename}: {str(e)}")

    # Process through ingestion pipeline
    chunks_created = 0
    if documents:
        pipeline = IngestionPipeline()
        for doc in documents:
            try:
                chunks = pipeline.ingest_document(doc)
                chunks_created += chunks
            except Exception as e:
                errors.append(f"Ingestion error for {doc.title}: {str(e)}")

        # Update state
        store = get_state_store()
        completed_at = datetime.utcnow()
        store.update_state(
            tenant_id=tenant_id,
            source="local_upload",
            patch={
                "last_sync_at": completed_at.isoformat(),
                "documents_synced": len(documents),
                "chunks_created": chunks_created,
                "folder_name": folder_name,
            },
        )

    completed_at = datetime.utcnow()
    duration = (completed_at - started_at).total_seconds()

    return UploadSyncResponse(
        status="completed",
        folder_name=folder_name,
        documents_found=len(supported_files),
        documents_processed=len(documents),
        chunks_created=chunks_created,
        errors=errors,
        duration_seconds=duration,
    )


class TranscribeResponse(BaseModel):
    text: str
    language: str | None = None
    duration_seconds: float | None = None


@app.post("/transcribe", response_model=TranscribeResponse)
@limiter.limit(settings.rate_limit_ask)
async def transcribe_audio_endpoint(
    request: Request,
    audio: UploadFile = File(...),
    language: str | None = None,
    user: dict = Depends(verify_supabase_token),
):
    """
    Transcribe audio using OpenAI Whisper.

    Accepts audio file (webm, mp3, wav, etc.) and returns transcribed text.
    Used by push-to-talk feature in the UI.
    """
    from app.transcription import transcribe_audio, SUPPORTED_FORMATS

    # Validate file extension
    filename = audio.filename or "audio.webm"
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {suffix}. Supported: {', '.join(SUPPORTED_FORMATS)}",
        )

    # Read audio content
    audio_bytes = await audio.read()

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Max 25MB (Whisper limit)
    if len(audio_bytes) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Audio file too large (max 25MB)")

    try:
        result = transcribe_audio(
            audio_bytes=audio_bytes,
            filename=filename,
            language=language,
        )
        return TranscribeResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Transcription failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Transcription failed")


@app.get("/health")
def health_check():
    """Health check endpoint. Returns minimal status information."""
    return {
        "status": "healthy",
    }
