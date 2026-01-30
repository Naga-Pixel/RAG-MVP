"""
Google Drive OAuth and Sync module.

Provides user-based Google Drive authentication via OAuth,
folder selection via Google Picker, and on-demand sync.
"""
import secrets
import tempfile
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any

import httpx
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import APIRouter, HTTPException, Depends, Header, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from cryptography.fernet import Fernet, InvalidToken
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings
from app.logging_config import get_logger

# Rate limiter for Drive endpoints
limiter = Limiter(key_func=get_remote_address)

logger = get_logger(__name__)

router = APIRouter()

# ============== Database ==============

# Error message for missing tables
MISSING_TABLES_MSG = "Drive tables not found. Please run migrations/001_drive_sources.sql in Supabase."


def _get_db():
    """Get database connection for Drive OAuth data."""
    if not settings.database_url:
        raise HTTPException(
            status_code=500,
            detail="DATABASE_URL not configured. Please set it to your Postgres connection string."
        )
    try:
        conn = psycopg2.connect(settings.database_url, cursor_factory=RealDictCursor)
        logger.debug("Connected to Postgres for Drive data")
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Failed to connect to Postgres: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to connect to database. Check DATABASE_URL configuration."
        )


def _handle_db_error(e: Exception):
    """Convert database errors to user-friendly HTTP exceptions."""
    error_str = str(e).lower()
    if "relation" in error_str and "does not exist" in error_str:
        raise HTTPException(status_code=500, detail=MISSING_TABLES_MSG)
    logger.error(f"Database error: {e}")
    raise HTTPException(status_code=500, detail="Database error occurred.")


# ============== Encryption ==============

def _get_fernet() -> Fernet:
    """Get Fernet instance for token encryption."""
    if not settings.drive_token_encryption_key:
        raise HTTPException(
            status_code=500,
            detail="DRIVE_TOKEN_ENCRYPTION_KEY not configured"
        )
    return Fernet(settings.drive_token_encryption_key.encode())


def _encrypt_token(token: str) -> str:
    """Encrypt a refresh token."""
    f = _get_fernet()
    return f.encrypt(token.encode()).decode()


def _decrypt_token(encrypted: str) -> str:
    """Decrypt a refresh token."""
    f = _get_fernet()
    try:
        return f.decrypt(encrypted.encode()).decode()
    except InvalidToken:
        raise HTTPException(status_code=500, detail="Failed to decrypt token")


# ============== Supabase JWT Auth ==============

# Cache for OpenID discovery and JWKS
_oidc_cache = {
    "jwks_uri": None,
    "jwks_uri_fetched_at": None,
    "jwks_keys": None,
    "jwks_keys_fetched_at": None,
}

# Cache TTL in seconds (1 hour)
_CACHE_TTL = 3600


def _discover_jwks_uri() -> str:
    """
    Discover the JWKS URI from Supabase OpenID Connect configuration.

    Fetches: SUPABASE_URL/auth/v1/.well-known/openid-configuration
    Returns the jwks_uri field.

    Raises HTTPException(401) on failure - fail closed.
    """
    import time
    import urllib.request
    import json

    # Return cached value if still valid
    if _oidc_cache["jwks_uri"] and _oidc_cache["jwks_uri_fetched_at"]:
        if time.time() - _oidc_cache["jwks_uri_fetched_at"] < _CACHE_TTL:
            return _oidc_cache["jwks_uri"]

    if not settings.supabase_url:
        logger.error("SUPABASE_URL not configured")
        raise HTTPException(status_code=401, detail="Invalid token")

    oidc_config_url = f"{settings.supabase_url}/auth/v1/.well-known/openid-configuration"

    try:
        logger.debug(f"Fetching OpenID configuration from Supabase")
        with urllib.request.urlopen(oidc_config_url, timeout=5) as response:
            config = json.loads(response.read().decode())

        jwks_uri = config.get("jwks_uri")
        if not jwks_uri:
            logger.error("OpenID configuration missing jwks_uri")
            raise HTTPException(status_code=401, detail="Invalid token")

        # Cache the discovered URI
        _oidc_cache["jwks_uri"] = jwks_uri
        _oidc_cache["jwks_uri_fetched_at"] = time.time()
        logger.info("Discovered JWKS URI from OpenID configuration")
        return jwks_uri

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch OpenID configuration: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")


def _get_jwks_public_key():
    """
    Fetch and cache Supabase JWKS public keys using discovered jwks_uri.
    Raises HTTPException(401) if fetch fails - fail closed, no fallback.
    """
    import time
    import urllib.request
    import json

    # Return cached keys if still valid
    if _oidc_cache["jwks_keys"] and _oidc_cache["jwks_keys_fetched_at"]:
        if time.time() - _oidc_cache["jwks_keys_fetched_at"] < _CACHE_TTL:
            return _oidc_cache["jwks_keys"]

    # Discover JWKS URI (also cached)
    jwks_uri = _discover_jwks_uri()

    try:
        logger.debug("Fetching JWKS from discovered URI")
        with urllib.request.urlopen(jwks_uri, timeout=5) as response:
            jwks = json.loads(response.read().decode())
            _oidc_cache["jwks_keys"] = jwks
            _oidc_cache["jwks_keys_fetched_at"] = time.time()
            logger.info("Fetched JWKS from Supabase")
            return jwks
    except Exception as e:
        logger.error(f"Failed to fetch JWKS: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")


async def verify_supabase_token(authorization: str | None = Header(None)) -> dict:
    """
    Verify Supabase JWT token and return user info.

    Verifies the JWT signature using the Supabase JWT secret,
    checks expiration, and extracts user information.
    """
    import jwt

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")

    token = authorization[7:]  # Remove "Bearer " prefix

    try:
        # First, peek at the token header to see the algorithm
        token_alg = None
        token_kid = None
        try:
            unverified_header = jwt.get_unverified_header(token)
            token_alg = unverified_header.get('alg')
            token_kid = unverified_header.get('kid')
        except Exception as e:
            logger.warning(f"Could not read JWT header: {e}")

        # Handle ES256 (asymmetric) - fetch public key from JWKS via OpenID discovery
        if token_alg == "ES256":
            # Discover JWKS URI and fetch keys (raises HTTPException(401) on failure)
            jwks_uri = _discover_jwks_uri()
            _get_jwks_public_key()

            # Get signing key using discovered JWKS URI
            from jwt import PyJWKClient
            try:
                jwks_client = PyJWKClient(jwks_uri)
                signing_key = jwks_client.get_signing_key_from_jwt(token)
            except Exception as e:
                logger.error(f"Failed to get signing key from JWKS: {e}")
                raise HTTPException(status_code=401, detail="Invalid token")

            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["ES256"],
                audience="authenticated",
                options={
                    "require": ["exp", "sub"],
                    "verify_exp": True,
                    "verify_aud": True,
                }
            )
        # Handle HS256 (symmetric) - use JWT secret
        elif settings.supabase_jwt_secret:
            payload = jwt.decode(
                token,
                settings.supabase_jwt_secret,
                algorithms=["HS256", "HS384", "HS512"],
                audience="authenticated",
                options={
                    "require": ["exp", "sub"],
                    "verify_exp": True,
                    "verify_aud": True,
                }
            )
        else:
            # No verification method available - fail closed
            logger.error("No JWT verification method configured (no SUPABASE_URL or SUPABASE_JWT_SECRET)")
            raise HTTPException(status_code=401, detail="Invalid token")

        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing user ID")

        return {
            "user_id": user_id,
            "email": payload.get("email"),
        }

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidAudienceError:
        raise HTTPException(status_code=401, detail="Invalid token audience")
    except jwt.InvalidSignatureError:
        logger.warning("JWT signature verification failed - possible token forgery attempt")
        raise HTTPException(status_code=401, detail="Invalid token signature")
    except jwt.InvalidAlgorithmError as e:
        logger.error(f"JWT algorithm mismatch: token uses '{token_alg}'")
        raise HTTPException(status_code=401, detail="Invalid token algorithm")
    except jwt.DecodeError as e:
        logger.error(f"JWT decode error: {e}")
        raise HTTPException(status_code=401, detail="Invalid token format")
    except Exception as e:
        logger.error(f"JWT verification failed (alg: {token_alg}): {e}")
        raise HTTPException(status_code=401, detail="Invalid token")


# ============== Request/Response Models ==============

class OAuthStartResponse(BaseModel):
    auth_url: str


class DriveStatusResponse(BaseModel):
    connected: bool
    folders: list[dict]


class AddFolderRequest(BaseModel):
    folder_id: str
    folder_name: str


class FoldersResponse(BaseModel):
    folders: list[dict]


class PickerTokenResponse(BaseModel):
    access_token: str


class DriveSyncResponse(BaseModel):
    status: str
    folders_synced: int
    documents_found: int
    documents_processed: int
    chunks_created: int
    errors: list[str]
    duration_seconds: float


# ============== OAuth Endpoints ==============

@router.post("/oauth/google/drive/start", response_model=OAuthStartResponse)
async def start_drive_oauth(user: dict = Depends(verify_supabase_token)):
    """
    Start Google Drive OAuth flow.
    Returns the auth URL to open in a popup.
    """
    if not settings.google_drive_client_id:
        raise HTTPException(status_code=500, detail="GOOGLE_DRIVE_CLIENT_ID not configured")
    if not settings.google_drive_client_secret:
        raise HTTPException(status_code=500, detail="GOOGLE_DRIVE_CLIENT_SECRET not configured")

    # Generate random state for CSRF protection
    state = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(minutes=10)

    # Store state in database
    conn = _get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO google_oauth_states (state, user_id, expires_at) VALUES (%s, %s, %s)",
            (state, user["user_id"], expires_at)
        )
        conn.commit()
    except psycopg2.Error as e:
        _handle_db_error(e)
    finally:
        cursor.close()
        conn.close()

    # Build OAuth URL
    params = {
        "client_id": settings.google_drive_client_id,
        "redirect_uri": settings.google_drive_redirect_uri,
        "response_type": "code",
        "scope": "https://www.googleapis.com/auth/drive.readonly",
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "state": state,
    }

    query = "&".join(f"{k}={v}" for k, v in params.items())
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{query}"

    return OAuthStartResponse(auth_url=auth_url)


@router.get("/oauth/google/drive/callback", response_class=HTMLResponse)
async def drive_oauth_callback(code: str | None = None, state: str | None = None, error: str | None = None):
    """
    Handle Google OAuth callback.
    Exchanges code for tokens and stores encrypted refresh token.
    """
    # Determine origin for postMessage
    origin = settings.google_drive_redirect_uri.rsplit("/oauth", 1)[0]

    if error:
        return HTMLResponse(f"""
            <html><body><script>
                window.opener.postMessage({{type: "drive_oauth_error", error: "{error}"}}, "{origin}");
                window.close();
            </script></body></html>
        """)

    if not code or not state:
        return HTMLResponse(f"""
            <html><body><script>
                window.opener.postMessage({{type: "drive_oauth_error", error: "Missing code or state"}}, "{origin}");
                window.close();
            </script></body></html>
        """)

    # Verify state and get user_id
    conn = _get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT user_id, expires_at FROM google_oauth_states WHERE state = %s",
            (state,)
        )
        row = cursor.fetchone()

        if not row:
            cursor.close()
            conn.close()
            return HTMLResponse(f"""
                <html><body><script>
                    window.opener.postMessage({{type: "drive_oauth_error", error: "Invalid state"}}, "{origin}");
                    window.close();
                </script></body></html>
            """)

        user_id = row["user_id"]
        expires_at = row["expires_at"]

        # Delete used state
        cursor.execute("DELETE FROM google_oauth_states WHERE state = %s", (state,))
        conn.commit()
    except psycopg2.Error as e:
        cursor.close()
        conn.close()
        logger.error(f"OAuth callback DB error: {e}")
        return HTMLResponse(f"""
            <html><body><script>
                window.opener.postMessage({{type: "drive_oauth_error", error: "Database error"}}, "{origin}");
                window.close();
            </script></body></html>
        """)
    finally:
        cursor.close()
        conn.close()

    # Check expiration (expires_at is already a datetime from Postgres)
    if datetime.utcnow() > expires_at.replace(tzinfo=None):
        return HTMLResponse(f"""
            <html><body><script>
                window.opener.postMessage({{type: "drive_oauth_error", error: "State expired"}}, "{origin}");
                window.close();
            </script></body></html>
        """)

    # Exchange code for tokens
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": settings.google_drive_client_id,
                    "client_secret": settings.google_drive_client_secret,
                    "redirect_uri": settings.google_drive_redirect_uri,
                    "grant_type": "authorization_code",
                }
            )

            if response.status_code != 200:
                error_msg = response.json().get("error_description", "Token exchange failed")
                return HTMLResponse(f"""
                    <html><body><script>
                        window.opener.postMessage({{type: "drive_oauth_error", error: "{error_msg}"}}, "{origin}");
                        window.close();
                    </script></body></html>
                """)

            tokens = response.json()
    except Exception as e:
        logger.error(f"Token exchange failed: {e}")
        return HTMLResponse(f"""
            <html><body><script>
                window.opener.postMessage({{type: "drive_oauth_error", error: "Token exchange failed"}}, "{origin}");
                window.close();
            </script></body></html>
        """)

    # Encrypt and store refresh token
    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        return HTMLResponse(f"""
            <html><body><script>
                window.opener.postMessage({{type: "drive_oauth_error", error: "No refresh token received"}}, "{origin}");
                window.close();
            </script></body></html>
        """)

    encrypted_token = _encrypt_token(refresh_token)
    scope = tokens.get("scope", "")

    conn = _get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO google_drive_tokens (user_id, refresh_token_enc, scope, created_at, updated_at)
            VALUES (%s, %s, %s, NOW(), NOW())
            ON CONFLICT (user_id) DO UPDATE SET
                refresh_token_enc = EXCLUDED.refresh_token_enc,
                scope = EXCLUDED.scope,
                updated_at = NOW()
        """, (user_id, encrypted_token, scope))
        conn.commit()
    except psycopg2.Error as e:
        logger.error(f"Failed to store Drive token: {e}")
        return HTMLResponse(f"""
            <html><body><script>
                window.opener.postMessage({{type: "drive_oauth_error", error: "Failed to save token"}}, "{origin}");
                window.close();
            </script></body></html>
        """)
    finally:
        cursor.close()
        conn.close()

    return HTMLResponse(f"""
        <html><body><script>
            window.opener.postMessage({{type: "drive_oauth_success"}}, "{origin}");
            window.close();
        </script></body></html>
    """)


# ============== Status & Folders Endpoints ==============

@router.get("/sources/drive/status", response_model=DriveStatusResponse)
async def get_drive_status(user: dict = Depends(verify_supabase_token)):
    """Get Google Drive connection status and folder list."""
    user_id = user["user_id"]

    conn = _get_db()
    cursor = conn.cursor()
    try:
        # Check if connected
        cursor.execute(
            "SELECT 1 FROM google_drive_tokens WHERE user_id = %s",
            (user_id,)
        )
        connected = cursor.fetchone() is not None

        # Get folders
        cursor.execute(
            "SELECT folder_id, folder_name FROM google_drive_folders WHERE user_id = %s",
            (user_id,)
        )
        folders = [dict(row) for row in cursor.fetchall()]
    except psycopg2.Error as e:
        _handle_db_error(e)
    finally:
        cursor.close()
        conn.close()

    return DriveStatusResponse(connected=connected, folders=folders)


@router.post("/sources/drive/folders", response_model=FoldersResponse)
async def add_drive_folder(request: AddFolderRequest, user: dict = Depends(verify_supabase_token)):
    """Add a Drive folder to sync list."""
    user_id = user["user_id"]

    conn = _get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO google_drive_folders (user_id, folder_id, folder_name)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id, folder_id) DO UPDATE SET folder_name = EXCLUDED.folder_name
        """, (user_id, request.folder_id, request.folder_name))
        conn.commit()

        # Return updated list
        cursor.execute(
            "SELECT folder_id, folder_name FROM google_drive_folders WHERE user_id = %s",
            (user_id,)
        )
        folders = [dict(row) for row in cursor.fetchall()]
    except psycopg2.Error as e:
        _handle_db_error(e)
    finally:
        cursor.close()
        conn.close()

    return FoldersResponse(folders=folders)


@router.delete("/sources/drive/folders/{folder_id}", response_model=FoldersResponse)
async def remove_drive_folder(folder_id: str, user: dict = Depends(verify_supabase_token)):
    """Remove a Drive folder from sync list."""
    user_id = user["user_id"]

    conn = _get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "DELETE FROM google_drive_folders WHERE user_id = %s AND folder_id = %s",
            (user_id, folder_id)
        )
        conn.commit()

        # Return updated list
        cursor.execute(
            "SELECT folder_id, folder_name FROM google_drive_folders WHERE user_id = %s",
            (user_id,)
        )
        folders = [dict(row) for row in cursor.fetchall()]
    except psycopg2.Error as e:
        _handle_db_error(e)
    finally:
        cursor.close()
        conn.close()

    return FoldersResponse(folders=folders)


# ============== Picker Token Endpoint ==============

@router.post("/drive/picker-token", response_model=PickerTokenResponse)
async def get_picker_token(user: dict = Depends(verify_supabase_token)):
    """
    Get a short-lived access token for Google Picker.
    Requires user to have connected Drive first.
    """
    user_id = user["user_id"]

    # Get encrypted refresh token
    conn = _get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT refresh_token_enc FROM google_drive_tokens WHERE user_id = %s",
            (user_id,)
        )
        row = cursor.fetchone()
    except psycopg2.Error as e:
        _handle_db_error(e)
    finally:
        cursor.close()
        conn.close()

    if not row:
        raise HTTPException(
            status_code=400,
            detail="Google Drive not connected. Please connect first."
        )

    # Decrypt refresh token
    refresh_token = _decrypt_token(row["refresh_token_enc"])

    # Exchange refresh token for access token
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": settings.google_drive_client_id,
                    "client_secret": settings.google_drive_client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                }
            )

            if response.status_code != 200:
                error_msg = response.json().get("error_description", "Token refresh failed")
                raise HTTPException(status_code=400, detail=error_msg)

            tokens = response.json()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to refresh access token")

    return PickerTokenResponse(access_token=tokens["access_token"])


# ============== Public Config Endpoint ==============

@router.get("/config/public")
async def get_public_config():
    """
    Return public configuration for frontend.
    Only exposes values safe to expose publicly (API keys, not secrets).
    """
    return {
        "google_picker_api_key": settings.google_picker_api_key,
        "google_drive_client_id": settings.google_drive_client_id,
    }


# ============== Sync Endpoint ==============

@router.post("/sync/drive", response_model=DriveSyncResponse)
@limiter.limit(settings.rate_limit_sync)
async def sync_drive(request: Request, user: dict = Depends(verify_supabase_token)):
    """
    Sync documents from all user's connected Drive folders.
    On-demand sync - processes all supported files in selected folders.
    Rate limited to prevent API abuse.
    """
    user_id = user["user_id"]
    start_time = time.time()

    # Get user's refresh token and folders
    conn = _get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT refresh_token_enc FROM google_drive_tokens WHERE user_id = %s",
            (user_id,)
        )
        token_row = cursor.fetchone()

        if not token_row:
            raise HTTPException(
                status_code=400,
                detail="Google Drive not connected. Please connect first."
            )

        # Get folders
        cursor.execute(
            "SELECT folder_id, folder_name FROM google_drive_folders WHERE user_id = %s",
            (user_id,)
        )
        folders = [dict(row) for row in cursor.fetchall()]
    except psycopg2.Error as e:
        _handle_db_error(e)
    finally:
        cursor.close()
        conn.close()

    logger.info(f"Starting Drive sync for user, {len(folders)} folders")

    if not folders:
        raise HTTPException(
            status_code=400,
            detail="No folders selected. Add at least one folder."
        )

    # Decrypt refresh token and get access token
    refresh_token = _decrypt_token(token_row["refresh_token_enc"])

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": settings.google_drive_client_id,
                    "client_secret": settings.google_drive_client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                }
            )

            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to refresh token")

            access_token = response.json()["access_token"]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to refresh access token")

    # Import here to avoid circular imports
    from connectors.base import Document, SourceType
    from ingest.pipeline import ingest
    from ingest.loaders import load_document as load_file_content

    # Build credentials for Google API client
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    import io

    credentials = Credentials(
        token=access_token,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=settings.google_drive_client_id,
        client_secret=settings.google_drive_client_secret,
    )

    service = build("drive", "v3", credentials=credentials)

    # Use user_id as tenant_id for isolation
    tenant_id = user_id

    # MIME types we support
    SUPPORTED_MIME_TYPES = {
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "text/plain": ".txt",
        "text/markdown": ".md",
        "application/vnd.google-apps.document": ".docx",
        "application/vnd.google-apps.spreadsheet": ".xlsx",
    }

    EXPORT_MIME_TYPES = {
        "application/vnd.google-apps.document": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.google-apps.spreadsheet": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }

    total_found = 0
    total_processed = 0
    total_chunks = 0
    errors = []

    temp_dir = tempfile.mkdtemp()

    def get_all_folder_ids(parent_id: str, parent_name: str) -> list[dict]:
        """Recursively get all folder IDs including sub-folders."""
        all_folders = [{"id": parent_id, "name": parent_name}]
        try:
            query = f"mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents and trashed=false"
            page_token = None
            while True:
                results = service.files().list(
                    q=query,
                    pageSize=100,
                    fields="nextPageToken, files(id, name)",
                    pageToken=page_token,
                ).execute()
                for subfolder in results.get("files", []):
                    subfolder_path = f"{parent_name}/{subfolder['name']}"
                    logger.info(f"Found sub-folder: {subfolder_path}")
                    all_folders.extend(get_all_folder_ids(subfolder["id"], subfolder_path))
                page_token = results.get("nextPageToken")
                if not page_token:
                    break
        except Exception as e:
            logger.error(f"Error listing sub-folders of {parent_name}: {e}")
        return all_folders

    try:
        for folder in folders:
            folder_id = folder["folder_id"]
            folder_name = folder["folder_name"]

            # Get all folders including sub-folders
            all_folder_ids = get_all_folder_ids(folder_id, folder_name)
            logger.info(f"Processing {len(all_folder_ids)} folders (including sub-folders)")

            for current_folder in all_folder_ids:
                current_id = current_folder["id"]
                current_name = current_folder["name"]

                try:
                    # Build query for supported file types
                    mime_queries = [f"mimeType='{mime}'" for mime in SUPPORTED_MIME_TYPES.keys()]
                    mime_filter = " or ".join(mime_queries)
                    query = f"({mime_filter}) and '{current_id}' in parents and trashed=false"
                    logger.info(f"Drive query for {current_name}")

                    # List files with pagination
                    page_token = None
                    files = []

                    while True:
                        try:
                            results = service.files().list(
                                q=query,
                                pageSize=100,
                                fields="nextPageToken, files(id, name, mimeType, modifiedTime, size)",
                                pageToken=page_token,
                            ).execute()

                            files.extend(results.get("files", []))
                            page_token = results.get("nextPageToken")

                            if not page_token:
                                break

                        except Exception as e:
                            if "429" in str(e):
                                # Rate limited - wait and retry
                                time.sleep(2)
                                continue
                            raise

                    total_found += len(files)
                    logger.info(f"Found {len(files)} files in folder {current_name}")
                    for f in files:
                        logger.info(f"  - {f['name']} ({f['mimeType']})")

                    # Process each file
                    for file in files:
                        try:
                            file_id = file["id"]
                            file_name = file["name"]
                            mime_type = file["mimeType"]
                            modified_at = file.get("modifiedTime", "")

                            # Download or export file
                            if mime_type in EXPORT_MIME_TYPES:
                                # Google native format - export
                                export_mime = EXPORT_MIME_TYPES[mime_type]
                                extension = SUPPORTED_MIME_TYPES[mime_type]
                                content = service.files().export(
                                    fileId=file_id,
                                    mimeType=export_mime
                                ).execute()
                            else:
                                # Regular file - download
                                extension = SUPPORTED_MIME_TYPES.get(mime_type, ".txt")
                                request = service.files().get_media(fileId=file_id)
                                buffer = io.BytesIO()
                                downloader = MediaIoBaseDownload(buffer, request)
                                done = False
                                while not done:
                                    _, done = downloader.next_chunk()
                                content = buffer.getvalue()

                            if not content:
                                continue

                            # Save to temp file
                            temp_path = Path(temp_dir) / f"{file_id}{extension}"
                            if isinstance(content, bytes):
                                temp_path.write_bytes(content)
                            else:
                                temp_path.write_bytes(content)

                            # Load and parse content
                            result = load_file_content(temp_path)

                            extra_metadata = {}
                            if isinstance(result, tuple):
                                text_content, sheet_metadata = result
                                if sheet_metadata:
                                    sheet_names = [s["sheet_name"] for s in sheet_metadata]
                                    extra_metadata["sheet_name"] = ", ".join(sheet_names)
                            else:
                                text_content = result

                            if not text_content or not text_content.strip():
                                temp_path.unlink()
                                continue

                            # Clean up temp file
                            temp_path.unlink()

                            # Create document
                            doc = Document(
                                content=text_content,
                                source_type=SourceType.GOOGLE_DRIVE,
                                source_id=file_id,
                                external_id=file_id,
                                title=Path(file_name).stem,
                                doc_id=Path(file_name).stem,
                                tenant_id=tenant_id,
                                metadata={
                                    "source_file": file_name,
                                    "mime_type": mime_type,
                                    "drive_file_id": file_id,
                                    "folder_id": current_id,
                                    "folder_name": current_name,
                                    **extra_metadata,
                                },
                                modified_at=datetime.fromisoformat(
                                    modified_at.replace("Z", "+00:00")
                                ) if modified_at else None,
                            )

                            # Ingest document using existing pipeline
                            from ingest.chunking import chunk_text
                            from ingest.embedder import get_embeddings
                            from app.qdrant_client import get_qdrant_client
                            from app.qdrant_client import upsert_points
                            import uuid
                            from uuid import uuid4

                            # Chunk the document
                            chunks = chunk_text(
                                doc.content,
                                doc.doc_id or doc.source_id,
                                chunk_size=settings.chunk_size,
                                chunk_overlap=settings.chunk_overlap,
                                metadata=doc.metadata,
                            )

                            if not chunks:
                                continue

                            # Generate embeddings
                            texts = [c["text"] for c in chunks]
                            embeddings, _ = get_embeddings(texts)

                            # Generate point IDs and prepare points
                            points = []
                            updated_at_str = doc.modified_at.isoformat() if doc.modified_at else datetime.utcnow().isoformat()

                            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                                key = f"{tenant_id}:google_drive:{doc.external_id}:{updated_at_str}:{i}"
                                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

                                payload = {
                                    "tenant_id": tenant_id,
                                    "source": "google_drive",
                                    "external_id": doc.external_id,
                                    "doc_id": doc.doc_id,
                                    "title": doc.title,
                                    "text": chunk["text"],
                                    "chunk_index": i,
                                    "updated_at": updated_at_str,
                                    **doc.metadata,
                                }

                                from qdrant_client.models import PointStruct
                                points.append(PointStruct(
                                    id=point_id,
                                    vector=embedding,
                                    payload=payload,
                                ))

                            # Upsert via canonical function (handles Qdrant + FTS shadow)
                            ingest_id = uuid4().hex[:12]
                            upsert_points(points, tenant_id, doc_id_fallback=doc.doc_id, ingest_id=ingest_id)

                            total_processed += 1
                            total_chunks += len(chunks)

                        except Exception as e:
                            error_msg = f"Error processing {file.get('name', 'unknown')}: {str(e)}"
                            logger.error(error_msg)
                            errors.append(error_msg)

                except Exception as e:
                    error_msg = f"Error listing folder {current_name}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)

    finally:
        # Clean up temp directory
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

    duration = time.time() - start_time

    return DriveSyncResponse(
        status="completed",
        folders_synced=len(folders),
        documents_found=total_found,
        documents_processed=total_processed,
        chunks_created=total_chunks,
        errors=errors,
        duration_seconds=round(duration, 2),
    )
