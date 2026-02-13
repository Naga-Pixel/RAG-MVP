"""
Google Drive connector.

Fetches documents from Google Drive using a service account.
"""
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Any

from connectors.base import BaseConnector, Document, SourceType
from ingest.loaders import load_document as load_file_content
from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

# Google API imports - will fail gracefully if not installed
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False


class GoogleDriveConnector(BaseConnector):
    """
    Connector for Google Drive documents.

    Requires:
    - google-api-python-client
    - google-auth

    Setup:
    1. Create a service account in Google Cloud Console
    2. Download the JSON key file
    3. Share your Drive folder with the service account email

    Images (.jpg, .png, etc.) supported when OCR_ENABLED=true.
    """

    # Base MIME types always supported
    BASE_MIME_TYPES = {
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "application/rtf": ".rtf",
        "text/plain": ".txt",
        "text/markdown": ".md",
        "text/csv": ".csv",
        # Google Docs native formats - will be exported
        "application/vnd.google-apps.document": ".docx",
        "application/vnd.google-apps.spreadsheet": ".xlsx",
    }

    # Image MIME types (only supported when OCR is enabled)
    IMAGE_MIME_TYPES = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/tiff": ".tiff",
        "image/bmp": ".bmp",
        "image/gif": ".gif",
        "image/webp": ".webp",
    }

    @classmethod
    def get_supported_mime_types(cls) -> dict[str, str]:
        """Get supported MIME types based on current settings."""
        mime_types = cls.BASE_MIME_TYPES.copy()
        if settings.ocr_enabled:
            mime_types.update(cls.IMAGE_MIME_TYPES)
        return mime_types

    # Legacy class variable for backward compatibility
    SUPPORTED_MIME_TYPES = BASE_MIME_TYPES
    
    EXPORT_MIME_TYPES = {
        "application/vnd.google-apps.document": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.google-apps.spreadsheet": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    
    def __init__(
        self,
        credentials_path: str | None = None,
        folder_id: str | None = None,
        tenant_id: str = "default",
    ):
        """
        Initialize Google Drive connector.
        
        Args:
            credentials_path: Path to service account JSON key file.
            folder_id: Google Drive folder ID to sync from.
            tenant_id: Tenant identifier for multi-tenancy.
        """
        super().__init__(tenant_id)
        self.credentials_path = credentials_path
        self.folder_id = folder_id
        self._service = None
        self._temp_dir = None
    
    @property
    def source_type(self) -> SourceType:
        return SourceType.GOOGLE_DRIVE
    
    @property
    def name(self) -> str:
        return "Google Drive"
    
    def connect(self) -> bool:
        """Establish connection to Google Drive API."""
        if not GOOGLE_API_AVAILABLE:
            print("Google API libraries not installed. Run: pip install google-api-python-client google-auth")
            return False
        
        if not self.credentials_path:
            print("No credentials path provided")
            return False
        
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=["https://www.googleapis.com/auth/drive.readonly"]
            )
            self._service = build("drive", "v3", credentials=credentials)
            self._temp_dir = tempfile.mkdtemp()
            return True
        except Exception as e:
            logger.error("Failed to connect to Google Drive: {e}")
            return False
    
    def disconnect(self) -> None:
        """Clean up resources."""
        self._service = None
        if self._temp_dir:
            import shutil
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass
            self._temp_dir = None
    
    def list_documents(self) -> list[dict]:
        """List all supported documents in the configured folder."""
        if not self._service:
            return []

        documents = []
        max_files = settings.max_sync_files

        try:
            # Build query for supported file types (dynamic based on OCR settings)
            supported_mimes = self.get_supported_mime_types()
            mime_queries = [f"mimeType='{mime}'" for mime in supported_mimes.keys()]
            mime_filter = " or ".join(mime_queries)

            query = f"({mime_filter}) and trashed=false"
            if self.folder_id:
                query += f" and '{self.folder_id}' in parents"

            # Limit to max_sync_files to prevent blocking
            results = self._service.files().list(
                q=query,
                pageSize=min(100, max_files),
                fields="files(id, name, mimeType, modifiedTime, size)",
            ).execute()

            for file in results.get("files", []):
                if len(documents) >= max_files:
                    logger.warning(f"Reached max sync limit ({max_files} files). Some files were skipped.")
                    break
                documents.append({
                    "source_id": file["id"],
                    "title": file["name"],
                    "mime_type": file["mimeType"],
                    "modified_at": datetime.fromisoformat(file["modifiedTime"].replace("Z", "+00:00")),
                    "size": int(file.get("size", 0)),
                })

        except Exception as e:
            logger.error("Error listing Drive files: {e}")

        return documents
    
    def fetch_document(self, source_id: str) -> Document | None:
        """Download and parse a document from Google Drive."""
        if not self._service or not self._temp_dir:
            return None
        
        try:
            # Get file metadata
            file_meta = self._service.files().get(
                fileId=source_id,
                fields="id, name, mimeType, modifiedTime"
            ).execute()
            
            mime_type = file_meta["mimeType"]
            file_name = file_meta["name"]
            
            # Determine extension and download method
            supported_mimes = self.get_supported_mime_types()
            if mime_type in self.EXPORT_MIME_TYPES:
                # Google native format - need to export
                export_mime = self.EXPORT_MIME_TYPES[mime_type]
                extension = supported_mimes[mime_type]
                content = self._export_file(source_id, export_mime)
            else:
                # Regular file - direct download
                extension = supported_mimes.get(mime_type, ".txt")
                content = self._download_file(source_id)
            
            if not content:
                return None
            
            # Save to temp file and load with existing loaders
            temp_path = Path(self._temp_dir) / f"{source_id}{extension}"
            temp_path.write_bytes(content)
            
            # Load content using existing loaders
            result = load_file_content(temp_path)
            
            # Handle xlsx files which return (content, sheet_metadata)
            extra_metadata = {}
            if isinstance(result, tuple):
                text_content, sheet_metadata = result
                if sheet_metadata:
                    sheet_names = [s["sheet_name"] for s in sheet_metadata]
                    extra_metadata["sheet_name"] = ", ".join(sheet_names)
            else:
                text_content = result
            
            if not text_content or not text_content.strip():
                return None
            
            # Clean up temp file
            temp_path.unlink()
            
            return Document(
                content=text_content,
                source_type=self.source_type,
                source_id=source_id,
                external_id=source_id,  # Google Drive file ID is stable
                title=Path(file_name).stem,
                doc_id=self._extract_doc_id(text_content, file_name),
                tenant_id=self.tenant_id,
                metadata={
                    "source_file": file_name,
                    "mime_type": mime_type,
                    "drive_file_id": source_id,
                    **extra_metadata,
                },
                modified_at=datetime.fromisoformat(
                    file_meta["modifiedTime"].replace("Z", "+00:00")
                ),
            )
            
        except Exception as e:
            logger.error("Error fetching Drive file {source_id}: {e}")
            return None
    
    def _download_file(self, file_id: str) -> bytes | None:
        """Download a file directly."""
        try:
            import io
            request = self._service.files().get_media(fileId=file_id)
            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)
            
            done = False
            while not done:
                _, done = downloader.next_chunk()
            
            return buffer.getvalue()
        except Exception as e:
            logger.error("Error downloading file: {e}")
            return None
    
    def _export_file(self, file_id: str, mime_type: str) -> bytes | None:
        """Export a Google native format file."""
        try:
            return self._service.files().export(
                fileId=file_id,
                mimeType=mime_type
            ).execute()
        except Exception as e:
            logger.error("Error exporting file: {e}")
            return None
    
    def _extract_doc_id(self, content: str, filename: str) -> str:
        """Extract document ID from content or use filename."""
        lines = content.split("\n")[:10]
        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith("doc-id:") or line_lower.startswith("doc_id:"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    return parts[1].strip()
        return Path(filename).stem
