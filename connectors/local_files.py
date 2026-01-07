"""
Local file system connector.

Wraps existing file loading logic into the connector interface.
"""
import hashlib
from pathlib import Path
from datetime import datetime

from connectors.base import BaseConnector, Document, SourceType
from ingest.loaders import load_document as load_file_content
from app.logging_config import get_logger

logger = get_logger(__name__)


class LocalFilesConnector(BaseConnector):
    """
    Connector for local file system documents.
    
    Supports: .md, .txt, .pdf, .docx, .xlsx
    """
    
    SUPPORTED_EXTENSIONS = (".md", ".txt", ".pdf", ".docx", ".xlsx")
    
    def __init__(
        self,
        directory: str | Path,
        tenant_id: str = "default",
        extensions: tuple[str, ...] | None = None,
    ):
        super().__init__(tenant_id)
        self.directory = Path(directory).resolve()
        self.extensions = extensions or self.SUPPORTED_EXTENSIONS
        self._connected = False
    
    @property
    def source_type(self) -> SourceType:
        return SourceType.LOCAL_FILE
    
    @property
    def name(self) -> str:
        return f"Local Files ({self.directory})"
    
    def connect(self) -> bool:
        """Verify directory exists."""
        if self.directory.exists() and self.directory.is_dir():
            self._connected = True
            return True
        return False
    
    def disconnect(self) -> None:
        """No cleanup needed for local files."""
        self._connected = False
    
    def _generate_external_id(self, path: Path) -> str:
        """
        Generate a stable external ID for a file.
        
        Uses relative path from base directory to ensure stability
        across different machines/environments.
        """
        try:
            relative = path.resolve().relative_to(self.directory)
            return hashlib.sha256(str(relative).encode()).hexdigest()[:16]
        except ValueError:
            # Path not relative to directory, use absolute path hash
            return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    
    def list_documents(self) -> list[dict]:
        """List all supported files in the directory."""
        if not self._connected:
            return []
        
        documents = []
        for ext in self.extensions:
            for path in self.directory.glob(f"*{ext}"):
                if path.is_file():
                    stat = path.stat()
                    documents.append({
                        "source_id": str(path),
                        "external_id": self._generate_external_id(path),
                        "title": path.stem,
                        "filename": path.name,
                        "extension": path.suffix.lower(),
                        "size": stat.st_size,
                        "modified_at": datetime.fromtimestamp(stat.st_mtime),
                    })
        
        return documents
    
    def fetch_document(self, source_id: str) -> Document | None:
        """Load a document from the file system."""
        path = Path(source_id)
        
        if not path.exists():
            return None
        
        try:
            result = load_file_content(path)
            
            # Handle xlsx files which return (content, sheet_metadata)
            extra_metadata = {}
            if isinstance(result, tuple):
                content, sheet_metadata = result
                if sheet_metadata:
                    sheet_names = [s["sheet_name"] for s in sheet_metadata]
                    extra_metadata["sheet_name"] = ", ".join(sheet_names)
            else:
                content = result
            
            if not content or not content.strip():
                return None
            
            # Extract doc_id from content if present
            doc_id = self._extract_doc_id(content, path)
            
            stat = path.stat()
            external_id = self._generate_external_id(path)
            
            return Document(
                content=content,
                source_type=self.source_type,
                source_id=str(path),
                external_id=external_id,
                title=path.stem,
                doc_id=doc_id,
                tenant_id=self.tenant_id,
                metadata={
                    "source_file": path.name,
                    "extension": path.suffix.lower(),
                    **extra_metadata,
                },
                modified_at=datetime.fromtimestamp(stat.st_mtime),
            )
            
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return None
    
    def _extract_doc_id(self, content: str, path: Path) -> str:
        """Extract document ID from content or use filename."""
        lines = content.split("\n")[:10]
        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith("doc-id:") or line_lower.startswith("doc_id:"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    return parts[1].strip()
        return path.stem
