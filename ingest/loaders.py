"""
Document loaders for ingestion pipeline.
Supports Markdown, Text, PDF, Word (.docx), Excel (.xlsx), RTF, CSV, and image files.

Can load from file paths or from bytes (for uploaded files).

OCR Support (ingest-time only):
- Scanned PDFs: Falls back to OCR if native extraction yields < OCR_MIN_TEXT_LENGTH chars
- Images (JPG, PNG): OCR extraction when OCR_ENABLED=true
- All OCR is fail-open: on error, returns empty string and logs

CSV Support:
- Deterministic text extraction (no ML/LLM)
- Format: "Row N: column_a = value1, column_b = value2"
"""
from io import BytesIO
from pathlib import Path
from typing import Iterator, NamedTuple

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

try:
    from openpyxl import load_workbook
except ImportError:
    load_workbook = None

try:
    from striprtf.striprtf import rtf_to_text
except ImportError:
    rtf_to_text = None


class ExtractionResult(NamedTuple):
    """Result of document extraction with metadata."""
    text: str
    extraction_method: str  # "native", "ocr", "native_csv"
    page_count: int | None = None
    metadata: dict | None = None


def load_text(path: Path) -> str:
    """Load plain text or markdown file."""
    return path.read_text(encoding="utf-8")


def load_pdf(path: Path) -> str | ExtractionResult:
    """
    Load PDF and extract text from all pages.

    OCR fallback: If native extraction yields < OCR_MIN_TEXT_LENGTH chars
    and OCR_ENABLED=true, falls back to OCR.

    Returns:
        str for backward compatibility, or ExtractionResult with metadata
    """
    if PdfReader is None:
        raise ImportError("pypdf is required for PDF loading. Install with: pip install pypdf")

    reader = PdfReader(path)
    page_count = len(reader.pages)
    texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text)
    native_text = "\n\n".join(texts)

    extraction_method = "native"

    # Check if we need OCR fallback
    if len(native_text.strip()) < settings.ocr_min_text_length and settings.ocr_enabled:
        logger.info(f"[loader] PDF native text too short ({len(native_text)} chars < {settings.ocr_min_text_length}), trying OCR | file={path.name}")

        try:
            from ingest.ocr import ocr_pdf_file
            ocr_text = ocr_pdf_file(path)

            if ocr_text and len(ocr_text.strip()) > len(native_text.strip()):
                logger.info(f"[loader] Using OCR text ({len(ocr_text)} chars) | file={path.name}")
                native_text = ocr_text
                extraction_method = "ocr"
            else:
                logger.info(f"[loader] OCR did not improve extraction, using native | file={path.name}")
        except Exception as e:
            # Capture in Sentry if available
            try:
                import sentry_sdk
                sentry_sdk.capture_exception(e)
            except ImportError:
                pass
            # Fail-open: log and continue with native text
            logger.warning(f"[loader] OCR failed, using native text | file={path.name} | error={e}")

    logger.info(f"[loader] PDF loaded | file={path.name} | method={extraction_method} | pages={page_count} | chars={len(native_text)}")

    return native_text


def load_docx(path: Path) -> str:
    """
    Load Word document and extract plain paragraph text in reading order.
    Ignores formatting, styles, headers, footers, and comments.
    """
    if DocxDocument is None:
        raise ImportError("python-docx is required for Word loading. Install with: pip install python-docx")
    
    doc = DocxDocument(path)
    texts = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            texts.append(text)
    return "\n\n".join(texts)


def load_xlsx(path: Path) -> tuple[str, list[dict]]:
    """
    Load Excel workbook and extract text from all sheets.

    Returns:
        Tuple of (full text content, list of sheet metadata dicts)
        Each sheet's content is prefixed with the sheet name.

    Note: Uses explicit cell iteration to capture pivot table display areas,
    which may not be included by default iter_rows().
    """
    if load_workbook is None:
        raise ImportError("openpyxl is required for Excel loading. Install with: pip install openpyxl")

    # Load with data_only=True to get computed/cached values instead of formulas
    wb = load_workbook(path, data_only=True)
    all_texts = []
    sheet_metadata = []

    for sheet_name in wb.sheetnames:
        sheet = wb[sheet_name]
        sheet_texts = []

        # Get explicit bounds - max_row/max_column include pivot table areas
        max_row = sheet.max_row or 0
        max_col = sheet.max_column or 0

        if max_row == 0 or max_col == 0:
            # Empty sheet, skip
            continue

        # Iterate with explicit bounds to ensure we capture all cells
        # including pivot table display areas
        for row in sheet.iter_rows(min_row=1, max_row=max_row,
                                   min_col=1, max_col=max_col):
            cell_values = []
            for cell in row:
                if cell.value is not None:
                    cell_values.append(str(cell.value))

            if cell_values:  # Skip empty rows
                row_text = "\t".join(cell_values)
                sheet_texts.append(row_text)

        # Check if sheet has pivot tables and note it in metadata
        has_pivot = len(getattr(sheet, '_pivots', [])) > 0

        if sheet_texts:
            sheet_content = f"[Sheet: {sheet_name}]\n" + "\n".join(sheet_texts)
            all_texts.append(sheet_content)
            sheet_metadata.append({
                "sheet_name": sheet_name,
                "has_pivot_table": has_pivot,
            })

    wb.close()
    return "\n\n".join(all_texts), sheet_metadata


def load_rtf(path: Path) -> str:
    """Load RTF file and convert to plain text."""
    if rtf_to_text is None:
        raise ImportError("striprtf is required for RTF loading. Install with: pip install striprtf")

    # Read as bytes first, then try different encodings
    raw_bytes = path.read_bytes()

    # Try common encodings
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            rtf_content = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        # Fallback to latin-1 which accepts any byte
        rtf_content = raw_bytes.decode("latin-1")

    text = rtf_to_text(rtf_content)
    return text.strip()


def load_image(path: Path) -> str:
    """
    Load image file and extract text using OCR.

    Only available when OCR_ENABLED=true.
    Supported formats: JPG, JPEG, PNG, TIFF, BMP, GIF, WEBP

    Args:
        path: Path to image file

    Returns:
        Extracted text.

    Raises:
        ValueError: If OCR is disabled or OCR fails/returns empty text
    """
    if not settings.ocr_enabled:
        raise ValueError(f"Image files require OCR. Set OCR_ENABLED=true to process images.")

    logger.info(f"[loader] Loading image with OCR | file={path.name}")

    try:
        from ingest.ocr import ocr_image_file
        text = ocr_image_file(path)

        # Do NOT silently ingest empty text for images
        if not text or not text.strip():
            raise ValueError(f"OCR returned empty text for image: {path.name}")

        logger.info(f"[loader] Image loaded | file={path.name} | method=ocr | chars={len(text)}")
        return text
    except ValueError:
        # Re-raise ValueError (includes empty text case)
        raise
    except Exception as e:
        # Capture in Sentry if available
        try:
            import sentry_sdk
            sentry_sdk.capture_exception(e)
        except ImportError:
            pass
        # Raise so image is SKIPPED, not ingested as empty
        logger.error(f"[loader] Image OCR failed | file={path.name} | error={e}")
        raise ValueError(f"OCR failed for image {path.name}: {e}")


def load_image_from_bytes(data: bytes, filename: str) -> str:
    """
    Load image from bytes and extract text using OCR.

    Only available when OCR_ENABLED=true.

    Args:
        data: Image file content as bytes
        filename: Original filename for logging

    Returns:
        Extracted text.

    Raises:
        ValueError: If OCR is disabled or OCR fails/returns empty text
    """
    if not settings.ocr_enabled:
        raise ValueError(f"Image files require OCR. Set OCR_ENABLED=true to process images.")

    logger.info(f"[loader] Loading image with OCR | file={filename}")

    try:
        from ingest.ocr import ocr_image_bytes
        text = ocr_image_bytes(data, filename=filename)

        # Do NOT silently ingest empty text for images
        if not text or not text.strip():
            raise ValueError(f"OCR returned empty text for image: {filename}")

        logger.info(f"[loader] Image loaded | file={filename} | method=ocr | chars={len(text)}")
        return text
    except ValueError:
        # Re-raise ValueError (includes empty text case)
        raise
    except Exception as e:
        # Capture in Sentry if available
        try:
            import sentry_sdk
            sentry_sdk.capture_exception(e)
        except ImportError:
            pass
        # Raise so image is SKIPPED, not ingested as empty
        logger.error(f"[loader] Image OCR failed | file={filename} | error={e}")
        raise ValueError(f"OCR failed for image {filename}: {e}")


def load_csv(path: Path) -> tuple[str, dict]:
    """
    Load CSV file and convert to readable text.

    Format: "Row N: column_a = value1, column_b = value2"

    Args:
        path: Path to CSV file

    Returns:
        Tuple of (text content, metadata dict)
    """
    logger.info(f"[loader] Loading CSV | file={path.name}")

    from ingest.csv_loader import load_csv_file
    text, metadata = load_csv_file(path)

    logger.info(f"[loader] CSV loaded | file={path.name} | method=native_csv | rows={metadata.row_count} | chars={len(text)}")

    return text, {
        "source_type": "csv",
        "extraction_method": "native_csv",
        "row_count": metadata.row_count,
        "column_count": metadata.column_count,
        "columns": metadata.columns,
        "has_header": metadata.has_header,
    }


def load_csv_from_bytes(data: bytes, filename: str) -> tuple[str, dict]:
    """
    Load CSV from bytes and convert to readable text.

    Args:
        data: CSV file content as bytes
        filename: Original filename for logging

    Returns:
        Tuple of (text content, metadata dict)
    """
    logger.info(f"[loader] Loading CSV | file={filename}")

    from ingest.csv_loader import load_csv_to_text
    text, metadata = load_csv_to_text(data, filename=filename)

    logger.info(f"[loader] CSV loaded | file={filename} | method=native_csv | rows={metadata.row_count} | chars={len(text)}")

    return text, {
        "source_type": "csv",
        "extraction_method": "native_csv",
        "row_count": metadata.row_count,
        "column_count": metadata.column_count,
        "columns": metadata.columns,
        "has_header": metadata.has_header,
    }


# Image file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif", ".webp"}


def load_document(path: Path) -> str | tuple[str, list[dict] | dict]:
    """
    Load a document based on its file extension.

    Supported formats:
    - .txt, .md: Plain text
    - .pdf: PDF (requires pypdf, OCR fallback if enabled)
    - .docx: Word (requires python-docx)
    - .xlsx: Excel (requires openpyxl) - returns tuple with sheet metadata
    - .rtf: Rich Text Format (requires striprtf)
    - .csv: CSV files - returns tuple with CSV metadata
    - .jpg, .png, etc.: Images (requires OCR_ENABLED=true)

    Returns:
        str for most formats, or tuple[str, metadata] for xlsx/csv
    """
    suffix = path.suffix.lower()

    if suffix in (".txt", ".md"):
        return load_text(path)
    elif suffix == ".pdf":
        return load_pdf(path)
    elif suffix == ".docx":
        return load_docx(path)
    elif suffix == ".xlsx":
        return load_xlsx(path)
    elif suffix == ".rtf":
        return load_rtf(path)
    elif suffix == ".csv":
        return load_csv(path)
    elif suffix in IMAGE_EXTENSIONS:
        return load_image(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


# =============================================================================
# Loaders from bytes (for uploaded files)
# =============================================================================

def load_text_from_bytes(data: bytes) -> str:
    """Load plain text or markdown from bytes."""
    return data.decode("utf-8")


def load_pdf_from_bytes(data: bytes, filename: str = "unknown.pdf") -> str:
    """
    Load PDF from bytes and extract text from all pages.

    OCR fallback: If native extraction yields < OCR_MIN_TEXT_LENGTH chars
    and OCR_ENABLED=true, falls back to OCR.

    Args:
        data: PDF file content as bytes
        filename: Original filename for logging

    Returns:
        Extracted text
    """
    if PdfReader is None:
        raise ImportError("pypdf is required for PDF loading. Install with: pip install pypdf")

    reader = PdfReader(BytesIO(data))
    page_count = len(reader.pages)
    texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text)
    native_text = "\n\n".join(texts)

    extraction_method = "native"

    # Check if we need OCR fallback
    if len(native_text.strip()) < settings.ocr_min_text_length and settings.ocr_enabled:
        logger.info(f"[loader] PDF native text too short ({len(native_text)} chars < {settings.ocr_min_text_length}), trying OCR | file={filename}")

        try:
            from ingest.ocr import ocr_pdf_bytes
            ocr_text = ocr_pdf_bytes(data, filename=filename)

            if ocr_text and len(ocr_text.strip()) > len(native_text.strip()):
                logger.info(f"[loader] Using OCR text ({len(ocr_text)} chars) | file={filename}")
                native_text = ocr_text
                extraction_method = "ocr"
            else:
                logger.info(f"[loader] OCR did not improve extraction, using native | file={filename}")
        except Exception as e:
            # Capture in Sentry if available
            try:
                import sentry_sdk
                sentry_sdk.capture_exception(e)
            except ImportError:
                pass
            # Fail-open: log and continue with native text
            logger.warning(f"[loader] OCR failed, using native text | file={filename} | error={e}")

    logger.info(f"[loader] PDF loaded | file={filename} | method={extraction_method} | pages={page_count} | chars={len(native_text)}")

    return native_text


def load_docx_from_bytes(data: bytes) -> str:
    """Load Word document from bytes and extract plain paragraph text."""
    if DocxDocument is None:
        raise ImportError("python-docx is required for Word loading. Install with: pip install python-docx")

    doc = DocxDocument(BytesIO(data))
    texts = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            texts.append(text)
    return "\n\n".join(texts)


def load_xlsx_from_bytes(data: bytes) -> tuple[str, list[dict]]:
    """Load Excel workbook from bytes and extract text from all sheets.

    Note: Uses explicit cell iteration to capture pivot table display areas.
    """
    if load_workbook is None:
        raise ImportError("openpyxl is required for Excel loading. Install with: pip install openpyxl")

    wb = load_workbook(BytesIO(data), data_only=True)
    all_texts = []
    sheet_metadata = []

    for sheet_name in wb.sheetnames:
        sheet = wb[sheet_name]
        sheet_texts = []

        # Get explicit bounds - max_row/max_column include pivot table areas
        max_row = sheet.max_row or 0
        max_col = sheet.max_column or 0

        if max_row == 0 or max_col == 0:
            continue

        # Iterate with explicit bounds to capture pivot table display areas
        for row in sheet.iter_rows(min_row=1, max_row=max_row,
                                   min_col=1, max_col=max_col):
            cell_values = []
            for cell in row:
                if cell.value is not None:
                    cell_values.append(str(cell.value))

            if cell_values:
                row_text = "\t".join(cell_values)
                sheet_texts.append(row_text)

        has_pivot = len(getattr(sheet, '_pivots', [])) > 0

        if sheet_texts:
            sheet_content = f"[Sheet: {sheet_name}]\n" + "\n".join(sheet_texts)
            all_texts.append(sheet_content)
            sheet_metadata.append({
                "sheet_name": sheet_name,
                "has_pivot_table": has_pivot,
            })

    wb.close()
    return "\n\n".join(all_texts), sheet_metadata


def load_rtf_from_bytes(data: bytes) -> str:
    """Load RTF from bytes and convert to plain text."""
    if rtf_to_text is None:
        raise ImportError("striprtf is required for RTF loading. Install with: pip install striprtf")

    # Try common encodings
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            rtf_content = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        # Fallback to latin-1 which accepts any byte
        rtf_content = data.decode("latin-1")

    text = rtf_to_text(rtf_content)
    return text.strip()


def load_document_from_bytes(data: bytes, filename: str) -> str | tuple[str, list[dict] | dict]:
    """
    Load a document from bytes based on the filename extension.

    Args:
        data: File content as bytes.
        filename: Original filename (used to determine file type).

    Returns:
        str for most formats, or tuple[str, metadata] for xlsx/csv.
    """
    suffix = Path(filename).suffix.lower()

    if suffix in (".txt", ".md"):
        return load_text_from_bytes(data)
    elif suffix == ".pdf":
        return load_pdf_from_bytes(data, filename=filename)
    elif suffix == ".docx":
        return load_docx_from_bytes(data)
    elif suffix == ".xlsx":
        return load_xlsx_from_bytes(data)
    elif suffix == ".rtf":
        return load_rtf_from_bytes(data)
    elif suffix == ".csv":
        return load_csv_from_bytes(data, filename=filename)
    elif suffix in IMAGE_EXTENSIONS:
        return load_image_from_bytes(data, filename=filename)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def iter_documents(
    directory: Path,
    extensions: tuple[str, ...] | None = None,
    include_images: bool = False,
) -> Iterator[tuple[Path, str | tuple[str, list[dict] | dict]]]:
    """
    Iterate over all documents in a directory.

    Args:
        directory: Directory to search
        extensions: File extensions to include (defaults to standard document types + CSV)
        include_images: Include image files (requires OCR_ENABLED=true)

    Yields:
        Tuples of (path, content) for each document.
        For xlsx/csv files, content is a tuple of (text, metadata).
    """
    # Default extensions
    if extensions is None:
        extensions = (".md", ".txt", ".pdf", ".docx", ".xlsx", ".rtf", ".csv")

    # Add image extensions if requested and OCR is enabled
    if include_images and settings.ocr_enabled:
        extensions = extensions + tuple(IMAGE_EXTENSIONS)
    elif include_images and not settings.ocr_enabled:
        logger.warning("[loader] include_images=True but OCR_ENABLED=false, skipping images")

    for ext in extensions:
        for path in directory.glob(f"*{ext}"):
            if path.is_file():
                try:
                    content = load_document(path)
                    yield path, content
                except ValueError as e:
                    logger.warning(f"[loader] Skipping unsupported file {path}: {e}")
                except Exception as e:
                    logger.error(f"[loader] Error loading {path}: {e}")

