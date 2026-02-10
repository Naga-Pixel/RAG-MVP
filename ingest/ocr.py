"""
OCR module for extracting text from images and scanned PDFs.

This module is INGEST-TIME ONLY. By the time chunking starts,
everything must be plain text. No multimodal retrieval.

Supports:
- Google Vision API (default, recommended)

Fail-open: If OCR fails, returns empty string and logs error.
"""
import os
from io import BytesIO
from pathlib import Path
from typing import Literal

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

# Type alias for supported providers
OCRProvider = Literal["google_vision"]


class OCRError(Exception):
    """Raised when OCR processing fails."""
    pass


def _check_ocr_enabled() -> bool:
    """Check if OCR is enabled in settings."""
    if not settings.ocr_enabled:
        logger.debug("OCR is disabled (OCR_ENABLED=false)")
        return False
    return True


def _get_google_vision_client():
    """
    Get Google Vision API client.

    Uses credentials from:
    1. OCR_GOOGLE_CREDENTIALS_PATH setting
    2. GOOGLE_APPLICATION_CREDENTIALS environment variable

    Returns:
        vision.ImageAnnotatorClient or None if unavailable
    """
    try:
        from google.cloud import vision
    except ImportError:
        logger.error("google-cloud-vision not installed. Install with: pip install google-cloud-vision")
        return None

    # Set credentials path if configured
    if settings.ocr_google_credentials_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.ocr_google_credentials_path

    try:
        client = vision.ImageAnnotatorClient()
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Google Vision client: {e}")
        return None


def ocr_image_bytes(
    image_bytes: bytes,
    filename: str = "unknown",
    provider: OCRProvider | None = None,
) -> str:
    """
    Extract text from image bytes using OCR.

    Args:
        image_bytes: Raw image data (JPG, PNG, etc.)
        filename: Original filename for logging
        provider: OCR provider to use (defaults to settings.ocr_provider)

    Returns:
        Extracted text as string. Empty string on failure (fail-open).

    Raises:
        Nothing - fail-open design, errors are logged.
    """
    if not _check_ocr_enabled():
        return ""

    provider = provider or settings.ocr_provider

    logger.info(f"[ocr] Starting OCR | file={filename} | provider={provider} | size={len(image_bytes)} bytes")

    if provider == "google_vision":
        return _ocr_google_vision(image_bytes, filename)
    else:
        logger.error(f"[ocr] Unknown OCR provider: {provider}")
        return ""


def _ocr_google_vision(image_bytes: bytes, filename: str) -> str:
    """
    Run OCR using Google Vision API.

    Args:
        image_bytes: Raw image data
        filename: Original filename for logging

    Returns:
        Extracted text. Empty string on failure.
    """
    try:
        from google.cloud import vision
    except ImportError:
        logger.error("[ocr] google-cloud-vision not installed")
        return ""

    client = _get_google_vision_client()
    if client is None:
        return ""

    try:
        logger.info(f"[ocr] Calling Vision API | file={filename} | image_size={len(image_bytes)} bytes")
        image = vision.Image(content=image_bytes)

        # Use document_text_detection for better results on documents
        response = client.document_text_detection(image=image)

        if response.error.message:
            logger.error(f"[ocr] Google Vision API error | file={filename} | error={response.error.message}")
            return ""

        # Log response details for debugging
        text_annotations_count = len(response.text_annotations) if response.text_annotations else 0
        logger.info(f"[ocr] Vision response | file={filename} | text_annotations={text_annotations_count} | has_full_text={bool(response.full_text_annotation)}")

        # Extract full text annotation
        if response.full_text_annotation:
            text = response.full_text_annotation.text
            logger.info(f"[ocr] Success | file={filename} | chars={len(text)}")
            return text
        elif response.text_annotations:
            # Fallback: use first text annotation (contains all text)
            text = response.text_annotations[0].description
            logger.info(f"[ocr] Success (fallback) | file={filename} | chars={len(text)}")
            return text
        else:
            logger.warning(f"[ocr] No text found in image | file={filename}")
            return ""

    except Exception as e:
        logger.error(f"[ocr] Google Vision failed | file={filename} | error={type(e).__name__}: {e}")
        return ""


def ocr_image_file(
    path: Path,
    provider: OCRProvider | None = None,
) -> str:
    """
    Extract text from an image file using OCR.

    Args:
        path: Path to image file (JPG, PNG, etc.)
        provider: OCR provider to use

    Returns:
        Extracted text. Empty string on failure.
    """
    if not _check_ocr_enabled():
        return ""

    if not path.exists():
        logger.error(f"[ocr] File not found: {path}")
        return ""

    image_bytes = path.read_bytes()
    return ocr_image_bytes(image_bytes, filename=path.name, provider=provider)


def ocr_pdf_bytes(
    pdf_bytes: bytes,
    filename: str = "unknown.pdf",
    provider: OCRProvider | None = None,
) -> str:
    """
    Extract text from a scanned PDF using OCR.

    Converts each PDF page to an image and runs OCR.

    Args:
        pdf_bytes: Raw PDF data
        filename: Original filename for logging
        provider: OCR provider to use

    Returns:
        Extracted text from all pages. Empty string on failure.
    """
    if not _check_ocr_enabled():
        return ""

    provider = provider or settings.ocr_provider

    logger.info(f"[ocr] Starting PDF OCR | file={filename} | provider={provider}")

    try:
        # Use pdf2image to convert PDF pages to images
        from pdf2image import convert_from_bytes
    except ImportError:
        logger.error("[ocr] pdf2image not installed. Install with: pip install pdf2image")
        logger.error("[ocr] Also requires poppler: brew install poppler (macOS) or apt-get install poppler-utils (Linux)")
        return ""

    try:
        # Convert PDF to images (one per page)
        # Use reasonable DPI for OCR (300 is good balance of quality/speed)
        images = convert_from_bytes(pdf_bytes, dpi=300, fmt="png")
        logger.info(f"[ocr] Converted PDF to {len(images)} page images | file={filename}")

        all_text = []
        for i, image in enumerate(images):
            # Convert PIL Image to bytes
            img_buffer = BytesIO()
            image.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()

            # OCR the page
            page_text = ocr_image_bytes(
                img_bytes,
                filename=f"{filename}_page_{i+1}",
                provider=provider,
            )

            if page_text:
                all_text.append(page_text)

        combined_text = "\n\n".join(all_text)
        logger.info(f"[ocr] PDF OCR complete | file={filename} | pages={len(images)} | chars={len(combined_text)}")
        return combined_text

    except Exception as e:
        logger.error(f"[ocr] PDF OCR failed | file={filename} | error={type(e).__name__}: {e}")
        return ""


def ocr_pdf_file(
    path: Path,
    provider: OCRProvider | None = None,
) -> str:
    """
    Extract text from a scanned PDF file using OCR.

    Args:
        path: Path to PDF file
        provider: OCR provider to use

    Returns:
        Extracted text. Empty string on failure.
    """
    if not _check_ocr_enabled():
        return ""

    if not path.exists():
        logger.error(f"[ocr] File not found: {path}")
        return ""

    pdf_bytes = path.read_bytes()
    return ocr_pdf_bytes(pdf_bytes, filename=path.name, provider=provider)


def is_ocr_available() -> bool:
    """
    Check if OCR is available and properly configured.

    Returns:
        True if OCR can be used, False otherwise.
    """
    if not settings.ocr_enabled:
        return False

    if settings.ocr_provider == "google_vision":
        client = _get_google_vision_client()
        return client is not None

    return False
