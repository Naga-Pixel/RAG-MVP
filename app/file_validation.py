"""
File content validation module.

Validates that file content matches the claimed file extension by checking magic bytes.
Prevents attacks where malicious files are renamed with safe extensions.
"""

# Magic byte signatures for supported file types
# Format: extension -> (magic_bytes, offset, description)
MAGIC_SIGNATURES: dict[str, list[tuple[bytes, int, str]]] = {
    # Documents
    ".pdf": [
        (b"%PDF", 0, "PDF document"),
    ],
    ".docx": [
        (b"PK\x03\x04", 0, "Office Open XML (ZIP)"),  # DOCX is a ZIP file
    ],
    ".xlsx": [
        (b"PK\x03\x04", 0, "Office Open XML (ZIP)"),  # XLSX is a ZIP file
    ],
    ".rtf": [
        (b"{\\rtf", 0, "Rich Text Format"),
    ],
    # Images
    ".png": [
        (b"\x89PNG\r\n\x1a\n", 0, "PNG image"),
    ],
    ".jpg": [
        (b"\xff\xd8\xff", 0, "JPEG image"),
    ],
    ".jpeg": [
        (b"\xff\xd8\xff", 0, "JPEG image"),
    ],
    ".gif": [
        (b"GIF87a", 0, "GIF image (87a)"),
        (b"GIF89a", 0, "GIF image (89a)"),
    ],
    ".tiff": [
        (b"II\x2a\x00", 0, "TIFF image (little-endian)"),
        (b"MM\x00\x2a", 0, "TIFF image (big-endian)"),
    ],
    ".tif": [
        (b"II\x2a\x00", 0, "TIFF image (little-endian)"),
        (b"MM\x00\x2a", 0, "TIFF image (big-endian)"),
    ],
    ".bmp": [
        (b"BM", 0, "BMP image"),
    ],
    ".webp": [
        (b"RIFF", 0, "WebP container"),  # Full check includes WEBP at offset 8
    ],
}

# Extensions that are plain text (no magic bytes to validate)
# These are validated by checking they're valid UTF-8 text
TEXT_EXTENSIONS = {".txt", ".md", ".csv"}

# Minimum file size for binary files (prevents empty file attacks)
MIN_BINARY_SIZE = 4


def validate_file_content(content: bytes, extension: str) -> tuple[bool, str]:
    """
    Validate that file content matches the claimed extension.

    Args:
        content: Raw file bytes
        extension: Claimed file extension (e.g., ".pdf")

    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is empty string.
    """
    ext = extension.lower()

    # Check minimum size for binary files
    if ext not in TEXT_EXTENSIONS and len(content) < MIN_BINARY_SIZE:
        return False, f"File too small to be a valid {ext} file"

    # Text files: validate UTF-8 encoding
    if ext in TEXT_EXTENSIONS:
        return _validate_text_file(content, ext)

    # WebP needs special handling (RIFF container with WEBP marker)
    if ext == ".webp":
        return _validate_webp(content)

    # Binary files: check magic bytes
    if ext in MAGIC_SIGNATURES:
        return _validate_magic_bytes(content, ext)

    # Unknown extension - allow through (filtered elsewhere)
    return True, ""


def _validate_text_file(content: bytes, ext: str) -> tuple[bool, str]:
    """Validate text file is valid UTF-8."""
    try:
        # Try to decode as UTF-8
        text = content.decode("utf-8")

        # Check for null bytes (indicates binary file)
        if "\x00" in text:
            return False, f"File contains binary data, not valid {ext}"

        return True, ""

    except UnicodeDecodeError:
        # Try other common encodings
        for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
            try:
                text = content.decode(encoding)
                if "\x00" not in text:
                    return True, ""
            except UnicodeDecodeError:
                continue

        return False, f"File is not valid text ({ext})"


def _validate_magic_bytes(content: bytes, ext: str) -> tuple[bool, str]:
    """Validate file has correct magic bytes for its extension."""
    signatures = MAGIC_SIGNATURES.get(ext, [])

    if not signatures:
        return True, ""  # No signature defined, allow

    for magic, offset, description in signatures:
        if len(content) >= offset + len(magic):
            if content[offset : offset + len(magic)] == magic:
                return True, ""

    # None of the signatures matched
    expected = " or ".join(sig[2] for sig in signatures)
    return False, f"File content doesn't match {ext} format (expected {expected})"


def _validate_webp(content: bytes) -> tuple[bool, str]:
    """Validate WebP file (RIFF container with WEBP marker)."""
    if len(content) < 12:
        return False, "File too small to be a valid WebP"

    # Check RIFF header
    if content[:4] != b"RIFF":
        return False, "File content doesn't match .webp format (missing RIFF header)"

    # Check WEBP marker at offset 8
    if content[8:12] != b"WEBP":
        return False, "File content doesn't match .webp format (missing WEBP marker)"

    return True, ""
