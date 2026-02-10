"""
CSV loader for deterministic text extraction.

Converts CSV files to readable text format suitable for RAG ingestion.
No ML, no LLM - pure deterministic parsing.

Output format:
    Row 1: column_a = value1, column_b = value2
    Row 2: column_a = value3, column_b = value4
    ...
"""
import csv
from io import BytesIO, StringIO
from pathlib import Path
from typing import NamedTuple

from app.logging_config import get_logger

logger = get_logger(__name__)


class CSVMetadata(NamedTuple):
    """Metadata about a parsed CSV file."""
    row_count: int
    column_count: int
    columns: list[str]
    has_header: bool


def _detect_encoding(data: bytes) -> str:
    """
    Detect encoding of CSV data.

    Tries common encodings in order of likelihood.
    Falls back to latin-1 which accepts any byte.

    Args:
        data: Raw CSV bytes

    Returns:
        Detected encoding name
    """
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            data.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue

    # latin-1 always works as fallback
    return "latin-1"


def _detect_dialect(sample: str) -> csv.Dialect:
    """
    Detect CSV dialect (delimiter, quoting, etc.)

    Args:
        sample: Sample of CSV text

    Returns:
        Detected dialect or excel as default
    """
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect
    except csv.Error:
        # Default to standard CSV
        return csv.excel


def _has_header(sample: str, dialect: csv.Dialect) -> bool:
    """
    Detect if CSV has a header row.

    Args:
        sample: Sample of CSV text
        dialect: CSV dialect

    Returns:
        True if header detected, False otherwise
    """
    try:
        return csv.Sniffer().has_header(sample)
    except csv.Error:
        # Assume header exists (safer for RAG)
        return True


def load_csv_to_text(
    data: bytes | str,
    filename: str = "unknown.csv",
    max_rows: int | None = None,
) -> tuple[str, CSVMetadata]:
    """
    Convert CSV data to readable text format.

    Output format:
        Row 1: column_a = value1, column_b = value2
        Row 2: column_a = value3, column_b = value4

    Args:
        data: CSV content as bytes or string
        filename: Original filename for logging
        max_rows: Maximum rows to process (None = all)

    Returns:
        Tuple of (text content, metadata)
    """
    logger.info(f"[csv] Loading CSV | file={filename}")

    # Handle bytes input
    if isinstance(data, bytes):
        encoding = _detect_encoding(data)
        text_data = data.decode(encoding)
        logger.debug(f"[csv] Detected encoding: {encoding}")
    else:
        text_data = data

    # Get sample for dialect detection
    sample_lines = text_data[:8192]  # First 8KB

    # Detect dialect and header
    dialect = _detect_dialect(sample_lines)
    has_header = _has_header(sample_lines, dialect)

    logger.debug(f"[csv] Dialect: delimiter='{dialect.delimiter}' | has_header={has_header}")

    # Parse CSV
    reader = csv.reader(StringIO(text_data), dialect)
    rows = list(reader)

    if not rows:
        logger.warning(f"[csv] Empty CSV file | file={filename}")
        return "", CSVMetadata(row_count=0, column_count=0, columns=[], has_header=False)

    # Extract headers
    if has_header:
        headers = rows[0]
        data_rows = rows[1:]
    else:
        # Generate column names: Col1, Col2, etc.
        num_cols = max(len(row) for row in rows) if rows else 0
        headers = [f"Col{i+1}" for i in range(num_cols)]
        data_rows = rows

    # Apply max_rows limit
    if max_rows is not None and len(data_rows) > max_rows:
        logger.info(f"[csv] Limiting to {max_rows} rows (total: {len(data_rows)})")
        data_rows = data_rows[:max_rows]

    # Convert to text format
    text_lines = []
    for row_num, row in enumerate(data_rows, start=1):
        # Build key=value pairs
        pairs = []
        for i, value in enumerate(row):
            if i < len(headers):
                col_name = headers[i]
            else:
                col_name = f"Col{i+1}"

            # Clean up value
            value = value.strip() if value else ""

            if value:  # Only include non-empty values
                pairs.append(f"{col_name} = {value}")

        if pairs:
            line = f"Row {row_num}: " + ", ".join(pairs)
            text_lines.append(line)

    text_content = "\n".join(text_lines)

    metadata = CSVMetadata(
        row_count=len(data_rows),
        column_count=len(headers),
        columns=headers,
        has_header=has_header,
    )

    logger.info(f"[csv] Loaded CSV | file={filename} | rows={metadata.row_count} | cols={metadata.column_count} | chars={len(text_content)}")

    return text_content, metadata


def load_csv_file(path: Path, max_rows: int | None = None) -> tuple[str, CSVMetadata]:
    """
    Load CSV file and convert to text.

    Args:
        path: Path to CSV file
        max_rows: Maximum rows to process

    Returns:
        Tuple of (text content, metadata)
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    data = path.read_bytes()
    return load_csv_to_text(data, filename=path.name, max_rows=max_rows)


def load_csv_from_bytes(data: bytes, filename: str, max_rows: int | None = None) -> tuple[str, CSVMetadata]:
    """
    Load CSV from bytes and convert to text.

    Args:
        data: CSV content as bytes
        filename: Original filename
        max_rows: Maximum rows to process

    Returns:
        Tuple of (text content, metadata)
    """
    return load_csv_to_text(data, filename=filename, max_rows=max_rows)
