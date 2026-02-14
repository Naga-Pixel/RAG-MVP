"""
Context formatter for improving LLM comprehension of structured documents.

Transforms retrieved chunk text at query time based on document type detection.
Works with existing documents - no re-ingestion required.
"""

import re
from typing import Literal

ContentType = Literal["legal", "spreadsheet", "general"]

# Legal document role patterns (parties to agreements)
LEGAL_ROLE_PATTERN = re.compile(
    r'\b(Landlord|Tenant|Lessor|Lessee|Buyer|Seller|Employer|Employee|'
    r'Licensor|Licensee|Borrower|Lender|Guarantor|Principal|Agent|'
    r'Party\s*[A-Z]|First\s*Party|Second\s*Party|Vendor|Purchaser|'
    r'Franchisor|Franchisee|Sublessor|Sublessee|Assignor|Assignee|'
    r'Mortgagor|Mortgagee|Settlor|Trustee|Beneficiary|Claimant|'
    r'Defendant|Plaintiff|Insurer|Insured|Contractor|Client|'
    r'Service\s*Provider|Customer|Supplier|Distributor)\s*:',
    re.IGNORECASE
)

# Legal document field patterns (contract terms)
LEGAL_FIELD_PATTERN = re.compile(
    r'\b(Property|Address|Premises|Term|Duration|Start\s*Date|End\s*Date|'
    r'Effective\s*Date|Commencement\s*Date|Expiration\s*Date|Rent|'
    r'Deposit|Security\s*Deposit|Monthly\s*Rent|Annual\s*Rent|'
    r'Payment\s*Terms?|Due\s*Date|Notice\s*Period|Governing\s*Law|'
    r'Jurisdiction|Witness(?:es)?|Amount|Price|Fee|Rate|'
    r'Termination|Renewal|Option|Covenant|Warranty|Indemnity)\s*:',
    re.IGNORECASE
)

# CSV/Spreadsheet row pattern
CSV_ROW_PATTERN = re.compile(r'^Row\s+\d+:\s*', re.MULTILINE)


def detect_content_type(text: str, payload: dict | None = None) -> ContentType:
    """
    Detect content type from text patterns and payload metadata.

    Priority:
    1. Payload metadata (most reliable)
    2. Text pattern matching (fallback)

    Args:
        text: The chunk text
        payload: Optional payload with metadata

    Returns:
        "legal", "spreadsheet", or "general"
    """
    payload = payload or {}

    # Check payload metadata first
    source_type = payload.get("source_type", "")
    extraction_method = payload.get("extraction_method", "")
    extension = payload.get("extension", "").lower()

    # Spreadsheet detection from metadata
    if source_type == "csv" or extraction_method == "native_csv":
        return "spreadsheet"
    if extension in [".csv", ".xlsx", ".xls"]:
        return "spreadsheet"

    # Spreadsheet detection from text pattern
    if CSV_ROW_PATTERN.search(text):
        return "spreadsheet"

    # Legal document detection - need at least 2 role matches for confidence
    role_matches = len(LEGAL_ROLE_PATTERN.findall(text))
    if role_matches >= 1:
        return "legal"

    # Also check field patterns for legal docs
    field_matches = len(LEGAL_FIELD_PATTERN.findall(text))
    if field_matches >= 2:
        return "legal"

    return "general"


def format_legal_document(text: str) -> str:
    """
    Format legal document text to make field labels stand out.

    Transforms dense inline text:
        "Landlord: John Smith 123 Main St AND Tenant: Jane Doe 456 Oak Ave"

    Into structured format:
        "**Landlord:** John Smith 123 Main St

         **Tenant:** Jane Doe 456 Oak Ave"

    Args:
        text: Raw legal document text

    Returns:
        Formatted text with visible structure
    """
    # Replace "AND" connectors between parties with newlines
    # Match AND followed by a capitalized word and colon (likely a role label)
    text = re.sub(
        r'\s+AND\s+(?=[A-Z][a-z]+\s*:)',
        '\n\n',
        text,
        flags=re.IGNORECASE
    )

    # Add newlines and bold before role labels
    def format_role(match):
        role = match.group(1).strip()
        return f'\n\n**{role}:** '

    text = LEGAL_ROLE_PATTERN.sub(format_role, text)

    # Add newlines before field labels (less prominent than roles)
    def format_field(match):
        field = match.group(1).strip()
        return f'\n**{field}:** '

    text = LEGAL_FIELD_PATTERN.sub(format_field, text)

    # Clean up multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text


def format_spreadsheet_data(text: str) -> str:
    """
    Format spreadsheet data with improved vertical structure.

    Transforms:
        "Row 1: Name = John, Age = 30, City = NYC"

    Into:
        "--- Row 1 ---
         Name: John
         Age: 30
         City: NYC"

    Args:
        text: CSV-formatted text (Row N: field = value, ...)

    Returns:
        Vertically structured text
    """
    lines = text.split('\n')
    formatted_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if it's a Row line
        row_match = re.match(r'^Row\s+(\d+):\s*(.*)$', line, re.IGNORECASE)
        if row_match:
            row_num = row_match.group(1)
            fields = row_match.group(2)

            # Add row header
            formatted_lines.append(f"\n--- Row {row_num} ---")

            # Parse field = value pairs
            # Handle comma-separated pairs, being careful with values containing commas
            pairs = re.split(r',\s*(?=[A-Za-z_][A-Za-z0-9_\s]*\s*=)', fields)

            for pair in pairs:
                pair = pair.strip()
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if value:  # Only include non-empty values
                        formatted_lines.append(f"  {key}: {value}")
        else:
            # Keep non-row lines as-is
            formatted_lines.append(line)

    return '\n'.join(formatted_lines).strip()


def format_context_block(text: str, doc_id: str, payload: dict | None = None) -> str:
    """
    Format a context block based on its detected content type.

    This is the main entry point for context formatting.

    Args:
        text: The chunk text
        doc_id: Document identifier for citation
        payload: Full payload with metadata (from Qdrant)

    Returns:
        Formatted context block: [doc_id]\n{formatted_text}
    """
    content_type = detect_content_type(text, payload)

    if content_type == "legal":
        formatted_text = format_legal_document(text)
    elif content_type == "spreadsheet":
        formatted_text = format_spreadsheet_data(text)
    else:
        # General content - return unchanged
        formatted_text = text

    return f"[{doc_id}]\n{formatted_text}"
