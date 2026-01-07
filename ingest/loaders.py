"""
Document loaders for ingestion pipeline.
Supports Markdown, Text, PDF, Word (.docx), and Excel (.xlsx) files.
"""
from pathlib import Path
from typing import Iterator

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


def load_text(path: Path) -> str:
    """Load plain text or markdown file."""
    return path.read_text(encoding="utf-8")


def load_pdf(path: Path) -> str:
    """Load PDF and extract text from all pages."""
    if PdfReader is None:
        raise ImportError("pypdf is required for PDF loading. Install with: pip install pypdf")
    
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text)
    return "\n\n".join(texts)


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
    """
    if load_workbook is None:
        raise ImportError("openpyxl is required for Excel loading. Install with: pip install openpyxl")
    
    wb = load_workbook(path, data_only=True)  # data_only=True to get computed values, not formulas
    all_texts = []
    sheet_metadata = []
    
    for sheet_name in wb.sheetnames:
        sheet = wb[sheet_name]
        sheet_texts = []
        
        for row in sheet.iter_rows():
            cell_values = []
            for cell in row:
                if cell.value is not None:
                    cell_values.append(str(cell.value))
            
            if cell_values:  # Skip empty rows
                row_text = "\t".join(cell_values)
                sheet_texts.append(row_text)
        
        if sheet_texts:
            sheet_content = f"[Sheet: {sheet_name}]\n" + "\n".join(sheet_texts)
            all_texts.append(sheet_content)
            sheet_metadata.append({"sheet_name": sheet_name})
    
    wb.close()
    return "\n\n".join(all_texts), sheet_metadata


def load_document(path: Path) -> str | tuple[str, list[dict]]:
    """
    Load a document based on its file extension.
    
    Supported formats:
    - .txt, .md: Plain text
    - .pdf: PDF (requires pypdf)
    - .docx: Word (requires python-docx)
    - .xlsx: Excel (requires openpyxl) - returns tuple with sheet metadata
    
    Returns:
        str for most formats, or tuple[str, list[dict]] for xlsx with sheet metadata
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
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def iter_documents(
    directory: Path, 
    extensions: tuple[str, ...] = (".md", ".txt", ".pdf", ".docx", ".xlsx")
) -> Iterator[tuple[Path, str | tuple[str, list[dict]]]]:
    """
    Iterate over all documents in a directory.
    
    Yields:
        Tuples of (path, content) for each document.
        For xlsx files, content is a tuple of (text, sheet_metadata).
    """
    for ext in extensions:
        for path in directory.glob(f"*{ext}"):
            if path.is_file():
                try:
                    content = load_document(path)
                    yield path, content
                except ValueError as e:
                    print(f"Skipping unsupported file {path}: {e}")
                except Exception as e:
                    print(f"Error loading {path}: {e}")

