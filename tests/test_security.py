"""
Security tests for Oku RAG.

Tests the security fixes:
- File content validation (magic bytes)
- SQL identifier validation
- Fernet key validation
"""

import pytest


class TestFileValidation:
    """Tests for file content validation (magic bytes)."""

    def test_valid_pdf(self):
        from app.file_validation import validate_file_content

        content = b"%PDF-1.4 some pdf content here"
        is_valid, error = validate_file_content(content, ".pdf")
        assert is_valid is True
        assert error == ""

    def test_fake_pdf_rejected(self):
        """Renamed executable should be rejected."""
        from app.file_validation import validate_file_content

        # MZ header = Windows executable
        content = b"MZ\x90\x00\x03\x00\x00\x00 this is an exe"
        is_valid, error = validate_file_content(content, ".pdf")
        assert is_valid is False
        assert "doesn't match .pdf format" in error

    def test_valid_png(self):
        from app.file_validation import validate_file_content

        content = b"\x89PNG\r\n\x1a\n" + b"x" * 100
        is_valid, error = validate_file_content(content, ".png")
        assert is_valid is True

    def test_fake_png_rejected(self):
        from app.file_validation import validate_file_content

        content = b"not a png file at all"
        is_valid, error = validate_file_content(content, ".png")
        assert is_valid is False
        assert "doesn't match .png format" in error

    def test_valid_jpeg(self):
        from app.file_validation import validate_file_content

        content = b"\xff\xd8\xff\xe0" + b"x" * 100
        is_valid, error = validate_file_content(content, ".jpg")
        assert is_valid is True

    def test_valid_docx(self):
        """DOCX is a ZIP file with PK header."""
        from app.file_validation import validate_file_content

        content = b"PK\x03\x04" + b"x" * 100
        is_valid, error = validate_file_content(content, ".docx")
        assert is_valid is True

    def test_valid_text_file(self):
        from app.file_validation import validate_file_content

        content = b"Hello world, this is plain text."
        is_valid, error = validate_file_content(content, ".txt")
        assert is_valid is True

    def test_text_file_with_binary_rejected(self):
        """Text file with null bytes should be rejected."""
        from app.file_validation import validate_file_content

        content = b"Hello\x00world"
        is_valid, error = validate_file_content(content, ".txt")
        assert is_valid is False
        assert "binary data" in error

    def test_valid_csv(self):
        from app.file_validation import validate_file_content

        content = b"name,age,city\nJohn,30,NYC\nJane,25,LA"
        is_valid, error = validate_file_content(content, ".csv")
        assert is_valid is True

    def test_empty_binary_file_rejected(self):
        from app.file_validation import validate_file_content

        content = b""
        is_valid, error = validate_file_content(content, ".pdf")
        assert is_valid is False
        assert "too small" in error

    def test_valid_webp(self):
        from app.file_validation import validate_file_content

        # RIFF....WEBP header
        content = b"RIFF\x00\x00\x00\x00WEBP" + b"x" * 100
        is_valid, error = validate_file_content(content, ".webp")
        assert is_valid is True

    def test_fake_webp_rejected(self):
        from app.file_validation import validate_file_content

        content = b"RIFF\x00\x00\x00\x00NOTW" + b"x" * 100
        is_valid, error = validate_file_content(content, ".webp")
        assert is_valid is False


class TestSQLIdentifierValidation:
    """Tests for SQL identifier validation (prevents SQL injection)."""

    def test_valid_identifier(self):
        from app.config import validate_sql_identifier

        result = validate_sql_identifier("my_table", "test_field")
        assert result == "my_table"

    def test_valid_identifier_with_numbers(self):
        from app.config import validate_sql_identifier

        result = validate_sql_identifier("table_123", "test_field")
        assert result == "table_123"

    def test_sql_injection_rejected(self):
        from app.config import validate_sql_identifier

        with pytest.raises(ValueError) as exc:
            validate_sql_identifier("table; DROP TABLE users;--", "test_field")
        assert "invalid characters" in str(exc.value)

    def test_sql_injection_with_quotes_rejected(self):
        from app.config import validate_sql_identifier

        with pytest.raises(ValueError) as exc:
            validate_sql_identifier("table'--", "test_field")
        assert "invalid characters" in str(exc.value)

    def test_empty_identifier_rejected(self):
        from app.config import validate_sql_identifier

        with pytest.raises(ValueError) as exc:
            validate_sql_identifier("", "test_field")
        assert "cannot be empty" in str(exc.value)

    def test_too_long_identifier_rejected(self):
        from app.config import validate_sql_identifier

        with pytest.raises(ValueError) as exc:
            validate_sql_identifier("a" * 64, "test_field")
        assert "exceeds maximum length" in str(exc.value)

    def test_identifier_starting_with_number_rejected(self):
        from app.config import validate_sql_identifier

        with pytest.raises(ValueError) as exc:
            validate_sql_identifier("123table", "test_field")
        assert "invalid characters" in str(exc.value)


class TestFernetKeyValidation:
    """Tests for Fernet encryption key validation."""

    def test_valid_fernet_key(self):
        from cryptography.fernet import Fernet
        from app.config import validate_fernet_key

        # Generate a valid key
        valid_key = Fernet.generate_key().decode()
        result = validate_fernet_key(valid_key, "test_key")
        assert result == valid_key

    def test_invalid_fernet_key_rejected(self):
        from app.config import validate_fernet_key

        with pytest.raises(ValueError) as exc:
            validate_fernet_key("not-a-valid-key", "test_key")
        assert "not a valid Fernet key" in str(exc.value)

    def test_empty_fernet_key_rejected(self):
        from app.config import validate_fernet_key

        with pytest.raises(ValueError) as exc:
            validate_fernet_key("", "test_key")
        assert "not a valid Fernet key" in str(exc.value)

    def test_wrong_length_fernet_key_rejected(self):
        from app.config import validate_fernet_key

        # Too short
        with pytest.raises(ValueError) as exc:
            validate_fernet_key("dGVzdA==", "test_key")
        assert "not a valid Fernet key" in str(exc.value)
