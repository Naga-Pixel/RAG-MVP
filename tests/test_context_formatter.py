"""
Tests for context_formatter module.

Tests structured formatting for legal documents and spreadsheets.
"""

import pytest
from app.context_formatter import (
    detect_content_type,
    format_legal_document,
    format_spreadsheet_data,
    format_context_block,
)


class TestDetectContentType:
    """Tests for content type detection."""

    def test_detects_spreadsheet_from_metadata(self):
        """Should detect spreadsheet from source_type metadata."""
        payload = {"source_type": "csv"}
        assert detect_content_type("any text", payload) == "spreadsheet"

    def test_detects_spreadsheet_from_extension(self):
        """Should detect spreadsheet from file extension."""
        payload = {"extension": ".xlsx"}
        assert detect_content_type("any text", payload) == "spreadsheet"

    def test_detects_spreadsheet_from_csv_extension(self):
        """Should detect spreadsheet from .csv extension."""
        payload = {"extension": ".csv"}
        assert detect_content_type("any text", payload) == "spreadsheet"

    def test_detects_spreadsheet_from_row_pattern(self):
        """Should detect spreadsheet from Row N: pattern in text."""
        text = "Row 1: Name = John, Age = 30\nRow 2: Name = Jane, Age = 25"
        assert detect_content_type(text) == "spreadsheet"

    def test_detects_legal_from_landlord_tenant(self):
        """Should detect legal document from Landlord/Tenant labels."""
        text = "PARTIES Landlord: Charlotte Whitcombe AND Tenant: Daniel Harrington"
        assert detect_content_type(text) == "legal"

    def test_detects_legal_from_buyer_seller(self):
        """Should detect legal document from Buyer/Seller labels."""
        text = "The Buyer: ABC Corp agrees to purchase from Seller: XYZ Ltd"
        assert detect_content_type(text) == "legal"

    def test_detects_legal_from_multiple_fields(self):
        """Should detect legal document from multiple field labels."""
        text = "Property: 123 Main St, Rent: 1500, Term: 12 months"
        assert detect_content_type(text) == "legal"

    def test_returns_general_for_plain_text(self):
        """Should return general for text without legal/spreadsheet patterns."""
        text = "This is a general document about various topics."
        assert detect_content_type(text) == "general"

    def test_handles_none_payload(self):
        """Should handle None payload gracefully."""
        text = "Landlord: John Smith"
        assert detect_content_type(text, None) == "legal"


class TestFormatLegalDocument:
    """Tests for legal document formatting."""

    def test_formats_landlord_tenant(self):
        """Should add newlines and bold around Landlord/Tenant."""
        text = "PARTIES Landlord: Charlotte Whitcombe AND Tenant: Daniel Harrington"
        result = format_legal_document(text)

        assert "**Landlord:**" in result
        assert "**Tenant:**" in result
        assert "Charlotte Whitcombe" in result
        assert "Daniel Harrington" in result

    def test_splits_on_and_connector(self):
        """Should split parties connected by AND."""
        text = "Landlord: Alice Smith 123 Main St AND Tenant: Bob Jones 456 Oak Ave"
        result = format_legal_document(text)

        # Should have separation between the two parties
        landlord_pos = result.find("**Landlord:**")
        tenant_pos = result.find("**Tenant:**")
        assert landlord_pos < tenant_pos
        assert "\n" in result[landlord_pos:tenant_pos]

    def test_formats_buyer_seller(self):
        """Should format Buyer and Seller labels."""
        text = "Buyer: ABC Corp Seller: XYZ Ltd"
        result = format_legal_document(text)

        assert "**Buyer:**" in result
        assert "**Seller:**" in result

    def test_formats_contract_fields(self):
        """Should format contract field labels."""
        text = "Property: 123 Main Street Rent: 2000 per month Term: 12 months"
        result = format_legal_document(text)

        assert "**Property:**" in result
        assert "**Rent:**" in result
        assert "**Term:**" in result

    def test_preserves_content_values(self):
        """Should preserve the actual values after labels."""
        text = "Tenant: John Smith 42 Baker Street London SW1A 1AA"
        result = format_legal_document(text)

        assert "John Smith" in result
        assert "42 Baker Street" in result
        assert "London SW1A 1AA" in result


class TestFormatSpreadsheetData:
    """Tests for spreadsheet data formatting."""

    def test_formats_single_row(self):
        """Should format a single row with vertical layout."""
        text = "Row 1: Name = John, Age = 30, City = NYC"
        result = format_spreadsheet_data(text)

        assert "--- Row 1 ---" in result
        assert "Name: John" in result
        assert "Age: 30" in result
        assert "City: NYC" in result

    def test_formats_multiple_rows(self):
        """Should format multiple rows."""
        text = "Row 1: Name = John, Age = 30\nRow 2: Name = Jane, Age = 25"
        result = format_spreadsheet_data(text)

        assert "--- Row 1 ---" in result
        assert "--- Row 2 ---" in result
        assert "Name: John" in result
        assert "Name: Jane" in result

    def test_handles_empty_values(self):
        """Should skip fields with empty values."""
        text = "Row 1: Name = John, Age = , City = NYC"
        result = format_spreadsheet_data(text)

        assert "Name: John" in result
        assert "City: NYC" in result
        # Should not have empty Age field

    def test_preserves_non_row_lines(self):
        """Should preserve lines that aren't Row N: format."""
        text = "Header information\nRow 1: Name = John\nFooter information"
        result = format_spreadsheet_data(text)

        assert "Header information" in result
        assert "Footer information" in result
        assert "--- Row 1 ---" in result

    def test_handles_underscore_field_names(self):
        """Should handle field names with underscores."""
        text = "Row 1: first_name = John, last_name = Smith, employee_id = 123"
        result = format_spreadsheet_data(text)

        assert "first_name: John" in result
        assert "last_name: Smith" in result
        assert "employee_id: 123" in result


class TestFormatContextBlock:
    """Tests for the main context block formatter."""

    def test_formats_legal_document_with_doc_id(self):
        """Should include doc_id and format legal content."""
        text = "Landlord: John Smith Tenant: Jane Doe"
        result = format_context_block(text, "lease_001", {})

        assert result.startswith("[lease_001]")
        assert "**Landlord:**" in result
        assert "**Tenant:**" in result

    def test_formats_spreadsheet_with_doc_id(self):
        """Should include doc_id and format spreadsheet content."""
        text = "Row 1: Name = John, Age = 30"
        result = format_context_block(text, "data_001", {"source_type": "csv"})

        assert result.startswith("[data_001]")
        assert "--- Row 1 ---" in result

    def test_returns_general_unchanged(self):
        """Should return general text unchanged except for doc_id wrapper."""
        text = "This is general text without special formatting."
        result = format_context_block(text, "doc_001", {})

        assert result == "[doc_001]\nThis is general text without special formatting."

    def test_uses_payload_for_detection(self):
        """Should use payload metadata for content type detection."""
        text = "Some text that could be anything"
        result = format_context_block(text, "sheet_001", {"extension": ".xlsx"})

        # Should be treated as spreadsheet due to extension
        assert result.startswith("[sheet_001]")


class TestRealWorldExamples:
    """Tests using realistic document examples."""

    def test_tenancy_agreement_parties(self):
        """Should correctly format a real tenancy agreement PARTIES section."""
        text = (
            "ASSURED SHORTHOLD TENANCY AGREEMENT PARTIES "
            "Landlord: Charlotte Eleanor Whitcombe 42 Addison Crescent London, W14 8JP "
            "AND Tenant: Daniel Thomas Harrington 18 Cranley Gardens London, SW7 3DE"
        )
        result = format_legal_document(text)

        # The two parties should be clearly separated
        assert "**Landlord:**" in result
        assert "**Tenant:**" in result
        assert "Charlotte Eleanor Whitcombe" in result
        assert "Daniel Thomas Harrington" in result

        # They should be on different "sections" (separated by newlines)
        landlord_section = result[result.find("**Landlord:**"):result.find("**Tenant:**")]
        assert "Charlotte Eleanor Whitcombe" in landlord_section
        assert "Daniel Thomas Harrington" not in landlord_section

    def test_invoice_spreadsheet(self):
        """Should correctly format invoice spreadsheet data."""
        text = (
            "Row 1: invoice_number = INV-001, amount = 1500.00, vendor = ABC Corp, status = paid\n"
            "Row 2: invoice_number = INV-002, amount = 2300.50, vendor = XYZ Ltd, status = pending"
        )
        result = format_spreadsheet_data(text)

        assert "--- Row 1 ---" in result
        assert "--- Row 2 ---" in result
        assert "invoice_number: INV-001" in result
        assert "amount: 1500.00" in result
        assert "vendor: ABC Corp" in result
