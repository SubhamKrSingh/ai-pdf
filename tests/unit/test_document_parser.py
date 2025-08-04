"""
Unit tests for main document parser.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.utils.parsers.document_parser import (
    DocumentParser, 
    DocumentParseError, 
    UnsupportedDocumentTypeError
)


class TestDocumentParser:
    """Test cases for DocumentParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = DocumentParser()
    
    def test_get_supported_content_types(self):
        """Test getting supported content types."""
        supported_types = self.parser.get_supported_content_types()
        
        assert 'pdf' in supported_types
        assert 'docx' in supported_types
        assert 'email' in supported_types
        
        # Check that PDF types are included
        assert 'application/pdf' in supported_types['pdf']
        
        # Check that DOCX types are included
        docx_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        assert docx_type in supported_types['docx']
        
        # Check that email types are included
        assert 'message/rfc822' in supported_types['email']
    
    def test_can_parse_content_type_pdf(self):
        """Test content type checking for PDF."""
        assert self.parser.can_parse_content_type('application/pdf') is True
        assert self.parser.can_parse_content_type('application/x-pdf') is True
    
    def test_can_parse_content_type_docx(self):
        """Test content type checking for DOCX."""
        docx_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        assert self.parser.can_parse_content_type(docx_type) is True
    
    def test_can_parse_content_type_email(self):
        """Test content type checking for email."""
        assert self.parser.can_parse_content_type('message/rfc822') is True
    
    def test_can_parse_content_type_unsupported(self):
        """Test content type checking for unsupported types."""
        assert self.parser.can_parse_content_type('image/jpeg') is False
        assert self.parser.can_parse_content_type('application/json') is False
    
    def test_get_parser_for_content_type_pdf(self):
        """Test getting parser for PDF content type."""
        parser_info = self.parser._get_parser_for_content_type('application/pdf')
        
        assert parser_info is not None
        parser_name, parser_instance = parser_info
        assert parser_name == 'pdf'
        assert parser_instance is self.parser.parsers['pdf']
    
    def test_get_parser_for_content_type_docx(self):
        """Test getting parser for DOCX content type."""
        docx_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        parser_info = self.parser._get_parser_for_content_type(docx_type)
        
        assert parser_info is not None
        parser_name, parser_instance = parser_info
        assert parser_name == 'docx'
        assert parser_instance is self.parser.parsers['docx']
    
    def test_get_parser_for_content_type_email(self):
        """Test getting parser for email content type."""
        parser_info = self.parser._get_parser_for_content_type('message/rfc822')
        
        assert parser_info is not None
        parser_name, parser_instance = parser_info
        assert parser_name == 'email'
        assert parser_instance is self.parser.parsers['email']
    
    def test_get_parser_for_content_type_unsupported(self):
        """Test getting parser for unsupported content type."""
        parser_info = self.parser._get_parser_for_content_type('image/jpeg')
        assert parser_info is None
    
    def test_parse_document_pdf_success(self):
        """Test successful PDF document parsing."""
        content = b"fake pdf content"
        content_type = "application/pdf"
        expected_text = "Extracted PDF text"
        
        # Mock the PDF parser
        with patch.object(self.parser.parsers['pdf'], 'parse', return_value=expected_text):
            result = self.parser.parse_document(content, content_type)
            assert result == expected_text
    
    def test_parse_document_docx_success(self):
        """Test successful DOCX document parsing."""
        content = b"fake docx content"
        content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        expected_text = "Extracted DOCX text"
        
        # Mock the DOCX parser
        with patch.object(self.parser.parsers['docx'], 'parse', return_value=expected_text):
            result = self.parser.parse_document(content, content_type)
            assert result == expected_text
    
    def test_parse_document_email_success(self):
        """Test successful email document parsing."""
        content = b"fake email content"
        content_type = "message/rfc822"
        expected_text = "Extracted email text"
        
        # Mock the email parser
        with patch.object(self.parser.parsers['email'], 'parse', return_value=expected_text):
            result = self.parser.parse_document(content, content_type)
            assert result == expected_text
    
    def test_parse_document_unsupported_type(self):
        """Test parsing document with unsupported content type."""
        content = b"fake content"
        content_type = "image/jpeg"
        
        with pytest.raises(UnsupportedDocumentTypeError, match="Unsupported document type"):
            self.parser.parse_document(content, content_type)
    
    def test_parse_document_parser_error(self):
        """Test parsing document when parser raises an error."""
        content = b"fake pdf content"
        content_type = "application/pdf"
        
        # Mock the PDF parser to raise an error
        with patch.object(self.parser.parsers['pdf'], 'parse', 
                         side_effect=Exception("Parser failed")):
            with pytest.raises(DocumentParseError, match="Unexpected error parsing document"):
                self.parser.parse_document(content, content_type)
    
    def test_parse_document_empty_result(self):
        """Test parsing document when parser returns empty content."""
        content = b"fake pdf content"
        content_type = "application/pdf"
        
        # Mock the PDF parser to return empty content
        with patch.object(self.parser.parsers['pdf'], 'parse', return_value=""):
            with pytest.raises(DocumentParseError, match="No text content extracted"):
                self.parser.parse_document(content, content_type)
    
    def test_parse_document_whitespace_only_result(self):
        """Test parsing document when parser returns only whitespace."""
        content = b"fake pdf content"
        content_type = "application/pdf"
        
        # Mock the PDF parser to return whitespace only
        with patch.object(self.parser.parsers['pdf'], 'parse', return_value="   \n\t  "):
            with pytest.raises(DocumentParseError, match="No text content extracted"):
                self.parser.parse_document(content, content_type)
    
    def test_parse_document_case_insensitive_content_type(self):
        """Test parsing document with case-insensitive content type matching."""
        content = b"fake pdf content"
        content_type = "APPLICATION/PDF"
        expected_text = "Extracted PDF text"
        
        # Mock the PDF parser
        with patch.object(self.parser.parsers['pdf'], 'parse', return_value=expected_text):
            result = self.parser.parse_document(content, content_type)
            assert result == expected_text
    
    def test_parse_document_specific_parser_error_types(self):
        """Test parsing document with specific parser error types."""
        from app.utils.parsers.pdf_parser import PDFParseError
        
        content = b"fake pdf content"
        content_type = "application/pdf"
        
        # Mock the PDF parser to raise a specific error
        with patch.object(self.parser.parsers['pdf'], 'parse', 
                         side_effect=PDFParseError("Specific PDF error")):
            with pytest.raises(DocumentParseError, match="Failed to parse document with pdf parser"):
                self.parser.parse_document(content, content_type)