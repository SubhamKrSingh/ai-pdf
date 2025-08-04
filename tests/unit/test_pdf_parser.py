"""
Unit tests for PDF parser.
"""

import pytest
import io
from unittest.mock import patch, MagicMock

from app.utils.parsers.pdf_parser import PDFParser, PDFParseError


class TestPDFParser:
    """Test cases for PDFParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = PDFParser()
    
    def test_can_parse_valid_content_types(self):
        """Test that parser recognizes valid PDF content types."""
        valid_types = [
            'application/pdf',
            'application/x-pdf',
            'application/acrobat',
            'applications/vnd.pdf',
            'text/pdf',
            'text/x-pdf'
        ]
        
        for content_type in valid_types:
            assert self.parser.can_parse(content_type) is True
    
    def test_can_parse_invalid_content_types(self):
        """Test that parser rejects invalid content types."""
        invalid_types = [
            'application/msword',
            'text/plain',
            'application/json',
            'image/jpeg'
        ]
        
        for content_type in invalid_types:
            assert self.parser.can_parse(content_type) is False
    
    def test_can_parse_case_insensitive(self):
        """Test that content type matching is case insensitive."""
        assert self.parser.can_parse('APPLICATION/PDF') is True
        assert self.parser.can_parse('Application/Pdf') is True
    
    @patch('app.utils.parsers.pdf_parser.PdfReader')
    def test_parse_success(self, mock_pdf_reader):
        """Test successful PDF parsing."""
        # Mock PDF reader and pages
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 content"
        
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 content"
        
        mock_reader = MagicMock()
        mock_reader.is_encrypted = False
        mock_reader.pages = [mock_page1, mock_page2]
        mock_pdf_reader.return_value = mock_reader
        
        content = b"fake pdf content"
        result = self.parser.parse(content)
        
        assert result == "Page 1 content\n\nPage 2 content"
        mock_pdf_reader.assert_called_once()
    
    @patch('app.utils.parsers.pdf_parser.PdfReader')
    def test_parse_encrypted_pdf_success(self, mock_pdf_reader):
        """Test parsing encrypted PDF with empty password."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Decrypted content"
        
        mock_reader = MagicMock()
        mock_reader.is_encrypted = True
        mock_reader.decrypt.return_value = True
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader
        
        content = b"fake encrypted pdf content"
        result = self.parser.parse(content)
        
        assert result == "Decrypted content"
        mock_reader.decrypt.assert_called_once_with("")
    
    @patch('app.utils.parsers.pdf_parser.PdfReader')
    def test_parse_encrypted_pdf_failure(self, mock_pdf_reader):
        """Test parsing encrypted PDF that cannot be decrypted."""
        mock_reader = MagicMock()
        mock_reader.is_encrypted = True
        mock_reader.decrypt.return_value = False
        mock_pdf_reader.return_value = mock_reader
        
        content = b"fake encrypted pdf content"
        
        with pytest.raises(PDFParseError, match="PDF is password protected"):
            self.parser.parse(content)
    
    @patch('app.utils.parsers.pdf_parser.PdfReader')
    def test_parse_empty_pdf(self, mock_pdf_reader):
        """Test parsing PDF with no extractable text."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        
        mock_reader = MagicMock()
        mock_reader.is_encrypted = False
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader
        
        content = b"fake pdf content"
        
        with pytest.raises(PDFParseError, match="No text content could be extracted"):
            self.parser.parse(content)
    
    @patch('app.utils.parsers.pdf_parser.PdfReader')
    def test_parse_with_page_extraction_errors(self, mock_pdf_reader):
        """Test parsing PDF where some pages fail to extract."""
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 content"
        
        mock_page2 = MagicMock()
        mock_page2.extract_text.side_effect = Exception("Page extraction failed")
        
        mock_page3 = MagicMock()
        mock_page3.extract_text.return_value = "Page 3 content"
        
        mock_reader = MagicMock()
        mock_reader.is_encrypted = False
        mock_reader.pages = [mock_page1, mock_page2, mock_page3]
        mock_pdf_reader.return_value = mock_reader
        
        content = b"fake pdf content"
        result = self.parser.parse(content)
        
        # Should skip the failed page but include others
        assert result == "Page 1 content\n\nPage 3 content"
    
    @patch('app.utils.parsers.pdf_parser.PdfReader')
    def test_parse_pdf_reader_exception(self, mock_pdf_reader):
        """Test parsing when PdfReader raises an exception."""
        mock_pdf_reader.side_effect = Exception("Invalid PDF format")
        
        content = b"invalid pdf content"
        
        with pytest.raises(PDFParseError, match="Failed to parse PDF"):
            self.parser.parse(content)
    
    @patch('app.utils.parsers.pdf_parser.PdfReader')
    def test_parse_filters_empty_pages(self, mock_pdf_reader):
        """Test that parser filters out pages with only whitespace."""
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Real content"
        
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "   \n\t  "  # Only whitespace
        
        mock_page3 = MagicMock()
        mock_page3.extract_text.return_value = "More content"
        
        mock_reader = MagicMock()
        mock_reader.is_encrypted = False
        mock_reader.pages = [mock_page1, mock_page2, mock_page3]
        mock_pdf_reader.return_value = mock_reader
        
        content = b"fake pdf content"
        result = self.parser.parse(content)
        
        # Should skip the whitespace-only page
        assert result == "Real content\n\nMore content"