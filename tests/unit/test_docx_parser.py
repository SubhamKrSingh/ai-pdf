"""
Unit tests for DOCX parser.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.utils.parsers.docx_parser import DOCXParser, DOCXParseError


class TestDOCXParser:
    """Test cases for DOCXParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = DOCXParser()
    
    def test_can_parse_valid_content_types(self):
        """Test that parser recognizes valid DOCX content types."""
        valid_types = [
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.ms-word.document.macroenabled.12',
            'application/msword',
            'application/x-msword'
        ]
        
        for content_type in valid_types:
            assert self.parser.can_parse(content_type) is True
    
    def test_can_parse_invalid_content_types(self):
        """Test that parser rejects invalid content types."""
        invalid_types = [
            'application/pdf',
            'text/plain',
            'application/json',
            'image/jpeg'
        ]
        
        for content_type in invalid_types:
            assert self.parser.can_parse(content_type) is False
    
    def test_can_parse_case_insensitive(self):
        """Test that content type matching is case insensitive."""
        docx_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        assert self.parser.can_parse(docx_type.upper()) is True
        assert self.parser.can_parse(docx_type.title()) is True
    
    @patch('app.utils.parsers.docx_parser.Document')
    def test_parse_success(self, mock_document):
        """Test successful DOCX parsing."""
        # Mock paragraphs
        mock_para1 = MagicMock()
        mock_para1.text = "First paragraph"
        
        mock_para2 = MagicMock()
        mock_para2.text = "Second paragraph"
        
        mock_para3 = MagicMock()
        mock_para3.text = ""  # Empty paragraph
        
        # Mock document
        mock_doc = MagicMock()
        mock_doc.paragraphs = [mock_para1, mock_para2, mock_para3]
        mock_doc.tables = []  # No tables
        mock_document.return_value = mock_doc
        
        content = b"fake docx content"
        result = self.parser.parse(content)
        
        assert result == "First paragraph\nSecond paragraph"
        mock_document.assert_called_once()
    
    @patch('app.utils.parsers.docx_parser.Document')
    def test_parse_with_tables(self, mock_document):
        """Test DOCX parsing with tables."""
        # Mock paragraphs
        mock_para = MagicMock()
        mock_para.text = "Document text"
        
        # Mock table cells
        mock_cell1 = MagicMock()
        mock_cell1.text = "Cell 1"
        
        mock_cell2 = MagicMock()
        mock_cell2.text = "Cell 2"
        
        mock_cell3 = MagicMock()
        mock_cell3.text = ""  # Empty cell
        
        # Mock table row
        mock_row = MagicMock()
        mock_row.cells = [mock_cell1, mock_cell2, mock_cell3]
        
        # Mock table
        mock_table = MagicMock()
        mock_table.rows = [mock_row]
        
        # Mock document
        mock_doc = MagicMock()
        mock_doc.paragraphs = [mock_para]
        mock_doc.tables = [mock_table]
        mock_document.return_value = mock_doc
        
        content = b"fake docx content"
        result = self.parser.parse(content)
        
        assert result == "Document text\nCell 1 | Cell 2"
    
    @patch('app.utils.parsers.docx_parser.Document')
    def test_parse_empty_document(self, mock_document):
        """Test parsing DOCX with no extractable text."""
        mock_para = MagicMock()
        mock_para.text = ""
        
        mock_doc = MagicMock()
        mock_doc.paragraphs = [mock_para]
        mock_doc.tables = []
        mock_document.return_value = mock_doc
        
        content = b"fake docx content"
        
        with pytest.raises(DOCXParseError, match="No text content could be extracted"):
            self.parser.parse(content)
    
    @patch('app.utils.parsers.docx_parser.Document')
    def test_parse_only_whitespace(self, mock_document):
        """Test parsing DOCX with only whitespace content."""
        mock_para1 = MagicMock()
        mock_para1.text = "   "
        
        mock_para2 = MagicMock()
        mock_para2.text = "\n\t"
        
        mock_doc = MagicMock()
        mock_doc.paragraphs = [mock_para1, mock_para2]
        mock_doc.tables = []
        mock_document.return_value = mock_doc
        
        content = b"fake docx content"
        
        with pytest.raises(DOCXParseError, match="No text content could be extracted"):
            self.parser.parse(content)
    
    @patch('app.utils.parsers.docx_parser.Document')
    def test_parse_document_exception(self, mock_document):
        """Test parsing when Document raises an exception."""
        mock_document.side_effect = Exception("Invalid DOCX format")
        
        content = b"invalid docx content"
        
        with pytest.raises(DOCXParseError, match="Failed to parse DOCX"):
            self.parser.parse(content)
    
    @patch('app.utils.parsers.docx_parser.Document')
    def test_parse_complex_table(self, mock_document):
        """Test parsing DOCX with complex table structure."""
        # Mock paragraphs
        mock_para = MagicMock()
        mock_para.text = "Header text"
        
        # Mock table with multiple rows
        mock_cell1 = MagicMock()
        mock_cell1.text = "Row1 Col1"
        
        mock_cell2 = MagicMock()
        mock_cell2.text = "Row1 Col2"
        
        mock_cell3 = MagicMock()
        mock_cell3.text = "Row2 Col1"
        
        mock_cell4 = MagicMock()
        mock_cell4.text = "Row2 Col2"
        
        mock_row1 = MagicMock()
        mock_row1.cells = [mock_cell1, mock_cell2]
        
        mock_row2 = MagicMock()
        mock_row2.cells = [mock_cell3, mock_cell4]
        
        mock_table = MagicMock()
        mock_table.rows = [mock_row1, mock_row2]
        
        # Mock document
        mock_doc = MagicMock()
        mock_doc.paragraphs = [mock_para]
        mock_doc.tables = [mock_table]
        mock_document.return_value = mock_doc
        
        content = b"fake docx content"
        result = self.parser.parse(content)
        
        expected = "Header text\nRow1 Col1 | Row1 Col2\nRow2 Col1 | Row2 Col2"
        assert result == expected
    
    @patch('app.utils.parsers.docx_parser.Document')
    def test_parse_filters_empty_content(self, mock_document):
        """Test that parser filters out empty paragraphs and cells."""
        # Mock paragraphs with mixed content
        mock_para1 = MagicMock()
        mock_para1.text = "Real content"
        
        mock_para2 = MagicMock()
        mock_para2.text = ""  # Empty
        
        mock_para3 = MagicMock()
        mock_para3.text = "More content"
        
        # Mock table with empty cells
        mock_cell1 = MagicMock()
        mock_cell1.text = "Data"
        
        mock_cell2 = MagicMock()
        mock_cell2.text = ""  # Empty cell
        
        mock_row = MagicMock()
        mock_row.cells = [mock_cell1, mock_cell2]
        
        mock_table = MagicMock()
        mock_table.rows = [mock_row]
        
        # Mock document
        mock_doc = MagicMock()
        mock_doc.paragraphs = [mock_para1, mock_para2, mock_para3]
        mock_doc.tables = [mock_table]
        mock_document.return_value = mock_doc
        
        content = b"fake docx content"
        result = self.parser.parse(content)
        
        # Should filter out empty paragraph and empty cell
        assert result == "Real content\nMore content\nData"