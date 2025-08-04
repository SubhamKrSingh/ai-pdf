"""
Unit tests for email parser.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.utils.parsers.email_parser import EmailParser, EmailParseError


class TestEmailParser:
    """Test cases for EmailParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = EmailParser()
    
    def test_can_parse_valid_content_types(self):
        """Test that parser recognizes valid email content types."""
        valid_types = [
            'message/rfc822',
            'text/rfc822-headers',
            'application/mbox',
            'text/plain',
            'message/partial'
        ]
        
        for content_type in valid_types:
            assert self.parser.can_parse(content_type) is True
    
    def test_can_parse_invalid_content_types(self):
        """Test that parser rejects invalid content types."""
        invalid_types = [
            'application/pdf',
            'application/msword',
            'application/json',
            'image/jpeg'
        ]
        
        for content_type in invalid_types:
            assert self.parser.can_parse(content_type) is False
    
    def test_can_parse_case_insensitive(self):
        """Test that content type matching is case insensitive."""
        assert self.parser.can_parse('MESSAGE/RFC822') is True
        assert self.parser.can_parse('Message/Rfc822') is True
    
    @patch('app.utils.parsers.email_parser.email.message_from_string')
    def test_parse_simple_email(self, mock_message_from_string):
        """Test parsing simple single-part email."""
        # Mock email message
        mock_msg = MagicMock()
        mock_msg.get.side_effect = lambda header, default=None: {
            'From': 'sender@example.com',
            'Subject': 'Test Subject'
        }.get(header, default)
        
        mock_msg.is_multipart.return_value = False
        mock_msg.get_content_type.return_value = 'text/plain'
        
        mock_message_from_string.return_value = mock_msg
        
        # Mock the _extract_text_from_payload method
        with patch.object(self.parser, '_extract_text_from_payload', return_value='This is the email body'):
            content = b"fake email content"
            result = self.parser.parse(content)
            
            assert "From: sender@example.com" in result
            assert "Subject: Test Subject" in result
            assert "This is the email body" in result
    
    @patch('app.utils.parsers.email_parser.email.message_from_string')
    def test_parse_multipart_email(self, mock_message_from_string):
        """Test parsing multipart email."""
        # Mock email parts
        mock_text_part = MagicMock()
        mock_text_part.get_content_type.return_value = 'text/plain'
        mock_text_part.get.side_effect = lambda header, default=None: None
        
        # Mock main message
        mock_msg = MagicMock()
        mock_msg.get.side_effect = lambda header, default=None: {
            'From': 'sender@example.com',
            'Subject': 'Multipart Test'
        }.get(header, default)
        
        mock_msg.is_multipart.return_value = True
        mock_msg.walk.return_value = [mock_msg, mock_text_part]
        
        mock_message_from_string.return_value = mock_msg
        
        # Mock the _extract_text_from_payload method
        with patch.object(self.parser, '_extract_text_from_payload', return_value='Plain text content'):
            content = b"fake multipart email"
            result = self.parser.parse(content)
            
            assert "From: sender@example.com" in result
            assert "Subject: Multipart Test" in result
            assert "Plain text content" in result
    
    @patch('app.utils.parsers.email_parser.email.message_from_string')
    def test_parse_email_with_attachment(self, mock_message_from_string):
        """Test parsing email with attachment (should skip attachment)."""
        # Mock text part
        mock_text_part = MagicMock()
        mock_text_part.get_content_type.return_value = 'text/plain'
        mock_text_part.get.side_effect = lambda header, default=None: None
        
        # Mock main message
        mock_msg = MagicMock()
        mock_msg.get.side_effect = lambda header, default=None: {
            'Subject': 'Email with attachment'
        }.get(header, default)
        
        mock_msg.is_multipart.return_value = True
        mock_msg.walk.return_value = [mock_msg, mock_text_part]
        
        mock_message_from_string.return_value = mock_msg
        
        # Mock the _extract_text_from_payload method
        with patch.object(self.parser, '_extract_text_from_payload', return_value='Email body'):
            content = b"fake email with attachment"
            result = self.parser.parse(content)
            
            assert "Subject: Email with attachment" in result
            assert "Email body" in result
    
    @patch('app.utils.parsers.email_parser.email.message_from_string')
    def test_parse_email_unicode_decode_error(self, mock_message_from_string):
        """Test parsing email with unicode decode error."""
        mock_msg = MagicMock()
        mock_msg.get.side_effect = lambda header, default=None: {
            'Subject': 'Unicode Test'
        }.get(header, default)
        
        mock_msg.is_multipart.return_value = False
        mock_msg.get_content_type.return_value = 'text/plain'
        
        mock_message_from_string.return_value = mock_msg
        
        # Mock the _extract_text_from_payload method
        with patch.object(self.parser, '_extract_text_from_payload', return_value='Decoded content'):
            content = b"fake email with unicode issues"
            result = self.parser.parse(content)
            
            # Should handle decode error gracefully
            assert "Subject: Unicode Test" in result
            assert "Decoded content" in result
    
    @patch('app.utils.parsers.email_parser.email.message_from_string')
    def test_parse_empty_email(self, mock_message_from_string):
        """Test parsing email with no extractable content."""
        mock_msg = MagicMock()
        mock_msg.get.return_value = None  # No headers
        mock_msg.is_multipart.return_value = False
        mock_msg.get_content_type.return_value = 'text/plain'
        mock_msg.get_payload.return_value = b''
        mock_msg.get_content_charset.return_value = 'utf-8'
        
        mock_message_from_string.return_value = mock_msg
        
        content = b"fake empty email"
        
        with pytest.raises(EmailParseError, match="No text content could be extracted"):
            self.parser.parse(content)
    
    @patch('app.utils.parsers.email_parser.email.message_from_string')
    def test_parse_email_exception(self, mock_message_from_string):
        """Test parsing when email parsing raises an exception."""
        mock_message_from_string.side_effect = Exception("Invalid email format")
        
        content = b"invalid email content"
        
        with pytest.raises(EmailParseError, match="Failed to parse email"):
            self.parser.parse(content)
    
    def test_extract_text_from_payload_success(self):
        """Test successful text extraction from payload."""
        mock_part = MagicMock()
        mock_part.get_payload.return_value = b'Test content'
        mock_part.get_content_charset.return_value = 'utf-8'
        
        result = self.parser._extract_text_from_payload(mock_part)
        assert result == 'Test content'
    
    def test_extract_text_from_payload_no_charset(self):
        """Test text extraction when no charset is specified."""
        mock_part = MagicMock()
        mock_part.get_payload.return_value = b'Test content'
        mock_part.get_content_charset.return_value = None
        
        result = self.parser._extract_text_from_payload(mock_part)
        assert result == 'Test content'
    
    def test_extract_text_from_payload_decode_error(self):
        """Test text extraction with decode error."""
        mock_part = MagicMock()
        mock_part.get_payload.return_value = b'\xff\xfe invalid'
        mock_part.get_content_charset.return_value = 'utf-8'
        
        result = self.parser._extract_text_from_payload(mock_part)
        # Should handle decode error gracefully
        assert isinstance(result, str)
    
    def test_extract_text_from_payload_exception(self):
        """Test text extraction when payload extraction fails."""
        mock_part = MagicMock()
        mock_part.get_payload.side_effect = Exception("Payload error")
        
        result = self.parser._extract_text_from_payload(mock_part)
        assert result == ""