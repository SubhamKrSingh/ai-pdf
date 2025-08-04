"""
Email parser for extracting content from email messages.
"""

import email
import logging
from typing import Optional
from email.message import EmailMessage

logger = logging.getLogger(__name__)


class EmailParseError(Exception):
    """Exception raised when email parsing fails."""
    pass


class EmailParser:
    """Parser for email messages."""
    
    def __init__(self):
        """Initialize the email parser."""
        pass
    
    def _extract_text_from_payload(self, part) -> str:
        """
        Extract text content from email part payload.
        
        Args:
            part: Email message part
            
        Returns:
            Extracted text content
        """
        try:
            payload = part.get_payload(decode=True)
            if payload:
                # Try to decode with the specified charset
                charset = part.get_content_charset() or 'utf-8'
                try:
                    return payload.decode(charset)
                except (UnicodeDecodeError, LookupError):
                    # Fallback to utf-8 with error handling
                    return payload.decode('utf-8', errors='ignore')
            return ""
        except Exception as e:
            logger.warning(f"Failed to extract payload from email part: {str(e)}")
            return ""
    
    def parse(self, content: bytes) -> str:
        """
        Parse email content and extract text.
        
        Args:
            content: Email content as bytes
            
        Returns:
            Extracted text content including headers and body
            
        Raises:
            EmailParseError: If parsing fails
        """
        try:
            # Parse the email message
            try:
                # Try to decode as string first
                email_str = content.decode('utf-8')
            except UnicodeDecodeError:
                # Fallback with error handling
                email_str = content.decode('utf-8', errors='ignore')
            
            msg = email.message_from_string(email_str)
            
            # Extract headers
            text_parts = []
            
            # Add important headers
            headers_to_extract = ['From', 'To', 'Cc', 'Bcc', 'Subject', 'Date']
            for header in headers_to_extract:
                value = msg.get(header)
                if value:
                    text_parts.append(f"{header}: {value}")
            
            text_parts.append("")  # Empty line after headers
            
            # Extract body content
            if msg.is_multipart():
                # Handle multipart messages
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition", ""))
                    
                    # Skip attachments
                    if "attachment" in content_disposition:
                        continue
                    
                    # Extract text content
                    if content_type == "text/plain":
                        text_content = self._extract_text_from_payload(part)
                        if text_content.strip():
                            text_parts.append(text_content)
                    elif content_type == "text/html":
                        # For HTML content, we'll extract it but note it's HTML
                        html_content = self._extract_text_from_payload(part)
                        if html_content.strip():
                            text_parts.append(f"[HTML Content]\n{html_content}")
            else:
                # Handle single-part messages
                content_type = msg.get_content_type()
                if content_type in ["text/plain", "text/html"]:
                    body_content = self._extract_text_from_payload(msg)
                    if body_content.strip():
                        if content_type == "text/html":
                            text_parts.append(f"[HTML Content]\n{body_content}")
                        else:
                            text_parts.append(body_content)
            
            # Check if we have any body content (after headers and empty line)
            body_parts = text_parts[len([h for h in ['From', 'To', 'Cc', 'Bcc', 'Subject', 'Date'] if msg.get(h)]) + 1:]
            if not any(part.strip() for part in body_parts):
                raise EmailParseError("No text content could be extracted from email")
            
            # Join all parts
            full_text = "\n".join(text_parts)
            
            logger.info(f"Successfully parsed email: {len(full_text)} characters extracted")
            
            return full_text
            
        except EmailParseError:
            raise
        except Exception as e:
            raise EmailParseError(f"Failed to parse email: {str(e)}")
    
    def can_parse(self, content_type: str) -> bool:
        """
        Check if this parser can handle the given content type.
        
        Args:
            content_type: MIME type of the content
            
        Returns:
            True if parser can handle this content type
        """
        return content_type.lower() in [
            'message/rfc822',
            'text/rfc822-headers',
            'application/mbox',
            'text/plain',  # Sometimes emails are served as plain text
            'message/partial'
        ]