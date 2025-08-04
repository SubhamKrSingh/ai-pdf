"""
PDF document parser using pypdf library.
"""

import io
import logging
from typing import Optional
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PDFParseError(Exception):
    """Exception raised when PDF parsing fails."""
    pass


class PDFParser:
    """Parser for PDF documents using pypdf library."""
    
    def __init__(self):
        """Initialize the PDF parser."""
        pass
    
    def parse(self, content: bytes) -> str:
        """
        Parse PDF content and extract text.
        
        Args:
            content: PDF file content as bytes
            
        Returns:
            Extracted text content
            
        Raises:
            PDFParseError: If parsing fails
        """
        try:
            # Create a BytesIO object from the content
            pdf_stream = io.BytesIO(content)
            
            # Create PDF reader
            reader = PdfReader(pdf_stream)
            
            # Check if PDF is encrypted
            if reader.is_encrypted:
                # Try to decrypt with empty password
                if not reader.decrypt(""):
                    raise PDFParseError("PDF is password protected and cannot be decrypted")
            
            # Extract text from all pages
            text_content = []
            total_pages = len(reader.pages)
            
            logger.info(f"Parsing PDF with {total_pages} pages")
            
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        text_content.append(page_text)
                        logger.debug(f"Extracted text from page {page_num}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {str(e)}")
                    continue
            
            if not text_content:
                raise PDFParseError("No text content could be extracted from PDF")
            
            # Join all pages with double newlines
            full_text = "\n\n".join(text_content)
            
            logger.info(f"Successfully parsed PDF: {len(full_text)} characters extracted")
            
            return full_text
            
        except PDFParseError:
            raise
        except Exception as e:
            raise PDFParseError(f"Failed to parse PDF: {str(e)}")
    
    def can_parse(self, content_type: str) -> bool:
        """
        Check if this parser can handle the given content type.
        
        Args:
            content_type: MIME type of the content
            
        Returns:
            True if parser can handle this content type
        """
        return content_type.lower() in [
            'application/pdf',
            'application/x-pdf',
            'application/acrobat',
            'applications/vnd.pdf',
            'text/pdf',
            'text/x-pdf'
        ]