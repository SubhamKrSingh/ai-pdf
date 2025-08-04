"""
Main document parser that handles content type detection and routing to appropriate parsers.
"""

import logging
from typing import Dict, Type, Optional

from .pdf_parser import PDFParser, PDFParseError
from .docx_parser import DOCXParser, DOCXParseError
from .email_parser import EmailParser, EmailParseError

logger = logging.getLogger(__name__)


class UnsupportedDocumentTypeError(Exception):
    """Exception raised when document type is not supported."""
    pass


class DocumentParseError(Exception):
    """General exception for document parsing errors."""
    pass


class DocumentParser:
    """
    Main document parser that routes content to appropriate specialized parsers
    based on content type detection.
    """
    
    def __init__(self):
        """Initialize the document parser with all available parsers."""
        self.parsers = {
            'pdf': PDFParser(),
            'docx': DOCXParser(),
            'email': EmailParser()
        }
    
    def _get_parser_for_content_type(self, content_type: str) -> Optional[tuple]:
        """
        Get the appropriate parser for the given content type.
        
        Args:
            content_type: MIME type of the content
            
        Returns:
            Tuple of (parser_name, parser_instance) or None if no parser found
        """
        for parser_name, parser in self.parsers.items():
            if parser.can_parse(content_type):
                return parser_name, parser
        return None
    
    def get_supported_content_types(self) -> Dict[str, list]:
        """
        Get all supported content types organized by parser.
        
        Returns:
            Dictionary mapping parser names to their supported content types
        """
        supported_types = {}
        
        # Get supported types for each parser
        test_types = [
            # PDF types
            'application/pdf',
            'application/x-pdf',
            'application/acrobat',
            'applications/vnd.pdf',
            'text/pdf',
            'text/x-pdf',
            
            # DOCX types
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.ms-word.document.macroEnabled.12',
            'application/msword',
            'application/x-msword',
            
            # Email types
            'message/rfc822',
            'text/rfc822-headers',
            'application/mbox',
            'message/partial'
        ]
        
        for parser_name, parser in self.parsers.items():
            supported_types[parser_name] = [
                content_type for content_type in test_types 
                if parser.can_parse(content_type)
            ]
        
        return supported_types
    
    def parse_document(self, content: bytes, content_type: str) -> str:
        """
        Parse document content using the appropriate parser based on content type.
        
        Args:
            content: Document content as bytes
            content_type: MIME type of the content
            
        Returns:
            Extracted text content
            
        Raises:
            UnsupportedDocumentTypeError: If content type is not supported
            DocumentParseError: If parsing fails
        """
        logger.info(f"Parsing document with content type: {content_type}")
        
        # Find appropriate parser
        parser_info = self._get_parser_for_content_type(content_type)
        if not parser_info:
            supported_types = self.get_supported_content_types()
            raise UnsupportedDocumentTypeError(
                f"Unsupported document type: {content_type}. "
                f"Supported types: {supported_types}"
            )
        
        parser_name, parser = parser_info
        logger.info(f"Using {parser_name} parser for content type: {content_type}")
        
        try:
            # Parse the document
            text_content = parser.parse(content)
            
            if not text_content or not text_content.strip():
                raise DocumentParseError("No text content extracted from document")
            
            logger.info(
                f"Successfully parsed document using {parser_name} parser: "
                f"{len(text_content)} characters extracted"
            )
            
            return text_content
            
        except (PDFParseError, DOCXParseError, EmailParseError) as e:
            raise DocumentParseError(f"Failed to parse document with {parser_name} parser: {str(e)}")
        except Exception as e:
            raise DocumentParseError(f"Unexpected error parsing document: {str(e)}")
    
    def can_parse_content_type(self, content_type: str) -> bool:
        """
        Check if the given content type can be parsed.
        
        Args:
            content_type: MIME type to check
            
        Returns:
            True if content type is supported
        """
        return self._get_parser_for_content_type(content_type) is not None