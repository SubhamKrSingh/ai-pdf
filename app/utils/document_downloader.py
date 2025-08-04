"""
Document downloader utility for handling URL validation and HTTP requests.

This module implements document downloading with comprehensive error handling
according to requirements 2.1, 8.2.
"""

import httpx
import mimetypes
from typing import Tuple, Optional
from urllib.parse import urlparse
import logging

from app.exceptions import DocumentDownloadError, DocumentNotFoundError, UnsupportedDocumentTypeError
from app.utils.retry import with_retry, DOWNLOAD_RETRY_CONFIG

logger = logging.getLogger(__name__)


class DocumentDownloader:
    """Handles document downloading from URLs with proper validation and error handling."""
    
    def __init__(self, timeout: int = 30, max_size_mb: int = 50):
        """
        Initialize the document downloader.
        
        Args:
            timeout: Request timeout in seconds
            max_size_mb: Maximum allowed document size in MB
        """
        self.timeout = timeout
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
    def _validate_url(self, url: str) -> bool:
        """
        Validate if the URL is properly formatted and uses allowed schemes.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is valid, False otherwise
        """
        try:
            parsed = urlparse(url)
            return parsed.scheme in ('http', 'https') and bool(parsed.netloc)
        except Exception:
            return False
    
    def _detect_content_type(self, response: httpx.Response, url: str) -> str:
        """
        Detect content type from response headers or URL extension.
        
        Args:
            response: HTTP response object
            url: Original URL
            
        Returns:
            Detected content type
        """
        # First try to get from response headers
        content_type = response.headers.get('content-type', '').lower()
        
        if content_type:
            # Extract main content type (ignore charset, etc.)
            main_type = content_type.split(';')[0].strip()
            if main_type:
                return main_type
        
        # Fallback to URL extension
        guessed_type, _ = mimetypes.guess_type(url)
        if guessed_type:
            return guessed_type
            
        return 'application/octet-stream'
    
    @with_retry(DOWNLOAD_RETRY_CONFIG, context={"component": "document_downloader"})
    async def download_document(self, url: str) -> Tuple[bytes, str]:
        """
        Download document from URL with validation and error handling.
        
        Args:
            url: URL to download document from
            
        Returns:
            Tuple of (document_content, content_type)
            
        Raises:
            DocumentDownloadError: If download fails for any reason
            DocumentNotFoundError: If document is not found (404)
            UnsupportedDocumentTypeError: If document type is not supported
        """
        if not self._validate_url(url):
            raise DocumentDownloadError(
                url=url,
                reason="Invalid URL format",
                details={"validation_error": "URL format is invalid"}
            )
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info(f"Downloading document from: {url}")
                
                # Make HEAD request first to check content length
                try:
                    head_response = await client.head(url, follow_redirects=True)
                    content_length = head_response.headers.get('content-length')
                    
                    if content_length and int(content_length) > self.max_size_bytes:
                        raise DocumentDownloadError(
                            url=url,
                            reason=f"Document too large: {content_length} bytes (max: {self.max_size_bytes} bytes)",
                            details={
                                "content_length": int(content_length),
                                "max_size_bytes": self.max_size_bytes
                            }
                        )
                except httpx.HTTPStatusError:
                    # Some servers don't support HEAD, continue with GET
                    pass
                
                # Download the document
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                
                # Check content length after download
                content = response.content
                if len(content) > self.max_size_bytes:
                    raise DocumentDownloadError(
                        url=url,
                        reason=f"Document too large: {len(content)} bytes (max: {self.max_size_bytes} bytes)",
                        details={
                            "actual_size": len(content),
                            "max_size_bytes": self.max_size_bytes
                        }
                    )
                
                # Detect content type
                content_type = self._detect_content_type(response, url)
                
                # Validate content type is supported
                supported_types = [
                    'application/pdf',
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    'message/rfc822',
                    'text/plain'
                ]
                
                if content_type not in supported_types:
                    raise UnsupportedDocumentTypeError(
                        content_type=content_type,
                        supported_types=supported_types,
                        details={"url": url}
                    )
                
                logger.info(
                    f"Successfully downloaded document: {len(content)} bytes, "
                    f"content-type: {content_type}"
                )
                
                return content, content_type
                
        except httpx.TimeoutException:
            raise DocumentDownloadError(
                url=url,
                reason="Request timeout",
                details={"timeout_seconds": self.timeout}
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise DocumentNotFoundError(
                    url=url,
                    details={"status_code": e.response.status_code}
                )
            else:
                raise DocumentDownloadError(
                    url=url,
                    status_code=e.response.status_code,
                    reason=f"HTTP error {e.response.status_code}",
                    details={"response_text": e.response.text[:500] if e.response.text else None}
                )
        except httpx.RequestError as e:
            raise DocumentDownloadError(
                url=url,
                reason=f"Request error: {str(e)}",
                details={"error_type": type(e).__name__}
            )
        except (DocumentDownloadError, DocumentNotFoundError, UnsupportedDocumentTypeError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            raise DocumentDownloadError(
                url=url,
                reason=f"Unexpected error: {str(e)}",
                details={"error_type": type(e).__name__}
            )