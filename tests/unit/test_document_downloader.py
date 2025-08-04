"""
Unit tests for document downloader.
"""

import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from app.utils.document_downloader import DocumentDownloader, DocumentDownloadError


class TestDocumentDownloader:
    """Test cases for DocumentDownloader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.downloader = DocumentDownloader(timeout=30, max_size_mb=50)
    
    def test_validate_url_valid_http(self):
        """Test URL validation with valid HTTP URL."""
        assert self.downloader._validate_url("http://example.com/document.pdf") is True
    
    def test_validate_url_valid_https(self):
        """Test URL validation with valid HTTPS URL."""
        assert self.downloader._validate_url("https://example.com/document.pdf") is True
    
    def test_validate_url_invalid_scheme(self):
        """Test URL validation with invalid scheme."""
        assert self.downloader._validate_url("ftp://example.com/document.pdf") is False
    
    def test_validate_url_no_netloc(self):
        """Test URL validation with missing netloc."""
        assert self.downloader._validate_url("http://") is False
    
    def test_validate_url_malformed(self):
        """Test URL validation with malformed URL."""
        assert self.downloader._validate_url("not-a-url") is False
    
    def test_detect_content_type_from_header(self):
        """Test content type detection from response headers."""
        mock_response = MagicMock()
        mock_response.headers = {'content-type': 'application/pdf; charset=utf-8'}
        
        content_type = self.downloader._detect_content_type(mock_response, "http://example.com/doc")
        assert content_type == "application/pdf"
    
    def test_detect_content_type_from_url_extension(self):
        """Test content type detection from URL extension."""
        mock_response = MagicMock()
        mock_response.headers = {}
        
        content_type = self.downloader._detect_content_type(mock_response, "http://example.com/doc.pdf")
        assert content_type == "application/pdf"
    
    def test_detect_content_type_fallback(self):
        """Test content type detection fallback."""
        mock_response = MagicMock()
        mock_response.headers = {}
        
        content_type = self.downloader._detect_content_type(mock_response, "http://example.com/doc")
        assert content_type == "application/octet-stream"
    
    @pytest.mark.asyncio
    async def test_download_document_success(self):
        """Test successful document download."""
        mock_content = b"PDF content here"
        mock_response = MagicMock()
        mock_response.content = mock_content
        mock_response.headers = {'content-type': 'application/pdf'}
        mock_response.raise_for_status = MagicMock()
        
        mock_head_response = MagicMock()
        mock_head_response.headers = {'content-length': '100'}
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            mock_client_instance.head.return_value = mock_head_response
            mock_client_instance.get.return_value = mock_response
            
            content, content_type = await self.downloader.download_document("https://example.com/doc.pdf")
            
            assert content == mock_content
            assert content_type == "application/pdf"
    
    @pytest.mark.asyncio
    async def test_download_document_invalid_url(self):
        """Test download with invalid URL."""
        with pytest.raises(DocumentDownloadError, match="Invalid URL format"):
            await self.downloader.download_document("not-a-url")
    
    @pytest.mark.asyncio
    async def test_download_document_too_large_from_header(self):
        """Test download failure when document is too large (detected from header)."""
        mock_head_response = MagicMock()
        mock_head_response.headers = {'content-length': str(100 * 1024 * 1024)}  # 100MB
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            mock_client_instance.head.return_value = mock_head_response
            
            with pytest.raises(DocumentDownloadError, match="Document too large"):
                await self.downloader.download_document("https://example.com/large.pdf")
    
    @pytest.mark.asyncio
    async def test_download_document_too_large_from_content(self):
        """Test download failure when document is too large (detected from content)."""
        large_content = b"x" * (60 * 1024 * 1024)  # 60MB
        mock_response = MagicMock()
        mock_response.content = large_content
        mock_response.headers = {'content-type': 'application/pdf'}
        mock_response.raise_for_status = MagicMock()
        
        mock_head_response = MagicMock()
        mock_head_response.headers = {}  # No content-length in HEAD
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            mock_client_instance.head.return_value = mock_head_response
            mock_client_instance.get.return_value = mock_response
            
            with pytest.raises(DocumentDownloadError, match="Document too large"):
                await self.downloader.download_document("https://example.com/large.pdf")
    
    @pytest.mark.asyncio
    async def test_download_document_http_error(self):
        """Test download failure with HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            mock_client_instance.head.return_value = MagicMock()
            mock_client_instance.get.side_effect = httpx.HTTPStatusError(
                "Not Found", request=MagicMock(), response=mock_response
            )
            
            with pytest.raises(DocumentDownloadError, match="HTTP error 404"):
                await self.downloader.download_document("https://example.com/notfound.pdf")
    
    @pytest.mark.asyncio
    async def test_download_document_timeout(self):
        """Test download failure with timeout."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            mock_client_instance.head.return_value = MagicMock()
            mock_client_instance.get.side_effect = httpx.TimeoutException("Timeout")
            
            with pytest.raises(DocumentDownloadError, match="Timeout while downloading"):
                await self.downloader.download_document("https://example.com/slow.pdf")