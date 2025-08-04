"""
Unit tests for the embedding service.

Tests the Jina embedding service integration with mocked API responses,
error handling, retry logic, and caching functionality.
"""

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from typing import List, Dict, Any

import pytest
import httpx
from pydantic import ValidationError

from app.services.embedding_service import (
    EmbeddingService,
    EmbeddingServiceError,
    CachedEmbedding,
    get_embedding_service,
    cleanup_embedding_service
)
from app.models.schemas import DocumentChunk
from app.config import Settings


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock(spec=Settings)
    settings.jina_api_key = "test-api-key"
    settings.jina_model = "jina-embeddings-v4"
    settings.request_timeout = 30
    settings.max_retries = 3
    settings.retry_delay = 1.0
    return settings


@pytest.fixture
def embedding_service(mock_settings):
    """Create embedding service instance for testing."""
    with patch('app.services.embedding_service.get_settings', return_value=mock_settings):
        service = EmbeddingService()
        yield service


@pytest.fixture
def sample_embedding_response():
    """Sample API response from Jina embeddings."""
    return {
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
            },
            {
                "object": "embedding", 
                "index": 1,
                "embedding": [0.6, 0.7, 0.8, 0.9, 1.0]
            }
        ],
        "model": "jina-embeddings-v4",
        "usage": {
            "total_tokens": 10,
            "prompt_tokens": 10
        }
    }


@pytest.fixture
def sample_document_chunks():
    """Sample document chunks for testing."""
    return [
        DocumentChunk(
            document_id="doc1",
            content="This is the first chunk of text.",
            chunk_index=0,
            metadata={"page": 1}
        ),
        DocumentChunk(
            document_id="doc1", 
            content="This is the second chunk of text.",
            chunk_index=1,
            metadata={"page": 1}
        )
    ]


class TestEmbeddingService:
    """Test cases for EmbeddingService class."""
    
    def test_initialization(self, embedding_service):
        """Test service initialization."""
        assert embedding_service.base_url == "https://api.jina.ai/v1/embeddings"
        assert "Bearer test-api-key" in embedding_service.headers["Authorization"]
        assert embedding_service.headers["Content-Type"] == "application/json"
        assert len(embedding_service._cache) == 0
    
    def test_cache_key_generation(self, embedding_service):
        """Test cache key generation."""
        key1 = embedding_service._generate_cache_key("test text", "model1")
        key2 = embedding_service._generate_cache_key("test text", "model1")
        key3 = embedding_service._generate_cache_key("test text", "model2")
        key4 = embedding_service._generate_cache_key("different text", "model1")
        
        assert key1 == key2  # Same text and model
        assert key1 != key3  # Different model
        assert key1 != key4  # Different text
        assert len(key1) == 64  # SHA256 hash length
    
    def test_cache_validity(self, embedding_service):
        """Test cache validity checking."""
        # Valid cache item
        valid_item = CachedEmbedding(
            embedding=[0.1, 0.2, 0.3],
            created_at=datetime.now(),
            model="test-model"
        )
        assert embedding_service._is_cache_valid(valid_item)
        
        # Expired cache item
        expired_item = CachedEmbedding(
            embedding=[0.1, 0.2, 0.3],
            created_at=datetime.now() - timedelta(hours=25),
            model="test-model"
        )
        assert not embedding_service._is_cache_valid(expired_item)
    
    def test_cache_operations(self, embedding_service):
        """Test cache store and retrieve operations."""
        text = "test text"
        model = "test-model"
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Initially no cache
        assert embedding_service._get_from_cache(text, model) is None
        
        # Store in cache
        embedding_service._store_in_cache(text, model, embedding)
        
        # Retrieve from cache
        cached_embedding = embedding_service._get_from_cache(text, model)
        assert cached_embedding == embedding
        
        # Different text should not be cached
        assert embedding_service._get_from_cache("different text", model) is None
    
    @pytest.mark.asyncio
    async def test_successful_api_request(self, embedding_service, sample_embedding_response):
        """Test successful API request."""
        with patch.object(embedding_service.client, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_embedding_response
            mock_post.return_value = mock_response
            
            texts = ["text1", "text2"]
            response = await embedding_service._make_api_request(texts, "jina-embeddings-v4")
            
            assert response.model == "jina-embeddings-v4"
            assert len(response.data) == 2
            assert response.data[0]["embedding"] == [0.1, 0.2, 0.3, 0.4, 0.5]
            
            # Verify API call
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]["json"]["input"] == texts
            assert call_args[1]["json"]["model"] == "jina-embeddings-v4"
    
    @pytest.mark.asyncio
    async def test_api_request_retry_on_rate_limit(self, embedding_service, sample_embedding_response):
        """Test API request retry on rate limit."""
        with patch.object(embedding_service.client, 'post') as mock_post:
            # First call returns 429, second call succeeds
            rate_limit_response = Mock()
            rate_limit_response.status_code = 429
            
            success_response = Mock()
            success_response.status_code = 200
            success_response.json.return_value = sample_embedding_response
            
            mock_post.side_effect = [rate_limit_response, success_response]
            
            with patch('asyncio.sleep') as mock_sleep:
                texts = ["text1"]
                response = await embedding_service._make_api_request(texts, "jina-embeddings-v4")
                
                assert response.model == "jina-embeddings-v4"
                assert mock_post.call_count == 2
                mock_sleep.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_api_request_max_retries_exceeded(self, embedding_service):
        """Test API request failure after max retries."""
        with patch.object(embedding_service.client, 'post') as mock_post:
            # All calls return 500
            server_error_response = Mock()
            server_error_response.status_code = 500
            mock_post.return_value = server_error_response
            
            with patch('asyncio.sleep'):
                with pytest.raises(EmbeddingServiceError) as exc_info:
                    await embedding_service._make_api_request(["text1"], "jina-embeddings-v4")
                
                assert exc_info.value.error_code == "API_MAX_RETRIES_EXCEEDED"
                assert mock_post.call_count == 3  # max_retries
    
    @pytest.mark.asyncio
    async def test_api_request_client_error_no_retry(self, embedding_service):
        """Test API request with client error (no retry)."""
        with patch.object(embedding_service.client, 'post') as mock_post:
            client_error_response = Mock()
            client_error_response.status_code = 400
            client_error_response.text = "Bad Request"
            mock_post.return_value = client_error_response
            
            with pytest.raises(EmbeddingServiceError) as exc_info:
                await embedding_service._make_api_request(["text1"], "jina-embeddings-v4")
            
            assert exc_info.value.error_code == "API_CLIENT_ERROR"
            assert mock_post.call_count == 1  # No retry for client errors
    
    @pytest.mark.asyncio
    async def test_api_request_timeout(self, embedding_service):
        """Test API request timeout handling."""
        with patch.object(embedding_service.client, 'post') as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Request timeout")
            
            with patch('asyncio.sleep'):
                with pytest.raises(EmbeddingServiceError) as exc_info:
                    await embedding_service._make_api_request(["text1"], "jina-embeddings-v4")
                
                assert exc_info.value.error_code == "API_TIMEOUT_ERROR"
                assert mock_post.call_count == 3  # max_retries
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_success(self, embedding_service, sample_embedding_response):
        """Test successful embedding generation."""
        with patch.object(embedding_service, '_make_api_request') as mock_api:
            mock_api.return_value = Mock(data=sample_embedding_response["data"])
            
            texts = ["text1", "text2"]
            embeddings = await embedding_service.generate_embeddings(texts)
            
            assert len(embeddings) == 2
            assert embeddings[0] == [0.1, 0.2, 0.3, 0.4, 0.5]
            assert embeddings[1] == [0.6, 0.7, 0.8, 0.9, 1.0]
            
            # Check caching
            cached_embedding = embedding_service._get_from_cache("text1", "jina-embeddings-v4")
            assert cached_embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_with_cache(self, embedding_service):
        """Test embedding generation with cache hits."""
        # Pre-populate cache
        embedding_service._store_in_cache("text1", "jina-embeddings-v4", [0.1, 0.2, 0.3])
        
        with patch.object(embedding_service, '_make_api_request') as mock_api:
            mock_api.return_value = Mock(data=[{
                "embedding": [0.4, 0.5, 0.6]
            }])
            
            texts = ["text1", "text2"]  # text1 cached, text2 not cached
            embeddings = await embedding_service.generate_embeddings(texts)
            
            assert len(embeddings) == 2
            assert embeddings[0] == [0.1, 0.2, 0.3]  # From cache
            assert embeddings[1] == [0.4, 0.5, 0.6]  # From API
            
            # API should only be called for uncached text
            mock_api.assert_called_once()
            call_args = mock_api.call_args[0]
            assert call_args[0] == ["text2"]  # Only uncached text
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_input(self, embedding_service):
        """Test embedding generation with empty input."""
        embeddings = await embedding_service.generate_embeddings([])
        assert embeddings == []
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_invalid_input(self, embedding_service):
        """Test embedding generation with invalid input."""
        with pytest.raises(EmbeddingServiceError) as exc_info:
            await embedding_service.generate_embeddings(["", "valid text"])
        
        assert exc_info.value.error_code == "INVALID_INPUT"
        assert "index 0" in exc_info.value.message
    
    @pytest.mark.asyncio
    async def test_generate_query_embedding(self, embedding_service):
        """Test single query embedding generation."""
        with patch.object(embedding_service, 'generate_embeddings') as mock_generate:
            mock_generate.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
            
            query = "What is the main topic?"
            embedding = await embedding_service.generate_query_embedding(query)
            
            assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
            mock_generate.assert_called_once_with([query])
    
    @pytest.mark.asyncio
    async def test_generate_query_embedding_empty_query(self, embedding_service):
        """Test query embedding with empty query."""
        with pytest.raises(EmbeddingServiceError) as exc_info:
            await embedding_service.generate_query_embedding("")
        
        assert exc_info.value.error_code == "INVALID_INPUT"
    
    @pytest.mark.asyncio
    async def test_generate_batch_embeddings(self, embedding_service, sample_document_chunks):
        """Test batch embedding generation for document chunks."""
        with patch.object(embedding_service, 'generate_embeddings') as mock_generate:
            mock_generate.return_value = [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.6, 0.7, 0.8, 0.9, 1.0]
            ]
            
            chunks = await embedding_service.generate_batch_embeddings(
                sample_document_chunks, 
                batch_size=2
            )
            
            assert len(chunks) == 2
            assert chunks[0].embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
            assert chunks[1].embedding == [0.6, 0.7, 0.8, 0.9, 1.0]
            
            mock_generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_batch_embeddings_multiple_batches(self, embedding_service):
        """Test batch embedding generation with multiple batches."""
        # Create 5 chunks to test batching with batch_size=2
        chunks = [
            DocumentChunk(
                document_id="doc1",
                content=f"Chunk {i} content",
                chunk_index=i,
                metadata={}
            )
            for i in range(5)
        ]
        
        with patch.object(embedding_service, 'generate_embeddings') as mock_generate:
            # Mock returns for each batch
            mock_generate.side_effect = [
                [[0.1, 0.2], [0.3, 0.4]],  # Batch 1: chunks 0, 1
                [[0.5, 0.6], [0.7, 0.8]],  # Batch 2: chunks 2, 3
                [[0.9, 1.0]]               # Batch 3: chunk 4
            ]
            
            result_chunks = await embedding_service.generate_batch_embeddings(
                chunks, 
                batch_size=2
            )
            
            assert len(result_chunks) == 5
            assert mock_generate.call_count == 3  # 3 batches
            
            # Check embeddings were assigned correctly
            expected_embeddings = [
                [0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]
            ]
            for i, chunk in enumerate(result_chunks):
                assert chunk.embedding == expected_embeddings[i]
    
    @pytest.mark.asyncio
    async def test_generate_batch_embeddings_empty_input(self, embedding_service):
        """Test batch embedding generation with empty input."""
        chunks = await embedding_service.generate_batch_embeddings([])
        assert chunks == []
    
    @pytest.mark.asyncio
    async def test_generate_batch_embeddings_error(self, embedding_service, sample_document_chunks):
        """Test batch embedding generation with error."""
        with patch.object(embedding_service, 'generate_embeddings') as mock_generate:
            mock_generate.side_effect = Exception("API Error")
            
            with pytest.raises(EmbeddingServiceError) as exc_info:
                await embedding_service.generate_batch_embeddings(sample_document_chunks)
            
            assert exc_info.value.error_code == "BATCH_PROCESSING_ERROR"
    
    def test_clear_cache(self, embedding_service):
        """Test cache clearing."""
        # Add some items to cache
        embedding_service._store_in_cache("text1", "model1", [0.1, 0.2])
        embedding_service._store_in_cache("text2", "model1", [0.3, 0.4])
        
        assert len(embedding_service._cache) == 2
        
        embedding_service.clear_cache()
        assert len(embedding_service._cache) == 0
    
    def test_get_cache_stats(self, embedding_service):
        """Test cache statistics."""
        # Add valid and expired items
        embedding_service._store_in_cache("text1", "model1", [0.1, 0.2])
        
        # Manually add expired item
        expired_key = embedding_service._generate_cache_key("text2", "model1")
        embedding_service._cache[expired_key] = CachedEmbedding(
            embedding=[0.3, 0.4],
            created_at=datetime.now() - timedelta(hours=25),
            model="model1"
        )
        
        stats = embedding_service.get_cache_stats()
        
        assert stats["total_entries"] == 2
        assert stats["valid_entries"] == 1
        assert stats["expired_entries"] == 1
        assert stats["cache_ttl_hours"] == 24


class TestEmbeddingServiceGlobal:
    """Test cases for global embedding service functions."""
    
    @pytest.mark.asyncio
    async def test_get_embedding_service(self):
        """Test getting global embedding service instance."""
        with patch('app.services.embedding_service.get_settings'):
            service1 = await get_embedding_service()
            service2 = await get_embedding_service()
            
            assert service1 is service2  # Same instance
            assert isinstance(service1, EmbeddingService)
    
    @pytest.mark.asyncio
    async def test_cleanup_embedding_service(self):
        """Test cleanup of global embedding service."""
        with patch('app.services.embedding_service.get_settings'):
            service = await get_embedding_service()
            
            with patch.object(service.client, 'aclose') as mock_close:
                await cleanup_embedding_service()
                mock_close.assert_called_once()


class TestEmbeddingModels:
    """Test cases for embedding-related data models."""
    
    def test_cached_embedding_model(self):
        """Test CachedEmbedding model validation."""
        embedding = CachedEmbedding(
            embedding=[0.1, 0.2, 0.3],
            created_at=datetime.now(),
            model="test-model"
        )
        
        assert len(embedding.embedding) == 3
        assert embedding.model == "test-model"
        assert isinstance(embedding.created_at, datetime)
    
    def test_embedding_service_error(self):
        """Test EmbeddingServiceError exception."""
        error = EmbeddingServiceError(
            "Test error",
            "TEST_ERROR",
            {"detail": "test detail"}
        )
        
        assert str(error) == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.details["detail"] == "test detail"


@pytest.mark.asyncio
async def test_context_manager(mock_settings):
    """Test embedding service as async context manager."""
    with patch('app.services.embedding_service.get_settings', return_value=mock_settings):
        async with EmbeddingService() as service:
            assert isinstance(service, EmbeddingService)
            
        # Client should be closed after context exit
        # This is tested implicitly by the context manager implementation


if __name__ == "__main__":
    pytest.main([__file__])