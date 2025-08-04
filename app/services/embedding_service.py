"""
Jina embedding service integration with async API calls, batch processing, and caching.

This module implements the embedding service for converting text to vector embeddings
using Jina embeddings v4 API. Includes error handling, retry logic, and caching.

Requirements implemented: 4.1, 4.5, 8.3
"""

import asyncio
import hashlib
import json
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

import httpx
from pydantic import BaseModel

from app.config import get_settings
from app.models.schemas import DocumentChunk
from app.exceptions import EmbeddingServiceError
from app.utils.retry import with_retry, EMBEDDING_RETRY_CONFIG
from app.middleware.error_handler import error_logger


# Configure logging
logger = logging.getLogger(__name__)


class EmbeddingRequest(BaseModel):
    """Request model for Jina embeddings API."""
    input: List[str]
    model: str


class EmbeddingResponse(BaseModel):
    """Response model from Jina embeddings API."""
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]


class CachedEmbedding(BaseModel):
    """Model for cached embedding data."""
    embedding: List[float]
    created_at: datetime
    model: str


class EmbeddingService:
    """
    Service for generating embeddings using Jina embeddings v4 API.
    
    Provides async batch processing, caching, and comprehensive error handling.
    """
    
    def __init__(self):
        """Initialize the embedding service with configuration."""
        self.settings = get_settings()
        self.base_url = "https://api.jina.ai/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.settings.jina_api_key}",
            "Content-Type": "application/json"
        }
        
        # In-memory cache for embeddings (in production, use Redis or similar)
        self._cache: Dict[str, CachedEmbedding] = {}
        self._cache_ttl = timedelta(hours=24)  # Cache for 24 hours
        
        # HTTP client with extended timeout for embedding operations
        # Embedding generation can take longer than regular API calls
        embedding_timeout = max(self.settings.request_timeout * 2, 60)  # At least 60 seconds
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,  # Connection timeout
                read=embedding_timeout,  # Read timeout (main bottleneck)
                write=10.0,  # Write timeout
                pool=5.0  # Pool timeout
            ),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()
    
    def _generate_cache_key(self, text: str, model: str) -> str:
        """Generate a cache key for the given text and model."""
        content = f"{text}:{model}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _is_cache_valid(self, cached_item: CachedEmbedding) -> bool:
        """Check if a cached embedding is still valid."""
        return datetime.now() - cached_item.created_at < self._cache_ttl
    
    def _get_from_cache(self, text: str, model: str) -> Optional[List[float]]:
        """Retrieve embedding from cache if available and valid."""
        cache_key = self._generate_cache_key(text, model)
        cached_item = self._cache.get(cache_key)
        
        if cached_item and self._is_cache_valid(cached_item):
            logger.debug(f"Cache hit for text: {text[:50]}...")
            return cached_item.embedding
        
        # Remove expired cache entry
        if cached_item:
            del self._cache[cache_key]
        
        return None
    
    def _store_in_cache(self, text: str, model: str, embedding: List[float]) -> None:
        """Store embedding in cache."""
        cache_key = self._generate_cache_key(text, model)
        self._cache[cache_key] = CachedEmbedding(
            embedding=embedding,
            created_at=datetime.now(),
            model=model
        )
        logger.debug(f"Cached embedding for text: {text[:50]}...")
    
    @with_retry(EMBEDDING_RETRY_CONFIG, context={"component": "embedding_service", "operation": "api_request"})
    async def _make_api_request(self, texts: List[str], model: str) -> EmbeddingResponse:
        """
        Make API request to Jina embeddings with comprehensive error handling.
        
        Args:
            texts: List of texts to embed
            model: Model name to use
            
        Returns:
            EmbeddingResponse: API response with embeddings
            
        Raises:
            EmbeddingServiceError: If API request fails
        """
        request_data = EmbeddingRequest(input=texts, model=model)
        
        try:
            logger.debug(f"Making embedding API request for {len(texts)} texts")
            
            response = await self.client.post(
                self.base_url,
                headers=self.headers,
                json=request_data.model_dump()
            )
            
            if response.status_code == 200:
                response_data = response.json()
                return EmbeddingResponse(**response_data)
            
            elif response.status_code == 429:  # Rate limit
                raise EmbeddingServiceError(
                    operation="generate_embeddings",
                    reason="Rate limit exceeded",
                    details={
                        "status_code": response.status_code,
                        "retry_after": response.headers.get("retry-after"),
                        "texts_count": len(texts)
                    }
                )
            
            elif response.status_code >= 500:  # Server error
                raise EmbeddingServiceError(
                    operation="generate_embeddings",
                    reason=f"Server error: {response.status_code}",
                    details={
                        "status_code": response.status_code,
                        "response_text": response.text[:500] if response.text else None,
                        "texts_count": len(texts)
                    }
                )
            
            else:  # Client error
                raise EmbeddingServiceError(
                    operation="generate_embeddings",
                    reason=f"Client error: {response.status_code}",
                    details={
                        "status_code": response.status_code,
                        "response_text": response.text[:500] if response.text else None,
                        "texts_count": len(texts)
                    }
                )
                
        except httpx.TimeoutException as e:
            raise EmbeddingServiceError(
                operation="generate_embeddings",
                reason="Request timeout",
                details={
                    "timeout_seconds": self.settings.request_timeout,
                    "texts_count": len(texts)
                }
            )
        except httpx.RequestError as e:
            raise EmbeddingServiceError(
                operation="generate_embeddings",
                reason=f"Request error: {str(e)}",
                details={
                    "error_type": type(e).__name__,
                    "texts_count": len(texts)
                }
            )
        except EmbeddingServiceError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            error_logger.log_error(
                e,
                {
                    "operation": "generate_embeddings",
                    "component": "embedding_service",
                    "texts_count": len(texts)
                }
            )
            raise EmbeddingServiceError(
                operation="generate_embeddings",
                reason=f"Unexpected error: {str(e)}",
                details={
                    "error_type": type(e).__name__,
                    "texts_count": len(texts)
                }
            )
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts with caching and batch processing.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
            
        Raises:
            EmbeddingServiceError: If embedding generation fails
        """
        if not texts:
            return []
        
        # Validate input
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise EmbeddingServiceError(
                    f"Empty text at index {i}",
                    "INVALID_INPUT",
                    {"index": i}
                )
        
        # Check cache for each text
        embeddings = []
        texts_to_process = []
        cache_indices = []
        
        for i, text in enumerate(texts):
            cached_embedding = self._get_from_cache(text.strip(), self.settings.jina_model)
            if cached_embedding:
                embeddings.append(cached_embedding)
                cache_indices.append(i)
            else:
                texts_to_process.append(text.strip())
        
        # Process uncached texts
        if texts_to_process:
            try:
                logger.info(f"Generating embeddings for {len(texts_to_process)} texts")
                response = await self._make_api_request(texts_to_process, self.settings.jina_model)
                
                # Extract embeddings from response
                new_embeddings = []
                for item in response.data:
                    if "embedding" in item:
                        embedding = item["embedding"]
                        new_embeddings.append(embedding)
                    else:
                        raise EmbeddingServiceError(
                            "Invalid API response format",
                            "API_RESPONSE_ERROR",
                            {"response_item": item}
                        )
                
                # Cache new embeddings
                for text, embedding in zip(texts_to_process, new_embeddings):
                    self._store_in_cache(text, self.settings.jina_model, embedding)
                
                # Merge cached and new embeddings in correct order
                result_embeddings = []
                new_idx = 0
                
                for i in range(len(texts)):
                    if i in cache_indices:
                        # Find the cached embedding for this index
                        cached_embedding = self._get_from_cache(texts[i].strip(), self.settings.jina_model)
                        result_embeddings.append(cached_embedding)
                    else:
                        result_embeddings.append(new_embeddings[new_idx])
                        new_idx += 1
                
                return result_embeddings
                
            except Exception as e:
                if isinstance(e, EmbeddingServiceError):
                    raise
                else:
                    raise EmbeddingServiceError(
                        f"Unexpected error during embedding generation: {str(e)}",
                        "UNEXPECTED_ERROR",
                        {"error": str(e)}
                    )
        
        return embeddings
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a single query string.
        
        Args:
            query: Query text to embed
            
        Returns:
            List[float]: Embedding vector
            
        Raises:
            EmbeddingServiceError: If embedding generation fails
        """
        if not query or not query.strip():
            raise EmbeddingServiceError(
                "Query cannot be empty",
                "INVALID_INPUT",
                {"query": query}
            )
        
        embeddings = await self.generate_embeddings([query.strip()])
        return embeddings[0]
    
    async def generate_batch_embeddings(
        self, 
        chunks: List[DocumentChunk], 
        batch_size: int = 5  # Reduced batch size for better reliability
    ) -> List[DocumentChunk]:
        """
        Generate embeddings for document chunks in batches.
        
        Args:
            chunks: List of document chunks to embed
            batch_size: Number of chunks to process in each batch
            
        Returns:
            List[DocumentChunk]: Chunks with embeddings added
            
        Raises:
            EmbeddingServiceError: If batch processing fails
        """
        if not chunks:
            return []
        
        logger.info(f"Processing {len(chunks)} chunks in batches of {batch_size}")
        
        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_texts = [chunk.content for chunk in batch]
            
            try:
                batch_embeddings = await self.generate_embeddings(batch_texts)
                
                # Add embeddings to chunks
                for chunk, embedding in zip(batch, batch_embeddings):
                    chunk.embedding = embedding
                
                logger.debug(f"Processed batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                logger.error(f"Failed to process batch starting at index {i}: {str(e)}")
                raise EmbeddingServiceError(
                    f"Batch processing failed at index {i}",
                    "BATCH_PROCESSING_ERROR",
                    {"batch_start": i, "batch_size": len(batch), "error": str(e)}
                )
        
        return chunks
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        valid_entries = sum(1 for item in self._cache.values() if self._is_cache_valid(item))
        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self._cache) - valid_entries,
            "cache_ttl_hours": self._cache_ttl.total_seconds() / 3600
        }


# Global service instance
_embedding_service: Optional[EmbeddingService] = None


async def get_embedding_service() -> EmbeddingService:
    """
    Get or create the global embedding service instance.
    
    Returns:
        EmbeddingService: The embedding service instance
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


async def cleanup_embedding_service() -> None:
    """Clean up the global embedding service instance."""
    global _embedding_service
    if _embedding_service is not None:
        await _embedding_service.client.aclose()
        _embedding_service = None