"""
Document caching service to avoid reprocessing the same documents.

This service implements URL-based caching to dramatically reduce latency
for repeated document processing requests.
"""

import hashlib
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from app.config import get_settings
from app.data.repository import get_repository
from app.data.vector_store import get_vector_store

logger = logging.getLogger(__name__)


class DocumentCacheService:
    """
    Service for caching processed documents to avoid redundant processing.
    
    Uses URL hashing and metadata tracking to identify previously processed documents.
    """
    
    def __init__(self):
        self.settings = get_settings()
        # Cache TTL - how long to consider a cached document valid
        self.cache_ttl = timedelta(hours=24)  # Configurable
    
    def _generate_url_hash(self, url: str) -> str:
        """Generate a consistent hash for a document URL."""
        return hashlib.sha256(url.encode()).hexdigest()
    
    async def get_cached_document(self, url: str) -> Optional[str]:
        """
        Check if document is already processed and cached.
        
        Args:
            url: Document URL to check
            
        Returns:
            Optional[str]: Document ID if cached and valid, None otherwise
        """
        try:
            url_hash = self._generate_url_hash(url)
            repository = await get_repository()
            
            # Check if we have metadata for this URL hash
            metadata = await repository.get_document_by_url_hash(url_hash)
            
            if not metadata:
                logger.debug(f"No cached document found for URL: {url}")
                return None
            
            # Check if cache is still valid
            processed_at = datetime.fromisoformat(metadata["processed_at"])
            if datetime.utcnow() - processed_at > self.cache_ttl:
                logger.info(f"Cached document expired for URL: {url}")
                # Optionally clean up expired cache
                await self._cleanup_expired_document(metadata["id"])
                return None
            
            # Verify vectors still exist in vector store
            vector_store = await get_vector_store()
            try:
                # Quick check if vectors exist by trying to search
                test_results = await vector_store.similarity_search(
                    query_vector=[0.0] * 2048,  # Dummy vector
                    document_id=metadata["id"],
                    top_k=1
                )
                
                if not test_results:
                    logger.warning(f"Vectors missing for cached document: {metadata['id']}")
                    return None
                
            except Exception as e:
                logger.warning(f"Vector store check failed for {metadata['id']}: {e}")
                return None
            
            logger.info(f"Found valid cached document: {metadata['id']} for URL: {url}")
            return metadata["id"]
            
        except Exception as e:
            logger.error(f"Error checking document cache: {e}")
            return None
    
    async def cache_document(self, url: str, document_id: str) -> bool:
        """
        Cache a processed document for future use.
        
        Args:
            url: Original document URL
            document_id: Processed document ID
            
        Returns:
            bool: True if caching was successful
        """
        try:
            url_hash = self._generate_url_hash(url)
            repository = await get_repository()
            
            # Store URL hash mapping
            await repository.store_url_hash_mapping(url_hash, url, document_id)
            
            logger.info(f"Cached document {document_id} for URL: {url}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching document: {e}")
            return False
    
    async def _cleanup_expired_document(self, document_id: str) -> None:
        """Clean up expired cached document."""
        try:
            # Delete from vector store
            vector_store = await get_vector_store()
            await vector_store.delete_document_vectors(document_id)
            
            # Delete from database
            repository = await get_repository()
            await repository.delete_document(document_id)
            
            logger.info(f"Cleaned up expired document: {document_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up expired document {document_id}: {e}")
    
    async def invalidate_cache(self, url: str) -> bool:
        """
        Manually invalidate cache for a specific URL.
        
        Args:
            url: Document URL to invalidate
            
        Returns:
            bool: True if invalidation was successful
        """
        try:
            cached_doc_id = await self.get_cached_document(url)
            if cached_doc_id:
                await self._cleanup_expired_document(cached_doc_id)
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error invalidating cache for URL {url}: {e}")
            return False


# Global service instance
_cache_service: Optional[DocumentCacheService] = None


def get_document_cache_service() -> DocumentCacheService:
    """Get or create the global document cache service instance."""
    global _cache_service
    if _cache_service is None:
        _cache_service = DocumentCacheService()
    return _cache_service