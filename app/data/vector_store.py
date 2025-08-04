"""
Pinecone vector database integration for document embeddings storage and retrieval.

This module implements vector storage functionality with metadata preservation,
semantic search with similarity scoring, and vector deletion capabilities.
Implements requirements 4.2, 4.3, 4.4, 5.2, 5.3, 5.4, 8.4.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4
import time

import pinecone
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeException

from app.config import get_settings
from app.models.schemas import DocumentChunk, SearchResult


logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Custom exception for vector store operations."""
    
    def __init__(self, message: str, error_code: str = "VECTOR_STORE_ERROR", details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class PineconeVectorStore:
    """
    Pinecone vector database client with connection pooling and error handling.
    
    Provides functionality for storing document embeddings, performing semantic search,
    and managing vector data with comprehensive error handling and retry logic.
    """
    
    def __init__(self):
        """Initialize Pinecone client with environment configuration."""
        self.settings = get_settings()
        self._client: Optional[Pinecone] = None
        self._index = None
        self._connection_pool_size = 10
        self._max_retries = self.settings.max_retries
        self._retry_delay = self.settings.retry_delay
        
    async def _get_client(self) -> Pinecone:
        """
        Get or create Pinecone client with connection pooling.
        
        Returns:
            Pinecone: Configured Pinecone client
            
        Raises:
            VectorStoreError: If client initialization fails
        """
        if self._client is None:
            try:
                self._client = Pinecone(
                    api_key=self.settings.pinecone_api_key,
                    environment=self.settings.pinecone_environment
                )
                logger.info("Pinecone client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone client: {str(e)}")
                raise VectorStoreError(
                    "Failed to initialize Pinecone client",
                    "PINECONE_CLIENT_ERROR",
                    {"error": str(e)}
                )
        return self._client
    
    async def _get_index(self):
        """
        Get or create Pinecone index with proper configuration.
        
        Returns:
            Index: Pinecone index instance
            
        Raises:
            VectorStoreError: If index access fails
        """
        if self._index is None:
            try:
                client = await self._get_client()
                
                # Check if index exists, create if it doesn't
                existing_indexes = client.list_indexes()
                index_names = [idx.name for idx in existing_indexes]
                
                if self.settings.pinecone_index_name not in index_names:
                    logger.info(f"Creating Pinecone index: {self.settings.pinecone_index_name}")
                    client.create_index(
                        name=self.settings.pinecone_index_name,
                        dimension=2048,  # Jina embeddings v4 dimension (corrected)
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud="aws",
                            region="us-east-1"
                        )
                    )
                    # Wait for index to be ready
                    await asyncio.sleep(10)
                
                self._index = client.Index(self.settings.pinecone_index_name)
                logger.info(f"Connected to Pinecone index: {self.settings.pinecone_index_name}")
                
            except Exception as e:
                logger.error(f"Failed to access Pinecone index: {str(e)}")
                raise VectorStoreError(
                    "Failed to access Pinecone index",
                    "PINECONE_INDEX_ERROR",
                    {"error": str(e), "index_name": self.settings.pinecone_index_name}
                )
        
        return self._index
    
    async def _retry_operation(self, operation, *args, **kwargs):
        """
        Execute operation with retry logic and exponential backoff.
        
        Args:
            operation: Function to execute
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Any: Result of the operation
            
        Raises:
            VectorStoreError: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self._max_retries):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self._max_retries - 1:
                    delay = self._retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Operation failed (attempt {attempt + 1}/{self._max_retries}), retrying in {delay}s: {str(e)}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Operation failed after {self._max_retries} attempts: {str(e)}")
        
        raise VectorStoreError(
            f"Operation failed after {self._max_retries} attempts",
            "RETRY_EXHAUSTED",
            {"last_error": str(last_exception)}
        )
    
    async def store_vectors(self, chunks: List[DocumentChunk]) -> bool:
        """
        Store document chunks with their embeddings in Pinecone.
        
        Args:
            chunks: List of document chunks with embeddings
            
        Returns:
            bool: True if storage was successful
            
        Raises:
            VectorStoreError: If storage operation fails
        """
        if not chunks:
            logger.warning("No chunks provided for storage")
            return True
        
        # Validate that all chunks have embeddings
        for chunk in chunks:
            if not chunk.embedding:
                raise VectorStoreError(
                    f"Chunk {chunk.id} missing embedding",
                    "MISSING_EMBEDDING",
                    {"chunk_id": chunk.id}
                )
        
        async def _store_operation():
            index = await self._get_index()
            
            # Prepare vectors for upsert
            vectors = []
            for chunk in chunks:
                vector_data = {
                    "id": chunk.id,
                    "values": chunk.embedding,
                    "metadata": {
                        "document_id": chunk.document_id,
                        "content": chunk.content[:1000],  # Limit content size in metadata
                        "chunk_index": chunk.chunk_index,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        **chunk.metadata
                    }
                }
                vectors.append(vector_data)
            
            # Batch upsert vectors (Pinecone handles batching internally)
            try:
                upsert_response = index.upsert(vectors=vectors)
                logger.info(f"Successfully stored {len(vectors)} vectors. Upserted count: {upsert_response.upserted_count}")
                return True
            except PineconeException as e:
                logger.error(f"Pinecone upsert failed: {str(e)}")
                raise VectorStoreError(
                    "Failed to store vectors in Pinecone",
                    "PINECONE_UPSERT_ERROR",
                    {"error": str(e), "vector_count": len(vectors)}
                )
        
        return await self._retry_operation(_store_operation)
    
    async def similarity_search(
        self, 
        query_vector: List[float], 
        document_id: Optional[str] = None,
        top_k: int = 10,
        score_threshold: float = 0.0
    ) -> List[SearchResult]:
        """
        Perform semantic search with similarity scoring and ranking.
        
        Args:
            query_vector: Query embedding vector
            document_id: Optional document ID to filter results
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List[SearchResult]: Ranked search results with similarity scores
            
        Raises:
            VectorStoreError: If search operation fails
        """
        if not query_vector:
            raise VectorStoreError(
                "Query vector cannot be empty",
                "INVALID_QUERY_VECTOR"
            )
        
        if top_k <= 0 or top_k > 100:
            raise VectorStoreError(
                "top_k must be between 1 and 100",
                "INVALID_TOP_K",
                {"top_k": top_k}
            )
        
        async def _search_operation():
            index = await self._get_index()
            
            # Prepare query parameters
            query_params = {
                "vector": query_vector,
                "top_k": top_k,
                "include_metadata": True,
                "include_values": False
            }
            
            # Add document filter if specified
            if document_id:
                query_params["filter"] = {"document_id": {"$eq": document_id}}
            
            try:
                search_response = index.query(**query_params)
                
                # Process results
                results = []
                for match in search_response.matches:
                    # Apply score threshold
                    if match.score < score_threshold:
                        continue
                    
                    metadata = match.metadata or {}
                    
                    result = SearchResult(
                        chunk_id=match.id,
                        content=metadata.get("content", ""),
                        score=float(match.score),
                        document_id=metadata.get("document_id", ""),
                        metadata={
                            k: v for k, v in metadata.items() 
                            if k not in ["content", "document_id"]
                        }
                    )
                    results.append(result)
                
                # Sort by score (highest first)
                results.sort(key=lambda x: x.score, reverse=True)
                
                logger.info(f"Similarity search returned {len(results)} results (top_k={top_k}, threshold={score_threshold})")
                return results
                
            except PineconeException as e:
                logger.error(f"Pinecone query failed: {str(e)}")
                raise VectorStoreError(
                    "Failed to perform similarity search",
                    "PINECONE_QUERY_ERROR",
                    {"error": str(e)}
                )
        
        return await self._retry_operation(_search_operation)
    
    async def delete_document_vectors(self, document_id: str) -> bool:
        """
        Delete all vectors associated with a specific document.
        
        Args:
            document_id: ID of the document whose vectors should be deleted
            
        Returns:
            bool: True if deletion was successful
            
        Raises:
            VectorStoreError: If deletion operation fails
        """
        if not document_id:
            raise VectorStoreError(
                "Document ID cannot be empty",
                "INVALID_DOCUMENT_ID"
            )
        
        async def _delete_operation():
            index = await self._get_index()
            
            try:
                # Delete vectors by document_id filter
                delete_response = index.delete(
                    filter={"document_id": {"$eq": document_id}}
                )
                
                logger.info(f"Successfully deleted vectors for document: {document_id}")
                return True
                
            except PineconeException as e:
                logger.error(f"Pinecone delete failed: {str(e)}")
                raise VectorStoreError(
                    "Failed to delete document vectors",
                    "PINECONE_DELETE_ERROR",
                    {"error": str(e), "document_id": document_id}
                )
        
        return await self._retry_operation(_delete_operation)
    
    async def delete_vectors_by_ids(self, vector_ids: List[str]) -> bool:
        """
        Delete specific vectors by their IDs.
        
        Args:
            vector_ids: List of vector IDs to delete
            
        Returns:
            bool: True if deletion was successful
            
        Raises:
            VectorStoreError: If deletion operation fails
        """
        if not vector_ids:
            logger.warning("No vector IDs provided for deletion")
            return True
        
        async def _delete_operation():
            index = await self._get_index()
            
            try:
                # Delete vectors by IDs
                delete_response = index.delete(ids=vector_ids)
                
                logger.info(f"Successfully deleted {len(vector_ids)} vectors")
                return True
                
            except PineconeException as e:
                logger.error(f"Pinecone delete by IDs failed: {str(e)}")
                raise VectorStoreError(
                    "Failed to delete vectors by IDs",
                    "PINECONE_DELETE_IDS_ERROR",
                    {"error": str(e), "vector_count": len(vector_ids)}
                )
        
        return await self._retry_operation(_delete_operation)
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Pinecone index.
        
        Returns:
            Dict[str, Any]: Index statistics including vector count and dimension
            
        Raises:
            VectorStoreError: If stats retrieval fails
        """
        async def _stats_operation():
            index = await self._get_index()
            
            try:
                stats_response = index.describe_index_stats()
                
                return {
                    "total_vector_count": stats_response.total_vector_count,
                    "dimension": stats_response.dimension,
                    "index_fullness": stats_response.index_fullness,
                    "namespaces": dict(stats_response.namespaces) if stats_response.namespaces else {}
                }
                
            except PineconeException as e:
                logger.error(f"Failed to get index stats: {str(e)}")
                raise VectorStoreError(
                    "Failed to retrieve index statistics",
                    "PINECONE_STATS_ERROR",
                    {"error": str(e)}
                )
        
        return await self._retry_operation(_stats_operation)
    
    async def health_check(self) -> bool:
        """
        Perform a health check on the Pinecone connection.
        
        Returns:
            bool: True if connection is healthy
            
        Raises:
            VectorStoreError: If health check fails
        """
        try:
            stats = await self.get_index_stats()
            logger.info("Pinecone health check passed")
            return True
        except Exception as e:
            logger.error(f"Pinecone health check failed: {str(e)}")
            raise VectorStoreError(
                "Pinecone health check failed",
                "HEALTH_CHECK_FAILED",
                {"error": str(e)}
            )
    
    async def close(self):
        """Close connections and cleanup resources."""
        if self._client:
            # Pinecone client doesn't require explicit closing
            self._client = None
            self._index = None
            logger.info("Pinecone client connections closed")


# Global vector store instance
_vector_store: Optional[PineconeVectorStore] = None


async def get_vector_store() -> PineconeVectorStore:
    """
    Get or create the global vector store instance.
    
    Returns:
        PineconeVectorStore: Configured vector store instance
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = PineconeVectorStore()
    return _vector_store


async def close_vector_store():
    """Close the global vector store instance."""
    global _vector_store
    if _vector_store:
        await _vector_store.close()
        _vector_store = None