"""
Integration tests for Pinecone vector store operations.

These tests verify the complete functionality of the vector store including
connection management, vector storage, similarity search, and error handling.
"""

import asyncio
import pytest
import os
from typing import List
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4

from app.data.vector_store import PineconeVectorStore, VectorStoreError, get_vector_store
from app.models.schemas import DocumentChunk, SearchResult
from app.config import get_settings


@pytest.fixture
def sample_chunks():
    """Create sample document chunks for testing."""
    chunks = []
    for i in range(3):
        chunk = DocumentChunk(
            id=f"chunk_{i}",
            document_id="test_doc_123",
            content=f"This is test chunk number {i} with some sample content.",
            chunk_index=i,
            start_char=i * 100,
            end_char=(i + 1) * 100,
            embedding=[0.1 * j for j in range(1024)],  # Mock embedding
            metadata={"page": i + 1, "section": f"Section {i}"}
        )
        chunks.append(chunk)
    return chunks


@pytest.fixture
def mock_pinecone_client():
    """Mock Pinecone client for testing."""
    with patch('app.data.vector_store.Pinecone') as mock_pinecone:
        mock_client = Mock()
        mock_index = Mock()
        
        # Mock client initialization
        mock_pinecone.return_value = mock_client
        mock_client.list_indexes.return_value = [Mock(name="test-index")]
        mock_client.Index.return_value = mock_index
        
        # Mock index operations
        mock_index.upsert.return_value = Mock(upserted_count=3)
        mock_index.query.return_value = Mock(
            matches=[
                Mock(
                    id="chunk_0",
                    score=0.95,
                    metadata={
                        "document_id": "test_doc_123",
                        "content": "This is test chunk number 0",
                        "chunk_index": 0,
                        "page": 1
                    }
                ),
                Mock(
                    id="chunk_1", 
                    score=0.85,
                    metadata={
                        "document_id": "test_doc_123",
                        "content": "This is test chunk number 1",
                        "chunk_index": 1,
                        "page": 2
                    }
                )
            ]
        )
        mock_index.delete.return_value = Mock()
        mock_index.describe_index_stats.return_value = Mock(
            total_vector_count=100,
            dimension=1024,
            index_fullness=0.1,
            namespaces={}
        )
        
        yield mock_client, mock_index


class TestPineconeVectorStore:
    """Test cases for PineconeVectorStore class."""
    
    @pytest.mark.asyncio
    async def test_client_initialization(self, mock_pinecone_client):
        """Test Pinecone client initialization."""
        mock_client, mock_index = mock_pinecone_client
        
        vector_store = PineconeVectorStore()
        client = await vector_store._get_client()
        
        assert client is not None
        assert vector_store._client is not None
    
    @pytest.mark.asyncio
    async def test_index_access(self, mock_pinecone_client):
        """Test Pinecone index access and creation."""
        mock_client, mock_index = mock_pinecone_client
        
        vector_store = PineconeVectorStore()
        index = await vector_store._get_index()
        
        assert index is not None
        assert vector_store._index is not None
        mock_client.list_indexes.assert_called_once()
        mock_client.Index.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_vectors_success(self, mock_pinecone_client, sample_chunks):
        """Test successful vector storage."""
        mock_client, mock_index = mock_pinecone_client
        
        vector_store = PineconeVectorStore()
        result = await vector_store.store_vectors(sample_chunks)
        
        assert result is True
        mock_index.upsert.assert_called_once()
        
        # Verify upsert was called with correct data
        call_args = mock_index.upsert.call_args
        vectors = call_args.kwargs['vectors']
        assert len(vectors) == 3
        assert vectors[0]['id'] == 'chunk_0'
        assert vectors[0]['metadata']['document_id'] == 'test_doc_123'
    
    @pytest.mark.asyncio
    async def test_store_vectors_empty_list(self, mock_pinecone_client):
        """Test storing empty list of vectors."""
        mock_client, mock_index = mock_pinecone_client
        
        vector_store = PineconeVectorStore()
        result = await vector_store.store_vectors([])
        
        assert result is True
        mock_index.upsert.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_store_vectors_missing_embedding(self, mock_pinecone_client):
        """Test error handling for chunks without embeddings."""
        mock_client, mock_index = mock_pinecone_client
        
        chunk_without_embedding = DocumentChunk(
            id="chunk_no_embedding",
            document_id="test_doc",
            content="Test content",
            chunk_index=0
        )
        
        vector_store = PineconeVectorStore()
        
        with pytest.raises(VectorStoreError) as exc_info:
            await vector_store.store_vectors([chunk_without_embedding])
        
        assert exc_info.value.error_code == "MISSING_EMBEDDING"
        assert "chunk_no_embedding" in exc_info.value.message
    
    @pytest.mark.asyncio
    async def test_similarity_search_success(self, mock_pinecone_client):
        """Test successful similarity search."""
        mock_client, mock_index = mock_pinecone_client
        
        query_vector = [0.1 * i for i in range(1024)]
        
        vector_store = PineconeVectorStore()
        results = await vector_store.similarity_search(query_vector, top_k=5)
        
        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].chunk_id == "chunk_0"
        assert results[0].score == 0.95
        assert results[0].document_id == "test_doc_123"
        
        # Verify results are sorted by score (highest first)
        assert results[0].score >= results[1].score
        
        mock_index.query.assert_called_once()
        call_args = mock_index.query.call_args
        assert call_args.kwargs['vector'] == query_vector
        assert call_args.kwargs['top_k'] == 5
    
    @pytest.mark.asyncio
    async def test_similarity_search_with_document_filter(self, mock_pinecone_client):
        """Test similarity search with document ID filter."""
        mock_client, mock_index = mock_pinecone_client
        
        query_vector = [0.1 * i for i in range(1024)]
        document_id = "specific_doc_123"
        
        vector_store = PineconeVectorStore()
        results = await vector_store.similarity_search(
            query_vector, 
            document_id=document_id,
            top_k=3
        )
        
        mock_index.query.assert_called_once()
        call_args = mock_index.query.call_args
        assert call_args.kwargs['filter'] == {"document_id": {"$eq": document_id}}
    
    @pytest.mark.asyncio
    async def test_similarity_search_with_score_threshold(self, mock_pinecone_client):
        """Test similarity search with score threshold filtering."""
        mock_client, mock_index = mock_pinecone_client
        
        query_vector = [0.1 * i for i in range(1024)]
        
        vector_store = PineconeVectorStore()
        results = await vector_store.similarity_search(
            query_vector, 
            score_threshold=0.9,
            top_k=5
        )
        
        # Only results with score >= 0.9 should be returned
        assert len(results) == 1
        assert results[0].score >= 0.9
    
    @pytest.mark.asyncio
    async def test_similarity_search_invalid_parameters(self, mock_pinecone_client):
        """Test similarity search with invalid parameters."""
        mock_client, mock_index = mock_pinecone_client
        
        vector_store = PineconeVectorStore()
        
        # Test empty query vector
        with pytest.raises(VectorStoreError) as exc_info:
            await vector_store.similarity_search([])
        assert exc_info.value.error_code == "INVALID_QUERY_VECTOR"
        
        # Test invalid top_k
        with pytest.raises(VectorStoreError) as exc_info:
            await vector_store.similarity_search([0.1] * 1024, top_k=0)
        assert exc_info.value.error_code == "INVALID_TOP_K"
        
        with pytest.raises(VectorStoreError) as exc_info:
            await vector_store.similarity_search([0.1] * 1024, top_k=101)
        assert exc_info.value.error_code == "INVALID_TOP_K"
    
    @pytest.mark.asyncio
    async def test_delete_document_vectors_success(self, mock_pinecone_client):
        """Test successful deletion of document vectors."""
        mock_client, mock_index = mock_pinecone_client
        
        document_id = "test_doc_123"
        
        vector_store = PineconeVectorStore()
        result = await vector_store.delete_document_vectors(document_id)
        
        assert result is True
        mock_index.delete.assert_called_once()
        call_args = mock_index.delete.call_args
        assert call_args.kwargs['filter'] == {"document_id": {"$eq": document_id}}
    
    @pytest.mark.asyncio
    async def test_delete_document_vectors_empty_id(self, mock_pinecone_client):
        """Test error handling for empty document ID."""
        mock_client, mock_index = mock_pinecone_client
        
        vector_store = PineconeVectorStore()
        
        with pytest.raises(VectorStoreError) as exc_info:
            await vector_store.delete_document_vectors("")
        
        assert exc_info.value.error_code == "INVALID_DOCUMENT_ID"
    
    @pytest.mark.asyncio
    async def test_delete_vectors_by_ids_success(self, mock_pinecone_client):
        """Test successful deletion of vectors by IDs."""
        mock_client, mock_index = mock_pinecone_client
        
        vector_ids = ["chunk_1", "chunk_2", "chunk_3"]
        
        vector_store = PineconeVectorStore()
        result = await vector_store.delete_vectors_by_ids(vector_ids)
        
        assert result is True
        mock_index.delete.assert_called_once()
        call_args = mock_index.delete.call_args
        assert call_args.kwargs['ids'] == vector_ids
    
    @pytest.mark.asyncio
    async def test_delete_vectors_by_ids_empty_list(self, mock_pinecone_client):
        """Test deletion with empty vector ID list."""
        mock_client, mock_index = mock_pinecone_client
        
        vector_store = PineconeVectorStore()
        result = await vector_store.delete_vectors_by_ids([])
        
        assert result is True
        mock_index.delete.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_index_stats_success(self, mock_pinecone_client):
        """Test successful retrieval of index statistics."""
        mock_client, mock_index = mock_pinecone_client
        
        vector_store = PineconeVectorStore()
        stats = await vector_store.get_index_stats()
        
        assert stats['total_vector_count'] == 100
        assert stats['dimension'] == 1024
        assert stats['index_fullness'] == 0.1
        assert 'namespaces' in stats
        
        mock_index.describe_index_stats.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_pinecone_client):
        """Test successful health check."""
        mock_client, mock_index = mock_pinecone_client
        
        vector_store = PineconeVectorStore()
        result = await vector_store.health_check()
        
        assert result is True
        mock_index.describe_index_stats.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, mock_pinecone_client):
        """Test retry mechanism for failed operations."""
        mock_client, mock_index = mock_pinecone_client
        
        # Mock failure followed by success
        mock_index.describe_index_stats.side_effect = [
            Exception("Connection failed"),
            Exception("Still failing"),
            Mock(total_vector_count=100, dimension=1024, index_fullness=0.1, namespaces={})
        ]
        
        vector_store = PineconeVectorStore()
        vector_store._max_retries = 3
        vector_store._retry_delay = 0.01  # Fast retry for testing
        
        stats = await vector_store.get_index_stats()
        
        assert stats['total_vector_count'] == 100
        assert mock_index.describe_index_stats.call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_exhaustion(self, mock_pinecone_client):
        """Test behavior when all retry attempts are exhausted."""
        mock_client, mock_index = mock_pinecone_client
        
        # Mock consistent failures
        mock_index.describe_index_stats.side_effect = Exception("Persistent failure")
        
        vector_store = PineconeVectorStore()
        vector_store._max_retries = 2
        vector_store._retry_delay = 0.01  # Fast retry for testing
        
        with pytest.raises(VectorStoreError) as exc_info:
            await vector_store.get_index_stats()
        
        assert exc_info.value.error_code == "RETRY_EXHAUSTED"
        assert mock_index.describe_index_stats.call_count == 2
    
    @pytest.mark.asyncio
    async def test_close_connections(self, mock_pinecone_client):
        """Test closing vector store connections."""
        mock_client, mock_index = mock_pinecone_client
        
        vector_store = PineconeVectorStore()
        await vector_store._get_client()  # Initialize client
        
        assert vector_store._client is not None
        
        await vector_store.close()
        
        assert vector_store._client is None
        assert vector_store._index is None


class TestVectorStoreGlobalFunctions:
    """Test cases for global vector store functions."""
    
    @pytest.mark.asyncio
    async def test_get_vector_store_singleton(self, mock_pinecone_client):
        """Test that get_vector_store returns singleton instance."""
        mock_client, mock_index = mock_pinecone_client
        
        # Clear any existing global instance
        import app.data.vector_store
        app.data.vector_store._vector_store = None
        
        store1 = await get_vector_store()
        store2 = await get_vector_store()
        
        assert store1 is store2
        assert isinstance(store1, PineconeVectorStore)
    
    @pytest.mark.asyncio
    async def test_close_vector_store_global(self, mock_pinecone_client):
        """Test closing global vector store instance."""
        mock_client, mock_index = mock_pinecone_client
        
        from app.data.vector_store import close_vector_store
        
        # Get instance first
        store = await get_vector_store()
        assert store is not None
        
        # Close it
        await close_vector_store()
        
        # Verify global instance is cleared
        import app.data.vector_store
        assert app.data.vector_store._vector_store is None


class TestVectorStoreErrorHandling:
    """Test cases for error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_pinecone_client_initialization_error(self):
        """Test error handling during client initialization."""
        with patch('app.data.vector_store.Pinecone') as mock_pinecone:
            mock_pinecone.side_effect = Exception("API key invalid")
            
            vector_store = PineconeVectorStore()
            
            with pytest.raises(VectorStoreError) as exc_info:
                await vector_store._get_client()
            
            assert exc_info.value.error_code == "PINECONE_CLIENT_ERROR"
            assert "API key invalid" in str(exc_info.value.details)
    
    @pytest.mark.asyncio
    async def test_pinecone_index_access_error(self, mock_pinecone_client):
        """Test error handling during index access."""
        mock_client, mock_index = mock_pinecone_client
        mock_client.list_indexes.side_effect = Exception("Index access failed")
        
        vector_store = PineconeVectorStore()
        
        with pytest.raises(VectorStoreError) as exc_info:
            await vector_store._get_index()
        
        assert exc_info.value.error_code == "PINECONE_INDEX_ERROR"
    
    @pytest.mark.asyncio
    async def test_upsert_operation_error(self, mock_pinecone_client, sample_chunks):
        """Test error handling during vector upsert."""
        mock_client, mock_index = mock_pinecone_client
        
        from pinecone.exceptions import PineconeException
        mock_index.upsert.side_effect = PineconeException("Upsert failed")
        
        vector_store = PineconeVectorStore()
        vector_store._max_retries = 1  # Fast failure for testing
        
        with pytest.raises(VectorStoreError) as exc_info:
            await vector_store.store_vectors(sample_chunks)
        
        assert exc_info.value.error_code == "RETRY_EXHAUSTED"
    
    @pytest.mark.asyncio
    async def test_query_operation_error(self, mock_pinecone_client):
        """Test error handling during similarity search."""
        mock_client, mock_index = mock_pinecone_client
        
        from pinecone.exceptions import PineconeException
        mock_index.query.side_effect = PineconeException("Query failed")
        
        vector_store = PineconeVectorStore()
        vector_store._max_retries = 1  # Fast failure for testing
        
        query_vector = [0.1 * i for i in range(1024)]
        
        with pytest.raises(VectorStoreError) as exc_info:
            await vector_store.similarity_search(query_vector)
        
        assert exc_info.value.error_code == "RETRY_EXHAUSTED"
    
    @pytest.mark.asyncio
    async def test_delete_operation_error(self, mock_pinecone_client):
        """Test error handling during vector deletion."""
        mock_client, mock_index = mock_pinecone_client
        
        from pinecone.exceptions import PineconeException
        mock_index.delete.side_effect = PineconeException("Delete failed")
        
        vector_store = PineconeVectorStore()
        vector_store._max_retries = 1  # Fast failure for testing
        
        with pytest.raises(VectorStoreError) as exc_info:
            await vector_store.delete_document_vectors("test_doc")
        
        assert exc_info.value.error_code == "RETRY_EXHAUSTED"


if __name__ == "__main__":
    pytest.main([__file__])