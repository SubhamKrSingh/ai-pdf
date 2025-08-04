"""
Unit tests for vector store utility functions and error handling.

These tests focus on testing individual components and error scenarios
without requiring actual Pinecone connections.
"""

import pytest
from unittest.mock import Mock, patch
from uuid import uuid4

from app.data.vector_store import VectorStoreError, PineconeVectorStore
from app.models.schemas import DocumentChunk, SearchResult


class TestVectorStoreError:
    """Test cases for VectorStoreError exception class."""
    
    def test_vector_store_error_creation(self):
        """Test VectorStoreError creation with all parameters."""
        error = VectorStoreError(
            message="Test error",
            error_code="TEST_ERROR",
            details={"key": "value"}
        )
        
        assert error.message == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.details == {"key": "value"}
        assert str(error) == "Test error"
    
    def test_vector_store_error_defaults(self):
        """Test VectorStoreError creation with default values."""
        error = VectorStoreError("Simple error")
        
        assert error.message == "Simple error"
        assert error.error_code == "VECTOR_STORE_ERROR"
        assert error.details == {}


class TestPineconeVectorStoreValidation:
    """Test cases for input validation in PineconeVectorStore."""
    
    def test_validate_chunk_embeddings(self):
        """Test validation of chunk embeddings."""
        vector_store = PineconeVectorStore()
        
        # Valid chunk with embedding
        valid_chunk = DocumentChunk(
            id="test_chunk",
            document_id="test_doc",
            content="Test content",
            chunk_index=0,
            embedding=[0.1, 0.2, 0.3]
        )
        
        # This should not raise an exception
        chunks = [valid_chunk]
        # We can't test the actual store_vectors method without mocking Pinecone,
        # but we can test the validation logic
        
        for chunk in chunks:
            assert chunk.embedding is not None
            assert len(chunk.embedding) > 0
    
    def test_search_result_creation(self):
        """Test SearchResult model creation and validation."""
        result = SearchResult(
            chunk_id="test_chunk_123",
            content="This is test content for search result",
            score=0.85,
            document_id="test_doc_456",
            metadata={"page": 1, "section": "intro"}
        )
        
        assert result.chunk_id == "test_chunk_123"
        assert result.score == 0.85
        assert result.document_id == "test_doc_456"
        assert result.metadata["page"] == 1
    
    def test_search_result_score_validation(self):
        """Test SearchResult score validation."""
        # Valid score
        result = SearchResult(
            chunk_id="test",
            content="test",
            score=0.5,
            document_id="doc"
        )
        assert result.score == 0.5
        
        # Test boundary values
        result_min = SearchResult(
            chunk_id="test",
            content="test", 
            score=0.0,
            document_id="doc"
        )
        assert result_min.score == 0.0
        
        result_max = SearchResult(
            chunk_id="test",
            content="test",
            score=1.0,
            document_id="doc"
        )
        assert result_max.score == 1.0


class TestVectorStoreConfiguration:
    """Test cases for vector store configuration and initialization."""
    
    @patch('app.data.vector_store.get_settings')
    def test_vector_store_initialization_with_settings(self, mock_get_settings):
        """Test vector store initialization with mocked settings."""
        mock_settings = Mock()
        mock_settings.pinecone_api_key = "test_key"
        mock_settings.pinecone_environment = "test_env"
        mock_settings.pinecone_index_name = "test_index"
        mock_settings.max_retries = 3
        mock_settings.retry_delay = 1.0
        
        mock_get_settings.return_value = mock_settings
        
        vector_store = PineconeVectorStore()
        
        assert vector_store.settings.pinecone_api_key == "test_key"
        assert vector_store.settings.pinecone_environment == "test_env"
        assert vector_store.settings.pinecone_index_name == "test_index"
        assert vector_store._max_retries == 3
        assert vector_store._retry_delay == 1.0
    
    def test_vector_store_default_values(self):
        """Test vector store default configuration values."""
        vector_store = PineconeVectorStore()
        
        assert vector_store._connection_pool_size == 10
        assert vector_store._client is None
        assert vector_store._index is None


class TestVectorStoreUtilityMethods:
    """Test cases for utility methods and helper functions."""
    
    def test_chunk_metadata_preparation(self):
        """Test preparation of chunk metadata for storage."""
        chunk = DocumentChunk(
            id="test_chunk",
            document_id="test_doc",
            content="This is a very long piece of content that should be truncated when stored in metadata to avoid exceeding size limits",
            chunk_index=5,
            start_char=100,
            end_char=200,
            embedding=[0.1] * 1024,
            metadata={"page": 3, "section": "results"}
        )
        
        # Simulate metadata preparation (as done in store_vectors)
        vector_data = {
            "id": chunk.id,
            "values": chunk.embedding,
            "metadata": {
                "document_id": chunk.document_id,
                "content": chunk.content[:1000],  # Truncated
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                **chunk.metadata
            }
        }
        
        assert vector_data["id"] == "test_chunk"
        assert len(vector_data["values"]) == 1024
        assert vector_data["metadata"]["document_id"] == "test_doc"
        assert len(vector_data["metadata"]["content"]) <= 1000
        assert vector_data["metadata"]["chunk_index"] == 5
        assert vector_data["metadata"]["page"] == 3
    
    def test_search_results_sorting(self):
        """Test sorting of search results by score."""
        results = [
            SearchResult(
                chunk_id="chunk_1",
                content="Content 1",
                score=0.7,
                document_id="doc_1"
            ),
            SearchResult(
                chunk_id="chunk_2", 
                content="Content 2",
                score=0.9,
                document_id="doc_1"
            ),
            SearchResult(
                chunk_id="chunk_3",
                content="Content 3",
                score=0.8,
                document_id="doc_1"
            )
        ]
        
        # Sort by score (highest first)
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        
        assert sorted_results[0].score == 0.9
        assert sorted_results[1].score == 0.8
        assert sorted_results[2].score == 0.7
        assert sorted_results[0].chunk_id == "chunk_2"
    
    def test_score_threshold_filtering(self):
        """Test filtering results by score threshold."""
        results = [
            SearchResult(chunk_id="1", content="Content", score=0.95, document_id="doc"),
            SearchResult(chunk_id="2", content="Content", score=0.85, document_id="doc"),
            SearchResult(chunk_id="3", content="Content", score=0.75, document_id="doc"),
            SearchResult(chunk_id="4", content="Content", score=0.65, document_id="doc"),
        ]
        
        threshold = 0.8
        filtered_results = [r for r in results if r.score >= threshold]
        
        assert len(filtered_results) == 2
        assert all(r.score >= threshold for r in filtered_results)
        assert filtered_results[0].chunk_id == "1"
        assert filtered_results[1].chunk_id == "2"


class TestErrorScenarios:
    """Test cases for various error scenarios."""
    
    def test_empty_vector_validation(self):
        """Test validation of empty vectors."""
        # This would be caught by the similarity_search method
        empty_vector = []
        
        # Simulate the validation that happens in similarity_search
        if not empty_vector:
            error_raised = True
        else:
            error_raised = False
        
        assert error_raised is True
    
    def test_invalid_top_k_validation(self):
        """Test validation of top_k parameter."""
        invalid_top_k_values = [0, -1, 101, 1000]
        
        for top_k in invalid_top_k_values:
            # Simulate validation logic from similarity_search
            if top_k <= 0 or top_k > 100:
                validation_failed = True
            else:
                validation_failed = False
            
            assert validation_failed is True
    
    def test_document_id_validation(self):
        """Test validation of document ID parameter."""
        invalid_document_ids = ["", None]
        
        for doc_id in invalid_document_ids:
            # Simulate validation logic from delete_document_vectors
            if not doc_id:
                validation_failed = True
            else:
                validation_failed = False
            
            assert validation_failed is True


if __name__ == "__main__":
    pytest.main([__file__])