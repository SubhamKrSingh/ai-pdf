"""
Integration tests for the document processing service.

Tests the complete document processing workflow including download, parsing,
chunking, embedding generation, and storage operations.
"""

import asyncio
import pytest
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from app.services.document_service import (
    DocumentProcessingService, 
    ProcessingStage, 
    ProcessingStatus,
    DocumentProcessingError,
    get_document_service,
    process_document_from_url
)
from app.models.schemas import DocumentChunk
from app.utils.document_downloader import DocumentDownloadError
from app.utils.parsers.document_parser import DocumentParseError
from app.services.embedding_service import EmbeddingServiceError
from app.data.vector_store import VectorStoreError
from app.data.repository import DatabaseError


class TestDocumentProcessingService:
    """Test suite for DocumentProcessingService."""
    
    @pytest.fixture
    def service(self):
        """Create a document processing service instance."""
        with patch('app.services.document_service.get_settings') as mock_settings:
            # Mock settings to avoid configuration validation errors
            mock_settings.return_value = MagicMock(
                request_timeout=30,
                max_document_size_mb=50,
                max_chunk_size=1000,
                chunk_overlap=200,
                jina_model="jina-embeddings-v4"
            )
            return DocumentProcessingService()
    
    @pytest.fixture
    def mock_progress_callback(self):
        """Create a mock progress callback."""
        return MagicMock()
    
    @pytest.fixture
    def sample_pdf_content(self):
        """Create sample PDF content for testing."""
        # This would be actual PDF bytes in a real test
        return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
    
    @pytest.fixture
    def sample_text_content(self):
        """Create sample text content."""
        return "This is a sample document with multiple sentences. It contains various information that will be processed and chunked. The content is designed to test the document processing pipeline thoroughly."
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample document chunks."""
        return [
            DocumentChunk(
                id="chunk_1",
                document_id="doc_123",
                content="This is the first chunk of text.",
                chunk_index=0,
                start_char=0,
                end_char=35,
                metadata={"section": "intro"}
            ),
            DocumentChunk(
                id="chunk_2", 
                document_id="doc_123",
                content="This is the second chunk of text.",
                chunk_index=1,
                start_char=36,
                end_char=71,
                metadata={"section": "body"}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_successful_document_processing(
        self, 
        service, 
        mock_progress_callback,
        sample_pdf_content,
        sample_text_content
    ):
        """Test successful end-to-end document processing."""
        test_url = "https://example.com/test.pdf"
        
        # Mock all dependencies
        with patch.object(service.downloader, 'download_document') as mock_download, \
             patch.object(service.parser, 'parse_document') as mock_parse, \
             patch.object(service.chunker, 'chunk_text') as mock_chunk, \
             patch('app.services.document_service.get_embedding_service') as mock_get_embedding, \
             patch('app.services.document_service.get_vector_store') as mock_get_vector, \
             patch('app.services.document_service.get_repository') as mock_get_repo:
            
            # Setup mocks
            mock_download.return_value = (sample_pdf_content, "application/pdf")
            mock_parse.return_value = sample_text_content
            
            # Create mock chunks
            mock_chunks = [
                DocumentChunk(
                    id="chunk_1",
                    document_id="test_doc",
                    content="First chunk",
                    chunk_index=0,
                    start_char=0,
                    end_char=11,
                    metadata={}
                ),
                DocumentChunk(
                    id="chunk_2",
                    document_id="test_doc", 
                    content="Second chunk",
                    chunk_index=1,
                    start_char=12,
                    end_char=24,
                    metadata={}
                )
            ]
            mock_chunk.return_value = mock_chunks
            
            # Mock embedding service
            mock_embedding_service = AsyncMock()
            mock_embedding_service.generate_embeddings.return_value = [
                [0.1, 0.2, 0.3],  # First embedding
                [0.4, 0.5, 0.6]   # Second embedding
            ]
            mock_get_embedding.return_value = mock_embedding_service
            
            # Mock vector store
            mock_vector_store = AsyncMock()
            mock_vector_store.store_vectors.return_value = True
            mock_get_vector.return_value = mock_vector_store
            
            # Mock repository
            mock_repository = AsyncMock()
            mock_repository.store_document_metadata.return_value = True
            mock_get_repo.return_value = mock_repository
            
            # Process document
            document_id = await service.process_document(test_url, mock_progress_callback)
            
            # Verify result
            assert document_id is not None
            assert len(document_id) > 0
            
            # Verify all stages were called
            mock_download.assert_called_once_with(test_url)
            mock_parse.assert_called_once_with(sample_pdf_content, "application/pdf")
            mock_chunk.assert_called_once()
            mock_embedding_service.generate_embeddings.assert_called_once()
            mock_vector_store.store_vectors.assert_called_once()
            mock_repository.store_document_metadata.assert_called_once()
            
            # Verify progress callbacks were made
            assert mock_progress_callback.call_count >= 5  # At least 5 stage updates
            
            # Verify final status is completed
            final_call = mock_progress_callback.call_args_list[-1]
            final_status = final_call[0][0]
            assert final_status.stage == ProcessingStage.COMPLETED
            assert final_status.progress_percent == 100.0
    
    @pytest.mark.asyncio
    async def test_download_failure(self, service, mock_progress_callback):
        """Test handling of download failures."""
        test_url = "https://example.com/nonexistent.pdf"
        
        with patch.object(service.downloader, 'download_document') as mock_download:
            mock_download.side_effect = DocumentDownloadError("File not found")
            
            with pytest.raises(DocumentProcessingError) as exc_info:
                await service.process_document(test_url, mock_progress_callback)
            
            assert exc_info.value.stage == ProcessingStage.DOWNLOADING
            assert "Document download failed" in str(exc_info.value)
            
            # Verify error callback was made
            error_calls = [call for call in mock_progress_callback.call_args_list 
                          if call[0][0].stage == ProcessingStage.FAILED]
            assert len(error_calls) > 0
    
    @pytest.mark.asyncio
    async def test_parsing_failure(self, service, mock_progress_callback, sample_pdf_content):
        """Test handling of parsing failures."""
        test_url = "https://example.com/corrupt.pdf"
        
        with patch.object(service.downloader, 'download_document') as mock_download, \
             patch.object(service.parser, 'parse_document') as mock_parse:
            
            mock_download.return_value = (sample_pdf_content, "application/pdf")
            mock_parse.side_effect = DocumentParseError("Corrupt PDF file")
            
            with pytest.raises(DocumentProcessingError) as exc_info:
                await service.process_document(test_url, mock_progress_callback)
            
            assert exc_info.value.stage == ProcessingStage.PARSING
            assert "Document parsing failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_embedding_failure(
        self, 
        service, 
        mock_progress_callback,
        sample_pdf_content,
        sample_text_content,
        sample_chunks
    ):
        """Test handling of embedding generation failures."""
        test_url = "https://example.com/test.pdf"
        
        with patch.object(service.downloader, 'download_document') as mock_download, \
             patch.object(service.parser, 'parse_document') as mock_parse, \
             patch.object(service.chunker, 'chunk_text') as mock_chunk, \
             patch('app.services.document_service.get_embedding_service') as mock_get_embedding:
            
            mock_download.return_value = (sample_pdf_content, "application/pdf")
            mock_parse.return_value = sample_text_content
            mock_chunk.return_value = sample_chunks
            
            # Mock embedding service failure
            mock_embedding_service = AsyncMock()
            mock_embedding_service.generate_embeddings.side_effect = EmbeddingServiceError(
                "API rate limit exceeded", "RATE_LIMIT_ERROR"
            )
            mock_get_embedding.return_value = mock_embedding_service
            
            with pytest.raises(DocumentProcessingError) as exc_info:
                await service.process_document(test_url, mock_progress_callback)
            
            assert exc_info.value.stage == ProcessingStage.EMBEDDING
            assert "Embedding generation failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_vector_storage_failure(
        self,
        service,
        mock_progress_callback,
        sample_pdf_content,
        sample_text_content
    ):
        """Test handling of vector storage failures."""
        test_url = "https://example.com/test.pdf"
        
        with patch.object(service.downloader, 'download_document') as mock_download, \
             patch.object(service.parser, 'parse_document') as mock_parse, \
             patch.object(service.chunker, 'chunk_text') as mock_chunk, \
             patch('app.services.document_service.get_embedding_service') as mock_get_embedding, \
             patch('app.services.document_service.get_vector_store') as mock_get_vector:
            
            mock_download.return_value = (sample_pdf_content, "application/pdf")
            mock_parse.return_value = sample_text_content
            
            # Create chunks with embeddings
            chunks_with_embeddings = [
                DocumentChunk(
                    id="chunk_1",
                    document_id="test_doc",
                    content="Test chunk",
                    chunk_index=0,
                    start_char=0,
                    end_char=10,
                    metadata={},
                    embedding=[0.1, 0.2, 0.3]
                )
            ]
            mock_chunk.return_value = chunks_with_embeddings
            
            # Mock embedding service
            mock_embedding_service = AsyncMock()
            mock_embedding_service.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
            mock_get_embedding.return_value = mock_embedding_service
            
            # Mock vector store failure
            mock_vector_store = AsyncMock()
            mock_vector_store.store_vectors.side_effect = VectorStoreError(
                "Pinecone connection failed", "CONNECTION_ERROR"
            )
            mock_get_vector.return_value = mock_vector_store
            
            with pytest.raises(DocumentProcessingError) as exc_info:
                await service.process_document(test_url, mock_progress_callback)
            
            assert exc_info.value.stage == ProcessingStage.STORING
            assert "Vector storage failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_progress_tracking(self, service, sample_pdf_content, sample_text_content):
        """Test that progress is tracked correctly throughout processing."""
        test_url = "https://example.com/test.pdf"
        progress_updates = []
        
        def progress_callback(status: ProcessingStatus):
            progress_updates.append({
                'stage': status.stage,
                'progress': status.progress_percent,
                'elapsed_time': status.get_elapsed_time()
            })
        
        with patch.object(service.downloader, 'download_document') as mock_download, \
             patch.object(service.parser, 'parse_document') as mock_parse, \
             patch.object(service.chunker, 'chunk_text') as mock_chunk, \
             patch('app.services.document_service.get_embedding_service') as mock_get_embedding, \
             patch('app.services.document_service.get_vector_store') as mock_get_vector, \
             patch('app.services.document_service.get_repository') as mock_get_repo:
            
            # Setup all mocks for successful processing
            mock_download.return_value = (sample_pdf_content, "application/pdf")
            mock_parse.return_value = sample_text_content
            mock_chunk.return_value = [
                DocumentChunk(
                    id="chunk_1",
                    document_id="test_doc",
                    content="Test chunk",
                    chunk_index=0,
                    start_char=0,
                    end_char=10,
                    metadata={}
                )
            ]
            
            mock_embedding_service = AsyncMock()
            mock_embedding_service.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
            mock_get_embedding.return_value = mock_embedding_service
            
            mock_vector_store = AsyncMock()
            mock_vector_store.store_vectors.return_value = True
            mock_get_vector.return_value = mock_vector_store
            
            mock_repository = AsyncMock()
            mock_repository.store_document_metadata.return_value = True
            mock_get_repo.return_value = mock_repository
            
            # Process document with progress tracking
            document_id = await service.process_document(test_url, progress_callback)
            
            # Verify progress updates
            assert len(progress_updates) >= 5  # At least 5 stage updates
            
            # Verify progress increases
            for i in range(1, len(progress_updates)):
                current_progress = progress_updates[i]['progress']
                previous_progress = progress_updates[i-1]['progress']
                # Progress should generally increase (allowing for same values during stage transitions)
                assert current_progress >= previous_progress
            
            # Verify final progress is 100%
            assert progress_updates[-1]['progress'] == 100.0
            assert progress_updates[-1]['stage'] == ProcessingStage.COMPLETED
            
            # Verify all expected stages are present
            stages_seen = {update['stage'] for update in progress_updates}
            expected_stages = {
                ProcessingStage.DOWNLOADING,
                ProcessingStage.PARSING,
                ProcessingStage.CHUNKING,
                ProcessingStage.EMBEDDING,
                ProcessingStage.STORING,
                ProcessingStage.COMPLETED
            }
            assert expected_stages.issubset(stages_seen)
    
    @pytest.mark.asyncio
    async def test_session_tracking(self, service):
        """Test that processing sessions are tracked correctly."""
        test_url = "https://example.com/test.pdf"
        
        # Initially no active sessions
        assert len(service.get_active_sessions()) == 0
        
        with patch.object(service.downloader, 'download_document') as mock_download:
            # Make download hang to test session tracking
            download_event = asyncio.Event()
            
            async def hanging_download(url):
                await download_event.wait()
                return b"test content", "text/plain"
            
            mock_download.side_effect = hanging_download
            
            # Start processing (will hang at download)
            process_task = asyncio.create_task(service.process_document(test_url))
            
            # Give it a moment to start
            await asyncio.sleep(0.1)
            
            # Verify session is tracked
            active_sessions = service.get_active_sessions()
            assert len(active_sessions) == 1
            
            document_id = list(active_sessions.keys())[0]
            status = active_sessions[document_id]
            assert status.stage == ProcessingStage.DOWNLOADING
            
            # Test getting specific session status
            retrieved_status = service.get_processing_status(document_id)
            assert retrieved_status is not None
            assert retrieved_status.document_id == document_id
            
            # Cancel processing
            cancelled = await service.cancel_processing(document_id)
            assert cancelled is True
            
            # Verify session is cleaned up
            assert len(service.get_active_sessions()) == 0
            
            # Release the hanging download and clean up
            download_event.set()
            try:
                await process_task
            except DocumentProcessingError:
                pass  # Expected due to cancellation
    
    @pytest.mark.asyncio
    async def test_reprocess_document(self, service, sample_pdf_content, sample_text_content):
        """Test document reprocessing functionality."""
        test_url = "https://example.com/test.pdf"
        existing_document_id = "existing_doc_123"
        
        with patch.object(service.downloader, 'download_document') as mock_download, \
             patch.object(service.parser, 'parse_document') as mock_parse, \
             patch.object(service.chunker, 'chunk_text') as mock_chunk, \
             patch('app.services.document_service.get_embedding_service') as mock_get_embedding, \
             patch('app.services.document_service.get_vector_store') as mock_get_vector, \
             patch('app.services.document_service.get_repository') as mock_get_repo:
            
            # Setup mocks
            mock_download.return_value = (sample_pdf_content, "application/pdf")
            mock_parse.return_value = sample_text_content
            mock_chunk.return_value = [
                DocumentChunk(
                    id="chunk_1",
                    document_id=existing_document_id,
                    content="Reprocessed chunk",
                    chunk_index=0,
                    start_char=0,
                    end_char=17,
                    metadata={}
                )
            ]
            
            mock_embedding_service = AsyncMock()
            mock_embedding_service.generate_embeddings.return_value = [[0.7, 0.8, 0.9]]
            mock_get_embedding.return_value = mock_embedding_service
            
            mock_vector_store = AsyncMock()
            mock_vector_store.delete_document_vectors.return_value = True
            mock_vector_store.store_vectors.return_value = True
            mock_get_vector.return_value = mock_vector_store
            
            mock_repository = AsyncMock()
            mock_repository.store_document_metadata.return_value = True
            mock_get_repo.return_value = mock_repository
            
            # Reprocess document
            result_document_id = await service.reprocess_document(existing_document_id, test_url)
            
            # Verify result
            assert result_document_id == existing_document_id
            
            # Verify cleanup was called
            mock_vector_store.delete_document_vectors.assert_called_once_with(existing_document_id)
            
            # Verify processing pipeline was executed
            mock_download.assert_called_once_with(test_url)
            mock_parse.assert_called_once()
            mock_chunk.assert_called_once()
            mock_embedding_service.generate_embeddings.assert_called_once()
            mock_vector_store.store_vectors.assert_called_once()
            mock_repository.store_document_metadata.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check(self, service):
        """Test service health check functionality."""
        with patch('app.services.document_service.get_embedding_service') as mock_get_embedding, \
             patch('app.services.document_service.get_vector_store') as mock_get_vector, \
             patch('app.services.document_service.get_repository') as mock_get_repo:
            
            # Mock healthy components
            mock_embedding_service = AsyncMock()
            mock_embedding_service.get_cache_stats.return_value = {"total_entries": 10}
            mock_get_embedding.return_value = mock_embedding_service
            
            mock_vector_store = AsyncMock()
            mock_vector_store.health_check.return_value = True
            mock_vector_store.get_index_stats.return_value = {"total_vector_count": 100}
            mock_get_vector.return_value = mock_vector_store
            
            mock_repository = AsyncMock()
            mock_repository.health_check.return_value = {"status": "healthy"}
            mock_get_repo.return_value = mock_repository
            
            # Perform health check
            health_status = await service.health_check()
            
            # Verify results
            assert health_status["service"] == "document_processing"
            assert health_status["status"] == "healthy"
            assert "components" in health_status
            assert "embedding_service" in health_status["components"]
            assert "vector_store" in health_status["components"]
            assert "repository" in health_status["components"]
            
            # Verify all components are healthy
            for component in health_status["components"].values():
                assert component["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_health_check_with_failures(self, service):
        """Test health check with component failures."""
        with patch('app.services.document_service.get_embedding_service') as mock_get_embedding, \
             patch('app.services.document_service.get_vector_store') as mock_get_vector, \
             patch('app.services.document_service.get_repository') as mock_get_repo:
            
            # Mock embedding service failure
            mock_get_embedding.side_effect = Exception("Embedding service unavailable")
            
            # Mock healthy vector store
            mock_vector_store = AsyncMock()
            mock_vector_store.health_check.return_value = True
            mock_vector_store.get_index_stats.return_value = {"total_vector_count": 100}
            mock_get_vector.return_value = mock_vector_store
            
            # Mock healthy repository
            mock_repository = AsyncMock()
            mock_repository.health_check.return_value = {"status": "healthy"}
            mock_get_repo.return_value = mock_repository
            
            # Perform health check
            health_status = await service.health_check()
            
            # Verify overall status is degraded
            assert health_status["status"] == "degraded"
            
            # Verify embedding service is marked unhealthy
            assert health_status["components"]["embedding_service"]["status"] == "unhealthy"
            assert "error" in health_status["components"]["embedding_service"]
            
            # Verify other components are still healthy
            assert health_status["components"]["vector_store"]["status"] == "healthy"
            assert health_status["components"]["repository"]["status"] == "healthy"


class TestGlobalFunctions:
    """Test global utility functions."""
    
    def test_get_document_service_singleton(self):
        """Test that get_document_service returns the same instance."""
        with patch('app.services.document_service.get_settings') as mock_settings:
            # Mock settings to avoid configuration validation errors
            mock_settings.return_value = MagicMock(
                request_timeout=30,
                max_document_size_mb=50,
                max_chunk_size=1000,
                chunk_overlap=200,
                jina_model="jina-embeddings-v4"
            )
            
            # Reset the global service instance for clean test
            import app.services.document_service
            app.services.document_service._document_service = None
            
            service1 = get_document_service()
            service2 = get_document_service()
            
            assert service1 is service2
            assert isinstance(service1, DocumentProcessingService)
    
    @pytest.mark.asyncio
    async def test_process_document_from_url(self):
        """Test the convenience function for processing documents."""
        test_url = "https://example.com/test.pdf"
        
        with patch('app.services.document_service.get_document_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.process_document.return_value = "doc_123"
            mock_get_service.return_value = mock_service
            
            # Test without callback
            result = await process_document_from_url(test_url)
            assert result == "doc_123"
            mock_service.process_document.assert_called_once_with(test_url, None)
            
            # Test with callback
            mock_service.reset_mock()
            callback = MagicMock()
            result = await process_document_from_url(test_url, callback)
            assert result == "doc_123"
            mock_service.process_document.assert_called_once_with(test_url, callback)


class TestProcessingStatus:
    """Test ProcessingStatus class."""
    
    def test_processing_status_initialization(self):
        """Test ProcessingStatus initialization."""
        document_id = "test_doc_123"
        status = ProcessingStatus(document_id)
        
        assert status.document_id == document_id
        assert status.stage == ProcessingStage.INITIALIZING
        assert status.progress_percent == 0.0
        assert status.error_message is None
        assert isinstance(status.metadata, dict)
        assert len(status.callbacks) == 0
    
    def test_stage_updates(self):
        """Test stage update functionality."""
        status = ProcessingStatus("test_doc")
        callback = MagicMock()
        status.add_callback(callback)
        
        # Update stage
        status.update_stage(ProcessingStage.DOWNLOADING, 25.0)
        
        assert status.stage == ProcessingStage.DOWNLOADING
        assert status.progress_percent == 25.0
        callback.assert_called_once_with(status)
        
        # Update without progress
        callback.reset_mock()
        status.update_stage(ProcessingStage.PARSING)
        
        assert status.stage == ProcessingStage.PARSING
        assert status.progress_percent == 25.0  # Should remain unchanged
        callback.assert_called_once_with(status)
    
    def test_error_handling(self):
        """Test error state handling."""
        status = ProcessingStatus("test_doc")
        callback = MagicMock()
        status.add_callback(callback)
        
        error_message = "Processing failed"
        status.set_error(error_message)
        
        assert status.stage == ProcessingStage.FAILED
        assert status.error_message == error_message
        callback.assert_called_once_with(status)
    
    def test_timing_functions(self):
        """Test timing calculation functions."""
        status = ProcessingStatus("test_doc")
        
        # Test elapsed time
        elapsed = status.get_elapsed_time()
        assert elapsed >= 0
        
        # Test stage time
        stage_time = status.get_stage_time()
        assert stage_time >= 0
        
        # Update stage and test again
        import time
        time.sleep(0.01)  # Small delay
        status.update_stage(ProcessingStage.DOWNLOADING)
        
        new_elapsed = status.get_elapsed_time()
        new_stage_time = status.get_stage_time()
        
        assert new_elapsed > elapsed
        assert new_stage_time < new_elapsed  # Stage time should be smaller


if __name__ == "__main__":
    pytest.main([__file__])