"""
Unit tests for the query controller.

Tests the controller logic that orchestrates document and query services.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi import HTTPException

from app.controllers.query_controller import QueryController
from app.models.schemas import QueryRequest, QueryResponse


class TestQueryController:
    """Test cases for the QueryController class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.controller = QueryController()
    
    @pytest.mark.asyncio
    @patch('app.controllers.query_controller.get_document_service')
    @patch('app.controllers.query_controller.get_query_service')
    async def test_process_query_request_success(self, mock_get_query_service, mock_get_document_service):
        """Test successful query request processing."""
        # Mock services
        mock_doc_service = AsyncMock()
        mock_query_service = AsyncMock()
        mock_get_document_service.return_value = mock_doc_service
        mock_get_query_service.return_value = mock_query_service
        
        # Mock service responses
        mock_doc_service.process_document.return_value = "test_doc_id"
        mock_query_service.process_multiple_questions.return_value = [
            "This is answer 1",
            "This is answer 2"
        ]
        
        # Create test request
        request = QueryRequest(
            documents="https://example.com/test.pdf",
            questions=["Question 1?", "Question 2?"]
        )
        
        # Process request
        response = await self.controller.process_query_request(request)
        
        # Verify response
        assert isinstance(response, QueryResponse)
        assert len(response.answers) == 2
        assert response.answers[0] == "This is answer 1"
        assert response.answers[1] == "This is answer 2"
        
        # Verify service calls
        mock_doc_service.process_document.assert_called_once_with(str(request.documents))
        mock_query_service.process_multiple_questions.assert_called_once_with(
            request.questions, "test_doc_id"
        )
    
    @pytest.mark.asyncio
    @patch('app.controllers.query_controller.get_document_service')
    async def test_process_document_failure(self, mock_get_document_service):
        """Test document processing failure handling."""
        # Mock service failure
        mock_doc_service = AsyncMock()
        mock_get_document_service.return_value = mock_doc_service
        mock_doc_service.process_document.side_effect = Exception("Download failed")
        
        # Create test request
        request = QueryRequest(
            documents="https://example.com/invalid.pdf",
            questions=["Question 1?"]
        )
        
        # Process request should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await self.controller.process_query_request(request)
        
        assert exc_info.value.status_code == 500
        assert "Document processing error" in exc_info.value.detail
    
    @pytest.mark.asyncio
    @patch('app.controllers.query_controller.get_document_service')
    @patch('app.controllers.query_controller.get_query_service')
    async def test_question_processing_failure(self, mock_get_query_service, mock_get_document_service):
        """Test question processing failure handling."""
        # Mock services
        mock_doc_service = AsyncMock()
        mock_query_service = AsyncMock()
        mock_get_document_service.return_value = mock_doc_service
        mock_get_query_service.return_value = mock_query_service
        
        # Mock successful document processing but failed question processing
        mock_doc_service.process_document.return_value = "test_doc_id"
        mock_query_service.process_multiple_questions.side_effect = Exception("LLM service failed")
        
        # Create test request
        request = QueryRequest(
            documents="https://example.com/test.pdf",
            questions=["Question 1?"]
        )
        
        # Process request should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await self.controller.process_query_request(request)
        
        assert exc_info.value.status_code == 500
        assert "Question processing error" in exc_info.value.detail
    
    @pytest.mark.asyncio
    @patch('app.controllers.query_controller.get_document_service')
    @patch('app.controllers.query_controller.get_query_service')
    async def test_answer_count_mismatch(self, mock_get_query_service, mock_get_document_service):
        """Test handling of answer count mismatch."""
        # Mock services
        mock_doc_service = AsyncMock()
        mock_query_service = AsyncMock()
        mock_get_document_service.return_value = mock_doc_service
        mock_get_query_service.return_value = mock_query_service
        
        # Mock responses with mismatched counts
        mock_doc_service.process_document.return_value = "test_doc_id"
        mock_query_service.process_multiple_questions.return_value = ["Only one answer"]  # But 2 questions
        
        # Create test request
        request = QueryRequest(
            documents="https://example.com/test.pdf",
            questions=["Question 1?", "Question 2?"]
        )
        
        # Process request should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await self.controller.process_query_request(request)
        
        assert exc_info.value.status_code == 500
        assert "Answer count" in exc_info.value.detail
    
    @pytest.mark.asyncio
    @patch('app.controllers.query_controller.get_document_service')
    @patch('app.controllers.query_controller.get_query_service')
    async def test_health_check_success(self, mock_get_query_service, mock_get_document_service):
        """Test successful health check."""
        # Mock services
        mock_doc_service = AsyncMock()
        mock_query_service = AsyncMock()
        mock_get_document_service.return_value = mock_doc_service
        mock_get_query_service.return_value = mock_query_service
        
        # Mock health check responses
        mock_doc_service.health_check.return_value = {"status": "healthy"}
        mock_query_service.health_check.return_value = {"status": "healthy"}
        
        # Perform health check
        result = await self.controller.health_check()
        
        # Verify result
        assert result["status"] == "healthy"
        assert "services" in result
        assert result["services"]["document_service"]["status"] == "healthy"
        assert result["services"]["query_service"]["status"] == "healthy"
    
    @pytest.mark.asyncio
    @patch('app.controllers.query_controller.get_document_service')
    async def test_health_check_failure(self, mock_get_document_service):
        """Test health check failure handling."""
        # Mock service failure
        mock_doc_service = AsyncMock()
        mock_get_document_service.return_value = mock_doc_service
        mock_doc_service.health_check.side_effect = Exception("Service unavailable")
        
        # Perform health check
        result = await self.controller.health_check()
        
        # Verify result
        assert result["status"] == "unhealthy"
        assert "error" in result
    
    @pytest.mark.asyncio
    @patch('app.controllers.query_controller.get_document_service')
    async def test_download_error_handling(self, mock_get_document_service):
        """Test specific download error handling."""
        # Mock service with download error
        mock_doc_service = AsyncMock()
        mock_get_document_service.return_value = mock_doc_service
        mock_doc_service.process_document.side_effect = Exception("Failed to download document")
        
        # Create test request
        request = QueryRequest(
            documents="https://example.com/invalid.pdf",
            questions=["Question 1?"]
        )
        
        # Process request should raise HTTPException with 400 status
        with pytest.raises(HTTPException) as exc_info:
            await self.controller.process_query_request(request)
        
        assert exc_info.value.status_code == 400
        assert "Failed to download document" in exc_info.value.detail
    
    @pytest.mark.asyncio
    @patch('app.controllers.query_controller.get_document_service')
    async def test_parse_error_handling(self, mock_get_document_service):
        """Test specific parse error handling."""
        # Mock service with parse error
        mock_doc_service = AsyncMock()
        mock_get_document_service.return_value = mock_doc_service
        mock_doc_service.process_document.side_effect = Exception("Failed to parse document")
        
        # Create test request
        request = QueryRequest(
            documents="https://example.com/test.pdf",
            questions=["Question 1?"]
        )
        
        # Process request should raise HTTPException with 400 status
        with pytest.raises(HTTPException) as exc_info:
            await self.controller.process_query_request(request)
        
        assert exc_info.value.status_code == 400
        assert "Failed to parse document" in exc_info.value.detail