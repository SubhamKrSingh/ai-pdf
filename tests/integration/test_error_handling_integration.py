"""
Integration tests for comprehensive error handling system.

Tests the complete error handling flow including middleware, exceptions,
and retry mechanisms in realistic scenarios according to requirements 8.1, 8.2, 8.3, 8.4, 8.5.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.middleware.error_handler import setup_error_handling
from app.exceptions import (
    DocumentDownloadError,
    DocumentParsingError,
    EmbeddingServiceError,
    LLMServiceError,
    VectorStoreError,
    DatabaseError,
    ValidationError,
    AuthenticationError
)
from app.utils.retry import RetryConfig, retry_async, GracefulDegradation
from app.models.schemas import QueryRequest, ErrorResponse


class TestErrorHandlingIntegration:
    """Test complete error handling integration."""
    
    @pytest.fixture
    def app(self):
        """Create test FastAPI app with error handling."""
        app = FastAPI()
        setup_error_handling(app, enable_debug=True)
        
        # Add test endpoints that simulate various error scenarios
        @app.post("/api/v1/hackrx/run")
        async def process_query(request: QueryRequest):
            # Simulate different error scenarios based on request
            if "download-error" in str(request.documents):
                raise DocumentDownloadError(
                    str(request.documents),
                    status_code=404,
                    reason="Document not found"
                )
            elif "parsing-error" in str(request.documents):
                raise DocumentParsingError("PDF", "Corrupted file")
            elif "embedding-error" in str(request.documents):
                raise EmbeddingServiceError("generate_embeddings", "API timeout")
            elif "llm-error" in str(request.documents):
                raise LLMServiceError("generate_answer", "Rate limit exceeded")
            elif "vector-error" in str(request.documents):
                raise VectorStoreError("similarity_search", "Connection failed")
            elif "database-error" in str(request.documents):
                raise DatabaseError("insert_document", "Connection timeout")
            elif "auth-error" in str(request.documents):
                raise AuthenticationError("Invalid token")
            elif "validation-error" in str(request.documents):
                raise ValidationError("Invalid field", field="documents")
            elif "unexpected-error" in str(request.documents):
                raise ValueError("Unexpected error occurred")
            else:
                return {"answers": ["Test answer"]}
        
        @app.get("/test/retry-success")
        async def test_retry_success():
            """Endpoint that succeeds after retries."""
            if not hasattr(test_retry_success, 'call_count'):
                test_retry_success.call_count = 0
            
            test_retry_success.call_count += 1
            
            if test_retry_success.call_count < 3:
                raise ConnectionError("Temporary failure")
            
            return {"message": "success", "attempts": test_retry_success.call_count}
        
        @app.get("/test/graceful-degradation")
        async def test_graceful_degradation():
            """Endpoint that demonstrates graceful degradation."""
            async def primary_service():
                raise ConnectionError("Primary service unavailable")
            
            async def fallback_service():
                return {"message": "fallback response", "degraded": True}
            
            result = await GracefulDegradation.with_fallback(
                primary_service,
                fallback_service
            )
            
            return result
        
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    def test_document_download_error_handling(self, client):
        """Test handling of document download errors."""
        request_data = {
            "documents": "https://example.com/download-error.pdf",
            "questions": ["What is this document about?"]
        }
        
        response = client.post("/api/v1/hackrx/run", json=request_data)
        
        assert response.status_code == 502
        
        data = response.json()
        assert data["error_code"] == "DOCUMENT_DOWNLOAD_ERROR"
        assert data["details"]["category"] == "server_error"
        assert data["details"]["recoverable"] is True
        assert data["details"]["url"] == request_data["documents"]
        assert data["details"]["http_status_code"] == 404
        assert "X-Error-Category" in response.headers
    
    def test_document_parsing_error_handling(self, client):
        """Test handling of document parsing errors."""
        request_data = {
            "documents": "https://example.com/parsing-error.pdf",
            "questions": ["What is this document about?"]
        }
        
        response = client.post("/api/v1/hackrx/run", json=request_data)
        
        assert response.status_code == 500
        
        data = response.json()
        assert data["error_code"] == "DOCUMENT_PARSING_ERROR"
        assert data["details"]["category"] == "server_error"
        assert data["details"]["recoverable"] is False
        assert data["details"]["document_type"] == "PDF"
    
    def test_embedding_service_error_handling(self, client):
        """Test handling of embedding service errors."""
        request_data = {
            "documents": "https://example.com/embedding-error.pdf",
            "questions": ["What is this document about?"]
        }
        
        response = client.post("/api/v1/hackrx/run", json=request_data)
        
        assert response.status_code == 503
        
        data = response.json()
        assert data["error_code"] == "EMBEDDING_SERVICE_ERROR"
        assert data["details"]["recoverable"] is True
        assert data["details"]["operation"] == "generate_embeddings"
    
    def test_llm_service_error_handling(self, client):
        """Test handling of LLM service errors."""
        request_data = {
            "documents": "https://example.com/llm-error.pdf",
            "questions": ["What is this document about?"]
        }
        
        response = client.post("/api/v1/hackrx/run", json=request_data)
        
        assert response.status_code == 503
        
        data = response.json()
        assert data["error_code"] == "LLM_SERVICE_ERROR"
        assert data["details"]["recoverable"] is True
        assert data["details"]["operation"] == "generate_answer"
    
    def test_vector_store_error_handling(self, client):
        """Test handling of vector store errors."""
        request_data = {
            "documents": "https://example.com/vector-error.pdf",
            "questions": ["What is this document about?"]
        }
        
        response = client.post("/api/v1/hackrx/run", json=request_data)
        
        assert response.status_code == 503
        
        data = response.json()
        assert data["error_code"] == "VECTOR_STORE_ERROR"
        assert data["details"]["recoverable"] is True
        assert data["details"]["operation"] == "similarity_search"
    
    def test_database_error_handling(self, client):
        """Test handling of database errors."""
        request_data = {
            "documents": "https://example.com/database-error.pdf",
            "questions": ["What is this document about?"]
        }
        
        response = client.post("/api/v1/hackrx/run", json=request_data)
        
        assert response.status_code == 503
        
        data = response.json()
        assert data["error_code"] == "DATABASE_ERROR"
        assert data["details"]["recoverable"] is True
        assert data["details"]["operation"] == "insert_document"
    
    def test_authentication_error_handling(self, client):
        """Test handling of authentication errors."""
        request_data = {
            "documents": "https://example.com/auth-error.pdf",
            "questions": ["What is this document about?"]
        }
        
        response = client.post("/api/v1/hackrx/run", json=request_data)
        
        assert response.status_code == 401
        
        data = response.json()
        assert data["error_code"] == "AUTHENTICATION_ERROR"
        assert data["details"]["category"] == "client_error"
        assert data["details"]["recoverable"] is True
    
    def test_validation_error_handling(self, client):
        """Test handling of validation errors."""
        request_data = {
            "documents": "https://example.com/validation-error.pdf",
            "questions": ["What is this document about?"]
        }
        
        response = client.post("/api/v1/hackrx/run", json=request_data)
        
        assert response.status_code == 422
        
        data = response.json()
        assert data["error_code"] == "VALIDATION_ERROR"
        assert data["details"]["category"] == "client_error"
        assert data["details"]["field"] == "documents"
    
    def test_unexpected_error_handling(self, client):
        """Test handling of unexpected errors."""
        request_data = {
            "documents": "https://example.com/unexpected-error.pdf",
            "questions": ["What is this document about?"]
        }
        
        response = client.post("/api/v1/hackrx/run", json=request_data)
        
        assert response.status_code == 500
        
        data = response.json()
        assert data["error_code"] == "INTERNAL_SERVER_ERROR"
        assert data["details"]["exception_type"] == "ValueError"
        # In debug mode, should include traceback
        assert "traceback" in data["details"]
    
    def test_successful_request_handling(self, client):
        """Test successful request handling."""
        request_data = {
            "documents": "https://example.com/valid-document.pdf",
            "questions": ["What is this document about?"]
        }
        
        response = client.post("/api/v1/hackrx/run", json=request_data)
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["answers"] == ["Test answer"]
    
    def test_request_validation_error(self, client):
        """Test FastAPI request validation error handling."""
        # Send invalid request (missing required fields)
        response = client.post("/api/v1/hackrx/run", json={})
        
        assert response.status_code == 422
        
        data = response.json()
        assert data["error_code"] == "VALIDATION_ERROR"
        assert "validation_errors" in data["details"]
    
    def test_error_statistics_tracking(self, client):
        """Test that error statistics are tracked correctly."""
        # Generate some errors
        error_requests = [
            {"documents": "https://example.com/download-error.pdf", "questions": ["test"]},
            {"documents": "https://example.com/auth-error.pdf", "questions": ["test"]},
            {"documents": "https://example.com/llm-error.pdf", "questions": ["test"]},
        ]
        
        for request_data in error_requests:
            client.post("/api/v1/hackrx/run", json=request_data)
        
        # Check error statistics
        stats_response = client.get("/api/v1/errors/stats")
        assert stats_response.status_code == 200
        
        stats = stats_response.json()
        assert stats["total_errors"] >= 3
        assert stats["client_errors"] >= 1  # auth error
        assert stats["server_errors"] >= 2  # download and llm errors
        assert stats["recoverable_errors"] >= 3  # all test errors are recoverable


class TestRetryIntegration:
    """Test retry mechanism integration."""
    
    @pytest.mark.asyncio
    async def test_retry_with_eventual_success(self):
        """Test retry mechanism with eventual success."""
        call_count = 0
        
        async def flaky_service():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise EmbeddingServiceError("generate_embeddings", "Temporary failure")
            return {"embeddings": [0.1, 0.2, 0.3]}
        
        config = RetryConfig(max_attempts=3, initial_delay=0.01)
        result = await retry_async(flaky_service, config)
        
        assert result.success is True
        assert result.result["embeddings"] == [0.1, 0.2, 0.3]
        assert result.attempts_used == 3
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_with_non_recoverable_error(self):
        """Test retry mechanism with non-recoverable error."""
        async def failing_service():
            raise DocumentParsingError("PDF", "Corrupted file")
        
        config = RetryConfig(max_attempts=3, initial_delay=0.01)
        result = await retry_async(failing_service, config)
        
        assert result.success is False
        assert isinstance(result.error, DocumentParsingError)
        # Non-recoverable errors still use all attempts
        assert result.attempts_used == 3
    
    @pytest.mark.asyncio
    async def test_retry_with_max_attempts_exceeded(self):
        """Test retry mechanism when max attempts are exceeded."""
        async def persistent_failure():
            raise LLMServiceError("generate_answer", "Service unavailable")
        
        config = RetryConfig(max_attempts=2, initial_delay=0.01)
        result = await retry_async(persistent_failure, config)
        
        assert result.success is False
        assert isinstance(result.error, LLMServiceError)
        assert result.attempts_used == 2


class TestGracefulDegradationIntegration:
    """Test graceful degradation integration."""
    
    @pytest.fixture
    def app_with_degradation(self):
        """Create app with graceful degradation endpoints."""
        app = FastAPI()
        setup_error_handling(app)
        
        @app.get("/test/graceful-degradation")
        async def test_graceful_degradation():
            async def primary_service():
                raise EmbeddingServiceError("generate_embeddings", "Service unavailable")
            
            async def fallback_service():
                return {"message": "Using cached embeddings", "degraded": True}
            
            result = await GracefulDegradation.with_fallback(
                primary_service,
                fallback_service
            )
            
            return result
        
        @app.get("/test/fallback-value")
        async def test_fallback_value():
            async def failing_service():
                raise VectorStoreError("similarity_search", "Database unavailable")
            
            result = await GracefulDegradation.with_fallback(
                failing_service,
                fallback_value={"results": [], "degraded": True}
            )
            
            return result
        
        return app
    
    def test_graceful_degradation_with_fallback_function(self, app_with_degradation):
        """Test graceful degradation using fallback function."""
        client = TestClient(app_with_degradation)
        
        response = client.get("/test/graceful-degradation")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Using cached embeddings"
        assert data["degraded"] is True
    
    def test_graceful_degradation_with_fallback_value(self, app_with_degradation):
        """Test graceful degradation using fallback value."""
        client = TestClient(app_with_degradation)
        
        response = client.get("/test/fallback-value")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["results"] == []
        assert data["degraded"] is True


class TestErrorResponseSerialization:
    """Test error response serialization and validation."""
    
    def test_error_response_schema_validation(self):
        """Test that error responses conform to schema."""
        app = FastAPI()
        setup_error_handling(app)
        
        @app.get("/test-error")
        async def test_error():
            raise DocumentDownloadError(
                "https://example.com/doc.pdf",
                status_code=404,
                reason="Not found"
            )
        
        client = TestClient(app)
        response = client.get("/test-error")
        
        # Validate response against ErrorResponse schema
        data = response.json()
        error_response = ErrorResponse(**data)
        
        assert error_response.error_code == "DOCUMENT_DOWNLOAD_ERROR"
        assert error_response.details["category"] == "server_error"
        assert error_response.details["recoverable"] is True
        assert error_response.timestamp is not None
    
    def test_error_response_with_processing_time(self):
        """Test that error responses include processing time."""
        app = FastAPI()
        setup_error_handling(app)
        
        @app.get("/test-slow-error")
        async def test_slow_error():
            await asyncio.sleep(0.01)  # Small delay
            raise LLMServiceError("generate_answer", "Timeout")
        
        client = TestClient(app)
        response = client.get("/test-slow-error")
        
        data = response.json()
        assert "processing_time_ms" in data["details"]
        assert data["details"]["processing_time_ms"] > 0


class TestErrorLoggingIntegration:
    """Test error logging integration."""
    
    def test_error_logging_with_context(self):
        """Test that errors are logged with proper context."""
        app = FastAPI()
        setup_error_handling(app)
        
        @app.get("/test-logging")
        async def test_logging():
            raise DatabaseError("insert_document", "Connection failed")
        
        client = TestClient(app)
        
        with patch('app.middleware.error_handler.logger') as mock_logger:
            response = client.get("/test-logging")
            
            # Verify error was logged
            mock_logger.log.assert_called()
            
            # Check log call arguments
            call_args = mock_logger.log.call_args
            assert call_args[0][0] == 40  # ERROR level
            assert "System error" in call_args[0][1]
            
            # Check extra context
            extra = call_args[1]["extra"]
            assert extra["error_code"] == "DATABASE_ERROR"
            assert extra["category"] == "server_error"
            assert extra["recoverable"] is True


if __name__ == "__main__":
    pytest.main([__file__])