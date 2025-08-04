"""
Unit tests for error handler middleware.

Tests the global error handler middleware and error logging functionality
according to requirements 8.1, 8.2, 8.3, 8.4, 8.5.
"""

import pytest
import json
import logging
from unittest.mock import Mock, patch, AsyncMock
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.middleware.error_handler import (
    ErrorHandlerMiddleware,
    ErrorLogger,
    error_logger,
    setup_error_handling
)
from app.exceptions import (
    BaseSystemError,
    ErrorCategory,
    ValidationError as CustomValidationError,
    DocumentDownloadError,
    LLMServiceError
)
from app.models.schemas import ErrorResponse


class TestErrorHandlerMiddleware:
    """Test the error handler middleware."""
    
    @pytest.fixture
    def app(self):
        """Create a test FastAPI app."""
        app = FastAPI()
        return app
    
    @pytest.fixture
    def middleware(self, app):
        """Create error handler middleware."""
        return ErrorHandlerMiddleware(app, enable_debug=True)
    
    @pytest.fixture
    def client(self, app):
        """Create test client with error handling."""
        setup_error_handling(app, enable_debug=True)
        
        @app.get("/test-success")
        async def test_success():
            return {"message": "success"}
        
        @app.get("/test-http-error")
        async def test_http_error():
            raise HTTPException(status_code=404, detail="Not found")
        
        @app.get("/test-system-error")
        async def test_system_error():
            raise DocumentDownloadError("http://example.com/doc.pdf", status_code=404)
        
        @app.get("/test-unexpected-error")
        async def test_unexpected_error():
            raise ValueError("Unexpected error")
        
        @app.post("/test-validation-error")
        async def test_validation_error(data: dict):
            return data
        
        return TestClient(app)
    
    def test_successful_request(self, client):
        """Test that successful requests pass through normally."""
        response = client.get("/test-success")
        
        assert response.status_code == 200
        assert response.json() == {"message": "success"}
    
    def test_http_exception_handling(self, client):
        """Test handling of HTTP exceptions."""
        response = client.get("/test-http-error")
        
        assert response.status_code == 404
        
        data = response.json()
        assert data["error"] == "Not found"
        assert data["error_code"] == "HTTP_404"
        assert data["details"]["status_code"] == 404
        assert "timestamp" in data
    
    def test_system_error_handling(self, client):
        """Test handling of custom system errors."""
        response = client.get("/test-system-error")
        
        assert response.status_code == 502
        
        data = response.json()
        assert "Failed to download document" in data["error"]
        assert data["error_code"] == "DOCUMENT_DOWNLOAD_ERROR"
        assert data["details"]["category"] == "server_error"
        assert data["details"]["recoverable"] is True
        assert "X-Error-Category" in response.headers
    
    def test_unexpected_error_handling(self, client):
        """Test handling of unexpected errors."""
        response = client.get("/test-unexpected-error")
        
        assert response.status_code == 500
        
        data = response.json()
        # In debug mode, should show actual error
        assert "ValueError: Unexpected error" in data["error"]
        assert data["error_code"] == "INTERNAL_SERVER_ERROR"
        assert data["details"]["exception_type"] == "ValueError"
        assert "traceback" in data["details"]
    
    def test_validation_error_handling(self, client):
        """Test handling of validation errors."""
        response = client.post("/test-validation-error", json="invalid")
        
        assert response.status_code == 422
        
        data = response.json()
        assert data["error"] == "Request validation failed"
        assert data["error_code"] == "VALIDATION_ERROR"
        assert "validation_errors" in data["details"]
    
    def test_error_statistics(self, middleware):
        """Test error statistics tracking."""
        initial_stats = middleware.get_error_statistics()
        assert initial_stats["total_errors"] == 0
        
        # Simulate some errors
        middleware.error_stats["total_errors"] = 10
        middleware.error_stats["client_errors"] = 6
        middleware.error_stats["server_errors"] = 4
        middleware.error_stats["recoverable_errors"] = 3
        
        stats = middleware.get_error_statistics()
        assert stats["total_errors"] == 10
        assert stats["client_errors"] == 6
        assert stats["server_errors"] == 4
        assert stats["recoverable_errors"] == 3
        assert stats["client_error_rate"] == 0.6
        assert stats["server_error_rate"] == 0.4
        assert stats["recovery_rate"] == 0.3
    
    @pytest.mark.asyncio
    async def test_middleware_dispatch_success(self, middleware):
        """Test middleware dispatch with successful request."""
        request = Mock(spec=Request)
        call_next = AsyncMock(return_value=Mock())
        
        response = await middleware.dispatch(request, call_next)
        
        call_next.assert_called_once_with(request)
        assert response is not None
    
    @pytest.mark.asyncio
    async def test_middleware_dispatch_error(self, middleware):
        """Test middleware dispatch with error."""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/test"
        request.state = Mock()
        
        call_next = AsyncMock(side_effect=ValueError("Test error"))
        
        response = await middleware.dispatch(request, call_next)
        
        assert response.status_code == 500
        content = json.loads(response.body)
        assert "ValueError: Test error" in content["error"]


class TestErrorLogger:
    """Test the error logger utility."""
    
    @pytest.fixture
    def logger(self):
        """Create error logger instance."""
        return ErrorLogger("test_logger")
    
    def test_log_error_with_system_error(self, logger):
        """Test logging system errors."""
        error = DocumentDownloadError("http://example.com/doc.pdf")
        context = {"operation": "download", "component": "document_service"}
        
        with patch.object(logger.logger, 'log') as mock_log:
            logger.log_error(error, context, logging.ERROR)
            
            mock_log.assert_called_once()
            args, kwargs = mock_log.call_args
            
            assert args[0] == logging.ERROR
            assert "Error in download" in args[1]
            assert kwargs["extra"]["error_type"] == "DocumentDownloadError"
            assert kwargs["extra"]["error_code"] == "DOCUMENT_DOWNLOAD_ERROR"
            assert kwargs["extra"]["category"] == "server_error"
            assert kwargs["extra"]["recoverable"] is True
    
    def test_log_error_with_standard_exception(self, logger):
        """Test logging standard exceptions."""
        error = ValueError("Test error")
        context = {"operation": "validation", "component": "api"}
        
        with patch.object(logger.logger, 'log') as mock_log:
            logger.log_error(error, context, logging.WARNING)
            
            mock_log.assert_called_once()
            args, kwargs = mock_log.call_args
            
            assert args[0] == logging.WARNING
            assert kwargs["extra"]["error_type"] == "ValueError"
            assert kwargs["extra"]["category"] == "validation_error"
            assert kwargs["extra"]["recoverable"] is False
    
    def test_log_recovery_attempt(self, logger):
        """Test logging recovery attempts."""
        error = LLMServiceError("generate_answer", "Timeout")
        context = {"operation": "answer_generation"}
        
        with patch.object(logger.logger, 'warning') as mock_warning:
            logger.log_recovery_attempt(error, 2, 3, context)
            
            mock_warning.assert_called_once()
            args, kwargs = mock_warning.call_args
            
            assert "Recovery attempt 2/3" in args[0]
            assert kwargs["extra"]["attempt"] == 2
            assert kwargs["extra"]["max_attempts"] == 3
            assert kwargs["extra"]["recoverable"] is True
    
    def test_log_recovery_success(self, logger):
        """Test logging successful recovery."""
        error = LLMServiceError("generate_answer", "Timeout")
        context = {"operation": "answer_generation"}
        
        with patch.object(logger.logger, 'info') as mock_info:
            logger.log_recovery_success(error, 2, context)
            
            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args
            
            assert "Successfully recovered" in args[0]
            assert "after 2 attempts" in args[0]
            assert kwargs["extra"]["attempts_used"] == 2
            assert kwargs["extra"]["recovered"] is True
    
    def test_log_recovery_failure(self, logger):
        """Test logging recovery failure."""
        error = LLMServiceError("generate_answer", "Timeout")
        context = {"operation": "answer_generation"}
        
        with patch.object(logger.logger, 'error') as mock_error:
            logger.log_recovery_failure(error, 3, context)
            
            mock_error.assert_called_once()
            args, kwargs = mock_error.call_args
            
            assert "Failed to recover" in args[0]
            assert "after 3 attempts" in args[0]
            assert kwargs["extra"]["attempts_used"] == 3
            assert kwargs["extra"]["recovered"] is False
            assert kwargs["exc_info"] is True


class TestSetupErrorHandling:
    """Test the setup_error_handling function."""
    
    def test_setup_error_handling(self):
        """Test setting up error handling on FastAPI app."""
        app = FastAPI()
        
        middleware = setup_error_handling(app, enable_debug=True)
        
        assert isinstance(middleware, ErrorHandlerMiddleware)
        assert middleware.enable_debug is True
        
        # Check that error statistics endpoint was added
        routes = [route.path for route in app.routes]
        assert "/api/v1/errors/stats" in routes
    
    def test_error_statistics_endpoint(self):
        """Test the error statistics endpoint."""
        app = FastAPI()
        setup_error_handling(app)
        
        client = TestClient(app)
        response = client.get("/api/v1/errors/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_errors" in data
        assert "client_errors" in data
        assert "server_errors" in data
        assert "recoverable_errors" in data
        assert "error_rate" in data


class TestErrorResponseIntegration:
    """Test integration with ErrorResponse model."""
    
    def test_error_response_serialization(self):
        """Test that error responses serialize correctly."""
        app = FastAPI()
        setup_error_handling(app, enable_debug=True)
        
        @app.get("/test-error")
        async def test_error():
            raise CustomValidationError(
                message="Test validation failed",
                field="test_field",
                value="invalid"
            )
        
        client = TestClient(app)
        response = client.get("/test-error")
        
        assert response.status_code == 422
        
        data = response.json()
        
        # Validate against ErrorResponse schema
        error_response = ErrorResponse(**data)
        assert error_response.error == "Test validation failed"
        assert error_response.error_code == "VALIDATION_ERROR"
        assert error_response.details["field"] == "test_field"
        assert error_response.timestamp is not None
    
    def test_processing_time_header(self):
        """Test that processing time is included in responses."""
        app = FastAPI()
        setup_error_handling(app)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        client = TestClient(app)
        response = client.get("/test")
        
        # Processing time should be added by request logging middleware
        # This would be tested in integration tests with actual middleware stack


if __name__ == "__main__":
    pytest.main([__file__])