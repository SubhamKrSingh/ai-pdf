"""
Unit tests for the main FastAPI application.

Tests the main application setup, middleware configuration, and error handling.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

from main import app
from app.models.schemas import QueryRequest, QueryResponse, ErrorResponse


class TestMainApplication:
    """Test cases for the main FastAPI application."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
    
    def test_health_check_endpoint(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "LLM Query Retrieval System"
    
    def test_docs_endpoint_accessible(self):
        """Test that API documentation is accessible."""
        response = self.client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc_endpoint_accessible(self):
        """Test that ReDoc documentation is accessible."""
        response = self.client.get("/redoc")
        assert response.status_code == 200
    
    def test_main_endpoint_requires_auth(self):
        """Test that the main endpoint requires authentication."""
        request_data = {
            "documents": "https://example.com/test.pdf",
            "questions": ["What is this document about?"]
        }
        
        # Request without auth header should fail
        response = self.client.post("/api/v1/hackrx/run", json=request_data)
        assert response.status_code == 403  # FastAPI returns 403 for missing auth
    
    def test_main_endpoint_with_invalid_auth(self):
        """Test the main endpoint with invalid authentication."""
        request_data = {
            "documents": "https://example.com/test.pdf",
            "questions": ["What is this document about?"]
        }
        
        headers = {"Authorization": "Bearer invalid_token"}
        
        with patch('app.config.get_settings') as mock_settings:
            mock_settings.return_value.auth_token = "valid_token"
            
            response = self.client.post(
                "/api/v1/hackrx/run", 
                json=request_data, 
                headers=headers
            )
            assert response.status_code == 401
    
    @patch('app.controllers.query_controller.QueryController.process_query_request')
    def test_main_endpoint_with_valid_auth(self, mock_process):
        """Test the main endpoint with valid authentication."""
        request_data = {
            "documents": "https://example.com/test.pdf",
            "questions": ["What is this document about?"]
        }
        
        # Mock the controller response
        mock_response = QueryResponse(answers=["This is a test document."])
        mock_process.return_value = mock_response
        
        headers = {"Authorization": "Bearer test_token"}
        
        with patch('app.config.get_settings') as mock_settings:
            mock_settings.return_value.auth_token = "test_token"
            
            response = self.client.post(
                "/api/v1/hackrx/run", 
                json=request_data, 
                headers=headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "answers" in data
            assert len(data["answers"]) == 1
            assert data["answers"][0] == "This is a test document."
    
    def test_request_validation_error(self):
        """Test request validation error handling."""
        # Invalid request data (missing required fields)
        invalid_request = {
            "documents": "not_a_url",  # Invalid URL
            "questions": []  # Empty questions array
        }
        
        headers = {"Authorization": "Bearer test_token"}
        
        with patch('app.config.get_settings') as mock_settings:
            mock_settings.return_value.auth_token = "test_token"
            
            response = self.client.post(
                "/api/v1/hackrx/run", 
                json=invalid_request, 
                headers=headers
            )
            
            assert response.status_code == 422
            data = response.json()
            assert "error" in data
            assert data["error_code"] == "VALIDATION_ERROR"
    
    @patch('app.controllers.query_controller.QueryController.process_query_request')
    def test_internal_server_error_handling(self, mock_process):
        """Test internal server error handling."""
        request_data = {
            "documents": "https://example.com/test.pdf",
            "questions": ["What is this document about?"]
        }
        
        # Mock an internal error
        mock_process.side_effect = Exception("Internal processing error")
        
        headers = {"Authorization": "Bearer test_token"}
        
        with patch('app.config.get_settings') as mock_settings:
            mock_settings.return_value.auth_token = "test_token"
            
            response = self.client.post(
                "/api/v1/hackrx/run", 
                json=request_data, 
                headers=headers
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "Failed to process query" in data["error"]
    
    @patch('app.controllers.query_controller.QueryController.process_query_request')
    def test_http_exception_handling(self, mock_process):
        """Test HTTP exception handling."""
        request_data = {
            "documents": "https://example.com/test.pdf",
            "questions": ["What is this document about?"]
        }
        
        # Mock an HTTP exception
        mock_process.side_effect = HTTPException(
            status_code=400, 
            detail="Document download failed"
        )
        
        headers = {"Authorization": "Bearer test_token"}
        
        with patch('app.config.get_settings') as mock_settings:
            mock_settings.return_value.auth_token = "test_token"
            
            response = self.client.post(
                "/api/v1/hackrx/run", 
                json=request_data, 
                headers=headers
            )
            
            assert response.status_code == 400
            data = response.json()
            assert "error" in data
            assert data["error"] == "Document download failed"
    
    def test_cors_headers(self):
        """Test that CORS headers are properly set."""
        response = self.client.get("/health")
        
        # Check that CORS headers are present
        assert "access-control-allow-origin" in response.headers
    
    def test_process_time_header(self):
        """Test that process time header is added to responses."""
        response = self.client.get("/health")
        
        # Check that process time header is present
        assert "x-process-time" in response.headers
        
        # Verify it's a valid float
        process_time = float(response.headers["x-process-time"])
        assert process_time >= 0