"""
Simple unit tests for the main FastAPI application that don't require environment setup.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

# Mock the settings to avoid configuration validation
mock_settings = Mock()
mock_settings.auth_token = "test_token"
mock_settings.host = "0.0.0.0"
mock_settings.port = 8000

# Patch the settings before importing main
with patch('app.config.get_settings', return_value=mock_settings):
    from main import app


class TestMainApplicationSimple:
    """Simple test cases for the main FastAPI application."""
    
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
    
    def test_process_time_header(self):
        """Test that process time header is added to responses."""
        response = self.client.get("/health")
        
        # Check that process time header is present
        assert "x-process-time" in response.headers
        
        # Verify it's a valid float
        process_time = float(response.headers["x-process-time"])
        assert process_time >= 0
    
    def test_request_validation_error_structure(self):
        """Test request validation error response structure."""
        # Invalid request data (missing required fields)
        invalid_request = {
            "documents": "not_a_url",  # Invalid URL
            "questions": []  # Empty questions array
        }
        
        headers = {"Authorization": "Bearer test_token"}
        
        response = self.client.post(
            "/api/v1/hackrx/run", 
            json=invalid_request, 
            headers=headers
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert "error_code" in data
        assert "timestamp" in data
        assert data["error_code"] == "VALIDATION_ERROR"
    
    @patch('app.controllers.query_controller.QueryController.process_query_request')
    def test_successful_request_structure(self, mock_process):
        """Test successful request response structure."""
        from app.models.schemas import QueryResponse
        
        request_data = {
            "documents": "https://example.com/test.pdf",
            "questions": ["What is this document about?"]
        }
        
        # Mock the controller response
        mock_response = QueryResponse(answers=["This is a test document."])
        mock_process.return_value = mock_response
        
        headers = {"Authorization": "Bearer test_token"}
        
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
        
        # Verify controller was called with correct arguments
        mock_process.assert_called_once()
        call_args = mock_process.call_args[0][0]
        assert str(call_args.documents) == request_data["documents"]
        assert call_args.questions == request_data["questions"]