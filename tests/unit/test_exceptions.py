"""
Unit tests for custom exception classes.

Tests all custom exception types and their behavior according to requirements 8.1, 8.2, 8.3, 8.4, 8.5.
"""

import pytest
from typing import Dict, Any

from app.exceptions import (
    BaseSystemError,
    ErrorCategory,
    ClientError,
    ServerError,
    ValidationError,
    AuthenticationError,
    DocumentNotFoundError,
    UnsupportedDocumentTypeError,
    DocumentDownloadError,
    DocumentParsingError,
    EmbeddingServiceError,
    VectorStoreError,
    DatabaseError,
    LLMServiceError,
    ProcessingError,
    ConfigurationError,
    is_recoverable_error,
    get_error_category,
    create_error_context
)


class TestBaseSystemError:
    """Test the base system error class."""
    
    def test_base_system_error_creation(self):
        """Test creating a base system error."""
        error = BaseSystemError(
            message="Test error",
            error_code="TEST_ERROR",
            category=ErrorCategory.SERVER_ERROR,
            details={"key": "value"},
            recoverable=True,
            http_status_code=500
        )
        
        assert error.message == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.category == ErrorCategory.SERVER_ERROR
        assert error.details == {"key": "value"}
        assert error.recoverable is True
        assert error.http_status_code == 500
    
    def test_base_system_error_to_dict(self):
        """Test converting base system error to dictionary."""
        error = BaseSystemError(
            message="Test error",
            error_code="TEST_ERROR",
            category=ErrorCategory.CLIENT_ERROR,
            details={"field": "test"},
            recoverable=False
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["error"] == "Test error"
        assert error_dict["error_code"] == "TEST_ERROR"
        assert error_dict["category"] == "client_error"
        assert error_dict["details"] == {"field": "test"}
        assert error_dict["recoverable"] is False
    
    def test_base_system_error_defaults(self):
        """Test base system error with default values."""
        error = BaseSystemError(
            message="Test error",
            error_code="TEST_ERROR",
            category=ErrorCategory.SERVER_ERROR
        )
        
        assert error.details == {}
        assert error.recoverable is False
        assert error.http_status_code == 500


class TestClientErrors:
    """Test client error classes."""
    
    def test_client_error_creation(self):
        """Test creating a client error."""
        error = ClientError(
            message="Client error",
            error_code="CLIENT_ERROR",
            details={"field": "value"}
        )
        
        assert error.category == ErrorCategory.CLIENT_ERROR
        assert error.recoverable is True
        assert error.http_status_code == 400
    
    def test_validation_error(self):
        """Test validation error creation."""
        error = ValidationError(
            message="Invalid field",
            field="test_field",
            value="invalid_value"
        )
        
        assert error.error_code == "VALIDATION_ERROR"
        assert error.http_status_code == 422
        assert error.details["field"] == "test_field"
        assert error.details["invalid_value"] == "invalid_value"
    
    def test_validation_error_defaults(self):
        """Test validation error with defaults."""
        error = ValidationError()
        
        assert error.message == "Request validation failed"
        assert error.error_code == "VALIDATION_ERROR"
    
    def test_authentication_error(self):
        """Test authentication error creation."""
        error = AuthenticationError(
            message="Invalid token",
            details={"token_type": "bearer"}
        )
        
        assert error.error_code == "AUTHENTICATION_ERROR"
        assert error.http_status_code == 401
        assert error.details["token_type"] == "bearer"
    
    def test_document_not_found_error(self):
        """Test document not found error."""
        url = "https://example.com/missing.pdf"
        error = DocumentNotFoundError(url)
        
        assert error.error_code == "DOCUMENT_NOT_FOUND"
        assert error.http_status_code == 404
        assert error.details["url"] == url
        assert url in error.message
    
    def test_unsupported_document_type_error(self):
        """Test unsupported document type error."""
        content_type = "application/unknown"
        error = UnsupportedDocumentTypeError(content_type)
        
        assert error.error_code == "UNSUPPORTED_DOCUMENT_TYPE"
        assert error.http_status_code == 415
        assert error.details["content_type"] == content_type
        assert "supported_types" in error.details


class TestServerErrors:
    """Test server error classes."""
    
    def test_server_error_creation(self):
        """Test creating a server error."""
        error = ServerError(
            message="Server error",
            error_code="SERVER_ERROR",
            details={"component": "test"}
        )
        
        assert error.category == ErrorCategory.SERVER_ERROR
        assert error.recoverable is False
        assert error.http_status_code == 500
    
    def test_document_download_error(self):
        """Test document download error."""
        url = "https://example.com/document.pdf"
        error = DocumentDownloadError(
            url=url,
            status_code=404,
            reason="Not found"
        )
        
        assert error.error_code == "DOCUMENT_DOWNLOAD_ERROR"
        assert error.http_status_code == 502
        assert error.recoverable is True
        assert error.details["url"] == url
        assert error.details["http_status_code"] == 404
        assert error.details["reason"] == "Not found"
    
    def test_document_parsing_error(self):
        """Test document parsing error."""
        error = DocumentParsingError(
            document_type="PDF",
            reason="Corrupted file"
        )
        
        assert error.error_code == "DOCUMENT_PARSING_ERROR"
        assert error.recoverable is False
        assert error.details["document_type"] == "PDF"
        assert error.details["reason"] == "Corrupted file"
    
    def test_embedding_service_error(self):
        """Test embedding service error."""
        error = EmbeddingServiceError(
            operation="generate_embeddings",
            reason="API timeout"
        )
        
        assert error.error_code == "EMBEDDING_SERVICE_ERROR"
        assert error.http_status_code == 503
        assert error.recoverable is True
        assert error.details["operation"] == "generate_embeddings"
        assert error.details["reason"] == "API timeout"
    
    def test_vector_store_error(self):
        """Test vector store error."""
        error = VectorStoreError(
            operation="similarity_search",
            reason="Connection failed"
        )
        
        assert error.error_code == "VECTOR_STORE_ERROR"
        assert error.http_status_code == 503
        assert error.recoverable is True
        assert error.details["operation"] == "similarity_search"
    
    def test_database_error(self):
        """Test database error."""
        error = DatabaseError(
            operation="insert_document",
            reason="Connection timeout"
        )
        
        assert error.error_code == "DATABASE_ERROR"
        assert error.http_status_code == 503
        assert error.recoverable is True
        assert error.details["operation"] == "insert_document"
    
    def test_llm_service_error(self):
        """Test LLM service error."""
        error = LLMServiceError(
            operation="generate_answer",
            reason="Rate limit exceeded"
        )
        
        assert error.error_code == "LLM_SERVICE_ERROR"
        assert error.http_status_code == 503
        assert error.recoverable is True
        assert error.details["operation"] == "generate_answer"
    
    def test_processing_error(self):
        """Test processing error."""
        error = ProcessingError(
            stage="chunking",
            reason="Memory limit exceeded"
        )
        
        assert error.error_code == "PROCESSING_ERROR"
        assert error.recoverable is False
        assert error.details["processing_stage"] == "chunking"
    
    def test_configuration_error(self):
        """Test configuration error."""
        error = ConfigurationError(
            config_item="GEMINI_API_KEY",
            reason="Missing required environment variable"
        )
        
        assert error.error_code == "CONFIGURATION_ERROR"
        assert error.recoverable is False
        assert error.details["config_item"] == "GEMINI_API_KEY"


class TestErrorUtilities:
    """Test error utility functions."""
    
    def test_is_recoverable_error_with_system_error(self):
        """Test is_recoverable_error with system errors."""
        recoverable_error = DocumentDownloadError("http://example.com/doc.pdf")
        non_recoverable_error = DocumentParsingError("PDF")
        
        assert is_recoverable_error(recoverable_error) is True
        assert is_recoverable_error(non_recoverable_error) is False
    
    def test_is_recoverable_error_with_standard_exceptions(self):
        """Test is_recoverable_error with standard exceptions."""
        assert is_recoverable_error(ConnectionError()) is True
        assert is_recoverable_error(TimeoutError()) is True
        assert is_recoverable_error(OSError()) is True
        assert is_recoverable_error(ValueError()) is False
    
    def test_get_error_category_with_system_error(self):
        """Test get_error_category with system errors."""
        client_error = ValidationError()
        server_error = DatabaseError("test_operation")
        
        assert get_error_category(client_error) == ErrorCategory.CLIENT_ERROR
        assert get_error_category(server_error) == ErrorCategory.SERVER_ERROR
    
    def test_get_error_category_with_standard_exceptions(self):
        """Test get_error_category with standard exceptions."""
        assert get_error_category(ValueError()) == ErrorCategory.VALIDATION_ERROR
        assert get_error_category(TypeError()) == ErrorCategory.VALIDATION_ERROR
        assert get_error_category(ConnectionError()) == ErrorCategory.EXTERNAL_SERVICE_ERROR
        assert get_error_category(RuntimeError()) == ErrorCategory.SERVER_ERROR
    
    def test_create_error_context(self):
        """Test create_error_context function."""
        context = create_error_context(
            operation="test_operation",
            component="test_component",
            additional_context={"key": "value"}
        )
        
        assert context["operation"] == "test_operation"
        assert context["component"] == "test_component"
        assert context["key"] == "value"
        assert "timestamp" in context
    
    def test_create_error_context_without_additional(self):
        """Test create_error_context without additional context."""
        context = create_error_context(
            operation="test_operation",
            component="test_component"
        )
        
        assert context["operation"] == "test_operation"
        assert context["component"] == "test_component"
        assert "timestamp" in context
        assert len(context) == 3  # operation, component, timestamp


class TestErrorInheritance:
    """Test error class inheritance and behavior."""
    
    def test_client_error_inheritance(self):
        """Test that client errors inherit properly."""
        error = ValidationError()
        
        assert isinstance(error, ClientError)
        assert isinstance(error, BaseSystemError)
        assert isinstance(error, Exception)
    
    def test_server_error_inheritance(self):
        """Test that server errors inherit properly."""
        error = DatabaseError("test_operation")
        
        assert isinstance(error, ServerError)
        assert isinstance(error, BaseSystemError)
        assert isinstance(error, Exception)
    
    def test_error_string_representation(self):
        """Test string representation of errors."""
        error = ValidationError(message="Test validation error")
        
        assert str(error) == "Test validation error"
    
    def test_error_with_none_details(self):
        """Test error creation with None details."""
        error = BaseSystemError(
            message="Test error",
            error_code="TEST_ERROR",
            category=ErrorCategory.SERVER_ERROR,
            details=None
        )
        
        assert error.details == {}


if __name__ == "__main__":
    pytest.main([__file__])