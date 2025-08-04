"""
Custom exception classes for the LLM Query Retrieval System.

This module defines all custom exceptions used throughout the application
with proper categorization for client and server errors according to requirement 8.1.
"""

from typing import Optional, Dict, Any
from enum import Enum


class ErrorCategory(str, Enum):
    """Error categories for proper classification."""
    CLIENT_ERROR = "client_error"
    SERVER_ERROR = "server_error"
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    EXTERNAL_SERVICE_ERROR = "external_service_error"
    PROCESSING_ERROR = "processing_error"


class BaseSystemError(Exception):
    """
    Base exception class for all system errors.
    
    Provides structured error information with categorization and context.
    """
    
    def __init__(
        self,
        message: str,
        error_code: str,
        category: ErrorCategory,
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = False,
        http_status_code: int = 500
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.category = category
        self.details = details or {}
        self.recoverable = recoverable
        self.http_status_code = http_status_code
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "error": self.message,
            "error_code": self.error_code,
            "category": self.category.value,
            "details": self.details,
            "recoverable": self.recoverable
        }


# Client Error Exceptions (4xx)

class ClientError(BaseSystemError):
    """Base class for client errors (4xx status codes)."""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        details: Optional[Dict[str, Any]] = None,
        http_status_code: int = 400
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.CLIENT_ERROR,
            details=details,
            recoverable=True,
            http_status_code=http_status_code
        )


class ValidationError(ClientError):
    """Raised when request validation fails."""
    
    def __init__(
        self,
        message: str = "Request validation failed",
        field: Optional[str] = None,
        value: Any = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        if field:
            error_details["field"] = field
        if value is not None:
            error_details["invalid_value"] = str(value)
            
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=error_details,
            http_status_code=422
        )


class AuthenticationError(ClientError):
    """Raised when authentication fails."""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            details=details,
            http_status_code=401
        )


class DocumentNotFoundError(ClientError):
    """Raised when a document cannot be found or accessed."""
    
    def __init__(
        self,
        url: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_message = message or f"Document not found or inaccessible: {url}"
        error_details = details or {}
        error_details["url"] = url
        
        super().__init__(
            message=error_message,
            error_code="DOCUMENT_NOT_FOUND",
            details=error_details,
            http_status_code=404
        )


class UnsupportedDocumentTypeError(ClientError):
    """Raised when document type is not supported."""
    
    def __init__(
        self,
        content_type: str,
        supported_types: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        supported = supported_types or ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "message/rfc822"]
        error_details = details or {}
        error_details.update({
            "content_type": content_type,
            "supported_types": supported
        })
        
        super().__init__(
            message=f"Unsupported document type: {content_type}",
            error_code="UNSUPPORTED_DOCUMENT_TYPE",
            details=error_details,
            http_status_code=415
        )


# Server Error Exceptions (5xx)

class ServerError(BaseSystemError):
    """Base class for server errors (5xx status codes)."""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = False,
        http_status_code: int = 500
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.SERVER_ERROR,
            details=details,
            recoverable=recoverable,
            http_status_code=http_status_code
        )


class DocumentDownloadError(ServerError):
    """Raised when document download fails."""
    
    def __init__(
        self,
        url: str,
        status_code: Optional[int] = None,
        reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        error_details["url"] = url
        if status_code:
            error_details["http_status_code"] = status_code
        if reason:
            error_details["reason"] = reason
            
        message = f"Failed to download document from {url}"
        if status_code:
            message += f" (HTTP {status_code})"
        if reason:
            message += f": {reason}"
            
        super().__init__(
            message=message,
            error_code="DOCUMENT_DOWNLOAD_ERROR",
            details=error_details,
            recoverable=True,
            http_status_code=502
        )


class DocumentParsingError(ServerError):
    """Raised when document parsing fails."""
    
    def __init__(
        self,
        document_type: str,
        reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        error_details["document_type"] = document_type
        if reason:
            error_details["reason"] = reason
            
        message = f"Failed to parse {document_type} document"
        if reason:
            message += f": {reason}"
            
        super().__init__(
            message=message,
            error_code="DOCUMENT_PARSING_ERROR",
            details=error_details,
            recoverable=False
        )


class EmbeddingServiceError(ServerError):
    """Raised when embedding service fails."""
    
    def __init__(
        self,
        operation: str,
        reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        error_details["operation"] = operation
        if reason:
            error_details["reason"] = reason
            
        message = f"Embedding service failed during {operation}"
        if reason:
            message += f": {reason}"
            
        super().__init__(
            message=message,
            error_code="EMBEDDING_SERVICE_ERROR",
            details=error_details,
            recoverable=True,
            http_status_code=503
        )


class VectorStoreError(ServerError):
    """Raised when vector store operations fail."""
    
    def __init__(
        self,
        operation: str,
        reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        error_details["operation"] = operation
        if reason:
            error_details["reason"] = reason
            
        message = f"Vector store operation failed: {operation}"
        if reason:
            message += f" - {reason}"
            
        super().__init__(
            message=message,
            error_code="VECTOR_STORE_ERROR",
            details=error_details,
            recoverable=True,
            http_status_code=503
        )


class DatabaseError(ServerError):
    """Raised when database operations fail."""
    
    def __init__(
        self,
        operation: str,
        reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        error_details["operation"] = operation
        if reason:
            error_details["reason"] = reason
            
        message = f"Database operation failed: {operation}"
        if reason:
            message += f" - {reason}"
            
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            details=error_details,
            recoverable=True,
            http_status_code=503
        )


class LLMServiceError(ServerError):
    """Raised when LLM service fails."""
    
    def __init__(
        self,
        operation: str,
        reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        error_details["operation"] = operation
        if reason:
            error_details["reason"] = reason
            
        message = f"LLM service failed during {operation}"
        if reason:
            message += f": {reason}"
            
        super().__init__(
            message=message,
            error_code="LLM_SERVICE_ERROR",
            details=error_details,
            recoverable=True,
            http_status_code=503
        )


class ProcessingError(ServerError):
    """Raised when document processing pipeline fails."""
    
    def __init__(
        self,
        stage: str,
        reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        error_details["processing_stage"] = stage
        if reason:
            error_details["reason"] = reason
            
        message = f"Processing failed at stage: {stage}"
        if reason:
            message += f" - {reason}"
            
        super().__init__(
            message=message,
            error_code="PROCESSING_ERROR",
            details=error_details,
            recoverable=False
        )


class ConfigurationError(ServerError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(
        self,
        config_item: str,
        reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        error_details["config_item"] = config_item
        if reason:
            error_details["reason"] = reason
            
        message = f"Configuration error: {config_item}"
        if reason:
            message += f" - {reason}"
            
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=error_details,
            recoverable=False,
            http_status_code=500
        )


# Utility functions for error handling

def is_recoverable_error(error: Exception) -> bool:
    """
    Check if an error is recoverable and should be retried.
    
    Args:
        error: Exception to check
        
    Returns:
        bool: True if error is recoverable
    """
    if isinstance(error, BaseSystemError):
        return error.recoverable
    
    # Consider certain standard exceptions as recoverable
    recoverable_types = (
        ConnectionError,
        TimeoutError,
        OSError
    )
    
    return isinstance(error, recoverable_types)


def get_error_category(error: Exception) -> ErrorCategory:
    """
    Get the category of an error for proper handling.
    
    Args:
        error: Exception to categorize
        
    Returns:
        ErrorCategory: Category of the error
    """
    if isinstance(error, BaseSystemError):
        return error.category
    
    # Categorize standard exceptions
    if isinstance(error, (ValueError, TypeError)):
        return ErrorCategory.VALIDATION_ERROR
    elif isinstance(error, (ConnectionError, TimeoutError)):
        return ErrorCategory.EXTERNAL_SERVICE_ERROR
    else:
        return ErrorCategory.SERVER_ERROR


def create_error_context(
    operation: str,
    component: str,
    additional_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create standardized error context information.
    
    Args:
        operation: Operation being performed when error occurred
        component: System component where error occurred
        additional_context: Additional context information
        
    Returns:
        Dict containing error context
    """
    context = {
        "operation": operation,
        "component": component,
        "timestamp": "2024-01-15T10:30:00Z"  # This would be datetime.now() in real implementation
    }
    
    if additional_context:
        context.update(additional_context)
    
    return context