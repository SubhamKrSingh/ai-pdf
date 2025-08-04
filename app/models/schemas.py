"""
Pydantic data models for the LLM Query Retrieval System.

This module contains all data models used for API requests/responses
and internal data structures with comprehensive validation.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator, ConfigDict
from pydantic.types import PositiveInt


# API Request/Response Models

class QueryRequest(BaseModel):
    """
    Request model for the main query endpoint.
    
    Validates document URL and questions array according to requirements 1.1 and 1.2.
    """
    documents: HttpUrl = Field(
        ...,
        description="URL to the document to be processed",
        example="https://example.com/document.pdf"
    )
    questions: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Array of natural language questions to answer"
    )
    
    @field_validator('questions')
    @classmethod
    def validate_questions(cls, v):
        """Validate that questions are non-empty and reasonable length."""
        for i, question in enumerate(v):
            if not question.strip():
                raise ValueError(f"Question at index {i} cannot be empty")
            if len(question.strip()) < 3:
                raise ValueError(f"Question at index {i} is too short (minimum 3 characters)")
            if len(question.strip()) > 1000:
                raise ValueError(f"Question at index {i} is too long (maximum 1000 characters)")
        return [q.strip() for q in v]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "documents": "https://example.com/sample-document.pdf",
                "questions": [
                    "What is the main topic of this document?",
                    "What are the key findings mentioned?"
                ]
            }
        }
    )


class QueryResponse(BaseModel):
    """
    Response model for successful query processing.
    
    Ensures answers array corresponds to input questions according to requirement 7.2.
    """
    answers: List[str] = Field(
        ...,
        description="Array of answers corresponding to input questions"
    )
    
    @field_validator('answers')
    @classmethod
    def validate_answers(cls, v):
        """Validate that answers are non-empty."""
        for i, answer in enumerate(v):
            if not answer.strip():
                raise ValueError(f"Answer at index {i} cannot be empty")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answers": [
                    "The main topic of this document is artificial intelligence applications in healthcare.",
                    "The key findings include improved diagnostic accuracy and reduced processing time."
                ]
            }
        }
    )


class ErrorResponse(BaseModel):
    """
    Standardized error response model for all error scenarios.
    
    Provides structured error information according to requirement 7.3.
    """
    error: str = Field(
        ...,
        description="Human-readable error message"
    )
    error_code: str = Field(
        ...,
        description="Machine-readable error code for programmatic handling"
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details and context"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error occurrence timestamp"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Document download failed",
                "error_code": "DOCUMENT_DOWNLOAD_ERROR",
                "details": {
                    "url": "https://example.com/invalid-document.pdf",
                    "status_code": 404
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    )


# Internal Data Models

class DocumentChunk(BaseModel):
    """
    Internal model representing a processed document chunk.
    
    Used for storing and retrieving document segments with embeddings.
    """
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the chunk"
    )
    document_id: str = Field(
        ...,
        description="Identifier of the parent document"
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text content of the chunk"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the chunk"
    )
    embedding: Optional[List[float]] = Field(
        None,
        description="Vector embedding of the chunk content"
    )
    chunk_index: int = Field(
        ...,
        ge=0,
        description="Sequential index of this chunk within the document"
    )
    start_char: Optional[int] = Field(
        None,
        ge=0,
        description="Starting character position in original document"
    )
    end_char: Optional[int] = Field(
        None,
        ge=0,
        description="Ending character position in original document"
    )

    @field_validator('embedding')
    @classmethod
    def validate_embedding(cls, v):
        """Validate embedding dimensions if provided."""
        if v is not None:
            if len(v) == 0:
                raise ValueError("Embedding cannot be empty if provided")
            if not all(isinstance(x, (int, float)) for x in v):
                raise ValueError("Embedding must contain only numeric values")
        return v

    @model_validator(mode='after')
    def validate_char_positions(self):
        """Validate that end_char is greater than start_char if both provided."""
        if self.start_char is not None and self.end_char is not None:
            if self.end_char <= self.start_char:
                raise ValueError("end_char must be greater than start_char")
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "chunk_123e4567-e89b-12d3-a456-426614174000",
                "document_id": "doc_987fcdeb-51a2-43d7-8f9e-123456789abc",
                "content": "This is a sample chunk of text from the document...",
                "metadata": {
                    "page_number": 1,
                    "section": "Introduction"
                },
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 150
            }
        }
    )


class SearchResult(BaseModel):
    """
    Model representing a search result from vector similarity search.
    
    Contains chunk information with similarity score for ranking.
    """
    chunk_id: str = Field(
        ...,
        description="Identifier of the matching chunk"
    )
    content: str = Field(
        ...,
        description="Text content of the matching chunk"
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score between 0 and 1"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the chunk"
    )
    document_id: str = Field(
        ...,
        description="Identifier of the source document"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk_id": "chunk_123e4567-e89b-12d3-a456-426614174000",
                "content": "This chunk contains relevant information about the query...",
                "score": 0.85,
                "metadata": {
                    "page_number": 3,
                    "section": "Results"
                },
                "document_id": "doc_987fcdeb-51a2-43d7-8f9e-123456789abc"
            }
        }
    )


class DocumentMetadata(BaseModel):
    """
    Model for storing document metadata and processing information.
    
    Tracks document processing status and statistics.
    """
    document_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the document"
    )
    url: HttpUrl = Field(
        ...,
        description="Original URL of the document"
    )
    content_type: str = Field(
        ...,
        description="MIME type of the document"
    )
    processed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when document was processed"
    )
    chunk_count: PositiveInt = Field(
        ...,
        description="Number of chunks created from the document"
    )
    status: str = Field(
        default="processing",
        description="Current processing status"
    )
    file_size_bytes: Optional[int] = Field(
        None,
        ge=0,
        description="Size of the original document in bytes"
    )
    processing_time_ms: Optional[int] = Field(
        None,
        ge=0,
        description="Time taken to process the document in milliseconds"
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if processing failed"
    )

    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        """Validate that status is one of the allowed values."""
        allowed_statuses = ['processing', 'completed', 'failed', 'pending']
        if v not in allowed_statuses:
            raise ValueError(f"Status must be one of: {', '.join(allowed_statuses)}")
        return v

    @field_validator('content_type')
    @classmethod
    def validate_content_type(cls, v):
        """Validate that content type is supported."""
        supported_types = [
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'message/rfc822',
            'text/plain'
        ]
        if v not in supported_types:
            raise ValueError(f"Unsupported content type: {v}")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "doc_987fcdeb-51a2-43d7-8f9e-123456789abc",
                "url": "https://example.com/document.pdf",
                "content_type": "application/pdf",
                "processed_at": "2024-01-15T10:30:00Z",
                "chunk_count": 25,
                "status": "completed",
                "file_size_bytes": 1048576,
                "processing_time_ms": 5000
            }
        }
    )


# Utility Models for Error Handling

class ValidationError(BaseModel):
    """Model for validation error details."""
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    value: Any = Field(None, description="Invalid value that caused the error")


class ProcessingError(BaseModel):
    """Model for processing error details."""
    stage: str = Field(..., description="Processing stage where error occurred")
    component: str = Field(..., description="System component that failed")
    message: str = Field(..., description="Error message")
    recoverable: bool = Field(default=False, description="Whether the error is recoverable")


# Export all models for easy importing
__all__ = [
    'QueryRequest',
    'QueryResponse', 
    'ErrorResponse',
    'DocumentChunk',
    'SearchResult',
    'DocumentMetadata',
    'ValidationError',
    'ProcessingError'
]