"""
Unit tests for Pydantic data models.

Tests validation rules, error handling, and model behavior for all data models
in the LLM Query Retrieval System.
"""

import pytest
from datetime import datetime
from typing import Dict, Any
from uuid import uuid4

from pydantic import ValidationError

from app.models import (
    QueryRequest,
    QueryResponse,
    ErrorResponse,
    DocumentChunk,
    SearchResult,
    DocumentMetadata,
    ValidationError as CustomValidationError,
    ProcessingError
)


class TestQueryRequest:
    """Test cases for QueryRequest model."""
    
    def test_valid_query_request(self):
        """Test creating a valid QueryRequest."""
        request = QueryRequest(
            documents="https://example.com/document.pdf",
            questions=["What is this about?", "Who is the author?"]
        )
        
        assert str(request.documents) == "https://example.com/document.pdf"
        assert len(request.questions) == 2
        assert request.questions[0] == "What is this about?"
    
    def test_invalid_url(self):
        """Test that invalid URLs are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(
                documents="not-a-valid-url",
                questions=["What is this about?"]
            )
        
        assert "url" in str(exc_info.value).lower()
    
    def test_empty_questions_array(self):
        """Test that empty questions array is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(
                documents="https://example.com/document.pdf",
                questions=[]
            )
        
        assert "at least 1 item" in str(exc_info.value).lower()
    
    def test_too_many_questions(self):
        """Test that too many questions are rejected."""
        questions = [f"Question {i}" for i in range(51)]
        
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(
                documents="https://example.com/document.pdf",
                questions=questions
            )
        
        assert "at most 50 items" in str(exc_info.value).lower()
    
    def test_empty_question_validation(self):
        """Test that empty questions are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(
                documents="https://example.com/document.pdf",
                questions=["Valid question", "   ", "Another valid question"]
            )
        
        assert "cannot be empty" in str(exc_info.value)
    
    def test_short_question_validation(self):
        """Test that very short questions are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(
                documents="https://example.com/document.pdf",
                questions=["Hi"]
            )
        
        assert "too short" in str(exc_info.value)
    
    def test_long_question_validation(self):
        """Test that very long questions are rejected."""
        long_question = "x" * 1001
        
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(
                documents="https://example.com/document.pdf",
                questions=[long_question]
            )
        
        assert "too long" in str(exc_info.value)
    
    def test_question_whitespace_trimming(self):
        """Test that questions are trimmed of whitespace."""
        request = QueryRequest(
            documents="https://example.com/document.pdf",
            questions=["  What is this about?  ", "\tWho is the author?\n"]
        )
        
        assert request.questions[0] == "What is this about?"
        assert request.questions[1] == "Who is the author?"


class TestQueryResponse:
    """Test cases for QueryResponse model."""
    
    def test_valid_query_response(self):
        """Test creating a valid QueryResponse."""
        response = QueryResponse(
            answers=["This is about AI", "The author is John Doe"]
        )
        
        assert len(response.answers) == 2
        assert response.answers[0] == "This is about AI"
    
    def test_empty_answer_validation(self):
        """Test that empty answers are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            QueryResponse(
                answers=["Valid answer", "   ", "Another valid answer"]
            )
        
        assert "cannot be empty" in str(exc_info.value)


class TestErrorResponse:
    """Test cases for ErrorResponse model."""
    
    def test_valid_error_response(self):
        """Test creating a valid ErrorResponse."""
        error = ErrorResponse(
            error="Document not found",
            error_code="DOCUMENT_NOT_FOUND",
            details={"url": "https://example.com/missing.pdf"}
        )
        
        assert error.error == "Document not found"
        assert error.error_code == "DOCUMENT_NOT_FOUND"
        assert error.details["url"] == "https://example.com/missing.pdf"
        assert isinstance(error.timestamp, datetime)
    
    def test_error_response_without_details(self):
        """Test creating ErrorResponse without details."""
        error = ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR"
        )
        
        assert error.details is None
        assert isinstance(error.timestamp, datetime)


class TestDocumentChunk:
    """Test cases for DocumentChunk model."""
    
    def test_valid_document_chunk(self):
        """Test creating a valid DocumentChunk."""
        chunk = DocumentChunk(
            document_id="doc_123",
            content="This is a sample chunk of text.",
            chunk_index=0,
            start_char=0,
            end_char=30
        )
        
        assert chunk.document_id == "doc_123"
        assert chunk.content == "This is a sample chunk of text."
        assert chunk.chunk_index == 0
        assert chunk.id is not None  # Auto-generated UUID
    
    def test_chunk_with_embedding(self):
        """Test DocumentChunk with embedding vector."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        chunk = DocumentChunk(
            document_id="doc_123",
            content="Sample text",
            chunk_index=0,
            embedding=embedding
        )
        
        assert chunk.embedding == embedding
    
    def test_invalid_embedding_validation(self):
        """Test that invalid embeddings are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                document_id="doc_123",
                content="Sample text",
                chunk_index=0,
                embedding=[]  # Empty embedding
            )
        
        assert "cannot be empty" in str(exc_info.value)
    
    def test_non_numeric_embedding_validation(self):
        """Test that non-numeric embeddings are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                document_id="doc_123",
                content="Sample text",
                chunk_index=0,
                embedding=[0.1, "invalid", 0.3]
            )
        
        # In Pydantic V2, this is caught by the type system
        assert "valid number" in str(exc_info.value).lower()
    
    def test_invalid_char_positions(self):
        """Test that invalid character positions are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                document_id="doc_123",
                content="Sample text",
                chunk_index=0,
                start_char=100,
                end_char=50  # end_char < start_char
            )
        
        assert "must be greater than" in str(exc_info.value)
    
    def test_negative_chunk_index(self):
        """Test that negative chunk index is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                document_id="doc_123",
                content="Sample text",
                chunk_index=-1
            )
        
        assert "greater than or equal to 0" in str(exc_info.value)
    
    def test_content_length_validation(self):
        """Test content length validation."""
        # Test empty content
        with pytest.raises(ValidationError):
            DocumentChunk(
                document_id="doc_123",
                content="",
                chunk_index=0
            )
        
        # Test very long content
        long_content = "x" * 10001
        with pytest.raises(ValidationError):
            DocumentChunk(
                document_id="doc_123",
                content=long_content,
                chunk_index=0
            )


class TestSearchResult:
    """Test cases for SearchResult model."""
    
    def test_valid_search_result(self):
        """Test creating a valid SearchResult."""
        result = SearchResult(
            chunk_id="chunk_123",
            content="Relevant content found",
            score=0.85,
            document_id="doc_123",
            metadata={"page": 1}
        )
        
        assert result.chunk_id == "chunk_123"
        assert result.score == 0.85
        assert result.metadata["page"] == 1
    
    def test_score_validation(self):
        """Test that score is validated to be between 0 and 1."""
        # Test negative score
        with pytest.raises(ValidationError):
            SearchResult(
                chunk_id="chunk_123",
                content="Content",
                score=-0.1,
                document_id="doc_123"
            )
        
        # Test score > 1
        with pytest.raises(ValidationError):
            SearchResult(
                chunk_id="chunk_123",
                content="Content",
                score=1.5,
                document_id="doc_123"
            )
    
    def test_valid_boundary_scores(self):
        """Test that boundary scores (0 and 1) are valid."""
        # Test score = 0
        result1 = SearchResult(
            chunk_id="chunk_123",
            content="Content",
            score=0.0,
            document_id="doc_123"
        )
        assert result1.score == 0.0
        
        # Test score = 1
        result2 = SearchResult(
            chunk_id="chunk_123",
            content="Content",
            score=1.0,
            document_id="doc_123"
        )
        assert result2.score == 1.0


class TestDocumentMetadata:
    """Test cases for DocumentMetadata model."""
    
    def test_valid_document_metadata(self):
        """Test creating valid DocumentMetadata."""
        metadata = DocumentMetadata(
            url="https://example.com/document.pdf",
            content_type="application/pdf",
            chunk_count=25,
            status="completed"
        )
        
        assert str(metadata.url) == "https://example.com/document.pdf"
        assert metadata.content_type == "application/pdf"
        assert metadata.chunk_count == 25
        assert metadata.status == "completed"
        assert metadata.document_id is not None  # Auto-generated
    
    def test_invalid_status_validation(self):
        """Test that invalid status values are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentMetadata(
                url="https://example.com/document.pdf",
                content_type="application/pdf",
                chunk_count=25,
                status="invalid_status"
            )
        
        assert "Status must be one of" in str(exc_info.value)
    
    def test_unsupported_content_type(self):
        """Test that unsupported content types are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentMetadata(
                url="https://example.com/document.pdf",
                content_type="application/unsupported",
                chunk_count=25
            )
        
        assert "Unsupported content type" in str(exc_info.value)
    
    def test_negative_chunk_count(self):
        """Test that negative chunk count is rejected."""
        with pytest.raises(ValidationError):
            DocumentMetadata(
                url="https://example.com/document.pdf",
                content_type="application/pdf",
                chunk_count=-1
            )
    
    def test_zero_chunk_count(self):
        """Test that zero chunk count is rejected."""
        with pytest.raises(ValidationError):
            DocumentMetadata(
                url="https://example.com/document.pdf",
                content_type="application/pdf",
                chunk_count=0
            )
    
    def test_negative_file_size(self):
        """Test that negative file size is rejected."""
        with pytest.raises(ValidationError):
            DocumentMetadata(
                url="https://example.com/document.pdf",
                content_type="application/pdf",
                chunk_count=25,
                file_size_bytes=-1
            )
    
    def test_negative_processing_time(self):
        """Test that negative processing time is rejected."""
        with pytest.raises(ValidationError):
            DocumentMetadata(
                url="https://example.com/document.pdf",
                content_type="application/pdf",
                chunk_count=25,
                processing_time_ms=-1
            )


class TestUtilityModels:
    """Test cases for utility models."""
    
    def test_validation_error_model(self):
        """Test ValidationError model."""
        error = CustomValidationError(
            field="questions",
            message="Question cannot be empty",
            value=""
        )
        
        assert error.field == "questions"
        assert error.message == "Question cannot be empty"
        assert error.value == ""
    
    def test_processing_error_model(self):
        """Test ProcessingError model."""
        error = ProcessingError(
            stage="document_parsing",
            component="pdf_parser",
            message="Failed to parse PDF document",
            recoverable=True
        )
        
        assert error.stage == "document_parsing"
        assert error.component == "pdf_parser"
        assert error.recoverable is True


class TestModelSerialization:
    """Test JSON serialization and deserialization of models."""
    
    def test_query_request_serialization(self):
        """Test QueryRequest JSON serialization."""
        request = QueryRequest(
            documents="https://example.com/document.pdf",
            questions=["What is this about?"]
        )
        
        json_data = request.model_dump()
        # In Pydantic V2, HttpUrl objects are serialized as strings in mode='json'
        json_data_str = request.model_dump(mode='json')
        assert json_data_str["documents"] == "https://example.com/document.pdf"
        assert json_data_str["questions"] == ["What is this about?"]
        
        # Test deserialization
        new_request = QueryRequest.model_validate(json_data_str)
        assert new_request.questions == request.questions
    
    def test_document_chunk_serialization(self):
        """Test DocumentChunk JSON serialization with embedding."""
        chunk = DocumentChunk(
            document_id="doc_123",
            content="Sample content",
            chunk_index=0,
            embedding=[0.1, 0.2, 0.3]
        )
        
        json_data = chunk.model_dump()
        assert json_data["embedding"] == [0.1, 0.2, 0.3]
        
        # Test deserialization
        new_chunk = DocumentChunk.model_validate(json_data)
        assert new_chunk.embedding == chunk.embedding
    
    def test_error_response_serialization(self):
        """Test ErrorResponse JSON serialization."""
        error = ErrorResponse(
            error="Test error",
            error_code="TEST_ERROR",
            details={"key": "value"}
        )
        
        json_data = error.model_dump()
        assert json_data["error"] == "Test error"
        assert json_data["details"]["key"] == "value"
        assert "timestamp" in json_data