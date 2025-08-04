"""
Unit tests for the LLM service.

Tests the Gemini 2.0 Flash API integration with mocked responses,
error handling, retry logic, and response validation.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from app.services.llm_service import LLMService, LLMServiceError, get_llm_service
from app.models.schemas import DocumentChunk
from app.config import Settings


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock(spec=Settings)
    settings.gemini_api_key = "test_api_key"
    settings.gemini_model = "gemini-2.0-flash"
    settings.max_retries = 3
    settings.retry_delay = 0.1  # Short delay for tests
    settings.llm_timeout = 30
    return settings


@pytest.fixture
def sample_document_chunks():
    """Sample document chunks for testing."""
    return [
        DocumentChunk(
            id="chunk_1",
            document_id="doc_1",
            content="This is the first chunk of content about artificial intelligence.",
            metadata={"page_number": 1, "section": "Introduction"},
            chunk_index=0,
            start_char=0,
            end_char=50
        ),
        DocumentChunk(
            id="chunk_2", 
            document_id="doc_1",
            content="This chunk discusses machine learning applications in healthcare.",
            metadata={"page_number": 2, "section": "Applications"},
            chunk_index=1,
            start_char=51,
            end_char=120
        )
    ]


@pytest.fixture
def llm_service(mock_settings):
    """LLM service instance with mocked settings."""
    with patch('app.services.llm_service.get_settings', return_value=mock_settings):
        with patch('app.services.llm_service.genai.configure'):
            service = LLMService()
            return service


class TestLLMServiceInitialization:
    """Test LLM service initialization and configuration."""
    
    def test_init_configures_gemini_api(self, mock_settings):
        """Test that initialization configures Gemini API."""
        with patch('app.services.llm_service.get_settings', return_value=mock_settings):
            with patch('app.services.llm_service.genai.configure') as mock_configure:
                service = LLMService()
                mock_configure.assert_called_once_with(api_key="test_api_key")
    
    def test_init_handles_configuration_error(self, mock_settings):
        """Test that initialization handles Gemini API configuration errors."""
        with patch('app.services.llm_service.get_settings', return_value=mock_settings):
            with patch('app.services.llm_service.genai.configure', side_effect=Exception("API key invalid")):
                with pytest.raises(LLMServiceError) as exc_info:
                    LLMService()
                
                assert exc_info.value.error_code == "GEMINI_CONFIG_ERROR"
                assert "Failed to configure Gemini API" in exc_info.value.message
    
    def test_get_model_creates_model_with_correct_settings(self, llm_service):
        """Test that _get_model creates model with correct configuration."""
        with patch('app.services.llm_service.genai.GenerativeModel') as mock_model_class:
            mock_model = Mock()
            mock_model_class.return_value = mock_model
            
            model = llm_service._get_model()
            
            # Verify model was created with correct parameters
            mock_model_class.assert_called_once()
            call_args = mock_model_class.call_args
            
            assert call_args[1]['model_name'] == "gemini-2.0-flash"
            assert 'safety_settings' in call_args[1]
            assert 'generation_config' in call_args[1]
            assert model == mock_model
    
    def test_get_model_caches_instance(self, llm_service):
        """Test that _get_model caches the model instance."""
        with patch('app.services.llm_service.genai.GenerativeModel') as mock_model_class:
            mock_model = Mock()
            mock_model_class.return_value = mock_model
            
            # Call twice
            model1 = llm_service._get_model()
            model2 = llm_service._get_model()
            
            # Should only create model once
            mock_model_class.assert_called_once()
            assert model1 == model2


class TestPromptCreation:
    """Test prompt creation and formatting."""
    
    def test_create_context_prompt_formats_correctly(self, llm_service, sample_document_chunks):
        """Test that context prompt is formatted correctly."""
        question = "What is artificial intelligence?"
        prompt = llm_service._create_context_prompt(question, sample_document_chunks)
        
        # Check that prompt contains all expected elements
        assert question in prompt
        assert "Context Chunk 1" in prompt
        assert "Context Chunk 2" in prompt
        assert sample_document_chunks[0].content in prompt
        assert sample_document_chunks[1].content in prompt
        assert "(Page 1)" in prompt
        assert "(Page 2)" in prompt
        assert "Introduction" in prompt
        assert "Applications" in prompt
    
    def test_create_context_prompt_handles_missing_metadata(self, llm_service):
        """Test prompt creation with chunks missing metadata."""
        chunks = [
            DocumentChunk(
                id="chunk_1",
                document_id="doc_1", 
                content="Content without metadata",
                chunk_index=0
            )
        ]
        
        question = "Test question?"
        prompt = llm_service._create_context_prompt(question, chunks)
        
        assert question in prompt
        assert "Content without metadata" in prompt
        assert "Context Chunk 1" in prompt
    
    def test_create_context_prompt_empty_chunks(self, llm_service):
        """Test prompt creation with empty chunks list."""
        question = "Test question?"
        prompt = llm_service._create_context_prompt(question, [])
        
        assert question in prompt
        assert "CONTEXT FROM DOCUMENT:" in prompt


class TestAPICallsAndRetry:
    """Test API calls and retry logic."""
    
    @pytest.mark.asyncio
    async def test_successful_api_call(self, llm_service):
        """Test successful API call without retries."""
        mock_response = Mock()
        mock_response.text = "This is a test response"
        mock_response.prompt_feedback = None
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        
        with patch.object(llm_service, '_get_model', return_value=mock_model):
            result = await llm_service._make_api_call_with_retry("test prompt")
            
            assert result == "This is a test response"
            mock_model.generate_content.assert_called_once_with("test prompt")
    
    @pytest.mark.asyncio
    async def test_api_call_with_timeout_retry(self, llm_service):
        """Test API call retry on timeout."""
        mock_response = Mock()
        mock_response.text = "Success after retry"
        mock_response.prompt_feedback = None
        
        mock_model = Mock()
        # First call times out, second succeeds
        mock_model.generate_content.side_effect = [
            asyncio.TimeoutError("Timeout"),
            mock_response
        ]
        
        with patch.object(llm_service, '_get_model', return_value=mock_model):
            with patch('asyncio.sleep'):  # Speed up test
                result = await llm_service._make_api_call_with_retry("test prompt")
                
                assert result == "Success after retry"
                assert mock_model.generate_content.call_count == 2
    
    @pytest.mark.asyncio
    async def test_api_call_rate_limit_retry(self, llm_service):
        """Test API call retry on rate limit."""
        mock_response = Mock()
        mock_response.text = "Success after rate limit"
        mock_response.prompt_feedback = None
        
        mock_model = Mock()
        # First call hits rate limit, second succeeds
        mock_model.generate_content.side_effect = [
            google_exceptions.ResourceExhausted("Rate limit exceeded"),
            mock_response
        ]
        
        with patch.object(llm_service, '_get_model', return_value=mock_model):
            with patch('asyncio.sleep'):  # Speed up test
                result = await llm_service._make_api_call_with_retry("test prompt")
                
                assert result == "Success after rate limit"
                assert mock_model.generate_content.call_count == 2
    
    @pytest.mark.asyncio
    async def test_api_call_max_retries_exceeded(self, llm_service):
        """Test API call failure after max retries."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = google_exceptions.GoogleAPIError("Persistent error")
        
        with patch.object(llm_service, '_get_model', return_value=mock_model):
            with patch('asyncio.sleep'):  # Speed up test
                with pytest.raises(LLMServiceError) as exc_info:
                    await llm_service._make_api_call_with_retry("test prompt")
                
                assert exc_info.value.error_code == "GEMINI_API_FAILURE"
                assert "after 3 attempts" in exc_info.value.message
                assert mock_model.generate_content.call_count == 3
    
    @pytest.mark.asyncio
    async def test_api_call_empty_response(self, llm_service):
        """Test handling of empty response from API."""
        mock_response = Mock()
        mock_response.text = ""
        mock_response.prompt_feedback = None
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        
        with patch.object(llm_service, '_get_model', return_value=mock_model):
            with pytest.raises(LLMServiceError) as exc_info:
                await llm_service._make_api_call_with_retry("test prompt")
            
            assert exc_info.value.error_code == "GEMINI_EMPTY_RESPONSE"
    
    @pytest.mark.asyncio
    async def test_api_call_safety_block(self, llm_service):
        """Test handling of safety filter blocks."""
        mock_feedback = Mock()
        mock_feedback.block_reason = "SAFETY"
        
        mock_response = Mock()
        mock_response.text = "Blocked content"
        mock_response.prompt_feedback = mock_feedback
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        
        with patch.object(llm_service, '_get_model', return_value=mock_model):
            with pytest.raises(LLMServiceError) as exc_info:
                await llm_service._make_api_call_with_retry("test prompt")
            
            assert exc_info.value.error_code == "GEMINI_SAFETY_BLOCK"


class TestResponseValidation:
    """Test response validation and formatting."""
    
    def test_validate_and_format_response_success(self, llm_service):
        """Test successful response validation and formatting."""
        response = "  ANSWER: This is a valid response.  "
        question = "Test question?"
        
        result = llm_service._validate_and_format_response(response, question)
        
        assert result == "This is a valid response."
    
    def test_validate_and_format_response_removes_prefixes(self, llm_service):
        """Test that response prefixes are removed."""
        test_cases = [
            ("ANSWER: Test response", "Test response"),
            ("Answer: Test response", "Test response"), 
            ("Response: Test response", "Test response"),
            ("A: Test response", "Test response"),
            ("Test response", "Test response")  # No prefix
        ]
        
        for input_response, expected in test_cases:
            result = llm_service._validate_and_format_response(input_response, "question")
            assert result == expected
    
    def test_validate_and_format_response_empty(self, llm_service):
        """Test validation of empty response."""
        with pytest.raises(LLMServiceError) as exc_info:
            llm_service._validate_and_format_response("", "question")
        
        assert exc_info.value.error_code == "EMPTY_LLM_RESPONSE"
    
    def test_validate_and_format_response_too_long(self, llm_service):
        """Test truncation of very long responses."""
        long_response = "A" * 6000  # Longer than 5000 char limit
        result = llm_service._validate_and_format_response(long_response, "question")
        
        assert len(result) <= 5000
        assert result.endswith("... [Response truncated]")


class TestAnswerGeneration:
    """Test main answer generation functionality."""
    
    @pytest.mark.asyncio
    async def test_generate_contextual_answer_success(self, llm_service, sample_document_chunks):
        """Test successful contextual answer generation."""
        question = "What is artificial intelligence?"
        expected_answer = "AI is a field of computer science."
        
        with patch.object(llm_service, '_make_api_call_with_retry', return_value=expected_answer) as mock_api:
            result = await llm_service.generate_contextual_answer(question, sample_document_chunks)
            
            assert result == expected_answer
            mock_api.assert_called_once()
            
            # Verify prompt was created correctly
            call_args = mock_api.call_args[0][0]
            assert question in call_args
            assert sample_document_chunks[0].content in call_args
    
    @pytest.mark.asyncio
    async def test_generate_contextual_answer_empty_question(self, llm_service, sample_document_chunks):
        """Test error handling for empty question."""
        with pytest.raises(LLMServiceError) as exc_info:
            await llm_service.generate_contextual_answer("", sample_document_chunks)
        
        assert exc_info.value.error_code == "EMPTY_QUESTION"
    
    @pytest.mark.asyncio
    async def test_generate_contextual_answer_no_context(self, llm_service):
        """Test handling of no context chunks."""
        question = "What is AI?"
        result = await llm_service.generate_contextual_answer(question, [])
        
        assert "don't have enough context" in result
    
    @pytest.mark.asyncio
    async def test_generate_contextual_answer_api_error(self, llm_service, sample_document_chunks):
        """Test error handling when API call fails."""
        question = "What is AI?"
        
        with patch.object(llm_service, '_make_api_call_with_retry', side_effect=LLMServiceError("API failed", "API_ERROR")):
            with pytest.raises(LLMServiceError) as exc_info:
                await llm_service.generate_contextual_answer(question, sample_document_chunks)
            
            assert exc_info.value.error_code == "API_ERROR"


class TestMultipleAnswers:
    """Test multiple answer generation."""
    
    @pytest.mark.asyncio
    async def test_generate_multiple_answers_success(self, llm_service, sample_document_chunks):
        """Test successful generation of multiple answers."""
        questions = ["What is AI?", "How is ML used?"]
        context_lists = [sample_document_chunks, sample_document_chunks]
        expected_answers = ["AI is computer science", "ML is used in healthcare"]
        
        with patch.object(llm_service, 'generate_contextual_answer', side_effect=expected_answers):
            results = await llm_service.generate_multiple_answers(questions, context_lists)
            
            assert results == expected_answers
    
    @pytest.mark.asyncio
    async def test_generate_multiple_answers_mismatch(self, llm_service, sample_document_chunks):
        """Test error when questions and context lists don't match."""
        questions = ["What is AI?", "How is ML used?"]
        context_lists = [sample_document_chunks]  # Only one context list
        
        with pytest.raises(LLMServiceError) as exc_info:
            await llm_service.generate_multiple_answers(questions, context_lists)
        
        assert exc_info.value.error_code == "QUESTION_CONTEXT_MISMATCH"
    
    @pytest.mark.asyncio
    async def test_generate_multiple_answers_partial_failure(self, llm_service, sample_document_chunks):
        """Test handling when some answers fail to generate."""
        questions = ["What is AI?", "How is ML used?"]
        context_lists = [sample_document_chunks, sample_document_chunks]
        
        # First succeeds, second fails
        side_effects = ["AI is computer science", LLMServiceError("Failed", "ERROR")]
        
        with patch.object(llm_service, 'generate_contextual_answer', side_effect=side_effects):
            results = await llm_service.generate_multiple_answers(questions, context_lists)
            
            assert len(results) == 2
            assert results[0] == "AI is computer science"
            assert "encountered an error" in results[1]


class TestHealthCheck:
    """Test health check functionality."""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, llm_service):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.text = "OK"
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        
        with patch.object(llm_service, '_get_model', return_value=mock_model):
            result = await llm_service.health_check()
            
            assert result["status"] == "healthy"
            assert result["model"] == "gemini-2.0-flash"
            assert "timestamp" in result
            assert result["test_response_length"] == 2
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, llm_service):
        """Test health check failure."""
        with patch.object(llm_service, '_get_model', side_effect=Exception("Model failed")):
            result = await llm_service.health_check()
            
            assert result["status"] == "unhealthy"
            assert "error" in result
            assert "Model failed" in result["error"]


class TestServiceSingleton:
    """Test global service instance management."""
    
    def test_get_llm_service_returns_singleton(self):
        """Test that get_llm_service returns the same instance."""
        with patch('app.services.llm_service.LLMService') as mock_service_class:
            mock_instance = Mock()
            mock_service_class.return_value = mock_instance
            
            # Clear any existing instance
            import app.services.llm_service
            app.services.llm_service._llm_service = None
            
            service1 = get_llm_service()
            service2 = get_llm_service()
            
            assert service1 == service2
            mock_service_class.assert_called_once()


class TestLLMServiceError:
    """Test custom exception class."""
    
    def test_llm_service_error_creation(self):
        """Test LLMServiceError creation with all parameters."""
        details = {"key": "value"}
        error = LLMServiceError("Test message", "TEST_ERROR", details)
        
        assert error.message == "Test message"
        assert error.error_code == "TEST_ERROR"
        assert error.details == details
        assert str(error) == "Test message"
    
    def test_llm_service_error_without_details(self):
        """Test LLMServiceError creation without details."""
        error = LLMServiceError("Test message", "TEST_ERROR")
        
        assert error.message == "Test message"
        assert error.error_code == "TEST_ERROR"
        assert error.details == {}