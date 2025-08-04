"""
LLM service for generating contextual answers using Gemini 2.0 Flash API.

This module implements the LLM integration with comprehensive error handling,
retry logic, and prompt engineering for optimal answer quality.
Implements requirements 6.1, 6.2, 6.3, 6.5, 8.3.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core import exceptions as google_exceptions

from app.config import get_settings
from app.models.schemas import DocumentChunk, SearchResult, ErrorResponse


logger = logging.getLogger(__name__)


class LLMServiceError(Exception):
    """Custom exception for LLM service errors."""
    
    def __init__(self, message: str, error_code: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class LLMService:
    """
    Service for generating contextual answers using Google Gemini 2.0 Flash API.
    
    Provides async methods for answer generation with retry logic, error handling,
    and response validation.
    """
    
    def __init__(self):
        """Initialize the LLM service with Gemini API configuration."""
        self.settings = get_settings()
        self._configure_gemini()
        self._model = None
        
    def _configure_gemini(self) -> None:
        """Configure the Gemini API client with API key and safety settings."""
        try:
            genai.configure(api_key=self.settings.gemini_api_key)
            logger.info("Gemini API configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {str(e)}")
            raise LLMServiceError(
                "Failed to configure Gemini API",
                "GEMINI_CONFIG_ERROR",
                {"error": str(e)}
            )
    
    def _get_model(self):
        """Get or create the Gemini model instance with safety settings."""
        if self._model is None:
            try:
                # Configure safety settings to allow most content for document analysis
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                }
                
                # Configure generation parameters for concise, factual responses
                generation_config = genai.types.GenerationConfig(
                    temperature=0.05,  # Very low temperature for consistent, factual responses
                    top_p=0.7,
                    top_k=30,
                    max_output_tokens=1024,  # Reduced for more concise responses
                    candidate_count=1
                )
                
                self._model = genai.GenerativeModel(
                    model_name=self.settings.gemini_model,
                    safety_settings=safety_settings,
                    generation_config=generation_config
                )
                
                logger.info(f"Gemini model {self.settings.gemini_model} initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize Gemini model: {str(e)}")
                raise LLMServiceError(
                    "Failed to initialize Gemini model",
                    "GEMINI_MODEL_ERROR",
                    {"model": self.settings.gemini_model, "error": str(e)}
                )
        
        return self._model
    
    def _create_context_prompt(self, question: str, context_chunks: List[DocumentChunk]) -> str:
        """
        Create an optimized prompt for contextual answer generation.
        
        Args:
            question: The user's natural language question
            context_chunks: List of relevant document chunks for context
            
        Returns:
            str: Formatted prompt for the LLM
        """
        # Sort chunks by relevance if they have scores (from SearchResult)
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            context_text += f"\n--- Context Chunk {i} ---\n"
            context_text += chunk.content.strip()
            context_text += "\n"
        
        prompt = f"""You are a document analysis assistant. Provide concise, direct answers based on the document content.

INSTRUCTIONS:
1. Answer using ONLY the information from the context below
2. Be concise and direct - avoid unnecessary explanations
3. State facts clearly without excessive detail
4. If information is not available, state this briefly
5. Do not include phrases like "according to the document" or "based on the context"
6. Provide a single, focused answer

CONTEXT:
{context_text}

QUESTION: {question}

ANSWER:"""

        return prompt
    
    async def _make_api_call_with_retry(self, prompt: str) -> str:
        """
        Make API call to Gemini with retry logic and error handling.
        
        Args:
            prompt: The formatted prompt to send to the LLM
            
        Returns:
            str: Generated response from the LLM
            
        Raises:
            LLMServiceError: If all retry attempts fail
        """
        model = self._get_model()
        last_error = None
        
        for attempt in range(self.settings.max_retries):
            try:
                logger.debug(f"Making Gemini API call, attempt {attempt + 1}")
                
                # Make async API call with timeout
                response = await asyncio.wait_for(
                    asyncio.to_thread(model.generate_content, prompt),
                    timeout=self.settings.llm_timeout
                )
                
                # Validate response
                if not response or not response.text:
                    logger.warning(f"Empty response on attempt {attempt + 1}")
                    raise LLMServiceError(
                        "Empty response from Gemini API",
                        "GEMINI_EMPTY_RESPONSE",
                        {"attempt": attempt + 1}
                    )
                
                # Check if response was blocked by safety filters
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    logger.warning(f"Safety block on attempt {attempt + 1}: {response.prompt_feedback.block_reason}")
                    raise LLMServiceError(
                        f"Response blocked by safety filters: {response.prompt_feedback.block_reason}",
                        "GEMINI_SAFETY_BLOCK",
                        {"block_reason": str(response.prompt_feedback.block_reason)}
                    )
                
                logger.info(f"Gemini API call successful on attempt {attempt + 1}")
                return response.text.strip()
                
            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning(f"Gemini API call timeout on attempt {attempt + 1}")
                if attempt < self.settings.max_retries - 1:
                    await asyncio.sleep(self.settings.retry_delay * (2 ** attempt))
                    
            except google_exceptions.ResourceExhausted as e:
                last_error = e
                logger.warning(f"Gemini API rate limit exceeded on attempt {attempt + 1}")
                if attempt < self.settings.max_retries - 1:
                    # Exponential backoff for rate limiting
                    await asyncio.sleep(self.settings.retry_delay * (3 ** attempt))
                    
            except google_exceptions.GoogleAPIError as e:
                last_error = e
                logger.error(f"Gemini API error on attempt {attempt + 1}: {str(e)}")
                if attempt < self.settings.max_retries - 1:
                    await asyncio.sleep(self.settings.retry_delay * (2 ** attempt))
                    
            except LLMServiceError as e:
                # Handle specific LLM service errors
                if e.error_code in ["GEMINI_EMPTY_RESPONSE", "GEMINI_SAFETY_BLOCK"]:
                    last_error = e
                    if attempt < self.settings.max_retries - 1:
                        await asyncio.sleep(self.settings.retry_delay * (2 ** attempt))
                    else:
                        raise e
                else:
                    # Re-raise other LLM service errors immediately
                    raise e
                    
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
                if attempt < self.settings.max_retries - 1:
                    await asyncio.sleep(self.settings.retry_delay)
        
        # All retries failed
        error_message = f"Failed to get response from Gemini API after {self.settings.max_retries} attempts"
        logger.error(f"{error_message}. Last error: {str(last_error)}")
        
        raise LLMServiceError(
            error_message,
            "GEMINI_API_FAILURE",
            {
                "attempts": self.settings.max_retries,
                "last_error": str(last_error),
                "error_type": type(last_error).__name__
            }
        )
    
    def _validate_and_format_response(self, response: str, question: str) -> str:
        """
        Validate and format the LLM response for consistency.
        
        Args:
            response: Raw response from the LLM
            question: Original question for context
            
        Returns:
            str: Validated and formatted response
        """
        if not response or not response.strip():
            raise LLMServiceError(
                "Empty response from LLM",
                "EMPTY_LLM_RESPONSE",
                {"question": question}
            )
        
        # Clean up the response
        formatted_response = response.strip()
        
        # Remove any unwanted prefixes that might be added by the model
        prefixes_to_remove = ["ANSWER:", "Answer:", "Response:", "A:", "Based on the context:", "According to the document:"]
        for prefix in prefixes_to_remove:
            if formatted_response.startswith(prefix):
                formatted_response = formatted_response[len(prefix):].strip()
        
        # Remove verbose phrases that add no value
        verbose_phrases = [
            "Based on the provided context,",
            "According to the document,",
            "From the information provided,",
            "The document states that",
            "As mentioned in the context,",
            "The context indicates that"
        ]
        
        for phrase in verbose_phrases:
            if formatted_response.startswith(phrase):
                formatted_response = formatted_response[len(phrase):].strip()
        
        # Ensure the response starts with a capital letter
        if formatted_response and formatted_response[0].islower():
            formatted_response = formatted_response[0].upper() + formatted_response[1:]
        
        # Ensure maximum response length for conciseness
        if len(formatted_response) > 2000:
            logger.warning("Response truncated for conciseness")
            formatted_response = formatted_response[:1950] + "..."
        
        return formatted_response
    
    async def generate_contextual_answer(
        self, 
        question: str, 
        context_chunks: List[DocumentChunk]
    ) -> str:
        """
        Generate a contextual answer using retrieved document chunks.
        
        This method implements the core LLM functionality by:
        1. Creating an optimized prompt with context
        2. Making API calls with retry logic
        3. Validating and formatting the response
        
        Args:
            question: Natural language question from the user
            context_chunks: List of relevant document chunks for context
            
        Returns:
            str: Generated contextual answer
            
        Raises:
            LLMServiceError: If answer generation fails
        """
        if not question or not question.strip():
            raise LLMServiceError(
                "Question cannot be empty",
                "EMPTY_QUESTION",
                {"question": question}
            )
        
        if not context_chunks:
            logger.warning("No context chunks provided for question")
            # Generate a response indicating lack of context
            return "I don't have enough context from the document to answer this question. Please ensure the document contains relevant information."
        
        try:
            logger.info(f"Generating answer for question with {len(context_chunks)} context chunks")
            
            # Create optimized prompt
            prompt = self._create_context_prompt(question, context_chunks)
            
            # Make API call with retry logic
            raw_response = await self._make_api_call_with_retry(prompt)
            
            # Validate and format response
            formatted_response = self._validate_and_format_response(raw_response, question)
            
            logger.info("Answer generated successfully")
            return formatted_response
            
        except LLMServiceError:
            # Re-raise LLM service errors as-is
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating answer: {str(e)}")
            raise LLMServiceError(
                "Failed to generate contextual answer",
                "ANSWER_GENERATION_ERROR",
                {"question": question, "error": str(e)}
            )
    
    async def generate_multiple_answers(
        self, 
        questions: List[str], 
        context_chunks_per_question: List[List[DocumentChunk]]
    ) -> List[str]:
        """
        Generate answers for multiple questions efficiently.
        
        Args:
            questions: List of questions to answer
            context_chunks_per_question: List of context chunks for each question
            
        Returns:
            List[str]: Generated answers corresponding to input questions
            
        Raises:
            LLMServiceError: If any answer generation fails
        """
        if len(questions) != len(context_chunks_per_question):
            raise LLMServiceError(
                "Number of questions must match number of context chunk lists",
                "QUESTION_CONTEXT_MISMATCH",
                {
                    "questions_count": len(questions),
                    "context_lists_count": len(context_chunks_per_question)
                }
            )
        
        logger.info(f"Generating answers for {len(questions)} questions")
        
        # Process questions concurrently for better performance
        tasks = []
        for i, (question, context_chunks) in enumerate(zip(questions, context_chunks_per_question)):
            task = self.generate_contextual_answer(question, context_chunks)
            tasks.append(task)
        
        try:
            # Wait for all answers to be generated
            answers = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check for any exceptions and convert them to proper errors
            final_answers = []
            for i, answer in enumerate(answers):
                if isinstance(answer, Exception):
                    logger.error(f"Failed to generate answer for question {i}: {str(answer)}")
                    # Provide a fallback answer instead of failing completely
                    final_answers.append(
                        f"I apologize, but I encountered an error while processing this question: {str(answer)}"
                    )
                else:
                    final_answers.append(answer)
            
            logger.info(f"Successfully generated {len(final_answers)} answers")
            return final_answers
            
        except Exception as e:
            logger.error(f"Failed to generate multiple answers: {str(e)}")
            raise LLMServiceError(
                "Failed to generate answers for multiple questions",
                "MULTIPLE_ANSWERS_ERROR",
                {"questions_count": len(questions), "error": str(e)}
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the LLM service.
        
        Returns:
            Dict[str, Any]: Health check results
        """
        try:
            # Test basic model initialization
            model = self._get_model()
            
            # Test a simple API call
            test_prompt = "Please respond with 'OK' to confirm the service is working."
            response = await asyncio.wait_for(
                asyncio.to_thread(model.generate_content, test_prompt),
                timeout=10
            )
            
            return {
                "status": "healthy",
                "model": self.settings.gemini_model,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "test_response_length": len(response.text) if response.text else 0
            }
            
        except Exception as e:
            logger.error(f"LLM service health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "model": self.settings.gemini_model,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }


# Global service instance
_llm_service = None


def get_llm_service() -> LLMService:
    """
    Get the global LLM service instance.
    
    Returns:
        LLMService: Configured LLM service instance
    """
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service