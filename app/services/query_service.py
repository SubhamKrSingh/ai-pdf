"""
Query processing service for handling question processing and answer generation.

This module implements the core query processing functionality including:
- Question-to-embedding conversion and semantic search execution
- Relevant chunk retrieval using vector similarity search
- Answer generation pipeline combining retrieved context with LLM processing
- Multi-question processing with proper answer correspondence
- Query result ranking and relevance filtering

Requirements implemented: 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.2, 6.4
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from uuid import uuid4

from app.config import get_settings
from app.models.schemas import DocumentChunk, SearchResult, QueryRequest, QueryResponse
from app.services.embedding_service import get_embedding_service, EmbeddingServiceError
from app.services.llm_service import get_llm_service, LLMServiceError
from app.data.vector_store import get_vector_store, VectorStoreError
from app.data.repository import get_repository, DatabaseError


logger = logging.getLogger(__name__)


class QueryServiceError(Exception):
    """Custom exception for query service errors."""
    
    def __init__(self, message: str, error_code: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class QueryService:
    """
    Service for processing natural language queries against document content.
    
    Orchestrates the complete query processing pipeline from question embedding
    to contextual answer generation using retrieved document chunks.
    """
    
    def __init__(self):
        """Initialize the query service with configuration and dependencies."""
        self.settings = get_settings()
        self._embedding_service = None
        self._llm_service = None
        self._vector_store = None
        self._repository = None
        
        # Query processing configuration
        self.default_top_k = 10
        self.min_similarity_threshold = 0.3
        self.max_context_chunks = 5
        self.max_context_length = 4000
    
    async def _get_embedding_service(self):
        """Get embedding service instance."""
        if self._embedding_service is None:
            self._embedding_service = await get_embedding_service()
        return self._embedding_service
    
    async def _get_llm_service(self):
        """Get LLM service instance."""
        if self._llm_service is None:
            self._llm_service = get_llm_service()
        return self._llm_service
    
    async def _get_vector_store(self):
        """Get vector store instance."""
        if self._vector_store is None:
            self._vector_store = await get_vector_store()
        return self._vector_store
    
    async def _get_repository(self):
        """Get repository instance."""
        if self._repository is None:
            self._repository = await get_repository()
        return self._repository
    
    async def _convert_question_to_embedding(self, question: str) -> List[float]:
        """
        Convert a natural language question to an embedding vector.
        
        Args:
            question: Natural language question
            
        Returns:
            List[float]: Question embedding vector
            
        Raises:
            QueryServiceError: If embedding conversion fails
        """
        try:
            embedding_service = await self._get_embedding_service()
            embedding = await embedding_service.generate_query_embedding(question)
            
            logger.debug(f"Generated embedding for question: {question[:50]}...")
            return embedding
            
        except EmbeddingServiceError as e:
            logger.error(f"Failed to generate question embedding: {e.message}")
            raise QueryServiceError(
                f"Failed to convert question to embedding: {e.message}",
                "QUESTION_EMBEDDING_ERROR",
                {"question": question, "embedding_error": e.error_code}
            )
        except Exception as e:
            logger.error(f"Unexpected error generating question embedding: {str(e)}")
            raise QueryServiceError(
                "Unexpected error during question embedding",
                "QUESTION_EMBEDDING_UNEXPECTED",
                {"question": question, "error": str(e)}
            )
    
    async def _perform_semantic_search(
        self,
        query_embedding: List[float],
        document_id: str,
        top_k: int = None,
        score_threshold: float = None
    ) -> List[SearchResult]:
        """
        Perform semantic search against stored document embeddings.
        
        Args:
            query_embedding: Query embedding vector
            document_id: Document ID to search within
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List[SearchResult]: Ranked search results
            
        Raises:
            QueryServiceError: If semantic search fails
        """
        try:
            vector_store = await self._get_vector_store()
            
            # Use default values if not provided
            if top_k is None:
                top_k = self.default_top_k
            if score_threshold is None:
                score_threshold = self.min_similarity_threshold
            
            # Perform similarity search
            search_results = await vector_store.similarity_search(
                query_vector=query_embedding,
                document_id=document_id,
                top_k=top_k,
                score_threshold=score_threshold
            )
            
            logger.info(f"Semantic search returned {len(search_results)} results for document {document_id}")
            return search_results
            
        except VectorStoreError as e:
            logger.error(f"Vector store search failed: {e.message}")
            raise QueryServiceError(
                f"Semantic search failed: {e.message}",
                "SEMANTIC_SEARCH_ERROR",
                {"document_id": document_id, "vector_error": e.error_code}
            )
        except Exception as e:
            logger.error(f"Unexpected error during semantic search: {str(e)}")
            raise QueryServiceError(
                "Unexpected error during semantic search",
                "SEMANTIC_SEARCH_UNEXPECTED",
                {"document_id": document_id, "error": str(e)}
            )
    
    def _rank_and_filter_results(
        self,
        search_results: List[SearchResult],
        question: str
    ) -> List[SearchResult]:
        """
        Rank and filter search results for optimal context selection.
        
        Args:
            search_results: Raw search results from vector store
            question: Original question for context
            
        Returns:
            List[SearchResult]: Filtered and ranked results
        """
        if not search_results:
            return []
        
        # Filter by minimum similarity threshold
        filtered_results = [
            result for result in search_results 
            if result.score >= self.min_similarity_threshold
        ]
        
        # Sort by similarity score (highest first)
        filtered_results.sort(key=lambda x: x.score, reverse=True)
        
        # Apply diversity filtering to avoid too similar chunks
        diverse_results = []
        for result in filtered_results:
            # Check if this result is too similar to already selected ones
            is_diverse = True
            for selected in diverse_results:
                # Simple diversity check based on content overlap
                content_overlap = self._calculate_content_overlap(
                    result.content, selected.content
                )
                if content_overlap > 0.8:  # 80% overlap threshold
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_results.append(result)
            
            # Limit to maximum context chunks
            if len(diverse_results) >= self.max_context_chunks:
                break
        
        logger.debug(f"Filtered {len(search_results)} results to {len(diverse_results)} diverse results")
        return diverse_results
    
    def _calculate_content_overlap(self, content1: str, content2: str) -> float:
        """
        Calculate content overlap between two text chunks.
        
        Args:
            content1: First text chunk
            content2: Second text chunk
            
        Returns:
            float: Overlap ratio between 0 and 1
        """
        if not content1 or not content2:
            return 0.0
        
        # Simple word-based overlap calculation
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _convert_search_results_to_chunks(
        self,
        search_results: List[SearchResult]
    ) -> List[DocumentChunk]:
        """
        Convert search results to document chunks for LLM processing.
        
        Args:
            search_results: Search results from vector store
            
        Returns:
            List[DocumentChunk]: Document chunks with metadata
        """
        chunks = []
        for result in search_results:
            chunk = DocumentChunk(
                id=result.chunk_id,
                document_id=result.document_id,
                content=result.content,
                metadata={
                    **result.metadata,
                    "similarity_score": result.score
                },
                chunk_index=result.metadata.get("chunk_index", 0),
                start_char=result.metadata.get("start_char"),
                end_char=result.metadata.get("end_char")
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _generate_contextual_answer(
        self,
        question: str,
        context_chunks: List[DocumentChunk]
    ) -> str:
        """
        Generate a contextual answer using retrieved document chunks.
        
        Args:
            question: Natural language question
            context_chunks: Relevant document chunks for context
            
        Returns:
            str: Generated contextual answer
            
        Raises:
            QueryServiceError: If answer generation fails
        """
        try:
            llm_service = await self._get_llm_service()
            
            # Limit context length to avoid token limits
            limited_chunks = self._limit_context_length(context_chunks)
            
            # Generate answer using LLM service
            answer = await llm_service.generate_contextual_answer(
                question=question,
                context_chunks=limited_chunks
            )
            
            logger.debug(f"Generated answer for question: {question[:50]}...")
            return answer
            
        except LLMServiceError as e:
            logger.error(f"LLM answer generation failed: {e.message}")
            raise QueryServiceError(
                f"Failed to generate answer: {e.message}",
                "ANSWER_GENERATION_ERROR",
                {"question": question, "llm_error": e.error_code}
            )
        except Exception as e:
            logger.error(f"Unexpected error generating answer: {str(e)}")
            raise QueryServiceError(
                "Unexpected error during answer generation",
                "ANSWER_GENERATION_UNEXPECTED",
                {"question": question, "error": str(e)}
            )
    
    def _limit_context_length(
        self,
        context_chunks: List[DocumentChunk]
    ) -> List[DocumentChunk]:
        """
        Limit context chunks to stay within token limits.
        
        Args:
            context_chunks: Original context chunks
            
        Returns:
            List[DocumentChunk]: Limited context chunks
        """
        if not context_chunks:
            return []
        
        limited_chunks = []
        total_length = 0
        
        for chunk in context_chunks:
            chunk_length = len(chunk.content)
            
            if total_length + chunk_length <= self.max_context_length:
                limited_chunks.append(chunk)
                total_length += chunk_length
            else:
                # Try to include partial chunk if there's remaining space
                remaining_space = self.max_context_length - total_length
                if remaining_space > 100:  # Minimum useful chunk size
                    truncated_chunk = DocumentChunk(
                        id=chunk.id,
                        document_id=chunk.document_id,
                        content=chunk.content[:remaining_space] + "...",
                        metadata={**chunk.metadata, "truncated": True},
                        chunk_index=chunk.chunk_index,
                        start_char=chunk.start_char,
                        end_char=chunk.end_char
                    )
                    limited_chunks.append(truncated_chunk)
                break
        
        if len(limited_chunks) < len(context_chunks):
            logger.debug(f"Limited context from {len(context_chunks)} to {len(limited_chunks)} chunks")
        
        return limited_chunks
    
    async def retrieve_relevant_chunks(
        self,
        question: str,
        document_id: str,
        top_k: int = None,
        score_threshold: float = None
    ) -> List[DocumentChunk]:
        """
        Retrieve relevant document chunks for a given question.
        
        This method implements the core retrieval functionality by:
        1. Converting question to embedding
        2. Performing semantic search
        3. Ranking and filtering results
        
        Args:
            question: Natural language question
            document_id: Document ID to search within
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List[DocumentChunk]: Relevant document chunks
            
        Raises:
            QueryServiceError: If retrieval fails
        """
        if not question or not question.strip():
            raise QueryServiceError(
                "Question cannot be empty",
                "EMPTY_QUESTION",
                {"question": question}
            )
        
        if not document_id:
            raise QueryServiceError(
                "Document ID cannot be empty",
                "EMPTY_DOCUMENT_ID",
                {"document_id": document_id}
            )
        
        try:
            logger.info(f"Retrieving relevant chunks for question in document {document_id}")
            
            # Step 1: Convert question to embedding
            question_embedding = await self._convert_question_to_embedding(question)
            
            # Step 2: Perform semantic search
            search_results = await self._perform_semantic_search(
                query_embedding=question_embedding,
                document_id=document_id,
                top_k=top_k,
                score_threshold=score_threshold
            )
            
            # Step 3: Rank and filter results
            filtered_results = self._rank_and_filter_results(search_results, question)
            
            # Step 4: Convert to document chunks
            relevant_chunks = self._convert_search_results_to_chunks(filtered_results)
            
            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks for question")
            return relevant_chunks
            
        except QueryServiceError:
            # Re-raise query service errors as-is
            raise
        except Exception as e:
            logger.error(f"Unexpected error retrieving relevant chunks: {str(e)}")
            raise QueryServiceError(
                "Failed to retrieve relevant chunks",
                "CHUNK_RETRIEVAL_ERROR",
                {"question": question, "document_id": document_id, "error": str(e)}
            )
    
    async def process_single_question(
        self,
        question: str,
        document_id: str
    ) -> str:
        """
        Process a single question and generate an answer.
        
        Args:
            question: Natural language question
            document_id: Document ID to search within
            
        Returns:
            str: Generated answer
            
        Raises:
            QueryServiceError: If processing fails
        """
        try:
            logger.info(f"Processing single question for document {document_id}")
            
            # Retrieve relevant chunks
            relevant_chunks = await self.retrieve_relevant_chunks(question, document_id)
            
            # Generate contextual answer
            answer = await self._generate_contextual_answer(question, relevant_chunks)
            
            logger.info("Successfully processed single question")
            return answer
            
        except QueryServiceError:
            # Re-raise query service errors as-is
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing single question: {str(e)}")
            raise QueryServiceError(
                "Failed to process single question",
                "SINGLE_QUESTION_ERROR",
                {"question": question, "document_id": document_id, "error": str(e)}
            )
    
    async def process_multiple_questions(
        self,
        questions: List[str],
        document_id: str
    ) -> List[str]:
        """
        Process multiple questions with proper answer correspondence.
        
        This method implements multi-question processing by:
        1. Processing each question independently
        2. Maintaining answer correspondence with input questions
        3. Handling partial failures gracefully
        
        Args:
            questions: List of natural language questions
            document_id: Document ID to search within
            
        Returns:
            List[str]: Generated answers corresponding to input questions
            
        Raises:
            QueryServiceError: If processing fails
        """
        if not questions:
            raise QueryServiceError(
                "Questions list cannot be empty",
                "EMPTY_QUESTIONS_LIST"
            )
        
        if not document_id:
            raise QueryServiceError(
                "Document ID cannot be empty",
                "EMPTY_DOCUMENT_ID",
                {"document_id": document_id}
            )
        
        try:
            logger.info(f"Processing {len(questions)} questions for document {document_id}")
            
            # Process questions concurrently for better performance
            tasks = []
            for i, question in enumerate(questions):
                task = self.process_single_question(question, document_id)
                tasks.append(task)
            
            # Wait for all questions to be processed
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle any exceptions
            answers = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process question {i}: {str(result)}")
                    # Provide fallback answer instead of failing completely
                    fallback_answer = (
                        f"I apologize, but I encountered an error while processing this question: "
                        f"{str(result)}"
                    )
                    answers.append(fallback_answer)
                else:
                    answers.append(result)
            
            # Ensure answer correspondence
            if len(answers) != len(questions):
                raise QueryServiceError(
                    "Answer count does not match question count",
                    "ANSWER_CORRESPONDENCE_ERROR",
                    {"questions_count": len(questions), "answers_count": len(answers)}
                )
            
            logger.info(f"Successfully processed {len(questions)} questions")
            return answers
            
        except QueryServiceError:
            # Re-raise query service errors as-is
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing multiple questions: {str(e)}")
            raise QueryServiceError(
                "Failed to process multiple questions",
                "MULTIPLE_QUESTIONS_ERROR",
                {"questions_count": len(questions), "document_id": document_id, "error": str(e)}
            )
    
    async def process_query_request(
        self,
        request: QueryRequest,
        document_id: str
    ) -> QueryResponse:
        """
        Process a complete query request and return structured response.
        
        This is the main entry point for query processing that orchestrates
        the complete pipeline from request validation to response generation.
        
        Args:
            request: Validated query request
            document_id: Document ID for the processed document
            
        Returns:
            QueryResponse: Structured response with answers
            
        Raises:
            QueryServiceError: If query processing fails
        """
        try:
            logger.info(f"Processing query request with {len(request.questions)} questions")
            
            # Track processing time
            start_time = datetime.now(timezone.utc)
            
            # Process questions
            answers = await self.process_multiple_questions(
                questions=request.questions,
                document_id=document_id
            )
            
            # Create response
            response = QueryResponse(answers=answers)
            
            # Log query session
            try:
                end_time = datetime.now(timezone.utc)
                processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
                
                repository = await self._get_repository()
                await repository.log_query_session(
                    document_id=document_id,
                    questions=request.questions,
                    answers=answers,
                    processing_time_ms=processing_time_ms
                )
            except Exception as e:
                logger.warning(f"Failed to log query session: {str(e)}")
            
            logger.info("Successfully processed query request")
            return response
            
        except QueryServiceError:
            # Re-raise query service errors as-is
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing query request: {str(e)}")
            raise QueryServiceError(
                "Failed to process query request",
                "QUERY_REQUEST_ERROR",
                {"questions_count": len(request.questions), "error": str(e)}
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the query service and its dependencies.
        
        Returns:
            Dict[str, Any]: Health check results
        """
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dependencies": {}
        }
        
        try:
            # Check embedding service
            try:
                embedding_service = await self._get_embedding_service()
                test_embedding = await embedding_service.generate_query_embedding("test")
                health_status["dependencies"]["embedding_service"] = {
                    "status": "healthy",
                    "embedding_dimension": len(test_embedding)
                }
            except Exception as e:
                health_status["dependencies"]["embedding_service"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
            
            # Check LLM service
            try:
                llm_service = await self._get_llm_service()
                llm_health = await llm_service.health_check()
                health_status["dependencies"]["llm_service"] = llm_health
                if llm_health["status"] != "healthy":
                    health_status["status"] = "degraded"
            except Exception as e:
                health_status["dependencies"]["llm_service"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
            
            # Check vector store
            try:
                vector_store = await self._get_vector_store()
                await vector_store.health_check()
                health_status["dependencies"]["vector_store"] = {"status": "healthy"}
            except Exception as e:
                health_status["dependencies"]["vector_store"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
            
            # Check repository
            try:
                repository = await self._get_repository()
                await repository.health_check()
                health_status["dependencies"]["repository"] = {"status": "healthy"}
            except Exception as e:
                health_status["dependencies"]["repository"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
            
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
        
        return health_status


# Global service instance
_query_service: Optional[QueryService] = None


async def get_query_service() -> QueryService:
    """
    Get or create the global query service instance.
    
    Returns:
        QueryService: Configured query service instance
    """
    global _query_service
    if _query_service is None:
        _query_service = QueryService()
    return _query_service


async def cleanup_query_service() -> None:
    """Clean up the global query service instance."""
    global _query_service
    if _query_service is not None:
        # Query service doesn't require explicit cleanup
        _query_service = None