"""
Integration tests for the query processing service.

Tests the complete query processing workflow including:
- Question-to-embedding conversion and semantic search execution
- Relevant chunk retrieval using vector similarity search
- Answer generation pipeline combining retrieved context with LLM processing
- Multi-question processing with proper answer correspondence
- Query result ranking and relevance filtering
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any
import os

from app.services.query_service import QueryService, QueryServiceError, get_query_service, cleanup_query_service
from app.models.schemas import QueryRequest, QueryResponse, DocumentChunk, SearchResult
from app.services.embedding_service import EmbeddingServiceError
from app.services.llm_service import LLMServiceError
from app.data.vector_store import VectorStoreError
from app.data.repository import DatabaseError


@pytest.fixture
async def query_service():
    """Create a query service instance for testing."""
    # Mock environment variables for testing
    with patch.dict(os.environ, {
        'AUTH_TOKEN': 'test_token',
        'GEMINI_API_KEY': 'test_gemini_key',
        'JINA_API_KEY': 'test_jina_key',
        'PINECONE_API_KEY': 'test_pinecone_key',
        'PINECONE_ENVIRONMENT': 'test_env',
        'DATABASE_URL': 'postgresql://test:test@localhost:5432/test'
    }):
        service = QueryService()
        yield service
        # Cleanup if needed
        await cleanup_query_service()


@pytest.fixture
def sample_document_chunks():
    """Sample document chunks for testing."""
    return [
        DocumentChunk(
            id="chunk_1",
            document_id="doc_123",
            content="This is the first chunk about artificial intelligence applications in healthcare.",
            metadata={"page_number": 1, "section": "Introduction"},
            chunk_index=0,
            start_char=0,
            end_char=100
        ),
        DocumentChunk(
            id="chunk_2", 
            document_id="doc_123",
            content="Machine learning algorithms have shown significant improvements in diagnostic accuracy.",
            metadata={"page_number": 2, "section": "Results"},
            chunk_index=1,
            start_char=100,
            end_char=200
        ),
        DocumentChunk(
            id="chunk_3",
            document_id="doc_123", 
            content="The study concludes that AI can reduce processing time by 50% in medical imaging.",
            metadata={"page_number": 3, "section": "Conclusion"},
            chunk_index=2,
            start_char=200,
            end_char=300
        )
    ]


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return [
        SearchResult(
            chunk_id="chunk_1",
            content="This is the first chunk about artificial intelligence applications in healthcare.",
            score=0.95,
            metadata={"page_number": 1, "section": "Introduction", "chunk_index": 0},
            document_id="doc_123"
        ),
        SearchResult(
            chunk_id="chunk_2",
            content="Machine learning algorithms have shown significant improvements in diagnostic accuracy.",
            score=0.87,
            metadata={"page_number": 2, "section": "Results", "chunk_index": 1},
            document_id="doc_123"
        ),
        SearchResult(
            chunk_id="chunk_3",
            content="The study concludes that AI can reduce processing time by 50% in medical imaging.",
            score=0.82,
            metadata={"page_number": 3, "section": "Conclusion", "chunk_index": 2},
            document_id="doc_123"
        )
    ]


@pytest.fixture
def sample_query_request():
    """Sample query request for testing."""
    return QueryRequest(
        documents="https://example.com/test-document.pdf",
        questions=[
            "What are the applications of AI in healthcare?",
            "How much can AI reduce processing time?",
            "What improvements were shown in diagnostic accuracy?"
        ]
    )


class TestQueryServiceBasicFunctionality:
    """Test basic query service functionality."""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, query_service):
        """Test that query service initializes correctly."""
        assert query_service is not None
        assert query_service.default_top_k == 10
        assert query_service.min_similarity_threshold == 0.3
        assert query_service.max_context_chunks == 5
        assert query_service.max_context_length == 4000
    
    @pytest.mark.asyncio
    async def test_global_service_instance(self):
        """Test that global service instance works correctly."""
        with patch.dict(os.environ, {
            'AUTH_TOKEN': 'test_token',
            'GEMINI_API_KEY': 'test_gemini_key',
            'JINA_API_KEY': 'test_jina_key',
            'PINECONE_API_KEY': 'test_pinecone_key',
            'PINECONE_ENVIRONMENT': 'test_env',
            'DATABASE_URL': 'postgresql://test:test@localhost:5432/test'
        }):
            service1 = await get_query_service()
            service2 = await get_query_service()
            assert service1 is service2  # Should be the same instance


class TestQuestionEmbeddingConversion:
    """Test question-to-embedding conversion functionality."""
    
    @pytest.mark.asyncio
    async def test_convert_question_to_embedding_success(self, query_service):
        """Test successful question-to-embedding conversion."""
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 200  # 1000-dim embedding
        
        with patch.object(query_service, '_get_embedding_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.generate_query_embedding.return_value = mock_embedding
            mock_get_service.return_value = mock_service
            
            result = await query_service._convert_question_to_embedding("What is AI?")
            
            assert result == mock_embedding
            mock_service.generate_query_embedding.assert_called_once_with("What is AI?")
    
    @pytest.mark.asyncio
    async def test_convert_question_to_embedding_service_error(self, query_service):
        """Test handling of embedding service errors."""
        with patch.object(query_service, '_get_embedding_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.generate_query_embedding.side_effect = EmbeddingServiceError(
                "API error", "EMBEDDING_API_ERROR"
            )
            mock_get_service.return_value = mock_service
            
            with pytest.raises(QueryServiceError) as exc_info:
                await query_service._convert_question_to_embedding("What is AI?")
            
            assert exc_info.value.error_code == "QUESTION_EMBEDDING_ERROR"
            assert "Failed to convert question to embedding" in exc_info.value.message
    
    @pytest.mark.asyncio
    async def test_convert_question_to_embedding_unexpected_error(self, query_service):
        """Test handling of unexpected errors during embedding conversion."""
        with patch.object(query_service, '_get_embedding_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.generate_query_embedding.side_effect = Exception("Unexpected error")
            mock_get_service.return_value = mock_service
            
            with pytest.raises(QueryServiceError) as exc_info:
                await query_service._convert_question_to_embedding("What is AI?")
            
            assert exc_info.value.error_code == "QUESTION_EMBEDDING_UNEXPECTED"


class TestSemanticSearch:
    """Test semantic search functionality."""
    
    @pytest.mark.asyncio
    async def test_perform_semantic_search_success(self, query_service, sample_search_results):
        """Test successful semantic search execution."""
        query_embedding = [0.1, 0.2, 0.3] * 300  # Mock embedding
        
        with patch.object(query_service, '_get_vector_store') as mock_get_store:
            mock_store = AsyncMock()
            mock_store.similarity_search.return_value = sample_search_results
            mock_get_store.return_value = mock_store
            
            results = await query_service._perform_semantic_search(
                query_embedding=query_embedding,
                document_id="doc_123",
                top_k=5,
                score_threshold=0.3
            )
            
            assert len(results) == 3
            assert results[0].score == 0.95
            assert results[0].chunk_id == "chunk_1"
            
            mock_store.similarity_search.assert_called_once_with(
                query_vector=query_embedding,
                document_id="doc_123",
                top_k=5,
                score_threshold=0.3
            )
    
    @pytest.mark.asyncio
    async def test_perform_semantic_search_with_defaults(self, query_service, sample_search_results):
        """Test semantic search with default parameters."""
        query_embedding = [0.1, 0.2, 0.3] * 300
        
        with patch.object(query_service, '_get_vector_store') as mock_get_store:
            mock_store = AsyncMock()
            mock_store.similarity_search.return_value = sample_search_results
            mock_get_store.return_value = mock_store
            
            results = await query_service._perform_semantic_search(
                query_embedding=query_embedding,
                document_id="doc_123"
            )
            
            mock_store.similarity_search.assert_called_once_with(
                query_vector=query_embedding,
                document_id="doc_123",
                top_k=10,  # default
                score_threshold=0.3  # default
            )
    
    @pytest.mark.asyncio
    async def test_perform_semantic_search_vector_store_error(self, query_service):
        """Test handling of vector store errors during search."""
        query_embedding = [0.1, 0.2, 0.3] * 300
        
        with patch.object(query_service, '_get_vector_store') as mock_get_store:
            mock_store = AsyncMock()
            mock_store.similarity_search.side_effect = VectorStoreError(
                "Search failed", "VECTOR_SEARCH_ERROR"
            )
            mock_get_store.return_value = mock_store
            
            with pytest.raises(QueryServiceError) as exc_info:
                await query_service._perform_semantic_search(
                    query_embedding=query_embedding,
                    document_id="doc_123"
                )
            
            assert exc_info.value.error_code == "SEMANTIC_SEARCH_ERROR"


class TestResultRankingAndFiltering:
    """Test result ranking and filtering functionality."""
    
    def test_rank_and_filter_results_basic(self, query_service, sample_search_results):
        """Test basic result ranking and filtering."""
        # Test with results above threshold
        filtered = query_service._rank_and_filter_results(sample_search_results, "test question")
        
        assert len(filtered) == 3
        assert filtered[0].score >= filtered[1].score >= filtered[2].score
        assert all(result.score >= query_service.min_similarity_threshold for result in filtered)
    
    def test_rank_and_filter_results_threshold_filtering(self, query_service):
        """Test filtering by similarity threshold."""
        low_score_results = [
            SearchResult(
                chunk_id="chunk_1",
                content="Low relevance content",
                score=0.2,  # Below threshold
                metadata={},
                document_id="doc_123"
            ),
            SearchResult(
                chunk_id="chunk_2",
                content="High relevance content",
                score=0.8,  # Above threshold
                metadata={},
                document_id="doc_123"
            )
        ]
        
        filtered = query_service._rank_and_filter_results(low_score_results, "test question")
        
        assert len(filtered) == 1
        assert filtered[0].score == 0.8
    
    def test_rank_and_filter_results_diversity_filtering(self, query_service):
        """Test diversity filtering to avoid similar chunks."""
        similar_results = [
            SearchResult(
                chunk_id="chunk_1",
                content="artificial intelligence machine learning AI ML",
                score=0.9,
                metadata={},
                document_id="doc_123"
            ),
            SearchResult(
                chunk_id="chunk_2",
                content="artificial intelligence machine learning AI ML algorithms",  # Very similar
                score=0.85,
                metadata={},
                document_id="doc_123"
            ),
            SearchResult(
                chunk_id="chunk_3",
                content="healthcare medical diagnosis treatment",  # Different content
                score=0.8,
                metadata={},
                document_id="doc_123"
            )
        ]
        
        filtered = query_service._rank_and_filter_results(similar_results, "test question")
        
        # Should keep the highest scoring and the diverse one
        assert len(filtered) == 2
        assert filtered[0].chunk_id == "chunk_1"  # Highest score
        assert filtered[1].chunk_id == "chunk_3"  # Diverse content
    
    def test_rank_and_filter_results_max_chunks_limit(self, query_service):
        """Test limiting to maximum context chunks."""
        # Create more results than max_context_chunks
        many_results = []
        for i in range(10):
            result = SearchResult(
                chunk_id=f"chunk_{i}",
                content=f"Unique content for chunk {i}",
                score=0.9 - (i * 0.05),  # Decreasing scores
                metadata={},
                document_id="doc_123"
            )
            many_results.append(result)
        
        filtered = query_service._rank_and_filter_results(many_results, "test question")
        
        assert len(filtered) <= query_service.max_context_chunks
    
    def test_calculate_content_overlap(self, query_service):
        """Test content overlap calculation."""
        content1 = "artificial intelligence machine learning"
        content2 = "artificial intelligence deep learning"
        content3 = "healthcare medical diagnosis"
        
        # Test overlap between similar content
        overlap1 = query_service._calculate_content_overlap(content1, content2)
        assert 0.4 < overlap1 < 0.8  # Some overlap
        
        # Test overlap between different content
        overlap2 = query_service._calculate_content_overlap(content1, content3)
        assert overlap2 < 0.3  # Little overlap
        
        # Test identical content
        overlap3 = query_service._calculate_content_overlap(content1, content1)
        assert overlap3 == 1.0  # Complete overlap


class TestChunkRetrieval:
    """Test relevant chunk retrieval functionality."""
    
    @pytest.mark.asyncio
    async def test_retrieve_relevant_chunks_success(self, query_service, sample_search_results):
        """Test successful chunk retrieval workflow."""
        mock_embedding = [0.1, 0.2, 0.3] * 300
        
        with patch.object(query_service, '_convert_question_to_embedding') as mock_embed, \
             patch.object(query_service, '_perform_semantic_search') as mock_search, \
             patch.object(query_service, '_rank_and_filter_results') as mock_filter:
            
            mock_embed.return_value = mock_embedding
            mock_search.return_value = sample_search_results
            mock_filter.return_value = sample_search_results[:2]  # Return top 2
            
            chunks = await query_service.retrieve_relevant_chunks(
                question="What is AI?",
                document_id="doc_123"
            )
            
            assert len(chunks) == 2
            assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
            assert chunks[0].id == "chunk_1"
            assert chunks[1].id == "chunk_2"
            
            mock_embed.assert_called_once_with("What is AI?")
            mock_search.assert_called_once_with(
                query_embedding=mock_embedding,
                document_id="doc_123",
                top_k=None,
                score_threshold=None
            )
            mock_filter.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retrieve_relevant_chunks_empty_question(self, query_service):
        """Test handling of empty questions."""
        with pytest.raises(QueryServiceError) as exc_info:
            await query_service.retrieve_relevant_chunks("", "doc_123")
        
        assert exc_info.value.error_code == "EMPTY_QUESTION"
        
        with pytest.raises(QueryServiceError) as exc_info:
            await query_service.retrieve_relevant_chunks("   ", "doc_123")
        
        assert exc_info.value.error_code == "EMPTY_QUESTION"
    
    @pytest.mark.asyncio
    async def test_retrieve_relevant_chunks_empty_document_id(self, query_service):
        """Test handling of empty document ID."""
        with pytest.raises(QueryServiceError) as exc_info:
            await query_service.retrieve_relevant_chunks("What is AI?", "")
        
        assert exc_info.value.error_code == "EMPTY_DOCUMENT_ID"
    
    @pytest.mark.asyncio
    async def test_retrieve_relevant_chunks_with_parameters(self, query_service, sample_search_results):
        """Test chunk retrieval with custom parameters."""
        mock_embedding = [0.1, 0.2, 0.3] * 300
        
        with patch.object(query_service, '_convert_question_to_embedding') as mock_embed, \
             patch.object(query_service, '_perform_semantic_search') as mock_search, \
             patch.object(query_service, '_rank_and_filter_results') as mock_filter:
            
            mock_embed.return_value = mock_embedding
            mock_search.return_value = sample_search_results
            mock_filter.return_value = sample_search_results
            
            await query_service.retrieve_relevant_chunks(
                question="What is AI?",
                document_id="doc_123",
                top_k=15,
                score_threshold=0.5
            )
            
            mock_search.assert_called_once_with(
                query_embedding=mock_embedding,
                document_id="doc_123",
                top_k=15,
                score_threshold=0.5
            )


class TestAnswerGeneration:
    """Test answer generation functionality."""
    
    @pytest.mark.asyncio
    async def test_generate_contextual_answer_success(self, query_service, sample_document_chunks):
        """Test successful answer generation."""
        expected_answer = "AI has many applications in healthcare including diagnostic assistance."
        
        with patch.object(query_service, '_get_llm_service') as mock_get_service, \
             patch.object(query_service, '_limit_context_length') as mock_limit:
            
            mock_service = AsyncMock()
            mock_service.generate_contextual_answer.return_value = expected_answer
            mock_get_service.return_value = mock_service
            mock_limit.return_value = sample_document_chunks
            
            answer = await query_service._generate_contextual_answer(
                question="What are AI applications?",
                context_chunks=sample_document_chunks
            )
            
            assert answer == expected_answer
            mock_service.generate_contextual_answer.assert_called_once_with(
                question="What are AI applications?",
                context_chunks=sample_document_chunks
            )
    
    @pytest.mark.asyncio
    async def test_generate_contextual_answer_llm_error(self, query_service, sample_document_chunks):
        """Test handling of LLM service errors."""
        with patch.object(query_service, '_get_llm_service') as mock_get_service, \
             patch.object(query_service, '_limit_context_length') as mock_limit:
            
            mock_service = AsyncMock()
            mock_service.generate_contextual_answer.side_effect = LLMServiceError(
                "LLM API failed", "LLM_API_ERROR"
            )
            mock_get_service.return_value = mock_service
            mock_limit.return_value = sample_document_chunks
            
            with pytest.raises(QueryServiceError) as exc_info:
                await query_service._generate_contextual_answer(
                    question="What are AI applications?",
                    context_chunks=sample_document_chunks
                )
            
            assert exc_info.value.error_code == "ANSWER_GENERATION_ERROR"
    
    def test_limit_context_length_within_limit(self, query_service, sample_document_chunks):
        """Test context limiting when within limits."""
        # All chunks should fit within default limit
        limited = query_service._limit_context_length(sample_document_chunks)
        
        assert len(limited) == len(sample_document_chunks)
        assert limited == sample_document_chunks
    
    def test_limit_context_length_exceeds_limit(self, query_service):
        """Test context limiting when exceeding limits."""
        # Create chunks that exceed the limit
        large_chunks = []
        for i in range(5):
            chunk = DocumentChunk(
                id=f"chunk_{i}",
                document_id="doc_123",
                content="x" * 1000,  # 1000 chars each
                metadata={},
                chunk_index=i
            )
            large_chunks.append(chunk)
        
        # Set a small limit for testing
        query_service.max_context_length = 2500
        
        limited = query_service._limit_context_length(large_chunks)
        
        # Should only include first 2 chunks + partial third
        assert len(limited) <= 3
        total_length = sum(len(chunk.content) for chunk in limited)
        assert total_length <= query_service.max_context_length


class TestSingleQuestionProcessing:
    """Test single question processing functionality."""
    
    @pytest.mark.asyncio
    async def test_process_single_question_success(self, query_service, sample_document_chunks):
        """Test successful single question processing."""
        expected_answer = "AI applications include diagnostic assistance and treatment planning."
        
        with patch.object(query_service, 'retrieve_relevant_chunks') as mock_retrieve, \
             patch.object(query_service, '_generate_contextual_answer') as mock_generate:
            
            mock_retrieve.return_value = sample_document_chunks
            mock_generate.return_value = expected_answer
            
            answer = await query_service.process_single_question(
                question="What are AI applications?",
                document_id="doc_123"
            )
            
            assert answer == expected_answer
            mock_retrieve.assert_called_once_with("What are AI applications?", "doc_123")
            mock_generate.assert_called_once_with("What are AI applications?", sample_document_chunks)
    
    @pytest.mark.asyncio
    async def test_process_single_question_retrieval_error(self, query_service):
        """Test handling of chunk retrieval errors."""
        with patch.object(query_service, 'retrieve_relevant_chunks') as mock_retrieve:
            mock_retrieve.side_effect = QueryServiceError(
                "Retrieval failed", "CHUNK_RETRIEVAL_ERROR"
            )
            
            with pytest.raises(QueryServiceError) as exc_info:
                await query_service.process_single_question(
                    question="What are AI applications?",
                    document_id="doc_123"
                )
            
            assert exc_info.value.error_code == "CHUNK_RETRIEVAL_ERROR"


class TestMultipleQuestionProcessing:
    """Test multiple question processing functionality."""
    
    @pytest.mark.asyncio
    async def test_process_multiple_questions_success(self, query_service):
        """Test successful multiple question processing."""
        questions = [
            "What is AI?",
            "How does ML work?",
            "What are the benefits?"
        ]
        expected_answers = [
            "AI is artificial intelligence.",
            "ML uses algorithms to learn patterns.",
            "Benefits include efficiency and accuracy."
        ]
        
        with patch.object(query_service, 'process_single_question') as mock_process:
            mock_process.side_effect = expected_answers
            
            answers = await query_service.process_multiple_questions(
                questions=questions,
                document_id="doc_123"
            )
            
            assert len(answers) == len(questions)
            assert answers == expected_answers
            assert mock_process.call_count == 3
    
    @pytest.mark.asyncio
    async def test_process_multiple_questions_empty_list(self, query_service):
        """Test handling of empty questions list."""
        with pytest.raises(QueryServiceError) as exc_info:
            await query_service.process_multiple_questions([], "doc_123")
        
        assert exc_info.value.error_code == "EMPTY_QUESTIONS_LIST"
    
    @pytest.mark.asyncio
    async def test_process_multiple_questions_empty_document_id(self, query_service):
        """Test handling of empty document ID."""
        with pytest.raises(QueryServiceError) as exc_info:
            await query_service.process_multiple_questions(["What is AI?"], "")
        
        assert exc_info.value.error_code == "EMPTY_DOCUMENT_ID"
    
    @pytest.mark.asyncio
    async def test_process_multiple_questions_partial_failure(self, query_service):
        """Test handling of partial failures in multiple question processing."""
        questions = ["What is AI?", "How does ML work?", "What are benefits?"]
        
        def mock_process_side_effect(question, doc_id):
            if "ML" in question:
                raise QueryServiceError("Processing failed", "PROCESSING_ERROR")
            return f"Answer for: {question}"
        
        with patch.object(query_service, 'process_single_question') as mock_process:
            mock_process.side_effect = mock_process_side_effect
            
            answers = await query_service.process_multiple_questions(
                questions=questions,
                document_id="doc_123"
            )
            
            assert len(answers) == 3
            assert "Answer for: What is AI?" in answers[0]
            assert "error while processing" in answers[1].lower()  # Fallback answer
            assert "Answer for: What are benefits?" in answers[2]


class TestQueryRequestProcessing:
    """Test complete query request processing."""
    
    @pytest.mark.asyncio
    async def test_process_query_request_success(self, query_service, sample_query_request):
        """Test successful query request processing."""
        expected_answers = [
            "AI applications in healthcare include diagnostic assistance.",
            "AI can reduce processing time by 50%.",
            "Diagnostic accuracy improvements were significant."
        ]
        
        with patch.object(query_service, 'process_multiple_questions') as mock_process, \
             patch.object(query_service, '_get_repository') as mock_get_repo:
            
            mock_process.return_value = expected_answers
            mock_repo = AsyncMock()
            mock_get_repo.return_value = mock_repo
            
            response = await query_service.process_query_request(
                request=sample_query_request,
                document_id="doc_123"
            )
            
            assert isinstance(response, QueryResponse)
            assert len(response.answers) == 3
            assert response.answers == expected_answers
            
            # Verify repository logging was attempted
            mock_repo.log_query_session.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_query_request_processing_error(self, query_service, sample_query_request):
        """Test handling of processing errors in query request."""
        with patch.object(query_service, 'process_multiple_questions') as mock_process:
            mock_process.side_effect = QueryServiceError(
                "Processing failed", "PROCESSING_ERROR"
            )
            
            with pytest.raises(QueryServiceError) as exc_info:
                await query_service.process_query_request(
                    request=sample_query_request,
                    document_id="doc_123"
                )
            
            assert exc_info.value.error_code == "PROCESSING_ERROR"
    
    @pytest.mark.asyncio
    async def test_process_query_request_logging_failure(self, query_service, sample_query_request):
        """Test that logging failures don't break the main processing."""
        expected_answers = ["Answer 1", "Answer 2", "Answer 3"]
        
        with patch.object(query_service, 'process_multiple_questions') as mock_process, \
             patch.object(query_service, '_get_repository') as mock_get_repo:
            
            mock_process.return_value = expected_answers
            mock_repo = AsyncMock()
            mock_repo.log_query_session.side_effect = DatabaseError(
                "Logging failed", "LOG_ERROR"
            )
            mock_get_repo.return_value = mock_repo
            
            # Should still succeed despite logging failure
            response = await query_service.process_query_request(
                request=sample_query_request,
                document_id="doc_123"
            )
            
            assert isinstance(response, QueryResponse)
            assert response.answers == expected_answers


class TestHealthCheck:
    """Test health check functionality."""
    
    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self, query_service):
        """Test health check when all dependencies are healthy."""
        with patch.object(query_service, '_get_embedding_service') as mock_get_embed, \
             patch.object(query_service, '_get_llm_service') as mock_get_llm, \
             patch.object(query_service, '_get_vector_store') as mock_get_vector, \
             patch.object(query_service, '_get_repository') as mock_get_repo:
            
            # Mock embedding service
            mock_embed_service = AsyncMock()
            mock_embed_service.generate_query_embedding.return_value = [0.1] * 1024
            mock_get_embed.return_value = mock_embed_service
            
            # Mock LLM service
            mock_llm_service = AsyncMock()
            mock_llm_service.health_check.return_value = {"status": "healthy"}
            mock_get_llm.return_value = mock_llm_service
            
            # Mock vector store
            mock_vector_service = AsyncMock()
            mock_vector_service.health_check.return_value = True
            mock_get_vector.return_value = mock_vector_service
            
            # Mock repository
            mock_repo_service = AsyncMock()
            mock_repo_service.health_check.return_value = {"status": "healthy"}
            mock_get_repo.return_value = mock_repo_service
            
            health = await query_service.health_check()
            
            assert health["status"] == "healthy"
            assert "dependencies" in health
            assert health["dependencies"]["embedding_service"]["status"] == "healthy"
            assert health["dependencies"]["llm_service"]["status"] == "healthy"
            assert health["dependencies"]["vector_store"]["status"] == "healthy"
            assert health["dependencies"]["repository"]["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_health_check_degraded_service(self, query_service):
        """Test health check when some dependencies are unhealthy."""
        with patch.object(query_service, '_get_embedding_service') as mock_get_embed, \
             patch.object(query_service, '_get_llm_service') as mock_get_llm, \
             patch.object(query_service, '_get_vector_store') as mock_get_vector, \
             patch.object(query_service, '_get_repository') as mock_get_repo:
            
            # Mock healthy embedding service
            mock_embed_service = AsyncMock()
            mock_embed_service.generate_query_embedding.return_value = [0.1] * 1024
            mock_get_embed.return_value = mock_embed_service
            
            # Mock unhealthy LLM service
            mock_llm_service = AsyncMock()
            mock_llm_service.health_check.return_value = {"status": "unhealthy", "error": "API down"}
            mock_get_llm.return_value = mock_llm_service
            
            # Mock healthy vector store
            mock_vector_service = AsyncMock()
            mock_vector_service.health_check.return_value = True
            mock_get_vector.return_value = mock_vector_service
            
            # Mock healthy repository
            mock_repo_service = AsyncMock()
            mock_repo_service.health_check.return_value = {"status": "healthy"}
            mock_get_repo.return_value = mock_repo_service
            
            health = await query_service.health_check()
            
            assert health["status"] == "degraded"  # Should be degraded due to unhealthy LLM
            assert health["dependencies"]["llm_service"]["status"] == "unhealthy"


class TestErrorHandling:
    """Test comprehensive error handling."""
    
    @pytest.mark.asyncio
    async def test_query_service_error_creation(self):
        """Test QueryServiceError creation and attributes."""
        error = QueryServiceError(
            message="Test error",
            error_code="TEST_ERROR",
            details={"key": "value"}
        )
        
        assert error.message == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.details == {"key": "value"}
        assert str(error) == "Test error"
    
    @pytest.mark.asyncio
    async def test_error_propagation_chain(self, query_service):
        """Test that errors propagate correctly through the service chain."""
        with patch.object(query_service, '_get_embedding_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.generate_query_embedding.side_effect = Exception("Network error")
            mock_get_service.return_value = mock_service
            
            with pytest.raises(QueryServiceError) as exc_info:
                await query_service.retrieve_relevant_chunks("What is AI?", "doc_123")
            
            # Should wrap the original exception
            assert exc_info.value.error_code == "QUESTION_EMBEDDING_UNEXPECTED"
            assert "Network error" in str(exc_info.value.details.get("error", ""))


if __name__ == "__main__":
    pytest.main([__file__])