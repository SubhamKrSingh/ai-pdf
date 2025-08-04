"""End-to-end tests for the complete API workflow."""

import pytest
import asyncio
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch, MagicMock
import json
import tempfile
import os

from main import app
from tests.fixtures.sample_documents import (
    TestDocumentFactory, 
    SAMPLE_TEXT_CONTENT,
    SAMPLE_QUESTIONS,
    INSURANCE_QUESTIONS,
    SAMPLE_INSURANCE_CONTENT
)
from tests.fixtures.database_fixtures import (
    MockPinecone,
    test_settings,
    DatabaseTestHelper
)

class TestCompleteWorkflow:
    """Test the complete API workflow from document processing to answer generation."""
    
    @pytest.fixture
    async def client(self):
        """Create test client."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    @pytest.fixture
    def mock_external_services(self):
        """Mock all external services for e2e testing."""
        with patch('app.utils.document_downloader.aiohttp.ClientSession') as mock_session, \
             patch('app.services.embedding_service.aiohttp.ClientSession') as mock_embed_session, \
             patch('app.services.llm_service.aiohttp.ClientSession') as mock_llm_session, \
             patch('app.data.vector_store.pinecone') as mock_pinecone, \
             patch('app.data.repository.asyncpg') as mock_asyncpg:
            
            # Mock document download
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {'content-type': 'application/pdf'}
            mock_response.read.return_value = TestDocumentFactory.create_sample_pdf()
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            # Mock embedding service
            mock_embed_response = AsyncMock()
            mock_embed_response.status = 200
            mock_embed_response.json.return_value = {
                "data": [{"embedding": [0.1] * 512}] * 3  # Mock embeddings
            }
            mock_embed_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_embed_response
            
            # Mock LLM service
            mock_llm_response = AsyncMock()
            mock_llm_response.status = 200
            mock_llm_response.json.return_value = {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "Machine learning is a subset of artificial intelligence that focuses on algorithms."}]
                    }
                }]
            }
            mock_llm_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_llm_response
            
            # Mock Pinecone
            mock_pinecone_client = MockPinecone()
            mock_pinecone.Pinecone.return_value = mock_pinecone_client
            
            # Mock database
            mock_pool = AsyncMock()
            mock_conn = AsyncMock()
            mock_conn.execute.return_value = None
            mock_conn.fetchval.return_value = "test-doc-id"
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            mock_asyncpg.create_pool.return_value = mock_pool
            
            yield {
                'document_session': mock_session,
                'embedding_session': mock_embed_session,
                'llm_session': mock_llm_session,
                'pinecone': mock_pinecone_client,
                'database': mock_pool
            }
    
    @pytest.mark.asyncio
    async def test_complete_pdf_workflow(self, client, mock_external_services):
        """Test complete workflow with PDF document."""
        # Prepare request
        request_data = {
            "documents": "http://example.com/sample.pdf",
            "questions": ["What is machine learning?", "What is NLP?"]
        }
        
        headers = {"Authorization": "Bearer test-token"}
        
        # Make request
        response = await client.post("/api/v1/hackrx/run", json=request_data, headers=headers)
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        
        assert "answers" in response_data
        assert len(response_data["answers"]) == 2
        assert all(isinstance(answer, str) for answer in response_data["answers"])
        assert all(len(answer) > 0 for answer in response_data["answers"])
    
    @pytest.mark.asyncio
    async def test_complete_docx_workflow(self, client, mock_external_services):
        """Test complete workflow with DOCX document."""
        # Mock DOCX download
        mock_external_services['document_session'].return_value.__aenter__.return_value.get.return_value.__aenter__.return_value.read.return_value = TestDocumentFactory.create_sample_docx()
        mock_external_services['document_session'].return_value.__aenter__.return_value.get.return_value.__aenter__.return_value.headers = {'content-type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'}
        
        request_data = {
            "documents": "http://example.com/sample.docx",
            "questions": ["What is document retrieval?"]
        }
        
        headers = {"Authorization": "Bearer test-token"}
        
        response = await client.post("/api/v1/hackrx/run", json=request_data, headers=headers)
        
        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data["answers"]) == 1
    
    @pytest.mark.asyncio
    async def test_multiple_questions_workflow(self, client, mock_external_services):
        """Test workflow with multiple questions."""
        request_data = {
            "documents": "http://example.com/sample.pdf",
            "questions": SAMPLE_QUESTIONS
        }
        
        headers = {"Authorization": "Bearer test-token"}
        
        response = await client.post("/api/v1/hackrx/run", json=request_data, headers=headers)
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Verify answer count matches question count
        assert len(response_data["answers"]) == len(SAMPLE_QUESTIONS)
        
        # Verify all answers are non-empty strings
        for answer in response_data["answers"]:
            assert isinstance(answer, str)
            assert len(answer.strip()) > 0
    
    @pytest.mark.asyncio
    async def test_insurance_document_workflow(self, client, mock_external_services):
        """Test workflow with insurance document content."""
        # Mock insurance document
        mock_external_services['document_session'].return_value.__aenter__.return_value.get.return_value.__aenter__.return_value.read.return_value = TestDocumentFactory.create_sample_pdf(SAMPLE_INSURANCE_CONTENT)
        
        request_data = {
            "documents": "http://example.com/insurance.pdf",
            "questions": INSURANCE_QUESTIONS[:3]  # Test first 3 questions
        }
        
        headers = {"Authorization": "Bearer test-token"}
        
        response = await client.post("/api/v1/hackrx/run", json=request_data, headers=headers)
        
        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data["answers"]) == 3
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, client, mock_external_services):
        """Test error handling in the complete workflow."""
        # Mock document download failure
        mock_external_services['document_session'].return_value.__aenter__.return_value.get.return_value.__aenter__.return_value.status = 404
        
        request_data = {
            "documents": "http://example.com/nonexistent.pdf",
            "questions": ["What is this about?"]
        }
        
        headers = {"Authorization": "Bearer test-token"}
        
        response = await client.post("/api/v1/hackrx/run", json=request_data, headers=headers)
        
        # Should return error response
        assert response.status_code in [400, 500]
        response_data = response.json()
        assert "error" in response_data
    
    @pytest.mark.asyncio
    async def test_authentication_workflow(self, client, mock_external_services):
        """Test authentication in the workflow."""
        request_data = {
            "documents": "http://example.com/sample.pdf",
            "questions": ["What is this?"]
        }
        
        # Test without authentication
        response = await client.post("/api/v1/hackrx/run", json=request_data)
        assert response.status_code == 401
        
        # Test with wrong token
        headers = {"Authorization": "Bearer wrong-token"}
        response = await client.post("/api/v1/hackrx/run", json=request_data, headers=headers)
        assert response.status_code == 401
        
        # Test with correct token
        headers = {"Authorization": "Bearer test-token"}
        response = await client.post("/api/v1/hackrx/run", json=request_data, headers=headers)
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_request_validation_workflow(self, client, mock_external_services):
        """Test request validation in the workflow."""
        headers = {"Authorization": "Bearer test-token"}
        
        # Test missing documents
        response = await client.post("/api/v1/hackrx/run", json={"questions": ["What?"]}, headers=headers)
        assert response.status_code == 422
        
        # Test missing questions
        response = await client.post("/api/v1/hackrx/run", json={"documents": "http://example.com/test.pdf"}, headers=headers)
        assert response.status_code == 422
        
        # Test empty questions
        response = await client.post("/api/v1/hackrx/run", json={"documents": "http://example.com/test.pdf", "questions": []}, headers=headers)
        assert response.status_code == 422
        
        # Test invalid URL format
        response = await client.post("/api/v1/hackrx/run", json={"documents": "not-a-url", "questions": ["What?"]}, headers=headers)
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_large_document_workflow(self, client, mock_external_services):
        """Test workflow with a large document."""
        # Create large content
        large_content = SAMPLE_TEXT_CONTENT * 50  # Repeat content 50 times
        mock_external_services['document_session'].return_value.__aenter__.return_value.get.return_value.__aenter__.return_value.read.return_value = TestDocumentFactory.create_sample_pdf(large_content)
        
        # Mock multiple embedding calls for chunks
        mock_external_services['embedding_session'].return_value.__aenter__.return_value.post.return_value.__aenter__.return_value.json.return_value = {
            "data": [{"embedding": [0.1] * 512}] * 20  # More embeddings for more chunks
        }
        
        request_data = {
            "documents": "http://example.com/large.pdf",
            "questions": ["What is the main topic of this document?"]
        }
        
        headers = {"Authorization": "Bearer test-token"}
        
        response = await client.post("/api/v1/hackrx/run", json=request_data, headers=headers)
        
        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data["answers"]) == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_workflow(self, client, mock_external_services):
        """Test handling of concurrent requests."""
        request_data = {
            "documents": "http://example.com/sample.pdf",
            "questions": ["What is this document about?"]
        }
        
        headers = {"Authorization": "Bearer test-token"}
        
        # Make multiple concurrent requests
        tasks = []
        for i in range(5):
            task = client.post("/api/v1/hackrx/run", json=request_data, headers=headers)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            response_data = response.json()
            assert "answers" in response_data
            assert len(response_data["answers"]) == 1
    
    @pytest.mark.asyncio
    async def test_service_failure_recovery(self, client, mock_external_services):
        """Test recovery from service failures."""
        headers = {"Authorization": "Bearer test-token"}
        request_data = {
            "documents": "http://example.com/sample.pdf",
            "questions": ["What is this?"]
        }
        
        # Test embedding service failure and recovery
        mock_external_services['embedding_session'].return_value.__aenter__.return_value.post.return_value.__aenter__.return_value.status = 500
        
        response = await client.post("/api/v1/hackrx/run", json=request_data, headers=headers)
        
        # Should handle the error gracefully
        assert response.status_code in [500, 503]  # Server error or service unavailable
        
        # Reset to working state
        mock_external_services['embedding_session'].return_value.__aenter__.return_value.post.return_value.__aenter__.return_value.status = 200
        
        response = await client.post("/api/v1/hackrx/run", json=request_data, headers=headers)
        assert response.status_code == 200