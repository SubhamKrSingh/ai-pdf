"""Performance and load tests for the API."""

import pytest
import asyncio
import time
import statistics
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock
import concurrent.futures
from typing import List, Dict, Any

from main import app
from tests.fixtures.sample_documents import TestDocumentFactory, SAMPLE_TEXT_CONTENT
from tests.fixtures.database_fixtures import MockPinecone

class TestPerformance:
    """Performance and load testing for the API."""
    
    @pytest.fixture
    async def client(self):
        """Create test client."""
        async with AsyncClient(app=app, base_url="http://test", timeout=30.0) as ac:
            yield ac
    
    @pytest.fixture
    def mock_fast_services(self):
        """Mock external services with fast responses for performance testing."""
        with patch('app.utils.document_downloader.aiohttp.ClientSession') as mock_session, \
             patch('app.services.embedding_service.aiohttp.ClientSession') as mock_embed_session, \
             patch('app.services.llm_service.aiohttp.ClientSession') as mock_llm_session, \
             patch('app.data.vector_store.pinecone') as mock_pinecone, \
             patch('app.data.repository.asyncpg') as mock_asyncpg:
            
            # Mock fast document download
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {'content-type': 'application/pdf'}
            mock_response.read.return_value = TestDocumentFactory.create_sample_pdf()
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            # Mock fast embedding service
            mock_embed_response = AsyncMock()
            mock_embed_response.status = 200
            mock_embed_response.json.return_value = {
                "data": [{"embedding": [0.1] * 512}] * 5
            }
            mock_embed_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_embed_response
            
            # Mock fast LLM service
            mock_llm_response = AsyncMock()
            mock_llm_response.status = 200
            mock_llm_response.json.return_value = {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "This is a fast mock response for performance testing."}]
                    }
                }]
            }
            mock_llm_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_llm_response
            
            # Mock fast Pinecone
            mock_pinecone_client = MockPinecone()
            mock_pinecone.Pinecone.return_value = mock_pinecone_client
            
            # Mock fast database
            mock_pool = AsyncMock()
            mock_conn = AsyncMock()
            mock_conn.execute.return_value = None
            mock_conn.fetchval.return_value = "test-doc-id"
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            mock_asyncpg.create_pool.return_value = mock_pool
            
            yield mock_pinecone_client
    
    async def make_request(self, client: AsyncClient, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a single API request and measure response time."""
        headers = {"Authorization": "Bearer test-token"}
        start_time = time.time()
        
        try:
            response = await client.post("/api/v1/hackrx/run", json=request_data, headers=headers)
            end_time = time.time()
            
            return {
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "success": response.status_code == 200,
                "response_size": len(response.content) if response.content else 0
            }
        except Exception as e:
            end_time = time.time()
            return {
                "status_code": 0,
                "response_time": end_time - start_time,
                "success": False,
                "error": str(e),
                "response_size": 0
            }
    
    @pytest.mark.asyncio
    async def test_single_request_performance(self, client, mock_fast_services):
        """Test performance of a single request."""
        request_data = {
            "documents": "http://example.com/sample.pdf",
            "questions": ["What is machine learning?"]
        }
        
        result = await self.make_request(client, request_data)
        
        assert result["success"], f"Request failed: {result}"
        assert result["response_time"] < 10.0, f"Response time too slow: {result['response_time']}s"
        print(f"Single request response time: {result['response_time']:.3f}s")
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_performance(self, client, mock_fast_services):
        """Test performance with concurrent requests."""
        request_data = {
            "documents": "http://example.com/sample.pdf",
            "questions": ["What is this document about?"]
        }
        
        # Test with different concurrency levels
        concurrency_levels = [5, 10, 20]
        
        for concurrency in concurrency_levels:
            print(f"\nTesting with {concurrency} concurrent requests...")
            
            start_time = time.time()
            tasks = [self.make_request(client, request_data) for _ in range(concurrency)]
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            # Analyze results
            successful_requests = [r for r in results if r["success"]]
            failed_requests = [r for r in results if not r["success"]]
            
            success_rate = len(successful_requests) / len(results) * 100
            avg_response_time = statistics.mean([r["response_time"] for r in successful_requests]) if successful_requests else 0
            max_response_time = max([r["response_time"] for r in successful_requests]) if successful_requests else 0
            min_response_time = min([r["response_time"] for r in successful_requests]) if successful_requests else 0
            
            print(f"Concurrency: {concurrency}")
            print(f"Success rate: {success_rate:.1f}%")
            print(f"Total time: {total_time:.3f}s")
            print(f"Requests per second: {len(results) / total_time:.2f}")
            print(f"Average response time: {avg_response_time:.3f}s")
            print(f"Min response time: {min_response_time:.3f}s")
            print(f"Max response time: {max_response_time:.3f}s")
            
            # Assertions
            assert success_rate >= 90, f"Success rate too low: {success_rate}%"
            assert avg_response_time < 15.0, f"Average response time too slow: {avg_response_time}s"
            assert len(failed_requests) <= concurrency * 0.1, f"Too many failed requests: {len(failed_requests)}"
    
    @pytest.mark.asyncio
    async def test_multiple_questions_performance(self, client, mock_fast_services):
        """Test performance with multiple questions in a single request."""
        questions = [
            "What is machine learning?",
            "What is natural language processing?",
            "What are document retrieval systems?",
            "How do embeddings work?",
            "What is the purpose of this system?"
        ]
        
        request_data = {
            "documents": "http://example.com/sample.pdf",
            "questions": questions
        }
        
        result = await self.make_request(client, request_data)
        
        assert result["success"], f"Request failed: {result}"
        
        # Should handle multiple questions efficiently
        expected_max_time = len(questions) * 3.0  # 3 seconds per question max
        assert result["response_time"] < expected_max_time, f"Response time too slow for {len(questions)} questions: {result['response_time']}s"
        
        print(f"Multiple questions ({len(questions)}) response time: {result['response_time']:.3f}s")
        print(f"Average time per question: {result['response_time'] / len(questions):.3f}s")
    
    @pytest.mark.asyncio
    async def test_large_document_performance(self, client, mock_fast_services):
        """Test performance with large documents."""
        # Create a large document
        large_content = SAMPLE_TEXT_CONTENT * 100  # Very large document
        
        with patch('tests.fixtures.sample_documents.TestDocumentFactory.create_sample_pdf') as mock_pdf:
            mock_pdf.return_value = TestDocumentFactory.create_sample_pdf(large_content)
            
            request_data = {
                "documents": "http://example.com/large.pdf",
                "questions": ["What is the main topic?"]
            }
            
            result = await self.make_request(client, request_data)
            
            assert result["success"], f"Large document request failed: {result}"
            assert result["response_time"] < 30.0, f"Large document processing too slow: {result['response_time']}s"
            
            print(f"Large document response time: {result['response_time']:.3f}s")
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, client, mock_fast_services):
        """Test memory usage stability under load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        request_data = {
            "documents": "http://example.com/sample.pdf",
            "questions": ["What is this about?"]
        }
        
        # Make many requests to test memory stability
        for i in range(50):
            result = await self.make_request(client, request_data)
            assert result["success"], f"Request {i} failed: {result}"
            
            if i % 10 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                print(f"Request {i}: Memory usage: {current_memory:.1f}MB (increase: {memory_increase:.1f}MB)")
                
                # Memory shouldn't grow excessively
                assert memory_increase < 500, f"Memory usage increased too much: {memory_increase}MB"
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_increase = final_memory - initial_memory
        print(f"Total memory increase after 50 requests: {total_increase:.1f}MB")
    
    @pytest.mark.asyncio
    async def test_sustained_load_performance(self, client, mock_fast_services):
        """Test performance under sustained load."""
        request_data = {
            "documents": "http://example.com/sample.pdf",
            "questions": ["What is this document about?"]
        }
        
        duration_seconds = 30  # Run for 30 seconds
        start_time = time.time()
        request_count = 0
        results = []
        
        print(f"Running sustained load test for {duration_seconds} seconds...")
        
        while time.time() - start_time < duration_seconds:
            # Make requests in batches to avoid overwhelming
            batch_tasks = [self.make_request(client, request_data) for _ in range(5)]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            request_count += len(batch_tasks)
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        total_time = time.time() - start_time
        successful_requests = [r for r in results if r["success"]]
        
        success_rate = len(successful_requests) / len(results) * 100
        avg_response_time = statistics.mean([r["response_time"] for r in successful_requests]) if successful_requests else 0
        requests_per_second = len(results) / total_time
        
        print(f"Sustained load test results:")
        print(f"Total requests: {len(results)}")
        print(f"Successful requests: {len(successful_requests)}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Average response time: {avg_response_time:.3f}s")
        print(f"Requests per second: {requests_per_second:.2f}")
        
        # Assertions for sustained performance
        assert success_rate >= 95, f"Success rate under sustained load too low: {success_rate}%"
        assert avg_response_time < 10.0, f"Average response time under sustained load too slow: {avg_response_time}s"
        assert requests_per_second >= 1.0, f"Throughput too low: {requests_per_second} req/s"
    
    @pytest.mark.asyncio
    async def test_error_recovery_performance(self, client, mock_fast_services):
        """Test performance recovery after errors."""
        request_data = {
            "documents": "http://example.com/sample.pdf",
            "questions": ["What is this?"]
        }
        
        # First, establish baseline performance
        baseline_result = await self.make_request(client, request_data)
        assert baseline_result["success"]
        baseline_time = baseline_result["response_time"]
        
        # Introduce errors by mocking service failures
        with patch('app.services.embedding_service.aiohttp.ClientSession') as mock_embed_session:
            mock_embed_response = AsyncMock()
            mock_embed_response.status = 500  # Simulate service failure
            mock_embed_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_embed_response
            
            # Make requests during error condition
            error_results = []
            for _ in range(5):
                result = await self.make_request(client, request_data)
                error_results.append(result)
        
        # After errors are resolved, test recovery
        recovery_results = []
        for _ in range(10):
            result = await self.make_request(client, request_data)
            recovery_results.append(result)
        
        successful_recovery = [r for r in recovery_results if r["success"]]
        recovery_rate = len(successful_recovery) / len(recovery_results) * 100
        avg_recovery_time = statistics.mean([r["response_time"] for r in successful_recovery]) if successful_recovery else 0
        
        print(f"Error recovery test results:")
        print(f"Baseline response time: {baseline_time:.3f}s")
        print(f"Recovery rate: {recovery_rate:.1f}%")
        print(f"Average recovery response time: {avg_recovery_time:.3f}s")
        
        # System should recover quickly
        assert recovery_rate >= 90, f"Recovery rate too low: {recovery_rate}%"
        assert avg_recovery_time < baseline_time * 2, f"Recovery time too slow: {avg_recovery_time}s vs baseline {baseline_time}s"