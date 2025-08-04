# Test Suite Documentation

This directory contains the comprehensive test suite for the LLM Query Retrieval System.

## Test Structure

```
tests/
├── unit/                   # Unit tests for individual components
├── integration/            # Integration tests for component interactions
├── e2e/                   # End-to-end tests for complete workflows
├── fixtures/              # Test fixtures and sample data
├── conftest.py            # Global test configuration
└── README.md              # This file
```

## Test Categories

### Unit Tests (`tests/unit/`)
- Test individual functions and classes in isolation
- Mock all external dependencies
- Fast execution (< 1 second per test)
- High coverage of edge cases and error conditions

**Files:**
- `test_auth.py` - Authentication middleware tests
- `test_config.py` - Configuration management tests
- `test_models.py` - Data model validation tests
- `test_*_parser.py` - Document parser tests
- `test_*_service.py` - Service layer tests
- `test_vector_store.py` - Vector store operations tests
- `test_repository.py` - Database repository tests

### Integration Tests (`tests/integration/`)
- Test interactions between components
- Use real databases and external service mocks
- Medium execution time (1-10 seconds per test)
- Focus on data flow and component integration

**Files:**
- `test_document_service.py` - Document processing integration
- `test_query_service.py` - Query processing integration
- `test_vector_store.py` - Vector database integration
- `test_repository.py` - Database integration
- `test_error_handling_integration.py` - Error handling integration

### End-to-End Tests (`tests/e2e/`)
- Test complete user workflows
- Mock external APIs but test full request/response cycle
- Slower execution (5-30 seconds per test)
- Focus on user scenarios and system behavior

**Files:**
- `test_complete_workflow.py` - Complete API workflow tests
- `test_performance.py` - Performance and load tests

### Test Fixtures (`tests/fixtures/`)
- Reusable test data and utilities
- Sample documents in various formats
- Database and external service mocks

**Files:**
- `sample_documents.py` - Sample documents and test data
- `database_fixtures.py` - Database fixtures and mocks

## Running Tests

### Prerequisites

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-asyncio pytest-cov pytest-xdist
   ```

2. **Setup Test Database (Optional):**
   ```bash
   python scripts/setup_test_db.py
   ```
   
   If you don't set up a test database, tests will use mocks automatically.

3. **Set Environment Variables:**
   ```bash
   export AUTH_TOKEN=test-token
   export GEMINI_API_KEY=test-key
   export JINA_API_KEY=test-key
   export PINECONE_API_KEY=test-key
   ```

### Running Tests

#### Using the Test Runner Script (Recommended)

```bash
# Run all tests
python scripts/run_tests.py --all

# Run specific test categories
python scripts/run_tests.py --unit
python scripts/run_tests.py --integration
python scripts/run_tests.py --e2e
python scripts/run_tests.py --performance

# Run with coverage
python scripts/run_tests.py --unit --coverage

# Run specific test file
python scripts/run_tests.py --file tests/unit/test_auth.py

# Run specific test function
python scripts/run_tests.py --test test_valid_token

# Run tests in parallel
python scripts/run_tests.py --unit --parallel 4
```

#### Using Pytest Directly

```bash
# Run all tests
pytest

# Run specific categories
pytest -m unit
pytest -m integration
pytest -m e2e
pytest -m performance

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific file
pytest tests/unit/test_auth.py

# Run specific test
pytest tests/unit/test_auth.py::test_valid_token

# Run in parallel
pytest -n 4
```

### Test Markers

Tests are marked with the following markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.slow` - Slow running tests

## Test Configuration

### Environment Variables

The following environment variables can be set for testing:

```bash
# Authentication
AUTH_TOKEN=test-token

# External APIs (use test keys)
GEMINI_API_KEY=test-gemini-key
JINA_API_KEY=test-jina-key
PINECONE_API_KEY=test-pinecone-key

# Test Database (optional)
TEST_DB_HOST=localhost
TEST_DB_PORT=5432
TEST_DB_USER=postgres
TEST_DB_PASSWORD=postgres
TEST_DB_NAME=test_llm_system

# Pinecone Test Index
PINECONE_INDEX_NAME=test-index
```

### Pytest Configuration

The `pytest.ini` file contains:
- Async test configuration
- Coverage settings
- Test discovery patterns
- Marker definitions
- Warning filters

### Global Fixtures

The `conftest.py` file provides:
- Mock settings for all tests
- Temporary file handling
- External API mocks
- Test data cleanup
- Automatic test categorization

## Writing Tests

### Unit Test Example

```python
import pytest
from unittest.mock import AsyncMock, patch
from app.services.embedding_service import EmbeddingService

class TestEmbeddingService:
    @pytest.fixture
    def embedding_service(self):
        return EmbeddingService()
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_success(self, embedding_service):
        with patch('app.services.embedding_service.aiohttp.ClientSession') as mock_session:
            # Setup mock
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2]}]}
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            # Test
            result = await embedding_service.generate_embeddings(["test text"])
            
            # Assert
            assert len(result) == 1
            assert result[0] == [0.1, 0.2]
```

### Integration Test Example

```python
import pytest
from app.services.document_service import DocumentService
from tests.fixtures.database_fixtures import clean_test_db

class TestDocumentServiceIntegration:
    @pytest.mark.asyncio
    async def test_process_document_integration(self, clean_test_db, mock_external_apis):
        service = DocumentService()
        
        result = await service.process_document("http://example.com/test.pdf")
        
        assert result["status"] == "completed"
        assert result["chunk_count"] > 0
```

### End-to-End Test Example

```python
import pytest
from httpx import AsyncClient
from main import app

class TestE2E:
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/hackrx/run",
                json={
                    "documents": "http://example.com/test.pdf",
                    "questions": ["What is this about?"]
                },
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "answers" in data
            assert len(data["answers"]) == 1
```

## Test Data

### Sample Documents

The test suite includes sample documents in various formats:
- PDF documents with technical content
- DOCX documents with business content
- Email content for parsing tests
- Insurance and legal documents for domain-specific testing

### Mock Services

All external services are mocked by default:
- Document download (HTTP requests)
- Embedding generation (Jina API)
- LLM responses (Gemini API)
- Vector database (Pinecone)
- Relational database (PostgreSQL)

## Performance Testing

Performance tests measure:
- Single request response time
- Concurrent request handling
- Memory usage stability
- Sustained load performance
- Error recovery time

### Performance Benchmarks

- Single request: < 10 seconds
- Concurrent requests (10): > 90% success rate
- Memory growth: < 500MB over 50 requests
- Sustained load: > 95% success rate over 30 seconds

## Coverage Requirements

- Minimum coverage: 80%
- Unit tests should achieve > 90% coverage
- Integration tests should cover all service interactions
- E2E tests should cover all user workflows

## Troubleshooting

### Common Issues

1. **Database Connection Errors:**
   - Ensure PostgreSQL is running
   - Check connection parameters
   - Tests will use mocks if database is unavailable

2. **Slow Test Execution:**
   - Use parallel execution: `pytest -n 4`
   - Run specific test categories
   - Check for real API calls (should be mocked)

3. **Import Errors:**
   - Ensure all dependencies are installed
   - Check Python path configuration
   - Verify app module structure

4. **Async Test Issues:**
   - Ensure `pytest-asyncio` is installed
   - Use `@pytest.mark.asyncio` for async tests
   - Check event loop configuration

### Getting Help

- Check test output for detailed error messages
- Use `pytest -vv` for verbose output
- Review the test fixtures and mocks
- Ensure all environment variables are set correctly