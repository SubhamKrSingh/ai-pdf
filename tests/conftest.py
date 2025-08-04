"""Global test configuration and fixtures."""

import pytest
import asyncio
import os
import sys
from unittest.mock import patch, AsyncMock
import tempfile
import shutil

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.config import get_settings
from tests.fixtures.database_fixtures import (
    test_db_connection,
    test_db_pool,
    mock_pinecone,
    mock_pinecone_index,
    clean_test_db,
    test_settings,
    DatabaseTestHelper,
    VectorStoreTestHelper
)

# Set test environment variables
os.environ.setdefault("AUTH_TOKEN", "test-token")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("JINA_API_KEY", "test-jina-key")
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("PINECONE_INDEX_NAME", "test-index")
os.environ.setdefault("PINECONE_ENVIRONMENT", "test-env")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def mock_settings():
    """Mock settings for all tests."""
    with patch('app.config.get_settings') as mock_get_settings:
        settings = get_settings()
        settings.auth_token = "test-token"
        settings.gemini_api_key = "test-gemini-key"
        settings.jina_api_key = "test-jina-key"
        settings.pinecone_api_key = "test-pinecone-key"
        settings.pinecone_index_name = "test-index"
        settings.pinecone_environment = "test-env"
        settings.database_url = "postgresql://test:test@localhost:5432/test_db"
        mock_get_settings.return_value = settings
        yield settings

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_pdf_file(temp_dir):
    """Create a sample PDF file for testing."""
    from tests.fixtures.sample_documents import TestDocumentFactory
    
    pdf_content = TestDocumentFactory.create_sample_pdf()
    pdf_path = os.path.join(temp_dir, "sample.pdf")
    
    with open(pdf_path, "wb") as f:
        f.write(pdf_content)
    
    return pdf_path

@pytest.fixture
def sample_docx_file(temp_dir):
    """Create a sample DOCX file for testing."""
    from tests.fixtures.sample_documents import TestDocumentFactory
    
    docx_content = TestDocumentFactory.create_sample_docx()
    docx_path = os.path.join(temp_dir, "sample.docx")
    
    with open(docx_path, "wb") as f:
        f.write(docx_content)
    
    return docx_path

@pytest.fixture
def mock_external_apis():
    """Mock all external API calls for testing."""
    with patch('app.utils.document_downloader.aiohttp.ClientSession') as mock_download, \
         patch('app.services.embedding_service.aiohttp.ClientSession') as mock_embedding, \
         patch('app.services.llm_service.aiohttp.ClientSession') as mock_llm:
        
        # Mock document download
        mock_download_response = AsyncMock()
        mock_download_response.status = 200
        mock_download_response.headers = {'content-type': 'application/pdf'}
        mock_download_response.read.return_value = b"mock pdf content"
        mock_download.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_download_response
        
        # Mock embedding service
        mock_embedding_response = AsyncMock()
        mock_embedding_response.status = 200
        mock_embedding_response.json.return_value = {
            "data": [{"embedding": [0.1] * 512}]
        }
        mock_embedding.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_embedding_response
        
        # Mock LLM service
        mock_llm_response = AsyncMock()
        mock_llm_response.status = 200
        mock_llm_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Mock LLM response"}]
                }
            }]
        }
        mock_llm.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_llm_response
        
        yield {
            'download': mock_download,
            'embedding': mock_embedding,
            'llm': mock_llm
        }

# Pytest markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "e2e: mark test as an end-to-end test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "slow: mark test as slow running")

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Add performance marker for performance tests
        if "performance" in str(item.fspath) or "performance" in item.name:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)

# Test data cleanup
@pytest.fixture(autouse=True)
async def cleanup_test_data():
    """Clean up test data before and after each test."""
    # Setup - clean before test
    yield
    # Teardown - clean after test (if needed)
    pass