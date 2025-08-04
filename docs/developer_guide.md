# Developer Guide

This guide provides comprehensive information for developers who want to extend, modify, or contribute to the LLM Query Retrieval System.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Development Setup](#development-setup)
3. [Code Structure](#code-structure)
4. [Adding New Document Parsers](#adding-new-document-parsers)
5. [Extending the API](#extending-the-api)
6. [Custom LLM Integration](#custom-llm-integration)
7. [Adding New Vector Stores](#adding-new-vector-stores)
8. [Testing Guidelines](#testing-guidelines)
9. [Performance Optimization](#performance-optimization)
10. [Deployment Considerations](#deployment-considerations)

## Architecture Overview

The system follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                     API Layer (FastAPI)                     │
├─────────────────────────────────────────────────────────────┤
│                   Business Logic Layer                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Document Service│  │  Query Service  │  │Auth Service  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Data Access Layer                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  Vector Store   │  │   Repository    │  │  Embedding   │ │
│  │   (Pinecone)    │  │ (PostgreSQL)    │  │   Service    │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                   External Services                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   LLM Service   │  │Document Parsers │  │   Utilities  │ │
│  │   (Gemini)      │  │ (PDF/DOCX/EML)  │  │   & Helpers  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Dependency Injection**: Services are injected to enable testing and flexibility
2. **Async/Await**: All I/O operations are asynchronous for better performance
3. **Error Handling**: Comprehensive error handling with structured responses
4. **Configuration Management**: Environment-based configuration with validation
5. **Modular Design**: Easy to extend with new parsers, LLMs, or vector stores

## Development Setup

### Prerequisites

- Python 3.9+
- PostgreSQL 12+
- Git
- Virtual environment tool (venv, conda, or poetry)

### Local Development Environment

1. **Clone and setup:**
   ```bash
   git clone <repository-url>
   cd llm-query-retrieval-system
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Environment configuration:**
   ```bash
   # Copy example environment file
   cp .env.example .env
   
   # Edit .env with your API keys and configuration
   nano .env
   ```

3. **Database setup:**
   ```bash
   # Create database
   createdb llm_query_system
   
   # Run migrations
   python -c "from app.data.migrations import run_migrations; run_migrations()"
   ```

4. **Run development server:**
   ```bash
   # With auto-reload
   python main.py
   
   # Or using uvicorn directly
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Development Tools

**Recommended IDE setup:**
- VS Code with Python extension
- PyCharm Professional
- Vim/Neovim with Python LSP

**Useful development commands:**
```bash
# Format code
black app/ tests/

# Lint code
flake8 app/ tests/

# Type checking
mypy app/

# Run tests
pytest tests/

# Generate test coverage
pytest --cov=app tests/
```

## Code Structure

```
app/
├── __init__.py
├── auth.py                 # Authentication middleware
├── config.py              # Configuration management
├── exceptions.py          # Custom exception classes
├── security.py           # Security utilities
├── controllers/           # API controllers
│   └── query_controller.py
├── data/                  # Data access layer
│   ├── migrations.py      # Database migrations
│   ├── repository.py      # Database repository
│   └── vector_store.py    # Vector database operations
├── middleware/            # Custom middleware
│   └── error_handler.py   # Error handling middleware
├── models/                # Data models
│   └── schemas.py         # Pydantic models
├── services/              # Business logic services
│   ├── document_service.py
│   ├── embedding_service.py
│   ├── llm_service.py
│   └── query_service.py
└── utils/                 # Utility modules
    ├── document_downloader.py
    ├── logging_config.py
    ├── retry.py
    ├── text_chunker.py
    └── parsers/           # Document parsers
        ├── document_parser.py
        ├── docx_parser.py
        ├── email_parser.py
        └── pdf_parser.py
```

### Key Components

**Controllers**: Handle HTTP requests and coordinate service calls
**Services**: Contain business logic and orchestrate data operations
**Repositories**: Abstract database operations
**Models**: Define data structures and validation
**Utilities**: Reusable helper functions and classes

## Adding New Document Parsers

### 1. Create Parser Class

Create a new parser in `app/utils/parsers/`:

```python
# app/utils/parsers/html_parser.py
from typing import Dict, Any
from bs4 import BeautifulSoup
import logging

from .base_parser import BaseParser

logger = logging.getLogger(__name__)

class HTMLParser(BaseParser):
    """Parser for HTML documents."""
    
    @property
    def supported_content_types(self) -> list[str]:
        """Return list of supported MIME types."""
        return ['text/html', 'application/xhtml+xml']
    
    async def parse(self, content: bytes, metadata: Dict[str, Any] = None) -> str:
        """
        Parse HTML content and extract text.
        
        Args:
            content: Raw HTML content as bytes
            metadata: Optional metadata about the document
            
        Returns:
            Extracted text content
            
        Raises:
            DocumentParseError: If parsing fails
        """
        try:
            # Decode content
            text_content = content.decode('utf-8', errors='ignore')
            
            # Parse HTML
            soup = BeautifulSoup(text_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            if not text.strip():
                raise ValueError("No text content found in HTML document")
            
            logger.info(f"Successfully parsed HTML document, extracted {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"HTML parsing failed: {str(e)}")
            raise DocumentParseError(f"Failed to parse HTML document: {str(e)}")
```

### 2. Create Base Parser Interface

```python
# app/utils/parsers/base_parser.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseParser(ABC):
    """Base class for document parsers."""
    
    @property
    @abstractmethod
    def supported_content_types(self) -> list[str]:
        """Return list of supported MIME types."""
        pass
    
    @abstractmethod
    async def parse(self, content: bytes, metadata: Dict[str, Any] = None) -> str:
        """
        Parse document content and extract text.
        
        Args:
            content: Raw document content as bytes
            metadata: Optional metadata about the document
            
        Returns:
            Extracted text content
            
        Raises:
            DocumentParseError: If parsing fails
        """
        pass
```

### 3. Register Parser

Update `app/utils/parsers/document_parser.py`:

```python
from .html_parser import HTMLParser

class DocumentParser:
    def __init__(self):
        self.parsers = {
            'application/pdf': PDFParser(),
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DOCXParser(),
            'message/rfc822': EmailParser(),
            'text/html': HTMLParser(),  # Add new parser
            'application/xhtml+xml': HTMLParser(),  # Add new parser
        }
```

### 4. Add Tests

```python
# tests/unit/test_html_parser.py
import pytest
from app.utils.parsers.html_parser import HTMLParser
from app.exceptions import DocumentParseError

class TestHTMLParser:
    
    @pytest.fixture
    def parser(self):
        return HTMLParser()
    
    @pytest.fixture
    def sample_html(self):
        return b"""
        <html>
            <head><title>Test Document</title></head>
            <body>
                <h1>Main Heading</h1>
                <p>This is a paragraph with <strong>bold text</strong>.</p>
                <script>console.log('should be removed');</script>
            </body>
        </html>
        """
    
    def test_supported_content_types(self, parser):
        types = parser.supported_content_types
        assert 'text/html' in types
        assert 'application/xhtml+xml' in types
    
    @pytest.mark.asyncio
    async def test_parse_html_success(self, parser, sample_html):
        result = await parser.parse(sample_html)
        
        assert "Main Heading" in result
        assert "This is a paragraph with bold text" in result
        assert "console.log" not in result  # Script should be removed
    
    @pytest.mark.asyncio
    async def test_parse_empty_html(self, parser):
        empty_html = b"<html><body></body></html>"
        
        with pytest.raises(DocumentParseError):
            await parser.parse(empty_html)
    
    @pytest.mark.asyncio
    async def test_parse_invalid_html(self, parser):
        invalid_content = b"not html content"
        
        # Should still work with BeautifulSoup's lenient parsing
        result = await parser.parse(invalid_content)
        assert result == "not html content"
```

## Extending the API

### 1. Add New Endpoint

Create a new controller or extend existing one:

```python
# app/controllers/analytics_controller.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
import logging

from app.auth import verify_token
from app.models.schemas import AnalyticsResponse
from app.services.analytics_service import AnalyticsService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/analytics", tags=["Analytics"])

@router.get("/usage", response_model=AnalyticsResponse)
async def get_usage_analytics(
    days: int = 30,
    _: bool = Depends(verify_token)
) -> AnalyticsResponse:
    """
    Get usage analytics for the specified time period.
    
    Args:
        days: Number of days to analyze (default: 30)
        
    Returns:
        Analytics data including request counts, processing times, etc.
    """
    try:
        analytics_service = AnalyticsService()
        data = await analytics_service.get_usage_analytics(days)
        return AnalyticsResponse(**data)
        
    except Exception as e:
        logger.error(f"Analytics request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

### 2. Create Service Layer

```python
# app/services/analytics_service.py
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging

from app.data.repository import Repository

logger = logging.getLogger(__name__)

class AnalyticsService:
    """Service for generating analytics and usage statistics."""
    
    def __init__(self):
        self.repository = Repository()
    
    async def get_usage_analytics(self, days: int) -> Dict[str, Any]:
        """
        Generate usage analytics for the specified time period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary containing analytics data
        """
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Get query statistics
            query_stats = await self.repository.get_query_statistics(start_date, end_date)
            
            # Get document statistics
            doc_stats = await self.repository.get_document_statistics(start_date, end_date)
            
            # Calculate metrics
            analytics = {
                'period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'days': days
                },
                'queries': {
                    'total_queries': query_stats.get('total_queries', 0),
                    'unique_documents': query_stats.get('unique_documents', 0),
                    'avg_processing_time_ms': query_stats.get('avg_processing_time', 0),
                    'success_rate': query_stats.get('success_rate', 0.0)
                },
                'documents': {
                    'total_processed': doc_stats.get('total_processed', 0),
                    'by_type': doc_stats.get('by_type', {}),
                    'avg_size_mb': doc_stats.get('avg_size_mb', 0.0),
                    'processing_errors': doc_stats.get('processing_errors', 0)
                }
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to generate analytics: {str(e)}")
            raise
```

### 3. Add to Main Application

Update `main.py`:

```python
from app.controllers.analytics_controller import router as analytics_router

# Add router
app.include_router(analytics_router)
```

## Custom LLM Integration

### 1. Create LLM Service Interface

```python
# app/services/base_llm_service.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseLLMService(ABC):
    """Base class for LLM service implementations."""
    
    @abstractmethod
    async def generate_answer(
        self, 
        question: str, 
        context: str, 
        **kwargs
    ) -> str:
        """
        Generate an answer based on question and context.
        
        Args:
            question: The question to answer
            context: Relevant context from documents
            **kwargs: Additional parameters
            
        Returns:
            Generated answer
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check if the LLM service is available."""
        pass
```

### 2. Implement Custom LLM

```python
# app/services/openai_llm_service.py
import openai
from typing import Dict, Any
import logging

from .base_llm_service import BaseLLMService
from app.config import get_settings
from app.exceptions import LLMServiceError

logger = logging.getLogger(__name__)

class OpenAILLMService(BaseLLMService):
    """OpenAI GPT-based LLM service implementation."""
    
    def __init__(self):
        self.settings = get_settings()
        openai.api_key = self.settings.openai_api_key
        self.model = self.settings.openai_model or "gpt-4"
    
    async def generate_answer(
        self, 
        question: str, 
        context: str, 
        **kwargs
    ) -> str:
        """Generate answer using OpenAI GPT."""
        try:
            prompt = self._build_prompt(question, context)
            
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=kwargs.get('max_tokens', 500),
                temperature=kwargs.get('temperature', 0.1)
            )
            
            answer = response.choices[0].message.content.strip()
            
            if not answer:
                raise LLMServiceError("Empty response from OpenAI")
            
            return answer
            
        except Exception as e:
            logger.error(f"OpenAI LLM generation failed: {str(e)}")
            raise LLMServiceError(f"Failed to generate answer: {str(e)}")
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build prompt for OpenAI."""
        return f"""
        Based on the following context, please answer the question. If the context doesn't contain enough information to answer the question, please say so.

        Context:
        {context}

        Question: {question}

        Answer:
        """
    
    async def health_check(self) -> Dict[str, Any]:
        """Check OpenAI API availability."""
        try:
            response = await openai.Model.alist()
            return {
                "status": "healthy",
                "service": "openai",
                "models_available": len(response.data)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "service": "openai",
                "error": str(e)
            }
```

### 3. Configure LLM Selection

Update configuration to support multiple LLMs:

```python
# app/config.py
class Settings(BaseSettings):
    # LLM Configuration
    llm_provider: Literal["gemini", "openai", "anthropic"] = Field(
        default="gemini", 
        description="LLM provider to use"
    )
    
    # Provider-specific settings
    gemini_api_key: Optional[str] = Field(None, description="Google Gemini API key")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key")
```

### 4. Create LLM Factory

```python
# app/services/llm_factory.py
from typing import Dict, Type
from app.config import get_settings
from .base_llm_service import BaseLLMService
from .gemini_llm_service import GeminiLLMService
from .openai_llm_service import OpenAILLMService

class LLMFactory:
    """Factory for creating LLM service instances."""
    
    _services: Dict[str, Type[BaseLLMService]] = {
        "gemini": GeminiLLMService,
        "openai": OpenAILLMService,
    }
    
    @classmethod
    def create_llm_service(cls) -> BaseLLMService:
        """Create LLM service based on configuration."""
        settings = get_settings()
        provider = settings.llm_provider
        
        if provider not in cls._services:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        service_class = cls._services[provider]
        return service_class()
    
    @classmethod
    def register_service(cls, name: str, service_class: Type[BaseLLMService]):
        """Register a new LLM service."""
        cls._services[name] = service_class
```

## Adding New Vector Stores

### 1. Create Vector Store Interface

```python
# app/data/base_vector_store.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from app.models.schemas import SearchResult

class BaseVectorStore(ABC):
    """Base class for vector store implementations."""
    
    @abstractmethod
    async def store_vectors(
        self, 
        vectors: List[List[float]], 
        metadata: List[Dict[str, Any]],
        document_id: str
    ) -> bool:
        """Store vectors with metadata."""
        pass
    
    @abstractmethod
    async def similarity_search(
        self, 
        query_vector: List[float], 
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform similarity search."""
        pass
    
    @abstractmethod
    async def delete_document_vectors(self, document_id: str) -> bool:
        """Delete all vectors for a document."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check vector store health."""
        pass
```

### 2. Implement New Vector Store

```python
# app/data/weaviate_vector_store.py
import weaviate
from typing import List, Dict, Any, Optional
import logging

from .base_vector_store import BaseVectorStore
from app.models.schemas import SearchResult
from app.config import get_settings
from app.exceptions import VectorStoreError

logger = logging.getLogger(__name__)

class WeaviateVectorStore(BaseVectorStore):
    """Weaviate vector store implementation."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = weaviate.Client(
            url=self.settings.weaviate_url,
            auth_client_secret=weaviate.AuthApiKey(api_key=self.settings.weaviate_api_key)
        )
        self.class_name = "DocumentChunk"
    
    async def store_vectors(
        self, 
        vectors: List[List[float]], 
        metadata: List[Dict[str, Any]],
        document_id: str
    ) -> bool:
        """Store vectors in Weaviate."""
        try:
            objects = []
            for i, (vector, meta) in enumerate(zip(vectors, metadata)):
                obj = {
                    "class": self.class_name,
                    "properties": {
                        "document_id": document_id,
                        "chunk_id": meta.get("chunk_id"),
                        "content": meta.get("content", ""),
                        "chunk_index": meta.get("chunk_index", i),
                        **meta
                    },
                    "vector": vector
                }
                objects.append(obj)
            
            # Batch insert
            with self.client.batch as batch:
                batch.batch_size = 100
                for obj in objects:
                    batch.add_data_object(**obj)
            
            logger.info(f"Stored {len(vectors)} vectors for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store vectors in Weaviate: {str(e)}")
            raise VectorStoreError(f"Vector storage failed: {str(e)}")
    
    async def similarity_search(
        self, 
        query_vector: List[float], 
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform similarity search in Weaviate."""
        try:
            query = (
                self.client.query
                .get(self.class_name, ["document_id", "chunk_id", "content", "chunk_index"])
                .with_near_vector({"vector": query_vector})
                .with_limit(top_k)
                .with_additional(["certainty"])
            )
            
            # Add filters if provided
            if filter_metadata:
                where_filter = {"path": ["document_id"], "operator": "Equal", "valueString": filter_metadata.get("document_id")}
                query = query.with_where(where_filter)
            
            result = query.do()
            
            search_results = []
            for item in result["data"]["Get"][self.class_name]:
                search_results.append(SearchResult(
                    chunk_id=item["chunk_id"],
                    content=item["content"],
                    score=item["_additional"]["certainty"],
                    metadata={
                        "chunk_index": item.get("chunk_index"),
                        "document_id": item["document_id"]
                    },
                    document_id=item["document_id"]
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Weaviate similarity search failed: {str(e)}")
            raise VectorStoreError(f"Similarity search failed: {str(e)}")
    
    async def delete_document_vectors(self, document_id: str) -> bool:
        """Delete vectors for a document from Weaviate."""
        try:
            where_filter = {
                "path": ["document_id"],
                "operator": "Equal",
                "valueString": document_id
            }
            
            result = self.client.batch.delete_objects(
                class_name=self.class_name,
                where=where_filter
            )
            
            logger.info(f"Deleted vectors for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors from Weaviate: {str(e)}")
            raise VectorStoreError(f"Vector deletion failed: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Weaviate health."""
        try:
            ready = self.client.is_ready()
            live = self.client.is_live()
            
            return {
                "status": "healthy" if ready and live else "unhealthy",
                "service": "weaviate",
                "ready": ready,
                "live": live
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "service": "weaviate",
                "error": str(e)
            }
```

## Testing Guidelines

### Unit Testing Best Practices

1. **Test Structure:**
   ```python
   # tests/unit/test_service.py
   import pytest
   from unittest.mock import Mock, patch, AsyncMock
   
   class TestDocumentService:
       
       @pytest.fixture
       def mock_dependencies(self):
           """Mock external dependencies."""
           return {
               'downloader': Mock(),
               'parser': Mock(),
               'chunker': Mock(),
               'embedder': Mock(),
               'vector_store': Mock()
           }
       
       @pytest.fixture
       def service(self, mock_dependencies):
           """Create service with mocked dependencies."""
           with patch.multiple(
               'app.services.document_service',
               **mock_dependencies
           ):
               from app.services.document_service import DocumentService
               return DocumentService()
       
       @pytest.mark.asyncio
       async def test_process_document_success(self, service, mock_dependencies):
           # Setup mocks
           mock_dependencies['downloader'].download.return_value = b"content"
           mock_dependencies['parser'].parse.return_value = "text"
           
           # Test
           result = await service.process_document("http://example.com/doc.pdf")
           
           # Assertions
           assert result is not None
           mock_dependencies['downloader'].download.assert_called_once()
   ```

2. **Integration Testing:**
   ```python
   # tests/integration/test_api.py
   import pytest
   from fastapi.testclient import TestClient
   from main import app
   
   @pytest.fixture
   def client():
       return TestClient(app)
   
   def test_query_endpoint_success(client, auth_headers):
       response = client.post(
           "/api/v1/hackrx/run",
           json={
               "documents": "https://example.com/test.pdf",
               "questions": ["What is this about?"]
           },
           headers=auth_headers
       )
       
       assert response.status_code == 200
       data = response.json()
       assert "answers" in data
       assert len(data["answers"]) == 1
   ```

### Performance Testing

```python
# tests/performance/test_load.py
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import requests

async def load_test_endpoint(num_requests: int = 100):
    """Load test the main endpoint."""
    
    def make_request():
        response = requests.post(
            "http://localhost:8000/api/v1/hackrx/run",
            json={
                "documents": "https://example.com/test.pdf",
                "questions": ["What is this about?"]
            },
            headers={"Authorization": "Bearer test-token"}
        )
        return response.status_code, response.elapsed.total_seconds()
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]
        results = [future.result() for future in futures]
    
    end_time = time.time()
    
    # Analyze results
    success_count = sum(1 for status, _ in results if status == 200)
    response_times = [duration for _, duration in results]
    
    print(f"Total time: {end_time - start_time:.2f}s")
    print(f"Success rate: {success_count/num_requests*100:.1f}%")
    print(f"Average response time: {sum(response_times)/len(response_times):.2f}s")
    print(f"Max response time: {max(response_times):.2f}s")
```

## Performance Optimization

### 1. Caching Strategies

```python
# app/utils/cache.py
import asyncio
from typing import Any, Optional, Callable
from functools import wraps
import hashlib
import json

class AsyncLRUCache:
    """Async LRU cache implementation."""
    
    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self.cache = {}
        self.access_order = []
    
    async def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    async def set(self, key: str, value: Any):
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.maxsize:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        
        self.cache[key] = value
        self.access_order.append(key)

def cache_async(maxsize: int = 128, ttl: int = 3600):
    """Decorator for caching async function results."""
    cache = AsyncLRUCache(maxsize)
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            key_data = {
                'func': func.__name__,
                'args': args,
                'kwargs': kwargs
            }
            key = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
            
            # Try cache first
            result = await cache.get(key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(key, result)
            return result
        
        return wrapper
    return decorator

# Usage example
@cache_async(maxsize=100, ttl=1800)
async def generate_embedding(text: str) -> List[float]:
    # Expensive embedding generation
    pass
```

### 2. Connection Pooling

```python
# app/utils/connection_pool.py
import asyncio
import aiohttp
from typing import Optional

class HTTPConnectionPool:
    """HTTP connection pool for external API calls."""
    
    def __init__(self, max_connections: int = 100, timeout: int = 30):
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=20,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=self.timeout
            )
        return self.session
    
    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

# Global connection pool
http_pool = HTTPConnectionPool()
```

### 3. Async Processing Optimization

```python
# app/services/optimized_query_service.py
import asyncio
from typing import List
import logging

logger = logging.getLogger(__name__)

class OptimizedQueryService:
    """Optimized query service with concurrent processing."""
    
    async def process_questions_concurrent(
        self, 
        questions: List[str], 
        document_id: str
    ) -> List[str]:
        """Process multiple questions concurrently."""
        
        # Create semaphore to limit concurrent LLM calls
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent calls
        
        async def process_single_question(question: str) -> str:
            async with semaphore:
                return await self._process_single_question(question, document_id)
        
        # Process all questions concurrently
        tasks = [process_single_question(q) for q in questions]
        answers = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_answers = []
        for i, answer in enumerate(answers):
            if isinstance(answer, Exception):
                logger.error(f"Question {i+1} processing failed: {answer}")
                processed_answers.append(f"Error processing question: {str(answer)}")
            else:
                processed_answers.append(answer)
        
        return processed_answers
```

## Deployment Considerations

### 1. Production Configuration

```python
# app/config.py - Production settings
class ProductionSettings(Settings):
    """Production-specific settings."""
    
    # Security
    debug: bool = False
    allowed_hosts: List[str] = Field(default_factory=lambda: ["yourdomain.com"])
    cors_origins: List[str] = Field(default_factory=lambda: ["https://yourdomain.com"])
    
    # Performance
    worker_processes: int = Field(default=4)
    max_concurrent_requests: int = Field(default=1000)
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: str = "/var/log/llm-query-system/app.log"
    
    # Database
    database_pool_size: int = 20
    database_max_overflow: int = 40
```

### 2. Docker Optimization

```dockerfile
# Dockerfile.production
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=app:app . .

# Switch to non-root user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### 3. Monitoring and Observability

```python
# app/middleware/monitoring.py
import time
import logging
from fastapi import Request, Response
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')

async def monitoring_middleware(request: Request, call_next):
    """Middleware for collecting metrics."""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Record metrics
    duration = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    REQUEST_DURATION.observe(duration)
    
    return response

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")
```

This developer guide provides a comprehensive foundation for extending and maintaining the LLM Query Retrieval System. Follow these patterns and best practices to ensure your extensions are robust, testable, and maintainable.