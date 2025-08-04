# Design Document

## Overview

The LLM-Powered Intelligent Query-Retrieval System is designed as a microservice architecture using FastAPI, implementing a Retrieval-Augmented Generation (RAG) pattern. The system processes documents through a pipeline of parsing, chunking, embedding, and intelligent retrieval to answer natural language queries with contextual accuracy.

The architecture follows a layered approach with clear separation of concerns: API layer, business logic layer, data access layer, and external service integration layer.

## Architecture

### High-Level Architecture

```mermaid
graph TB
    Client[Client Application] --> API[FastAPI Application]
    API --> Auth[Authentication Middleware]
    Auth --> Controller[Query Controller]
    Controller --> DocService[Document Service]
    Controller --> QueryService[Query Service]
    
    DocService --> Parser[Document Parser]
    DocService --> Chunker[Text Chunker]
    DocService --> Embedder[Embedding Service]
    
    QueryService --> Retriever[Vector Retriever]
    QueryService --> LLM[LLM Service]
    
    Parser --> Storage[File Storage]
    Embedder --> Pinecone[(Pinecone Vector DB)]
    Retriever --> Pinecone
    LLM --> Gemini[Gemini 2.0 Flash API]
    
    Controller --> PostgreSQL[(PostgreSQL)]
```

### Component Architecture

The system is organized into the following main components:

1. **API Layer**: FastAPI application with authentication and request/response handling
2. **Service Layer**: Business logic for document processing and query handling
3. **Data Layer**: Vector database (Pinecone) and relational database (PostgreSQL) integration
4. **External Services**: LLM API (Gemini) and embedding model (Jina) integration

## Components and Interfaces

### 1. API Layer

#### FastAPI Application (`main.py`)
- **Purpose**: Main application entry point and API endpoint definition
- **Responsibilities**: 
  - Route definition and request handling
  - Authentication middleware integration
  - Error handling and response formatting
  - CORS and security configuration

#### Authentication Middleware (`auth.py`)
- **Purpose**: Bearer token validation
- **Interface**:
  ```python
  async def verify_token(authorization: str = Header(...)) -> bool
  ```

#### Query Controller (`controllers/query_controller.py`)
- **Purpose**: Handle the main API endpoint logic
- **Interface**:
  ```python
  async def process_query_request(request: QueryRequest) -> QueryResponse
  ```

### 2. Service Layer

#### Document Service (`services/document_service.py`)
- **Purpose**: Handle document download, parsing, and processing
- **Key Methods**:
  ```python
  async def download_document(url: str) -> bytes
  async def parse_document(content: bytes, content_type: str) -> str
  async def chunk_document(text: str) -> List[DocumentChunk]
  async def store_embeddings(chunks: List[DocumentChunk]) -> str
  ```

#### Query Service (`services/query_service.py`)
- **Purpose**: Handle query processing and answer generation
- **Key Methods**:
  ```python
  async def process_questions(questions: List[str], document_id: str) -> List[str]
  async def retrieve_relevant_chunks(question: str, document_id: str) -> List[DocumentChunk]
  async def generate_answer(question: str, context_chunks: List[DocumentChunk]) -> str
  ```

#### Embedding Service (`services/embedding_service.py`)
- **Purpose**: Generate embeddings using Jina model
- **Interface**:
  ```python
  async def generate_embeddings(texts: List[str]) -> List[List[float]]
  async def generate_query_embedding(query: str) -> List[float]
  ```

#### LLM Service (`services/llm_service.py`)
- **Purpose**: Interface with Gemini 2.0 Flash API
- **Interface**:
  ```python
  async def generate_contextual_answer(question: str, context: str) -> str
  ```

### 3. Data Layer

#### Vector Store (`data/vector_store.py`)
- **Purpose**: Pinecone vector database operations
- **Key Methods**:
  ```python
  async def store_vectors(vectors: List[Vector], metadata: List[dict]) -> bool
  async def similarity_search(query_vector: List[float], top_k: int) -> List[SearchResult]
  async def delete_document_vectors(document_id: str) -> bool
  ```

#### Database Repository (`data/repository.py`)
- **Purpose**: PostgreSQL operations for metadata and session management
- **Key Methods**:
  ```python
  async def store_document_metadata(document_id: str, metadata: dict) -> bool
  async def get_document_metadata(document_id: str) -> dict
  async def log_query_session(session_data: dict) -> str
  ```

### 4. Utility Components

#### Document Parsers (`utils/parsers/`)
- **PDF Parser**: Using pypdf for PDF document processing
- **DOCX Parser**: Using python-docx for Word document processing
- **Email Parser**: Using email library for email content extraction

#### Text Chunker (`utils/text_chunker.py`)
- **Purpose**: Split documents into semantic chunks
- **Strategy**: Recursive character splitting with overlap for context preservation

## Data Models

### Request/Response Models

```python
class QueryRequest(BaseModel):
    documents: str  # URL to document
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
```

### Internal Data Models

```python
class DocumentChunk(BaseModel):
    id: str
    document_id: str
    content: str
    metadata: dict
    embedding: Optional[List[float]] = None

class SearchResult(BaseModel):
    chunk_id: str
    content: str
    score: float
    metadata: dict

class DocumentMetadata(BaseModel):
    document_id: str
    url: str
    content_type: str
    processed_at: datetime
    chunk_count: int
```

### Database Schema

#### PostgreSQL Tables

```sql
-- Document metadata table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url TEXT NOT NULL,
    content_type VARCHAR(50),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    chunk_count INTEGER,
    status VARCHAR(20) DEFAULT 'processing'
);

-- Query sessions for logging and analytics
CREATE TABLE query_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    questions JSONB,
    answers JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_time_ms INTEGER
);
```

## Error Handling

### Error Categories

1. **Client Errors (4xx)**:
   - Invalid request format
   - Authentication failures
   - Invalid document URLs
   - Unsupported document formats

2. **Server Errors (5xx)**:
   - Document download failures
   - Embedding service failures
   - Vector database connection issues
   - LLM API failures

### Error Handling Strategy

```python
class SystemError(Exception):
    def __init__(self, message: str, error_code: str, details: Optional[dict] = None):
        self.message = message
        self.error_code = error_code
        self.details = details

# Error handler middleware
@app.exception_handler(SystemError)
async def system_error_handler(request: Request, exc: SystemError):
    return JSONResponse(
        status_code=500,
        content={
            "error": exc.message,
            "error_code": exc.error_code,
            "details": exc.details
        }
    )
```

### Retry Logic

- **LLM API calls**: Exponential backoff with 3 retry attempts
- **Vector database operations**: Linear backoff with 2 retry attempts
- **Document downloads**: Single retry with timeout handling

## Testing Strategy

### Unit Testing

- **Service Layer**: Mock external dependencies (APIs, databases)
- **Utility Functions**: Test document parsing, chunking, and embedding logic
- **Data Models**: Validate serialization and validation rules

### Integration Testing

- **API Endpoints**: Test complete request/response cycles
- **Database Operations**: Test with test database instances
- **External Service Integration**: Test with sandbox/test environments

### End-to-End Testing

- **Complete Workflow**: Test document processing through answer generation
- **Error Scenarios**: Test various failure modes and recovery
- **Performance Testing**: Load testing with concurrent requests

### Test Structure

```
tests/
├── unit/
│   ├── test_document_service.py
│   ├── test_query_service.py
│   ├── test_embedding_service.py
│   └── test_parsers.py
├── integration/
│   ├── test_api_endpoints.py
│   ├── test_database_operations.py
│   └── test_vector_store.py
└── e2e/
    ├── test_complete_workflow.py
    └── test_error_scenarios.py
```

### Configuration Management

Environment variables required:

```bash
# API Configuration
AUTH_TOKEN=your_bearer_token_here
PORT=8000
HOST=0.0.0.0

# LLM Configuration
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash

# Embedding Configuration
JINA_API_KEY=your_jina_api_key
JINA_MODEL=jina-embeddings-v4

# Vector Database
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=document-embeddings

# PostgreSQL Database
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# Document Processing
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_DOCUMENT_SIZE_MB=50
```

### Performance Considerations

1. **Caching Strategy**: 
   - Cache embeddings for frequently accessed documents
   - Cache LLM responses for identical questions

2. **Async Processing**: 
   - Use async/await throughout the pipeline
   - Implement concurrent processing for multiple questions

3. **Resource Management**:
   - Connection pooling for database operations
   - Rate limiting for external API calls
   - Memory management for large document processing

4. **Scalability**:
   - Stateless service design for horizontal scaling
   - Queue-based processing for heavy workloads
   - Database indexing for optimal query performance