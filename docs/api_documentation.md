# LLM Query Retrieval System - API Documentation

## Overview

The LLM Query Retrieval System is a FastAPI-based REST API that processes documents and answers natural language questions using advanced language models and vector search technology. The system implements a Retrieval-Augmented Generation (RAG) pattern to provide contextual and accurate answers based on document content.

## Base URL

```
http://localhost:8000
```

## Authentication

All API endpoints (except health checks) require Bearer token authentication.

**Header:**
```
Authorization: Bearer <your-auth-token>
```

The auth token is configured via the `AUTH_TOKEN` environment variable.

## API Endpoints

### 1. Process Query

**Endpoint:** `POST /api/v1/hackrx/run`

**Description:** Main endpoint for processing documents and answering natural language questions.

**Request Body:**
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the main topic of this document?",
    "What are the key findings mentioned?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "The main topic of this document is artificial intelligence applications in healthcare.",
    "The key findings include improved diagnostic accuracy and reduced processing time."
  ]
}
```

**Request Schema:**
- `documents` (string, required): URL to the document to be processed. Must be a valid HTTP/HTTPS URL.
- `questions` (array of strings, required): Array of natural language questions to answer. Minimum 1 question, maximum 50 questions. Each question must be 3-1000 characters.

**Response Schema:**
- `answers` (array of strings): Array of answers corresponding to input questions in the same order.

**Supported Document Formats:**
- PDF (`.pdf`)
- Microsoft Word (`.docx`)
- Email files (`.eml`)
- Plain text (`.txt`)

**Status Codes:**
- `200 OK`: Successful processing
- `400 Bad Request`: Invalid request format or unsupported document
- `401 Unauthorized`: Missing or invalid authentication token
- `422 Unprocessable Entity`: Validation errors in request data
- `500 Internal Server Error`: Server-side processing errors

**Error Response Format:**
```json
{
  "error": "Human-readable error message",
  "error_code": "MACHINE_READABLE_ERROR_CODE",
  "details": {
    "additional": "context information"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 2. Health Check

**Endpoint:** `GET /health`

**Description:** Basic health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "service": "LLM Query Retrieval System"
}
```

### 3. Detailed Health Check

**Endpoint:** `GET /health/detailed`

**Description:** Detailed health check with service status information.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "environment": "production",
  "services": {
    "database": {"status": "healthy"},
    "vector_store": {"status": "healthy"},
    "llm_service": {"status": "healthy"},
    "embedding_service": {"status": "healthy"}
  }
}
```

### 4. Configuration Validation

**Endpoint:** `GET /config/validate`

**Description:** Validate current system configuration (requires authentication).

**Response:**
```json
{
  "status": "valid",
  "errors": [],
  "warnings": [],
  "config_summary": {
    "environment": "production",
    "debug": false,
    "host": "0.0.0.0",
    "port": 8000,
    "log_level": "INFO"
  }
}
```

## Interactive Documentation

The API provides interactive documentation through:

- **Swagger UI**: Available at `/docs`
- **ReDoc**: Available at `/redoc`

These interfaces allow you to explore the API, view detailed schemas, and test endpoints directly from your browser.

## Rate Limiting and Performance

- **Maximum concurrent requests**: 100 (configurable)
- **Request timeout**: 30 seconds (configurable)
- **LLM timeout**: 60 seconds (configurable)
- **Maximum document size**: 50 MB (configurable)
- **Maximum questions per request**: 50

## Error Codes

| Error Code | Description |
|------------|-------------|
| `VALIDATION_ERROR` | Request validation failed |
| `AUTHENTICATION_ERROR` | Invalid or missing authentication |
| `DOCUMENT_DOWNLOAD_ERROR` | Failed to download document from URL |
| `DOCUMENT_PARSE_ERROR` | Failed to parse document content |
| `EMBEDDING_ERROR` | Failed to generate embeddings |
| `VECTOR_STORE_ERROR` | Vector database operation failed |
| `LLM_ERROR` | Language model API call failed |
| `DATABASE_ERROR` | Database operation failed |
| `INTERNAL_ERROR` | Unexpected server error |

## Request/Response Examples

### Example 1: PDF Document Analysis

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Authorization: Bearer your-auth-token" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/research-paper.pdf",
    "questions": [
      "What is the research methodology used?",
      "What are the main conclusions?",
      "What future work is suggested?"
    ]
  }'
```

**Response:**
```json
{
  "answers": [
    "The research methodology used is a randomized controlled trial with 500 participants over 12 months.",
    "The main conclusions are that the intervention showed significant improvement in patient outcomes with a 25% reduction in readmission rates.",
    "Future work should focus on expanding the study to multiple healthcare systems and investigating long-term effects beyond 12 months."
  ]
}
```

### Example 2: Error Response

**Request with invalid URL:**
```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Authorization: Bearer your-auth-token" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "invalid-url",
    "questions": ["What is this document about?"]
  }'
```

**Response:**
```json
{
  "error": "Invalid document URL provided",
  "error_code": "VALIDATION_ERROR",
  "details": {
    "field": "documents",
    "message": "URL scheme must be http or https"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Processing Pipeline

The system processes requests through the following pipeline:

1. **Authentication**: Validate Bearer token
2. **Request Validation**: Validate request format and constraints
3. **Document Download**: Download document from provided URL
4. **Document Parsing**: Extract text content based on file type
5. **Text Chunking**: Split document into semantic chunks
6. **Embedding Generation**: Generate vector embeddings for chunks
7. **Vector Storage**: Store embeddings in Pinecone vector database
8. **Query Processing**: For each question:
   - Generate query embedding
   - Perform semantic search
   - Retrieve relevant chunks
   - Generate contextual answer using LLM
9. **Response Formation**: Return structured JSON response

## Security Considerations

- All API endpoints require authentication
- CORS is configured for allowed origins
- Request size limits are enforced
- Input validation prevents injection attacks
- Sensitive configuration is managed through environment variables
- HTTPS is recommended for production deployments

## Monitoring and Logging

The system provides comprehensive logging and monitoring:

- **Request Logging**: All requests are logged with timing information
- **Error Logging**: Detailed error logging with stack traces
- **Performance Metrics**: Response times and processing statistics
- **Health Checks**: Regular health monitoring endpoints
- **Structured Logging**: JSON format for production environments

## API Versioning

The current API version is `v1`. The version is included in the URL path (`/api/v1/`). Future versions will maintain backward compatibility or provide migration guides.