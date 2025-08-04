# Comprehensive Error Handling System Guide

This guide explains how to use the comprehensive error handling system implemented for the LLM Query Retrieval System.

## Overview

The error handling system provides:

- **Custom Exception Classes**: Structured exceptions with proper categorization
- **Global Error Handler Middleware**: Automatic error catching and structured responses
- **Retry Mechanisms**: Configurable retry logic for recoverable errors
- **Graceful Degradation**: Fallback strategies for service failures
- **Comprehensive Logging**: Structured error logging with context
- **Error Monitoring**: Statistics and metrics for error tracking

## Custom Exception Classes

### Base Exception

All custom exceptions inherit from `BaseSystemError`:

```python
from app.exceptions import BaseSystemError, ErrorCategory

error = BaseSystemError(
    message="Something went wrong",
    error_code="CUSTOM_ERROR",
    category=ErrorCategory.SERVER_ERROR,
    details={"context": "additional info"},
    recoverable=True,
    http_status_code=500
)
```

### Client Errors (4xx)

For errors caused by client requests:

```python
from app.exceptions import ValidationError, AuthenticationError, DocumentNotFoundError

# Validation error
raise ValidationError(
    message="Invalid field value",
    field="document_url",
    value="invalid-url"
)

# Authentication error
raise AuthenticationError(
    message="Invalid token",
    details={"token_type": "bearer"}
)

# Document not found
raise DocumentNotFoundError(
    url="https://example.com/missing.pdf"
)
```

### Server Errors (5xx)

For internal server errors:

```python
from app.exceptions import (
    DocumentDownloadError,
    EmbeddingServiceError,
    LLMServiceError,
    VectorStoreError,
    DatabaseError
)

# Document download error
raise DocumentDownloadError(
    url="https://example.com/doc.pdf",
    status_code=404,
    reason="Not found"
)

# Service errors
raise EmbeddingServiceError(
    operation="generate_embeddings",
    reason="API timeout"
)

raise LLMServiceError(
    operation="generate_answer",
    reason="Rate limit exceeded"
)
```

## Global Error Handler Middleware

The middleware automatically catches all exceptions and provides structured responses:

```python
from app.middleware.error_handler import setup_error_handling

# Set up error handling for your FastAPI app
app = FastAPI()
error_middleware = setup_error_handling(app, enable_debug=False)
```

### Error Response Format

All errors return a structured JSON response:

```json
{
  "error": "Human-readable error message",
  "error_code": "MACHINE_READABLE_CODE",
  "details": {
    "category": "client_error|server_error",
    "recoverable": true,
    "processing_time_ms": 123.45,
    "additional_context": "..."
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Retry Mechanisms

### Using Retry Decorators

```python
from app.utils.retry import with_retry, RetryConfig, EMBEDDING_RETRY_CONFIG

# Use predefined config
@with_retry(EMBEDDING_RETRY_CONFIG)
async def call_embedding_service():
    # Your code here
    pass

# Use custom config
custom_config = RetryConfig(
    max_attempts=5,
    initial_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0
)

@with_retry(custom_config, context={"operation": "custom_operation"})
async def custom_function():
    # Your code here
    pass
```

### Manual Retry Usage

```python
from app.utils.retry import retry_async, RetryConfig

async def flaky_function():
    # Function that might fail
    pass

config = RetryConfig(max_attempts=3)
result = await retry_async(flaky_function, config)

if result.success:
    return result.result
else:
    raise result.error
```

### Predefined Retry Configurations

```python
from app.utils.retry import (
    DEFAULT_RETRY_CONFIG,    # General purpose: 3 attempts, 1s delay
    LLM_RETRY_CONFIG,        # LLM services: 3 attempts, 2s delay, 30s max
    EMBEDDING_RETRY_CONFIG,  # Embedding: 2 attempts, 1s delay, 10s max
    DATABASE_RETRY_CONFIG,   # Database: 2 attempts, 0.5s delay, 5s max
    DOWNLOAD_RETRY_CONFIG    # Downloads: 2 attempts, 1s delay, 10s max
)
```

## Graceful Degradation

### Using Fallback Functions

```python
from app.utils.retry import GracefulDegradation

async def primary_service():
    # Primary functionality that might fail
    raise ConnectionError("Service unavailable")

async def fallback_service():
    # Fallback functionality
    return {"message": "Using cached data", "degraded": True}

# Use fallback function
result = await GracefulDegradation.with_fallback(
    primary_service,
    fallback_service
)
```

### Using Fallback Values

```python
# Use fallback value
result = await GracefulDegradation.with_fallback(
    primary_service,
    fallback_value={"default": "response"}
)
```

### Circuit Breaker Pattern

```python
# Create circuit breaker
circuit_breaker = GracefulDegradation.create_circuit_breaker(
    failure_threshold=5,
    recovery_timeout=60.0,
    expected_exception=ConnectionError
)

@circuit_breaker
async def protected_service():
    # Service protected by circuit breaker
    pass
```

## Error Logging

### Structured Error Logging

```python
from app.middleware.error_handler import error_logger

# Log error with context
try:
    # Some operation
    pass
except Exception as e:
    error_logger.log_error(
        error=e,
        context={
            "operation": "document_processing",
            "component": "document_service",
            "document_id": "doc_123"
        }
    )
```

### Recovery Logging

```python
# Log recovery attempts
error_logger.log_recovery_attempt(
    error=exception,
    attempt=2,
    max_attempts=3,
    context={"operation": "embedding_generation"}
)

# Log successful recovery
error_logger.log_recovery_success(
    error=exception,
    attempts_used=2,
    context={"operation": "embedding_generation"}
)

# Log recovery failure
error_logger.log_recovery_failure(
    error=exception,
    attempts_used=3,
    context={"operation": "embedding_generation"}
)
```

## Error Monitoring

### Error Statistics

Access error statistics via the monitoring endpoint:

```bash
GET /api/v1/errors/stats
```

Response:
```json
{
  "total_errors": 150,
  "client_errors": 90,
  "server_errors": 60,
  "recoverable_errors": 45,
  "error_rate": 0.15,
  "client_error_rate": 0.6,
  "server_error_rate": 0.4,
  "recovery_rate": 0.3
}
```

### Programmatic Access

```python
# Get error statistics from middleware
stats = error_middleware.get_error_statistics()
print(f"Total errors: {stats['total_errors']}")
print(f"Recovery rate: {stats['recovery_rate']:.2%}")
```

## Best Practices

### 1. Use Appropriate Exception Types

```python
# Good: Use specific exception types
raise DocumentDownloadError(url, status_code=404)

# Bad: Use generic exceptions
raise Exception("Download failed")
```

### 2. Provide Context in Error Details

```python
# Good: Include relevant context
raise EmbeddingServiceError(
    operation="generate_embeddings",
    reason="API timeout",
    details={
        "texts_count": len(texts),
        "timeout_seconds": 30,
        "api_endpoint": "https://api.jina.ai/v1/embeddings"
    }
)

# Bad: Minimal context
raise EmbeddingServiceError("API failed")
```

### 3. Use Retry for Recoverable Errors

```python
# Good: Retry recoverable operations
@with_retry(EMBEDDING_RETRY_CONFIG)
async def generate_embeddings(texts):
    # Implementation
    pass

# Consider: Non-recoverable operations shouldn't be retried
def parse_document(content):
    # Document parsing failures are usually not recoverable
    pass
```

### 4. Implement Graceful Degradation

```python
# Good: Provide fallback for non-critical features
async def get_document_summary():
    return await GracefulDegradation.with_fallback(
        primary_func=generate_ai_summary,
        fallback_value="Summary not available"
    )
```

### 5. Log Errors with Context

```python
# Good: Include operation context
try:
    result = await process_document(doc_id)
except Exception as e:
    error_logger.log_error(
        error=e,
        context={
            "operation": "document_processing",
            "document_id": doc_id,
            "user_id": user_id
        }
    )
    raise
```

## Testing Error Handling

### Unit Tests

```python
import pytest
from app.exceptions import DocumentDownloadError

def test_document_download_error():
    error = DocumentDownloadError(
        url="https://example.com/doc.pdf",
        status_code=404
    )
    
    assert error.error_code == "DOCUMENT_DOWNLOAD_ERROR"
    assert error.recoverable is True
    assert error.details["url"] == "https://example.com/doc.pdf"
```

### Integration Tests

```python
from fastapi.testclient import TestClient

def test_error_handling_integration(client: TestClient):
    response = client.post("/api/v1/hackrx/run", json={
        "documents": "https://invalid-url",
        "questions": ["test"]
    })
    
    assert response.status_code == 502
    data = response.json()
    assert data["error_code"] == "DOCUMENT_DOWNLOAD_ERROR"
    assert "X-Error-Category" in response.headers
```

## Configuration

### Environment Variables

```bash
# Retry configuration
MAX_RETRIES=3
RETRY_DELAY=1.0

# Timeout configuration
REQUEST_TIMEOUT=30
LLM_TIMEOUT=60

# Debug mode (shows detailed error information)
DEBUG=false
```

### Application Settings

```python
from app.config import get_settings

settings = get_settings()
print(f"Max retries: {settings.max_retries}")
print(f"Request timeout: {settings.request_timeout}")
```

This comprehensive error handling system ensures robust operation of the LLM Query Retrieval System with proper error categorization, recovery mechanisms, and monitoring capabilities.