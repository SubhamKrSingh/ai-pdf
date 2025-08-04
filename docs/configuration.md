# Configuration Management Guide

This document provides comprehensive information about configuring the LLM Query Retrieval System for different environments and deployment scenarios.

## Overview

The system uses environment variables for configuration with comprehensive validation and structured logging. Configuration is managed through Pydantic settings with automatic validation and type checking.

## Configuration Files

### Environment Files

- `.env.example` - Template with all available configuration options
- `.env` - Your actual configuration (not committed to version control)
- `.env.test` - Test configuration for development

### Configuration Scripts

- `scripts/validate_config.py` - Validates current configuration
- `app/config.py` - Main configuration module with validation

## Environment Variables

### Required Variables

These variables must be set for the system to function:

```bash
# Authentication
AUTH_TOKEN=your_bearer_token_here

# API Keys
GEMINI_API_KEY=your_gemini_api_key_here
JINA_API_KEY=your_jina_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here

# Database
DATABASE_URL=postgresql://user:password@host:port/database
```

### Optional Variables

#### Application Settings
```bash
ENVIRONMENT=development          # development, staging, production
DEBUG=false                     # Enable debug mode
HOST=0.0.0.0                   # Server host
PORT=8000                      # Server port
```

#### Document Processing
```bash
MAX_CHUNK_SIZE=1000            # Maximum text chunk size
CHUNK_OVERLAP=200              # Overlap between chunks
MAX_DOCUMENT_SIZE_MB=50        # Maximum document size
```

#### Performance Settings
```bash
MAX_CONCURRENT_REQUESTS=100    # Maximum concurrent requests
WORKER_PROCESSES=1             # Number of worker processes
REQUEST_TIMEOUT=30             # HTTP request timeout
LLM_TIMEOUT=60                # LLM API timeout
MAX_RETRIES=3                 # Maximum API retries
RETRY_DELAY=1.0               # Retry delay in seconds
```

#### Database Configuration
```bash
DATABASE_POOL_SIZE=10          # Connection pool size
DATABASE_MAX_OVERFLOW=20       # Pool overflow limit
```

#### Logging Configuration
```bash
LOG_LEVEL=INFO                 # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=json                # json or text
LOG_FILE=                      # Log file path (empty for console)
ENABLE_ACCESS_LOGS=true        # Enable HTTP access logs
ENABLE_SQL_LOGS=false          # Enable SQL query logs (debug only)
```

#### Security Configuration
```bash
ALLOWED_HOSTS=["*"]            # Allowed hosts (JSON array)
CORS_ORIGINS=["*"]             # CORS origins (JSON array)
CORS_ALLOW_CREDENTIALS=true    # Allow credentials in CORS
ENABLE_HTTPS_REDIRECT=false    # Enable HTTPS redirect
```

#### Health Check Configuration
```bash
ENABLE_DETAILED_HEALTH=false   # Enable detailed health checks
HEALTH_CHECK_TIMEOUT=5         # Health check timeout
```

## Environment-Specific Configuration

### Development Environment

```bash
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
LOG_FORMAT=text
ALLOWED_HOSTS=["*"]
CORS_ORIGINS=["*"]
ENABLE_DETAILED_HEALTH=true
```

### Staging Environment

```bash
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=INFO
LOG_FORMAT=json
ALLOWED_HOSTS=["staging.yourdomain.com"]
CORS_ORIGINS=["https://staging.yourdomain.com"]
ENABLE_DETAILED_HEALTH=true
```

### Production Environment

```bash
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/app/logs/app.log
ALLOWED_HOSTS=["yourdomain.com", "www.yourdomain.com"]
CORS_ORIGINS=["https://yourdomain.com"]
CORS_ALLOW_CREDENTIALS=true
ENABLE_HTTPS_REDIRECT=true
ENABLE_DETAILED_HEALTH=false
```

## Configuration Validation

### Automatic Validation

The system automatically validates configuration on startup:

- Required fields are checked
- Data types are validated
- Value ranges are enforced
- Cross-field dependencies are verified

### Manual Validation

Use the validation script to check configuration:

```bash
python scripts/validate_config.py
```

This will:
- Validate all environment variables
- Check for missing required values
- Verify configuration consistency
- Provide warnings for potential issues
- Display configuration summary

### Validation Rules

#### API Keys
- Must not be empty
- Whitespace is trimmed
- Minimum length requirements

#### Database URL
- Must start with `postgresql://` or `postgres://`
- Connection string format validation

#### Numeric Values
- Port: 1-65535
- Chunk size: 100-10000
- Document size: 1-1000 MB
- Pool sizes: 1-100
- Concurrent requests: 1-1000
- Worker processes: 1-32

#### Environment-Specific Rules
- Production: Debug mode disabled
- Production: No wildcard hosts/origins
- Development: Relaxed security settings allowed

## Configuration Loading

### Loading Order

1. System environment variables
2. `.env` file (if present)
3. Default values from settings class

### Nested Configuration

List values can be specified as JSON arrays:
```bash
ALLOWED_HOSTS=["host1.com", "host2.com"]
CORS_ORIGINS=["https://app.com", "https://admin.com"]
```

Or as comma-separated values:
```bash
ALLOWED_HOSTS=host1.com,host2.com
CORS_ORIGINS=https://app.com,https://admin.com
```

## Logging Configuration

### Log Levels

- `DEBUG`: Detailed debugging information
- `INFO`: General operational information
- `WARNING`: Warning messages
- `ERROR`: Error conditions
- `CRITICAL`: Critical errors

### Log Formats

#### JSON Format (Production)
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "INFO",
  "logger": "app.services.query_service",
  "message": "Processing query",
  "module": "query_service",
  "function": "process_query",
  "line": 45
}
```

#### Text Format (Development)
```
2024-01-01 12:00:00 - app.services.query_service - INFO - Processing query
```

### Log Files

When `LOG_FILE` is set:
- Logs are written to the specified file
- File rotation is enabled (10MB, 5 backups)
- Console logging is reduced to warnings and above

## Security Configuration

### Production Security

For production deployments:

```bash
ENVIRONMENT=production
DEBUG=false
ALLOWED_HOSTS=["yourdomain.com"]
CORS_ORIGINS=["https://yourdomain.com"]
ENABLE_HTTPS_REDIRECT=true
```

### Development Security

For development:

```bash
ENVIRONMENT=development
DEBUG=true
ALLOWED_HOSTS=["*"]
CORS_ORIGINS=["*"]
ENABLE_HTTPS_REDIRECT=false
```

## Health Check Configuration

### Basic Health Check

Always available at `/health`:
```json
{
  "status": "healthy",
  "service": "LLM Query Retrieval System"
}
```

### Detailed Health Check

When `ENABLE_DETAILED_HEALTH=true`, available at `/health/detailed`:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
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

## Configuration API

### Validation Endpoint

`GET /config/validate` (requires authentication):
```json
{
  "status": "valid",
  "errors": [],
  "warnings": [],
  "config_summary": {
    "environment": "development",
    "debug": true,
    "host": "0.0.0.0",
    "port": 8000
  }
}
```

## Troubleshooting

### Common Issues

#### Missing Environment Variables
```
Configuration validation failed: Missing required environment variables: AUTH_TOKEN, GEMINI_API_KEY
```
**Solution**: Set all required environment variables in `.env` file.

#### Invalid Database URL
```
Database URL must start with postgresql:// or postgres://
```
**Solution**: Ensure database URL has correct format.

#### Production Security Warnings
```
Wildcard CORS origins not allowed in production
```
**Solution**: Set specific allowed origins for production.

### Debugging Configuration

1. **Check current configuration**:
   ```bash
   python scripts/validate_config.py
   ```

2. **Test configuration loading**:
   ```python
   from app.config import get_settings
   settings = get_settings()
   print(settings.model_dump())
   ```

3. **Validate specific settings**:
   ```python
   from app.config import validate_environment
   result = validate_environment()
   print(result)
   ```

## Best Practices

### Security
- Never commit `.env` files to version control
- Use strong, unique API keys
- Rotate keys regularly
- Use specific hosts/origins in production
- Enable HTTPS redirect in production

### Performance
- Adjust pool sizes based on load
- Monitor concurrent request limits
- Use appropriate timeout values
- Enable caching where possible

### Monitoring
- Enable detailed health checks in non-production
- Use structured JSON logging in production
- Set appropriate log levels
- Monitor log file sizes

### Deployment
- Validate configuration before deployment
- Use environment-specific configurations
- Test configuration changes in staging
- Document configuration changes

## Configuration Schema

The complete configuration schema is defined in `app/config.py` using Pydantic models with comprehensive validation rules. Refer to the source code for the most up-to-date field definitions and validation logic.