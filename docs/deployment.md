# Deployment Guide

This guide covers deployment options for the LLM Query Retrieval System, including Docker containerization, configuration management, and production best practices.

## Quick Start

### Development Deployment

1. **Clone and Setup**:
   ```bash
   git clone <repository-url>
   cd llm-query-retrieval-system
   cp .env.example .env
   ```

2. **Configure Environment**:
   Edit `.env` file with your API keys and configuration:
   ```bash
   # Required API Keys
   AUTH_TOKEN=your_bearer_token_here
   GEMINI_API_KEY=your_gemini_api_key_here
   JINA_API_KEY=your_jina_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_ENVIRONMENT=your_pinecone_environment_here
   DATABASE_URL=postgresql://postgres:password@localhost:5432/llm_query_db
   ```

3. **Deploy with Docker**:
   ```bash
   # Linux/Mac
   ./scripts/deploy.sh development
   
   # Windows
   .\scripts\deploy.ps1 development
   ```

### Production Deployment

1. **Configure for Production**:
   ```bash
   # Set production environment
   ENVIRONMENT=production
   DEBUG=false
   
   # Configure security
   SECURITY__ALLOWED_HOSTS=["yourdomain.com", "www.yourdomain.com"]
   SECURITY__CORS_ORIGINS=["https://yourdomain.com"]
   SECURITY__ENABLE_HTTPS_REDIRECT=true
   
   # Configure logging
   LOG__LOG_LEVEL=INFO
   LOG__LOG_FORMAT=json
   LOG__LOG_FILE=/app/logs/app.log
   ```

2. **Deploy**:
   ```bash
   # Linux/Mac
   ./scripts/deploy.sh production
   
   # Windows
   .\scripts\deploy.ps1 production
   ```

## Configuration Management

### Environment Variables

The system uses comprehensive environment variable configuration with validation:

#### Core Configuration
- `ENVIRONMENT`: Application environment (development/staging/production)
- `DEBUG`: Enable debug mode (true/false)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)

#### API Keys (Required)
- `AUTH_TOKEN`: Bearer token for API authentication
- `GEMINI_API_KEY`: Google Gemini API key
- `JINA_API_KEY`: Jina embeddings API key
- `PINECONE_API_KEY`: Pinecone vector database API key
- `PINECONE_ENVIRONMENT`: Pinecone environment name

#### Database Configuration
- `DATABASE_URL`: PostgreSQL connection URL
- `DATABASE_POOL_SIZE`: Connection pool size (default: 10)
- `DATABASE_MAX_OVERFLOW`: Max pool overflow (default: 20)

#### Logging Configuration
- `LOG__LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
- `LOG__LOG_FORMAT`: Log format (json/text)
- `LOG__LOG_FILE`: Log file path (optional, logs to stdout if not set)
- `LOG__ENABLE_ACCESS_LOGS`: Enable HTTP access logging
- `LOG__ENABLE_SQL_LOGS`: Enable SQL query logging (debug only)

#### Security Configuration
- `SECURITY__ALLOWED_HOSTS`: Allowed hosts for TrustedHostMiddleware
- `SECURITY__CORS_ORIGINS`: Allowed CORS origins
- `SECURITY__CORS_ALLOW_CREDENTIALS`: Allow credentials in CORS
- `SECURITY__ENABLE_HTTPS_REDIRECT`: Enable HTTPS redirect

#### Performance Configuration
- `MAX_CONCURRENT_REQUESTS`: Maximum concurrent requests
- `WORKER_PROCESSES`: Number of worker processes
- `REQUEST_TIMEOUT`: HTTP request timeout
- `LLM_TIMEOUT`: LLM API timeout

### Configuration Validation

The system includes comprehensive configuration validation:

```python
from app.config import validate_environment

# Validate configuration
try:
    result = validate_environment()
    print(f"Configuration status: {result['status']}")
    if result['warnings']:
        print("Warnings:", result['warnings'])
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Docker Deployment

### Docker Compose (Recommended)

The system includes Docker Compose configurations for different environments:

#### Development
```bash
docker-compose -f docker-compose.dev.yml up -d
```

Features:
- Hot reload enabled
- Debug logging
- SQL query logging
- Relaxed security settings

#### Production
```bash
docker-compose up -d
```

Features:
- Optimized build
- JSON logging
- Security hardening
- Health checks
- Automatic restarts

### Manual Docker Build

```bash
# Build image
docker build -t llm-query-system .

# Run container
docker run -d \
  --name llm-query-system \
  -p 8000:8000 \
  --env-file .env \
  llm-query-system
```

## Health Monitoring

### Health Check Endpoints

1. **Basic Health Check**:
   ```
   GET /health
   ```
   Returns basic service status.

2. **Detailed Health Check**:
   ```
   GET /health/detailed
   ```
   Returns detailed service information including dependency status.

3. **Configuration Validation**:
   ```
   GET /config/validate
   ```
   Validates current configuration (requires authentication).

### Docker Health Checks

The Docker containers include built-in health checks:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

### Monitoring Integration

For production monitoring, integrate with:
- **Prometheus**: Metrics collection
- **Grafana**: Visualization
- **ELK Stack**: Log aggregation
- **Sentry**: Error tracking

## Logging

### Structured Logging

The system supports structured JSON logging for production:

```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "INFO",
  "logger": "app.services.query_service",
  "message": "Processing query with 3 questions",
  "module": "query_service",
  "function": "process_questions",
  "line": 45
}
```

### Log Levels

- **DEBUG**: Detailed debugging information
- **INFO**: General information about system operation
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages for handled exceptions
- **CRITICAL**: Critical errors that may cause system failure

### Log Rotation

For production, configure log rotation:

```bash
# Using logrotate
/app/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 appuser appuser
}
```

## Security Considerations

### Production Security Checklist

- [ ] Set `ENVIRONMENT=production`
- [ ] Disable debug mode (`DEBUG=false`)
- [ ] Configure specific allowed hosts
- [ ] Set specific CORS origins
- [ ] Enable HTTPS redirect
- [ ] Use strong authentication tokens
- [ ] Secure API keys in environment variables
- [ ] Configure firewall rules
- [ ] Enable container security scanning
- [ ] Set up SSL/TLS certificates
- [ ] Configure rate limiting
- [ ] Enable audit logging

### API Key Management

Store API keys securely:
- Use environment variables
- Never commit keys to version control
- Rotate keys regularly
- Use key management services (AWS KMS, Azure Key Vault, etc.)

## Scaling and Performance

### Horizontal Scaling

Scale the application using multiple instances:

```bash
# Scale with Docker Compose
docker-compose up -d --scale app=3

# Use load balancer (nginx, HAProxy, etc.)
```

### Performance Optimization

1. **Database Connection Pooling**:
   ```bash
   DATABASE_POOL_SIZE=20
   DATABASE_MAX_OVERFLOW=40
   ```

2. **Concurrent Request Handling**:
   ```bash
   MAX_CONCURRENT_REQUESTS=200
   WORKER_PROCESSES=4
   ```

3. **Caching**:
   - Implement Redis for caching embeddings
   - Cache frequent queries
   - Use CDN for static assets

### Resource Requirements

#### Minimum Requirements
- CPU: 2 cores
- RAM: 4GB
- Storage: 20GB
- Network: 100Mbps

#### Recommended for Production
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 100GB+ SSD
- Network: 1Gbps+

## Troubleshooting

### Common Issues

1. **Configuration Validation Errors**:
   ```bash
   # Check configuration
   python -c "from app.config import validate_environment; validate_environment()"
   ```

2. **Database Connection Issues**:
   ```bash
   # Test database connection
   docker-compose exec app python -c "from app.data.repository import test_connection; test_connection()"
   ```

3. **API Key Issues**:
   ```bash
   # Validate API keys
   curl -H "Authorization: Bearer $AUTH_TOKEN" http://localhost:8000/config/validate
   ```

### Log Analysis

Check logs for issues:
```bash
# View application logs
docker-compose logs -f app

# View database logs
docker-compose logs -f db

# Search for errors
docker-compose logs app | grep ERROR
```

### Performance Issues

Monitor performance:
```bash
# Check resource usage
docker stats

# Monitor API response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health
```

## Backup and Recovery

### Database Backup

```bash
# Create backup
docker-compose exec db pg_dump -U postgres llm_query_db > backup.sql

# Restore backup
docker-compose exec -T db psql -U postgres llm_query_db < backup.sql
```

### Configuration Backup

- Backup `.env` files
- Document API key sources
- Version control deployment scripts
- Maintain infrastructure as code

## Support and Maintenance

### Regular Maintenance Tasks

1. **Update Dependencies**:
   ```bash
   pip install -r requirements.txt --upgrade
   docker-compose build --no-cache
   ```

2. **Database Maintenance**:
   ```bash
   # Run migrations
   docker-compose exec app python -c "from app.data.migrations import run_migrations; run_migrations()"
   
   # Vacuum database
   docker-compose exec db psql -U postgres -c "VACUUM ANALYZE;"
   ```

3. **Log Cleanup**:
   ```bash
   # Clean old logs
   find /app/logs -name "*.log" -mtime +30 -delete
   ```

### Monitoring Alerts

Set up alerts for:
- High error rates
- Slow response times
- Database connection issues
- High resource usage
- Failed health checks

For additional support, refer to the API documentation at `/docs` when the service is running.