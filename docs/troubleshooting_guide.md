# Troubleshooting Guide

This guide helps you diagnose and resolve common issues when using the LLM Query Retrieval System.

## Table of Contents

1. [Common Error Messages](#common-error-messages)
2. [Authentication Issues](#authentication-issues)
3. [Document Processing Issues](#document-processing-issues)
4. [Performance Issues](#performance-issues)
5. [Configuration Problems](#configuration-problems)
6. [Network and Connectivity Issues](#network-and-connectivity-issues)
7. [Debugging Tools](#debugging-tools)
8. [Getting Help](#getting-help)

## Common Error Messages

### 1. Authentication Errors

#### Error: "Authentication failed" (401 Unauthorized)

**Symptoms:**
```json
{
  "error": "Authentication failed",
  "error_code": "AUTHENTICATION_ERROR",
  "details": {
    "message": "Invalid or missing Bearer token"
  }
}
```

**Causes:**
- Missing `Authorization` header
- Invalid or expired auth token
- Incorrect token format

**Solutions:**
1. Verify your auth token is correct:
   ```bash
   echo $AUTH_TOKEN  # Check environment variable
   ```

2. Ensure proper header format:
   ```bash
   curl -H "Authorization: Bearer your-token-here" ...
   ```

3. Check token configuration in environment:
   ```bash
   # Verify token is set correctly
   grep AUTH_TOKEN .env
   ```

#### Error: "Bearer token required"

**Solution:**
Always include the Authorization header:
```python
headers = {
    'Authorization': f'Bearer {your_token}',
    'Content-Type': 'application/json'
}
```

### 2. Validation Errors

#### Error: "Invalid document URL" (422 Unprocessable Entity)

**Symptoms:**
```json
{
  "error": "Invalid document URL provided",
  "error_code": "VALIDATION_ERROR",
  "details": {
    "field": "documents",
    "message": "URL scheme must be http or https"
  }
}
```

**Solutions:**
1. Ensure URL starts with `http://` or `https://`
2. Verify URL is accessible and returns a valid document
3. Test URL in browser first

#### Error: "Question validation failed"

**Common issues:**
- Empty questions
- Questions too short (< 3 characters)
- Questions too long (> 1000 characters)
- Too many questions (> 50)

**Solution:**
```python
# Validate questions before sending
def validate_questions(questions):
    if not questions:
        raise ValueError("At least one question required")
    
    if len(questions) > 50:
        raise ValueError("Maximum 50 questions allowed")
    
    for i, q in enumerate(questions):
        q = q.strip()
        if not q:
            raise ValueError(f"Question {i+1} cannot be empty")
        if len(q) < 3:
            raise ValueError(f"Question {i+1} too short")
        if len(q) > 1000:
            raise ValueError(f"Question {i+1} too long")
```

### 3. Document Processing Errors

#### Error: "Document download failed" (DOCUMENT_DOWNLOAD_ERROR)

**Symptoms:**
```json
{
  "error": "Failed to download document from URL",
  "error_code": "DOCUMENT_DOWNLOAD_ERROR",
  "details": {
    "url": "https://example.com/document.pdf",
    "status_code": 404
  }
}
```

**Troubleshooting Steps:**

1. **Check URL accessibility:**
   ```bash
   curl -I "https://example.com/document.pdf"
   ```

2. **Verify document exists and is accessible:**
   - Open URL in browser
   - Check for authentication requirements
   - Verify file permissions

3. **Check document size:**
   ```bash
   curl -sI "https://example.com/document.pdf" | grep -i content-length
   ```
   - Default limit: 50MB
   - Configure with `MAX_DOCUMENT_SIZE_MB`

4. **Test with different document:**
   ```bash
   # Try with a known working document
   curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
     -H "Authorization: Bearer $AUTH_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf", "questions": ["What is this?"]}'
   ```

#### Error: "Document parsing failed" (DOCUMENT_PARSE_ERROR)

**Symptoms:**
```json
{
  "error": "Failed to parse document content",
  "error_code": "DOCUMENT_PARSE_ERROR",
  "details": {
    "content_type": "application/pdf",
    "message": "PDF parsing failed: corrupted file"
  }
}
```

**Solutions:**

1. **Verify document format:**
   - Supported: PDF, DOCX, EML, TXT
   - Check file extension matches content

2. **Test document integrity:**
   ```bash
   # For PDF files
   pdfinfo document.pdf
   
   # For DOCX files
   unzip -t document.docx
   ```

3. **Try with different document:**
   - Test with known good document
   - Check if issue is document-specific

4. **Check document size and complexity:**
   - Very large documents may timeout
   - Password-protected documents not supported
   - Scanned PDFs (images) may not extract text properly

#### Error: "Text chunking failed"

**Symptoms:**
- Document downloads but processing fails during chunking
- Timeout errors during processing

**Solutions:**

1. **Check chunk configuration:**
   ```bash
   # Verify chunk settings
   echo "MAX_CHUNK_SIZE: $MAX_CHUNK_SIZE"
   echo "CHUNK_OVERLAP: $CHUNK_OVERLAP"
   ```

2. **Adjust chunk parameters:**
   ```bash
   # In .env file
   MAX_CHUNK_SIZE=800  # Reduce if having issues
   CHUNK_OVERLAP=100   # Reduce overlap
   ```

3. **Check document content:**
   - Very repetitive content may cause issues
   - Documents with unusual formatting

### 4. External Service Errors

#### Error: "Embedding service failed" (EMBEDDING_ERROR)

**Symptoms:**
```json
{
  "error": "Failed to generate embeddings",
  "error_code": "EMBEDDING_ERROR",
  "details": {
    "service": "jina",
    "message": "API rate limit exceeded"
  }
}
```

**Solutions:**

1. **Check API key:**
   ```bash
   echo $JINA_API_KEY
   ```

2. **Verify API quota:**
   - Check Jina API dashboard
   - Monitor usage limits

3. **Test API connectivity:**
   ```bash
   curl -H "Authorization: Bearer $JINA_API_KEY" \
        "https://api.jina.ai/v1/embeddings" \
        -d '{"input": ["test"], "model": "jina-embeddings-v4"}'
   ```

#### Error: "Vector store operation failed" (VECTOR_STORE_ERROR)

**Solutions:**

1. **Check Pinecone configuration:**
   ```bash
   echo "PINECONE_API_KEY: $PINECONE_API_KEY"
   echo "PINECONE_ENVIRONMENT: $PINECONE_ENVIRONMENT"
   echo "PINECONE_INDEX_NAME: $PINECONE_INDEX_NAME"
   ```

2. **Verify index exists:**
   - Check Pinecone dashboard
   - Ensure index is created and active

3. **Test Pinecone connectivity:**
   ```python
   import pinecone
   
   pinecone.init(
       api_key="your-api-key",
       environment="your-environment"
   )
   
   # List indexes
   print(pinecone.list_indexes())
   ```

#### Error: "LLM service failed" (LLM_ERROR)

**Solutions:**

1. **Check Gemini API key:**
   ```bash
   echo $GEMINI_API_KEY
   ```

2. **Verify API quota and billing:**
   - Check Google AI Studio dashboard
   - Ensure billing is enabled

3. **Test Gemini API:**
   ```bash
   curl -H "Content-Type: application/json" \
        -d '{"contents":[{"parts":[{"text":"Hello"}]}]}' \
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=$GEMINI_API_KEY"
   ```

## Performance Issues

### Slow Response Times

**Symptoms:**
- Requests taking longer than expected
- Timeouts occurring frequently

**Diagnostic Steps:**

1. **Check system resources:**
   ```bash
   # CPU and memory usage
   top
   
   # Disk space
   df -h
   
   # Network connectivity
   ping google.com
   ```

2. **Monitor API response times:**
   ```bash
   # Test with timing
   time curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
     -H "Authorization: Bearer $AUTH_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"documents": "url", "questions": ["test"]}'
   ```

3. **Check logs for bottlenecks:**
   ```bash
   # View application logs
   tail -f logs/app.log
   
   # Look for slow operations
   grep "processing time" logs/app.log
   ```

**Optimization Solutions:**

1. **Reduce document size:**
   - Use smaller documents for testing
   - Split large documents into sections

2. **Optimize questions:**
   - Reduce number of questions per request
   - Make questions more specific

3. **Adjust timeouts:**
   ```bash
   # In .env file
   REQUEST_TIMEOUT=60
   LLM_TIMEOUT=120
   ```

4. **Scale resources:**
   - Increase server memory
   - Use faster storage (SSD)
   - Improve network bandwidth

### Memory Issues

**Symptoms:**
- Out of memory errors
- System becoming unresponsive
- Process killed by OS

**Solutions:**

1. **Monitor memory usage:**
   ```bash
   # Check memory usage
   free -h
   
   # Monitor specific process
   ps aux | grep python
   ```

2. **Reduce memory consumption:**
   ```bash
   # Reduce chunk size
   MAX_CHUNK_SIZE=500
   
   # Reduce concurrent requests
   MAX_CONCURRENT_REQUESTS=10
   ```

3. **Optimize processing:**
   - Process documents sequentially
   - Clear caches regularly
   - Restart service periodically

## Configuration Problems

### Environment Variable Issues

**Common Problems:**
- Missing required variables
- Incorrect variable names
- Invalid values

**Diagnostic Commands:**

```bash
# Check all environment variables
env | grep -E "(AUTH_TOKEN|GEMINI|JINA|PINECONE|DATABASE)"

# Validate configuration
curl -H "Authorization: Bearer $AUTH_TOKEN" \
     "http://localhost:8000/config/validate"

# Test database connection
python scripts/test_database.py

# Validate all settings
python scripts/validate_config.py
```

**Common Fixes:**

1. **Load environment file:**
   ```bash
   # Ensure .env file is loaded
   source .env
   
   # Or use dotenv in Python
   from dotenv import load_dotenv
   load_dotenv()
   ```

2. **Check variable format:**
   ```bash
   # Correct format in .env
   AUTH_TOKEN=your-token-here
   GEMINI_API_KEY=your-key-here
   DATABASE_URL=postgresql://user:pass@host:port/db
   ```

3. **Verify permissions:**
   ```bash
   # Check .env file permissions
   ls -la .env
   
   # Should be readable by application user
   chmod 600 .env
   ```

### Database Connection Issues

**Error: "Database connection failed"**

**Troubleshooting:**

1. **Test database connectivity:**
   ```bash
   # Test PostgreSQL connection
   psql $DATABASE_URL -c "SELECT 1;"
   
   # Or using Python
   python -c "
   import psycopg2
   conn = psycopg2.connect('$DATABASE_URL')
   print('Connection successful')
   conn.close()
   "
   ```

2. **Check database server:**
   ```bash
   # Check if PostgreSQL is running
   systemctl status postgresql
   
   # Check port availability
   netstat -an | grep 5432
   ```

3. **Verify credentials:**
   ```bash
   # Parse DATABASE_URL
   echo $DATABASE_URL | sed 's/.*:\/\/\([^:]*\):\([^@]*\)@\([^:]*\):\([^/]*\)\/\(.*\)/User: \1\nPassword: \2\nHost: \3\nPort: \4\nDatabase: \5/'
   ```

## Network and Connectivity Issues

### DNS Resolution Problems

**Symptoms:**
- "Name or service not known" errors
- Intermittent connection failures

**Solutions:**

1. **Test DNS resolution:**
   ```bash
   # Test domain resolution
   nslookup example.com
   dig example.com
   
   # Test specific services
   nslookup generativelanguage.googleapis.com
   nslookup api.jina.ai
   ```

2. **Use alternative DNS:**
   ```bash
   # Temporarily use Google DNS
   echo "nameserver 8.8.8.8" > /etc/resolv.conf
   ```

### Firewall and Proxy Issues

**Symptoms:**
- Connection timeouts
- "Connection refused" errors

**Solutions:**

1. **Check firewall rules:**
   ```bash
   # Check iptables
   iptables -L
   
   # Check ufw status
   ufw status
   ```

2. **Test port connectivity:**
   ```bash
   # Test outbound HTTPS
   telnet google.com 443
   
   # Test specific API endpoints
   curl -I https://api.jina.ai
   curl -I https://generativelanguage.googleapis.com
   ```

3. **Configure proxy if needed:**
   ```bash
   # Set proxy environment variables
   export HTTP_PROXY=http://proxy.company.com:8080
   export HTTPS_PROXY=http://proxy.company.com:8080
   ```

## Debugging Tools

### Enable Debug Logging

```bash
# In .env file
DEBUG=true
LOG_LEVEL=DEBUG
LOG_FORMAT=text

# Restart application to apply changes
```

### Health Check Endpoints

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl -H "Authorization: Bearer $AUTH_TOKEN" \
     http://localhost:8000/health/detailed

# Configuration validation
curl -H "Authorization: Bearer $AUTH_TOKEN" \
     http://localhost:8000/config/validate
```

### Log Analysis

```bash
# View recent logs
tail -f logs/app.log

# Search for errors
grep -i error logs/app.log

# Search for specific operations
grep "processing query" logs/app.log

# Monitor response times
grep "processing time" logs/app.log | tail -20
```

### Testing Scripts

```bash
# Test database connection
python scripts/test_database.py

# Validate configuration
python scripts/validate_config.py

# Run comprehensive tests
python scripts/run_tests.py
```

## Getting Help

### Before Contacting Support

1. **Gather system information:**
   ```bash
   # System info
   uname -a
   python --version
   pip list | grep -E "(fastapi|pydantic|requests)"
   
   # Configuration (without sensitive data)
   curl -H "Authorization: Bearer $AUTH_TOKEN" \
        http://localhost:8000/config/validate
   ```

2. **Collect relevant logs:**
   ```bash
   # Recent application logs
   tail -100 logs/app.log > debug_logs.txt
   
   # Error logs only
   grep -i error logs/app.log > error_logs.txt
   ```

3. **Document the issue:**
   - Exact error message
   - Steps to reproduce
   - Expected vs actual behavior
   - System configuration
   - Recent changes

### Support Channels

1. **Check documentation:**
   - API documentation: `/docs`
   - Configuration guide: `docs/configuration.md`
   - Deployment guide: `docs/deployment.md`

2. **Review logs:**
   - Application logs in `logs/` directory
   - System logs: `/var/log/`

3. **Test with minimal example:**
   ```bash
   # Simple test request
   curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
     -H "Authorization: Bearer $AUTH_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
       "questions": ["What is this document about?"]
     }'
   ```

### Emergency Procedures

**If system is completely unresponsive:**

1. **Restart the service:**
   ```bash
   # Stop the service
   pkill -f "python.*main.py"
   
   # Start fresh
   python main.py
   ```

2. **Check system resources:**
   ```bash
   # Free up memory
   sync && echo 3 > /proc/sys/vm/drop_caches
   
   # Check disk space
   df -h
   ```

3. **Reset to known good state:**
   ```bash
   # Restore from backup configuration
   cp .env.backup .env
   
   # Clear temporary files
   rm -rf logs/*.log
   rm -rf __pycache__/
   ```

This troubleshooting guide should help you resolve most common issues. For persistent problems, ensure you have collected the diagnostic information mentioned above before seeking additional support.