# Embedding Timeout Issue - Fix Applied

## Problem Analysis

Your system was failing with **embedding service timeouts** during document processing. Here's what was happening:

### Error Flow
1. Document processing starts → Text chunking succeeds
2. Embedding service processes chunks in batches of 10
3. Jina AI API takes longer than 30 seconds to respond
4. Request times out → Retry attempts also timeout
5. Entire document processing fails with 500 error

### Root Causes
- **30-second timeout too short** for embedding operations
- **Large batch sizes** (10 chunks) overwhelming the API
- **No graceful degradation** when some batches fail
- **All-or-nothing approach** - one failure kills entire process

## Fixes Applied

### 1. Extended Timeouts (`app/services/embedding_service.py`)
```python
# Before: 30 seconds
timeout=httpx.Timeout(self.settings.request_timeout)

# After: Minimum 60 seconds with granular control
embedding_timeout = max(self.settings.request_timeout * 2, 60)
timeout=httpx.Timeout(
    connect=10.0,  # Connection timeout
    read=embedding_timeout,  # Read timeout (main bottleneck) 
    write=10.0,  # Write timeout
    pool=5.0  # Pool timeout
)
```

### 2. Reduced Batch Sizes
```python
# Before: 10 chunks per batch
batch_size: int = 10

# After: 5 chunks per batch (more reliable)
batch_size: int = 5
```

### 3. Enhanced Retry Configuration (`app/utils/retry.py`)
```python
# Before: 2 attempts, 10s max delay
EMBEDDING_RETRY_CONFIG = RetryConfig(max_attempts=2, initial_delay=1.0, max_delay=10.0)

# After: 3 attempts, 30s max delay
EMBEDDING_RETRY_CONFIG = RetryConfig(max_attempts=3, initial_delay=2.0, max_delay=30.0)
```

### 4. Resilient Batch Processing (`app/services/document_service.py`)
- **Graceful failure handling**: Failed batches don't kill entire process
- **Individual chunk retry**: Failed batches retry with single chunks
- **Partial success support**: Process continues with successful embeddings
- **Better error reporting**: Shows success/failure statistics

### 5. Configuration Updates
- **Default timeout**: Increased from 30s to 60s
- **Environment variables**: Updated in `.env` and `.env.example`

## Expected Behavior Now

### Success Scenarios
1. **Normal processing**: Faster completion with smaller batches
2. **Partial timeouts**: System continues with successful chunks
3. **Individual recovery**: Failed batches retry chunk-by-chunk

### Error Handling
- **Graceful degradation**: Partial failures don't stop processing
- **Better logging**: Clear indication of success/failure rates
- **Meaningful errors**: Specific timeout and batch information

## Testing the Fix

### 1. Quick Test
```bash
# Test with your original request
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?...",
    "questions": ["What is the grace period for premium payment?"]
  }'
```

### 2. Monitor Logs
Look for these improved log messages:
```
INFO - Generated embeddings for 45 out of 50 chunks
WARNING - Failed to generate embeddings for 5 chunks, continuing with 45 valid chunks
DEBUG - Recovered chunk 23 individually
```

### 3. Expected Response Time
- **Before**: Timeout after 30s → 500 error
- **After**: Complete processing in 60-120s → 200 success

## Monitoring

### Key Metrics to Watch
1. **Success rate**: Should be >90% even with some timeouts
2. **Processing time**: 60-120s for large documents
3. **Partial failures**: Logged but don't stop processing
4. **Retry effectiveness**: Individual chunk recovery working

### Log Patterns
```bash
# Good signs
grep "Generated embeddings for.*out of" logs/
grep "Recovered chunk.*individually" logs/

# Issues to watch
grep "No valid embeddings generated" logs/
grep "Failed to process chunk.*individually" logs/
```

## Rollback Plan (if needed)

If issues persist, revert these changes:
```bash
# Restore original timeouts
sed -i 's/REQUEST_TIMEOUT=60/REQUEST_TIMEOUT=30/' .env
sed -i 's/batch_size: int = 5/batch_size: int = 10/' app/services/embedding_service.py
```

## Next Steps

1. **Deploy the fixes** and test with your original request
2. **Monitor success rates** - should be much higher now
3. **Adjust timeouts** if needed based on your API performance
4. **Consider caching** for frequently processed documents

The system is now much more resilient to embedding API timeouts and should handle your document processing successfully!