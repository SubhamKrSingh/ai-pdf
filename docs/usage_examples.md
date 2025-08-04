# Usage Examples

This document provides comprehensive examples of how to use the LLM Query Retrieval System API with various tools and programming languages.

## Table of Contents

1. [cURL Examples](#curl-examples)
2. [Python Client Examples](#python-client-examples)
3. [JavaScript/Node.js Examples](#javascriptnodejs-examples)
4. [Common Use Cases](#common-use-cases)
5. [Error Handling Examples](#error-handling-examples)

## cURL Examples

### Basic Query Processing

```bash
# Basic document analysis
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Authorization: Bearer your-auth-token-here" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/sample-document.pdf",
    "questions": [
      "What is the main topic of this document?",
      "What are the key findings?"
    ]
  }'
```

### Multiple Questions Analysis

```bash
# Comprehensive document analysis with multiple questions
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Authorization: Bearer your-auth-token-here" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/research-paper.pdf",
    "questions": [
      "What is the research methodology used in this study?",
      "What are the main findings and results?",
      "What are the limitations mentioned?",
      "What future research directions are suggested?",
      "Who are the target participants or subjects?"
    ]
  }'
```

### Health Check

```bash
# Basic health check
curl -X GET "http://localhost:8000/health"

# Detailed health check
curl -X GET "http://localhost:8000/health/detailed"
```

### Configuration Validation

```bash
# Validate system configuration (requires authentication)
curl -X GET "http://localhost:8000/config/validate" \
  -H "Authorization: Bearer your-auth-token-here"
```

## Python Client Examples

### Basic Python Client

```python
import requests
import json
from typing import List, Dict, Any

class LLMQueryClient:
    """Python client for the LLM Query Retrieval System."""
    
    def __init__(self, base_url: str, auth_token: str):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the API (e.g., "http://localhost:8000")
            auth_token: Bearer token for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.headers = {
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json'
        }
    
    def process_query(self, document_url: str, questions: List[str]) -> Dict[str, Any]:
        """
        Process a document and answer questions.
        
        Args:
            document_url: URL to the document to process
            questions: List of questions to answer
            
        Returns:
            Dictionary containing the answers
            
        Raises:
            requests.RequestException: If the API request fails
        """
        payload = {
            'documents': document_url,
            'questions': questions
        }
        
        response = requests.post(
            f'{self.base_url}/api/v1/hackrx/run',
            headers=self.headers,
            json=payload,
            timeout=120  # 2 minutes timeout for processing
        )
        
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict[str, Any]:
        """Get basic health status."""
        response = requests.get(f'{self.base_url}/health')
        response.raise_for_status()
        return response.json()
    
    def detailed_health_check(self) -> Dict[str, Any]:
        """Get detailed health status."""
        response = requests.get(f'{self.base_url}/health/detailed')
        response.raise_for_status()
        return response.json()
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate system configuration."""
        response = requests.get(
            f'{self.base_url}/config/validate',
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

# Usage example
if __name__ == "__main__":
    # Initialize client
    client = LLMQueryClient(
        base_url="http://localhost:8000",
        auth_token="your-auth-token-here"
    )
    
    try:
        # Process a document
        result = client.process_query(
            document_url="https://example.com/sample-document.pdf",
            questions=[
                "What is the main topic of this document?",
                "What are the key findings mentioned?",
                "What recommendations are provided?"
            ]
        )
        
        # Print results
        print("Query Results:")
        for i, answer in enumerate(result['answers'], 1):
            print(f"{i}. {answer}")
            
    except requests.RequestException as e:
        print(f"Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
```

### Advanced Python Client with Error Handling

```python
import requests
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

@dataclass
class QueryResult:
    """Structured result from query processing."""
    answers: List[str]
    processing_time: Optional[float] = None
    document_url: Optional[str] = None

@dataclass
class APIError:
    """Structured API error information."""
    error: str
    error_code: str
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None

class LLMQueryClientAdvanced:
    """Advanced Python client with comprehensive error handling and retry logic."""
    
    def __init__(self, base_url: str, auth_token: str, max_retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json'
        })
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with retry logic."""
        url = f'{self.base_url}{endpoint}'
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(method, url, **kwargs)
                
                if response.status_code < 500:
                    return response
                    
                # Server error - retry
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(f"Server error {response.status_code}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    response.raise_for_status()
                    
            except requests.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Request failed: {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
        
        return response
    
    def process_query(self, document_url: str, questions: List[str]) -> QueryResult:
        """
        Process a document and answer questions with comprehensive error handling.
        
        Args:
            document_url: URL to the document to process
            questions: List of questions to answer
            
        Returns:
            QueryResult object with answers and metadata
            
        Raises:
            ValueError: If input validation fails
            requests.RequestException: If the API request fails
        """
        # Input validation
        if not document_url or not document_url.startswith(('http://', 'https://')):
            raise ValueError("Invalid document URL")
        
        if not questions or len(questions) == 0:
            raise ValueError("At least one question is required")
        
        if len(questions) > 50:
            raise ValueError("Maximum 50 questions allowed")
        
        for i, question in enumerate(questions):
            if not question.strip():
                raise ValueError(f"Question {i+1} cannot be empty")
            if len(question.strip()) < 3:
                raise ValueError(f"Question {i+1} is too short (minimum 3 characters)")
            if len(question.strip()) > 1000:
                raise ValueError(f"Question {i+1} is too long (maximum 1000 characters)")
        
        payload = {
            'documents': document_url,
            'questions': questions
        }
        
        start_time = time.time()
        
        try:
            response = self._make_request(
                'POST',
                '/api/v1/hackrx/run',
                json=payload,
                timeout=120
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                return QueryResult(
                    answers=data['answers'],
                    processing_time=processing_time,
                    document_url=document_url
                )
            else:
                # Handle error response
                try:
                    error_data = response.json()
                    error = APIError(**error_data)
                    raise requests.RequestException(f"API Error: {error.error} ({error.error_code})")
                except json.JSONDecodeError:
                    response.raise_for_status()
                    
        except requests.RequestException as e:
            self.logger.error(f"Query processing failed: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Get system health status."""
        response = self._make_request('GET', '/health', timeout=10)
        response.raise_for_status()
        return response.json()
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate system configuration."""
        response = self._make_request('GET', '/config/validate', timeout=10)
        response.raise_for_status()
        return response.json()

# Usage example with error handling
if __name__ == "__main__":
    client = LLMQueryClientAdvanced(
        base_url="http://localhost:8000",
        auth_token="your-auth-token-here"
    )
    
    try:
        # Check system health first
        health = client.health_check()
        print(f"System status: {health['status']}")
        
        # Process document
        result = client.process_query(
            document_url="https://example.com/sample-document.pdf",
            questions=[
                "What is the main topic of this document?",
                "What are the key findings mentioned?",
                "What recommendations are provided?"
            ]
        )
        
        print(f"\nProcessing completed in {result.processing_time:.2f} seconds")
        print(f"Document: {result.document_url}")
        print("\nAnswers:")
        for i, answer in enumerate(result.answers, 1):
            print(f"{i}. {answer}")
            
    except ValueError as e:
        print(f"Input validation error: {e}")
    except requests.RequestException as e:
        print(f"API request failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
```

## JavaScript/Node.js Examples

### Basic Node.js Client

```javascript
const axios = require('axios');

class LLMQueryClient {
    constructor(baseUrl, authToken) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.client = axios.create({
            baseURL: this.baseUrl,
            headers: {
                'Authorization': `Bearer ${authToken}`,
                'Content-Type': 'application/json'
            },
            timeout: 120000 // 2 minutes
        });
    }

    async processQuery(documentUrl, questions) {
        try {
            const response = await this.client.post('/api/v1/hackrx/run', {
                documents: documentUrl,
                questions: questions
            });
            
            return response.data;
        } catch (error) {
            if (error.response) {
                throw new Error(`API Error: ${error.response.data.error || error.response.statusText}`);
            } else if (error.request) {
                throw new Error('Network error: No response received');
            } else {
                throw new Error(`Request error: ${error.message}`);
            }
        }
    }

    async healthCheck() {
        try {
            const response = await this.client.get('/health');
            return response.data;
        } catch (error) {
            throw new Error(`Health check failed: ${error.message}`);
        }
    }

    async validateConfig() {
        try {
            const response = await this.client.get('/config/validate');
            return response.data;
        } catch (error) {
            throw new Error(`Config validation failed: ${error.message}`);
        }
    }
}

// Usage example
async function main() {
    const client = new LLMQueryClient(
        'http://localhost:8000',
        'your-auth-token-here'
    );

    try {
        // Check health
        const health = await client.healthCheck();
        console.log('System status:', health.status);

        // Process document
        const result = await client.processQuery(
            'https://example.com/sample-document.pdf',
            [
                'What is the main topic of this document?',
                'What are the key findings mentioned?',
                'What recommendations are provided?'
            ]
        );

        console.log('\nAnswers:');
        result.answers.forEach((answer, index) => {
            console.log(`${index + 1}. ${answer}`);
        });

    } catch (error) {
        console.error('Error:', error.message);
    }
}

// Run the example
main();
```

## Common Use Cases

### 1. Research Paper Analysis

```python
# Analyze academic research papers
questions = [
    "What is the research question or hypothesis?",
    "What methodology was used in this study?",
    "What are the main findings and results?",
    "What are the limitations of this study?",
    "What future research directions are suggested?",
    "How large was the sample size?",
    "What statistical methods were used?"
]

result = client.process_query(
    "https://example.com/research-paper.pdf",
    questions
)
```

### 2. Legal Document Review

```python
# Analyze legal contracts or documents
questions = [
    "What are the key terms and conditions?",
    "What are the parties' obligations?",
    "What are the termination clauses?",
    "Are there any penalty or liability clauses?",
    "What is the governing law?",
    "What are the payment terms?",
    "Are there any confidentiality requirements?"
]

result = client.process_query(
    "https://example.com/contract.pdf",
    questions
)
```

### 3. Financial Report Analysis

```python
# Analyze financial reports or statements
questions = [
    "What is the company's revenue for this period?",
    "What are the major expenses or costs?",
    "What is the profit margin?",
    "Are there any significant risks mentioned?",
    "What are the future outlook and projections?",
    "What are the key performance indicators?",
    "Are there any regulatory compliance issues?"
]

result = client.process_query(
    "https://example.com/financial-report.pdf",
    questions
)
```

### 4. Technical Documentation Review

```python
# Analyze technical specifications or manuals
questions = [
    "What are the system requirements?",
    "What are the key features and capabilities?",
    "What are the installation or setup procedures?",
    "Are there any known limitations or constraints?",
    "What troubleshooting steps are provided?",
    "What are the security considerations?",
    "What maintenance procedures are required?"
]

result = client.process_query(
    "https://example.com/technical-manual.pdf",
    questions
)
```

## Error Handling Examples

### Handling Different Error Types

```python
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

def robust_query_processing(client, document_url, questions):
    """Example of comprehensive error handling."""
    
    try:
        result = client.process_query(document_url, questions)
        return result
        
    except ValueError as e:
        print(f"Input validation error: {e}")
        return None
        
    except Timeout:
        print("Request timed out. The document might be too large or the server is busy.")
        return None
        
    except ConnectionError:
        print("Connection error. Please check if the server is running.")
        return None
        
    except requests.HTTPError as e:
        if e.response.status_code == 401:
            print("Authentication failed. Please check your auth token.")
        elif e.response.status_code == 400:
            try:
                error_data = e.response.json()
                print(f"Bad request: {error_data.get('error', 'Unknown error')}")
            except:
                print("Bad request: Invalid input data")
        elif e.response.status_code == 422:
            try:
                error_data = e.response.json()
                print(f"Validation error: {error_data.get('error', 'Unknown validation error')}")
            except:
                print("Validation error: Request data is invalid")
        elif e.response.status_code == 500:
            print("Server error. Please try again later or contact support.")
        else:
            print(f"HTTP error {e.response.status_code}: {e.response.reason}")
        return None
        
    except RequestException as e:
        print(f"Request failed: {e}")
        return None
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Usage
result = robust_query_processing(
    client,
    "https://example.com/document.pdf",
    ["What is this document about?"]
)

if result:
    print("Success:", result.answers[0])
else:
    print("Failed to process the query")
```

### Batch Processing with Error Recovery

```python
def process_multiple_documents(client, documents_and_questions):
    """Process multiple documents with error recovery."""
    
    results = []
    failed_documents = []
    
    for doc_url, questions in documents_and_questions:
        try:
            print(f"Processing: {doc_url}")
            result = client.process_query(doc_url, questions)
            results.append({
                'document': doc_url,
                'success': True,
                'answers': result.answers
            })
            print(f"✓ Successfully processed {doc_url}")
            
        except Exception as e:
            print(f"✗ Failed to process {doc_url}: {e}")
            results.append({
                'document': doc_url,
                'success': False,
                'error': str(e)
            })
            failed_documents.append(doc_url)
    
    print(f"\nProcessing complete:")
    print(f"Successful: {len([r for r in results if r['success']])}")
    print(f"Failed: {len(failed_documents)}")
    
    if failed_documents:
        print(f"Failed documents: {failed_documents}")
    
    return results

# Usage example
documents_to_process = [
    ("https://example.com/doc1.pdf", ["What is the main topic?"]),
    ("https://example.com/doc2.pdf", ["What are the key findings?"]),
    ("https://example.com/doc3.pdf", ["What recommendations are made?"])
]

results = process_multiple_documents(client, documents_to_process)
```

## Performance Optimization Tips

1. **Batch Questions**: Include multiple questions in a single request to avoid repeated document processing.

2. **Reasonable Timeouts**: Set appropriate timeouts for large documents:
   ```python
   # For large documents, increase timeout
   response = requests.post(url, json=payload, timeout=300)  # 5 minutes
   ```

3. **Error Recovery**: Implement retry logic for transient failures:
   ```python
   import time
   
   def retry_request(func, max_retries=3, delay=1):
       for attempt in range(max_retries):
           try:
               return func()
           except requests.RequestException as e:
               if attempt < max_retries - 1:
                   time.sleep(delay * (2 ** attempt))  # Exponential backoff
               else:
                   raise
   ```

4. **Connection Pooling**: Use session objects for multiple requests:
   ```python
   session = requests.Session()
   session.headers.update({'Authorization': f'Bearer {token}'})
   # Reuse session for multiple requests
   ```

These examples provide a comprehensive guide for integrating with the LLM Query Retrieval System API across different programming languages and use cases.