# LLM Query Retrieval System

An intelligent document processing and query answering system that uses advanced language models and vector search technology to provide contextual answers from document content.

## 🚀 Features

- **🔍 Multi-format Document Support**: Process PDF, DOCX, email, and text documents
- **🤖 Intelligent Query Processing**: Answer natural language questions with contextual accuracy
- **⚡ Vector Search Technology**: Semantic search using Pinecone vector database
- **🧠 Advanced Language Models**: Powered by Gemini 2.0 Flash for answer generation
- **🚄 High Performance**: Async processing with concurrent request handling
- **🛡️ Comprehensive Error Handling**: Detailed error responses and logging
- **🔒 Secure Authentication**: Bearer token-based API security
- **📚 RESTful API**: Clean, well-documented REST endpoints with OpenAPI/Swagger
- **📊 Monitoring & Observability**: Health checks, metrics, and structured logging
- **🐳 Container Ready**: Docker and Docker Compose support

## 🎯 Use Cases

- **📄 Research Analysis**: Extract insights from academic papers and reports
- **⚖️ Legal Document Review**: Analyze contracts, agreements, and legal texts
- **💰 Financial Report Analysis**: Process earnings reports and financial statements
- **🔧 Technical Documentation**: Query manuals, specifications, and guides
- **📋 Compliance Review**: Analyze regulatory documents and policies
- **🏥 Healthcare Documentation**: Process medical reports and clinical studies

## 🏃‍♂️ Quick Start

### Prerequisites

- Python 3.9+
- PostgreSQL 12+
- API keys for:
  - Google Gemini API
  - Jina Embeddings API
  - Pinecone Vector Database

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd llm-query-retrieval-system
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Set up database:**
   ```bash
   # Create PostgreSQL database
   createdb llm_query_system
   
   # Run migrations
   python -c "from app.data.migrations import run_migrations; run_migrations()"
   ```

6. **Start the server:**
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

## 📖 API Usage

### Basic Example

```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Authorization: Bearer your-auth-token" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": [
      "What is the main topic of this document?",
      "What are the key findings mentioned?",
      "What recommendations are provided?"
    ]
  }'
```

### Response

```json
{
  "answers": [
    "The main topic of this document is artificial intelligence applications in healthcare, specifically focusing on diagnostic accuracy improvements.",
    "The key findings include a 25% improvement in diagnostic accuracy, 40% reduction in processing time, and 95% user satisfaction rate among healthcare professionals.",
    "The document recommends implementing AI-assisted diagnostics in clinical workflows, providing additional training for healthcare staff, and establishing quality assurance protocols."
  ]
}
```

### Python Client Example

```python
import requests

# Initialize client
client_config = {
    'base_url': 'http://localhost:8000',
    'auth_token': 'your-auth-token-here'
}

# Make request
response = requests.post(
    f"{client_config['base_url']}/api/v1/hackrx/run",
    headers={
        'Authorization': f"Bearer {client_config['auth_token']}",
        'Content-Type': 'application/json'
    },
    json={
        'documents': 'https://example.com/research-paper.pdf',
        'questions': [
            'What is the research methodology used?',
            'What are the main conclusions?',
            'What future work is suggested?'
        ]
    }
)

if response.status_code == 200:
    data = response.json()
    for i, answer in enumerate(data['answers'], 1):
        print(f"Q{i}: {answer}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

## ⚙️ Configuration

### Required Environment Variables

```bash
# Authentication
AUTH_TOKEN=your-bearer-token-here

# LLM Configuration
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.0-flash

# Embedding Configuration
JINA_API_KEY=your-jina-api-key
JINA_MODEL=jina-embeddings-v4

# Vector Database
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=your-pinecone-environment
PINECONE_INDEX_NAME=document-embeddings

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
```

### Optional Configuration

```bash
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false
ENVIRONMENT=production

# Processing Configuration
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_DOCUMENT_SIZE_MB=50
MAX_CONCURRENT_REQUESTS=100

# Timeouts
REQUEST_TIMEOUT=30
LLM_TIMEOUT=60

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/var/log/llm-query-system/app.log
```

## 🏗️ Architecture

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

### Key Components

- **API Layer**: FastAPI with authentication, validation, and error handling
- **Service Layer**: Business logic for document processing and query handling
- **Data Layer**: Vector database (Pinecone) and relational database (PostgreSQL)
- **External Services**: LLM (Gemini), embedding model (Jina), and document parsers

## 🧪 Testing

### Run Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# End-to-end tests
pytest tests/e2e/ -v

# All tests with coverage
pytest --cov=app --cov-report=html tests/

# Performance tests
pytest tests/performance/ -v
```

### Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions and external services
- **End-to-End Tests**: Test complete workflows from API to response
- **Performance Tests**: Load testing and performance benchmarks

## 🚀 Deployment

### Docker

```bash
# Build image
docker build -t llm-query-system .

# Run container
docker run -p 8000:8000 --env-file .env llm-query-system
```

### Docker Compose

```bash
# Start all services (app + PostgreSQL)
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

### Production Deployment

```bash
# Using Gunicorn with multiple workers
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# With environment file
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --env-file .env
```

## 📚 Documentation

### API Documentation
- **Interactive Swagger UI**: Available at `/docs`
- **ReDoc Documentation**: Available at `/redoc`
- **OpenAPI Specification**: Available at `/openapi.json`

### Comprehensive Guides
- **[API Documentation](docs/api_documentation.md)**: Complete API reference with examples
- **[Usage Examples](docs/usage_examples.md)**: cURL, Python, and JavaScript examples
- **[Configuration Guide](docs/configuration.md)**: Environment setup and configuration options
- **[Deployment Guide](docs/deployment.md)**: Production deployment instructions
- **[Troubleshooting Guide](docs/troubleshooting_guide.md)**: Common issues and solutions
- **[Developer Guide](docs/developer_guide.md)**: Extending and customizing the system
- **[Error Handling Guide](docs/error_handling_guide.md)**: Error codes and handling strategies

## 📊 Monitoring & Observability

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check with service status
curl -H "Authorization: Bearer $AUTH_TOKEN" \
     http://localhost:8000/health/detailed

# Configuration validation
curl -H "Authorization: Bearer $AUTH_TOKEN" \
     http://localhost:8000/config/validate
```

### Logging

The system provides comprehensive logging:

- **Request/Response Logging**: All API requests with timing information
- **Error Tracking**: Detailed error logs with stack traces
- **Performance Metrics**: Processing times and resource usage
- **Structured JSON Logging**: Machine-readable logs for production
- **Security Logging**: Authentication attempts and access patterns

### Metrics

- Response times and throughput
- Error rates by type and endpoint
- Resource utilization (CPU, memory, disk)
- External service response times
- Document processing statistics

## ⚡ Performance

### Specifications
- **Concurrent Requests**: Up to 100 concurrent requests (configurable)
- **Document Size**: Maximum 50MB per document (configurable)
- **Questions**: Up to 50 questions per request
- **Response Time**: Typically 10-60 seconds depending on document size and complexity
- **Throughput**: 10-50 requests per minute depending on document complexity

### Optimization Tips
- **Batch Questions**: Include multiple related questions in one request
- **Document Quality**: Text-based documents work better than scanned images
- **Specific Questions**: More specific questions yield better and faster answers
- **Caching**: Frequently accessed documents are cached for faster processing

## 🔒 Security

### Security Features
- **Bearer Token Authentication**: Secure API access control
- **Input Validation**: Comprehensive request validation and sanitization
- **CORS Configuration**: Configurable cross-origin resource sharing
- **Rate Limiting**: Configurable request rate limits
- **Secure Error Handling**: No sensitive data exposed in error responses
- **Environment-based Configuration**: Sensitive data managed through environment variables

### Security Best Practices
- Use HTTPS in production
- Rotate API tokens regularly
- Monitor authentication logs
- Configure appropriate CORS origins
- Use strong database passwords
- Keep dependencies updated

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** with proper tests and documentation
4. **Run the test suite**: `pytest tests/`
5. **Commit your changes**: `git commit -m 'Add amazing feature'`
6. **Push to the branch**: `git push origin feature/amazing-feature`
7. **Submit a pull request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add tests for new functionality
- Update documentation for API changes
- Use type hints for better code clarity
- Add docstrings for public methods

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help

1. **Check the Documentation**: Start with our comprehensive guides
2. **Review Common Issues**: See the [troubleshooting guide](docs/troubleshooting_guide.md)
3. **Search Existing Issues**: Check if your issue has been reported
4. **Create a New Issue**: Provide detailed information about your problem

### Support Channels

- **Documentation**: Comprehensive guides and API reference
- **GitHub Issues**: Bug reports and feature requests
- **Health Checks**: Built-in system diagnostics
- **Logging**: Detailed error and performance logs

### Issue Reporting

When reporting issues, please include:
- System information (OS, Python version)
- Configuration details (without sensitive data)
- Error messages and logs
- Steps to reproduce the issue
- Expected vs actual behavior

## 📈 Changelog

### Version 1.0.0 (Current)

#### ✨ Features
- Multi-format document processing (PDF, DOCX, Email, Text)
- Vector search integration with Pinecone
- LLM-powered answer generation with Gemini 2.0 Flash
- Comprehensive API documentation with OpenAPI/Swagger
- Docker deployment support with Docker Compose
- Comprehensive error handling and logging
- Health monitoring and configuration validation
- Performance optimization with async processing
- Security features with Bearer token authentication

#### 🔧 Technical Improvements
- Async/await throughout the application
- Connection pooling for external services
- Structured logging with JSON format
- Comprehensive test suite with 90%+ coverage
- Type hints and documentation for all public APIs
- Modular architecture for easy extension

#### 📚 Documentation
- Complete API documentation with examples
- Usage guides for multiple programming languages
- Deployment and configuration guides
- Troubleshooting and developer guides
- Performance optimization recommendations

---

**Built with ❤️ using FastAPI, Pinecone, and Gemini AI**