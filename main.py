"""
Main FastAPI application entry point for the LLM Query Retrieval System.

This module implements the FastAPI application with middleware configuration,
error handling, and the main API endpoint according to requirements 1.1, 1.2, 1.4, 7.1, 7.2, 7.3, 8.1.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
import uvicorn

from app.config import get_settings, Settings, get_health_check_info
from app.models.schemas import QueryRequest, QueryResponse, ErrorResponse
from app.controllers.query_controller import QueryController
from app.auth import verify_token
from app.middleware.error_handler import setup_error_handling
from app.exceptions import BaseSystemError
# Initialize settings and logging
settings = get_settings()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting LLM Query Retrieval System...")
    try:
        settings = get_settings()
        logger.info(f"Server configured to run on {settings.host}:{settings.port}")
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM Query Retrieval System...")


# Create FastAPI application with comprehensive OpenAPI documentation
app = FastAPI(
    title="LLM Query Retrieval System",
    description="""
    ## Intelligent Document Processing and Query Answering System

    This API provides intelligent document analysis and natural language query answering capabilities using:
    - **Advanced Language Models** (Gemini 2.0 Flash) for contextual answer generation
    - **Vector Search Technology** (Pinecone) for semantic document retrieval
    - **Multi-format Document Support** (PDF, DOCX, Email, Text)
    - **Retrieval-Augmented Generation (RAG)** for accurate, explainable answers

    ### Key Features
    - 🔍 **Semantic Search**: Find relevant information using vector similarity
    - 🤖 **AI-Powered Answers**: Generate contextual responses with LLM technology
    - 📄 **Multi-Format Support**: Process PDF, DOCX, email, and text documents
    - ⚡ **High Performance**: Async processing with concurrent request handling
    - 🔒 **Secure**: Bearer token authentication and input validation
    - 📊 **Comprehensive Logging**: Detailed monitoring and error tracking

    ### Use Cases
    - **Research Analysis**: Extract insights from academic papers and reports
    - **Legal Document Review**: Analyze contracts, agreements, and legal texts
    - **Financial Report Analysis**: Process earnings reports and financial statements
    - **Technical Documentation**: Query manuals, specifications, and guides
    - **Compliance Review**: Analyze regulatory documents and policies

    ### Getting Started
    1. Obtain your authentication token
    2. Prepare your document URL (must be publicly accessible)
    3. Formulate your questions (up to 50 per request)
    4. Send a POST request to `/api/v1/hackrx/run`
    5. Receive structured JSON responses with answers

    ### Rate Limits
    - Maximum 100 concurrent requests
    - Maximum 50 questions per request
    - Maximum 50MB document size
    - Request timeout: 30 seconds (configurable)

    For detailed examples and integration guides, see the [Usage Examples](/docs/usage_examples.md).
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    contact={
        "name": "LLM Query Retrieval System Support",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    servers=[
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        },
        {
            "url": "https://api.example.com",
            "description": "Production server"
        }
    ],
    tags_metadata=[
        {
            "name": "Query Processing",
            "description": "Main document processing and query answering functionality",
        },
        {
            "name": "Health",
            "description": "System health monitoring and status endpoints",
        },
        {
            "name": "Configuration",
            "description": "System configuration validation and management",
        },
    ]
)

# Set up comprehensive error handling
error_middleware = setup_error_handling(app, enable_debug=settings.debug)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all incoming requests with timing information.
    """
    start_time = time.time()
    
    # Log request
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"from {request.client.host if request.client else 'unknown'}"
    )
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(
        f"Response: {response.status_code} "
        f"processed in {process_time:.3f}s"
    )
    
    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# Note: Error handling is now managed by the ErrorHandlerMiddleware
# which provides comprehensive error handling with structured responses


# Health check endpoints
@app.get(
    "/health", 
    tags=["Health"],
    summary="Basic Health Check",
    description="""
    **Basic health check endpoint for monitoring and load balancers.**

    This endpoint provides a simple health status check that can be used by:
    - Load balancers for health monitoring
    - Container orchestration systems (Docker, Kubernetes)
    - Monitoring tools and dashboards
    - Automated health checks in CI/CD pipelines

    ### Response
    Returns a simple JSON object indicating the service is running and responsive.

    ### Use Cases
    - **Load Balancer Health Checks**: Configure your load balancer to check this endpoint
    - **Container Health Checks**: Use in Docker HEALTHCHECK instructions
    - **Monitoring Alerts**: Set up alerts when this endpoint becomes unavailable
    - **Service Discovery**: Verify service availability before routing traffic

    ### Performance
    - **Response Time**: Typically < 10ms
    - **No Dependencies**: Does not check external services
    - **Lightweight**: Minimal resource usage
    """,
    responses={
        200: {
            "description": "Service is healthy and responsive",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "service": "LLM Query Retrieval System"
                    }
                }
            }
        }
    }
)
async def health_check():
    """Basic health check endpoint for monitoring."""
    return {"status": "healthy", "service": "LLM Query Retrieval System"}


@app.get(
    "/health/detailed", 
    tags=["Health"],
    summary="Detailed Health Check",
    description="""
    **Comprehensive health check with dependency status information.**

    This endpoint provides detailed health information about the system and its dependencies:

    ### Checked Components
    - **Database Connection**: PostgreSQL connectivity and query performance
    - **Vector Store**: Pinecone index availability and response time
    - **LLM Service**: Gemini API accessibility and quota status
    - **Embedding Service**: Jina API connectivity and rate limits
    - **System Resources**: Memory usage, disk space, and CPU load

    ### Response Information
    - **Overall Status**: Aggregated health status (healthy/degraded/unhealthy)
    - **Service Details**: Individual component status and metrics
    - **Performance Metrics**: Response times and resource utilization
    - **Version Information**: System version and environment details
    - **Timestamp**: When the health check was performed

    ### Status Levels
    - **Healthy**: All services operational and performing within normal parameters
    - **Degraded**: Some non-critical services experiencing issues
    - **Unhealthy**: Critical services unavailable or failing

    ### Use Cases
    - **Operational Monitoring**: Detailed system status for operations teams
    - **Troubleshooting**: Identify which components are experiencing issues
    - **Performance Monitoring**: Track response times and resource usage
    - **Capacity Planning**: Monitor resource utilization trends

    ### Performance Impact
    - **Response Time**: 100-500ms (depends on external service checks)
    - **Resource Usage**: Minimal, but checks external dependencies
    - **Caching**: Results may be cached for 30 seconds to reduce load
    """,
    responses={
        200: {
            "description": "Detailed health status information",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "timestamp": "2024-01-15T10:30:00Z",
                        "version": "1.0.0",
                        "environment": "production",
                        "services": {
                            "database": {
                                "status": "healthy",
                                "response_time_ms": 15,
                                "connection_pool": {
                                    "active": 5,
                                    "idle": 15,
                                    "total": 20
                                }
                            },
                            "vector_store": {
                                "status": "healthy",
                                "response_time_ms": 45,
                                "index_stats": {
                                    "total_vectors": 150000,
                                    "dimensions": 1024
                                }
                            },
                            "llm_service": {
                                "status": "healthy",
                                "response_time_ms": 120,
                                "quota_remaining": 85.5
                            },
                            "embedding_service": {
                                "status": "healthy",
                                "response_time_ms": 80,
                                "rate_limit_remaining": 950
                            }
                        }
                    }
                }
            }
        },
        503: {
            "description": "Service unhealthy - one or more critical components failing",
            "content": {
                "application/json": {
                    "example": {
                        "status": "unhealthy",
                        "timestamp": "2024-01-15T10:30:00Z",
                        "version": "1.0.0",
                        "environment": "production",
                        "services": {
                            "database": {
                                "status": "unhealthy",
                                "error": "Connection timeout after 5 seconds"
                            },
                            "vector_store": {
                                "status": "healthy",
                                "response_time_ms": 45
                            }
                        }
                    }
                }
            }
        }
    }
)
async def detailed_health_check():
    """Detailed health check endpoint with service status information."""
    return get_health_check_info(settings)


@app.get(
    "/config/validate", 
    tags=["Configuration"],
    summary="Validate System Configuration",
    description="""
    **Validate current system configuration and environment setup.**

    This endpoint performs comprehensive validation of all system configuration:

    ### Validation Checks
    - **Environment Variables**: Verify all required variables are set
    - **API Keys**: Validate format and accessibility of external service keys
    - **Database Configuration**: Check database connection parameters
    - **Service Endpoints**: Verify external service connectivity
    - **Security Settings**: Validate authentication and CORS configuration
    - **Performance Settings**: Check resource limits and timeouts

    ### Validation Categories
    - **Required Configuration**: Critical settings that must be present
    - **Optional Configuration**: Settings with defaults that can be customized
    - **Security Configuration**: Authentication, CORS, and security headers
    - **Performance Configuration**: Timeouts, limits, and resource settings

    ### Response Information
    - **Overall Status**: `valid` or `invalid`
    - **Errors**: List of configuration errors that must be fixed
    - **Warnings**: Non-critical issues that should be addressed
    - **Config Summary**: Current configuration values (sensitive data excluded)

    ### Use Cases
    - **Deployment Validation**: Verify configuration before going live
    - **Troubleshooting**: Identify configuration issues causing problems
    - **Security Audit**: Review security-related configuration settings
    - **Environment Setup**: Validate new environment configurations

    ### Security
    - **Authentication Required**: Must provide valid Bearer token
    - **Sensitive Data Protection**: API keys and passwords are not included in response
    - **Access Logging**: Configuration access is logged for security monitoring

    ### Example Scenarios

    **Valid Configuration:**
    All required environment variables are set and valid.

    **Invalid Configuration:**
    Missing required API keys or invalid database connection string.

    **Configuration Warnings:**
    Development settings enabled in production environment.
    """,
    responses={
        200: {
            "description": "Configuration validation results",
            "content": {
                "application/json": {
                    "examples": {
                        "valid_config": {
                            "summary": "Valid Configuration",
                            "value": {
                                "status": "valid",
                                "errors": [],
                                "warnings": [],
                                "config_summary": {
                                    "environment": "production",
                                    "debug": False,
                                    "host": "0.0.0.0",
                                    "port": 8000,
                                    "log_level": "INFO",
                                    "max_chunk_size": 1000,
                                    "max_document_size_mb": 50,
                                    "database_pool_size": 20,
                                    "max_concurrent_requests": 100
                                }
                            }
                        },
                        "invalid_config": {
                            "summary": "Invalid Configuration",
                            "value": {
                                "status": "invalid",
                                "errors": [
                                    "Missing required environment variable: GEMINI_API_KEY",
                                    "Invalid database URL format"
                                ],
                                "warnings": [
                                    "Debug mode enabled in production environment"
                                ],
                                "config_summary": {
                                    "environment": "production",
                                    "debug": True,
                                    "host": "0.0.0.0",
                                    "port": 8000
                                }
                            }
                        }
                    }
                }
            }
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Authentication failed",
                        "error_code": "AUTHENTICATION_ERROR",
                        "details": {
                            "message": "Invalid or missing Bearer token"
                        },
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                }
            }
        },
        500: {
            "description": "Configuration validation failed due to system error",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Configuration validation failed",
                        "error_code": "INTERNAL_ERROR",
                        "details": {
                            "message": "Unable to access configuration system"
                        },
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                }
            }
        }
    }
)
async def validate_config(_: bool = Depends(verify_token)):
    """Validate current configuration (requires authentication)."""
    from app.config import validate_environment
    try:
        result = validate_environment()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Main API endpoint
@app.post(
    "/api/v1/hackrx/run",
    response_model=QueryResponse,
    responses={
        200: {
            "description": "Successful query processing",
            "content": {
                "application/json": {
                    "example": {
                        "answers": [
                            "The main topic of this document is artificial intelligence applications in healthcare, specifically focusing on diagnostic accuracy improvements.",
                            "The key findings include a 25% improvement in diagnostic accuracy, 40% reduction in processing time, and 95% user satisfaction rate among healthcare professionals."
                        ]
                    }
                }
            }
        },
        400: {
            "model": ErrorResponse,
            "description": "Bad request - Invalid input data",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Invalid document URL provided",
                        "error_code": "VALIDATION_ERROR",
                        "details": {
                            "field": "documents",
                            "message": "URL must start with http:// or https://"
                        },
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                }
            }
        },
        401: {
            "model": ErrorResponse,
            "description": "Unauthorized - Invalid or missing authentication token",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Authentication failed",
                        "error_code": "AUTHENTICATION_ERROR",
                        "details": {
                            "message": "Invalid or missing Bearer token"
                        },
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                }
            }
        },
        422: {
            "model": ErrorResponse,
            "description": "Validation error - Request data validation failed",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Question validation failed",
                        "error_code": "VALIDATION_ERROR",
                        "details": {
                            "field": "questions",
                            "message": "Question at index 0 is too short (minimum 3 characters)"
                        },
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                }
            }
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal server error - Processing failed",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Document processing failed",
                        "error_code": "DOCUMENT_PARSE_ERROR",
                        "details": {
                            "stage": "parsing",
                            "message": "Failed to extract text from PDF document"
                        },
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                }
            }
        }
    },
    tags=["Query Processing"],
    summary="Process Document and Answer Questions",
    description="""
    **Main endpoint for document processing and question answering.**

    This endpoint implements the complete document analysis pipeline:

    ### Processing Pipeline
    1. **Document Download**: Retrieves document from provided URL
    2. **Content Parsing**: Extracts text based on document format (PDF, DOCX, Email, Text)
    3. **Text Chunking**: Splits content into semantic chunks for optimal processing
    4. **Embedding Generation**: Creates vector embeddings using Jina v4 model
    5. **Vector Storage**: Stores embeddings in Pinecone for semantic search
    6. **Query Processing**: For each question:
       - Generates query embedding
       - Performs semantic similarity search
       - Retrieves most relevant document chunks
       - Generates contextual answer using Gemini 2.0 Flash LLM

    ### Supported Document Formats
    - **PDF** (`.pdf`): Text-based PDFs (scanned images not supported)
    - **Microsoft Word** (`.docx`): Modern Word documents
    - **Email** (`.eml`): RFC822 email format
    - **Plain Text** (`.txt`): UTF-8 encoded text files

    ### Input Constraints
    - **Document URL**: Must be publicly accessible HTTP/HTTPS URL
    - **Questions**: 1-50 questions per request, 3-1000 characters each
    - **Document Size**: Maximum 50MB (configurable)
    - **Processing Time**: Typically 10-60 seconds depending on document size

    ### Response Format
    The response contains an `answers` array with responses corresponding exactly to the input `questions` array order.

    ### Example Use Cases

    **Research Paper Analysis:**
    ```json
    {
      "documents": "https://arxiv.org/pdf/2301.00001.pdf",
      "questions": [
        "What is the research methodology used?",
        "What are the main findings?",
        "What are the limitations of this study?"
      ]
    }
    ```

    **Legal Document Review:**
    ```json
    {
      "documents": "https://example.com/contract.pdf",
      "questions": [
        "What are the key terms and conditions?",
        "What are the termination clauses?",
        "What are the liability limitations?"
      ]
    }
    ```

    **Financial Report Analysis:**
    ```json
    {
      "documents": "https://example.com/earnings-report.pdf",
      "questions": [
        "What is the revenue growth rate?",
        "What are the major risk factors?",
        "What is the outlook for next quarter?"
      ]
    }
    ```

    ### Performance Tips
    - **Batch Questions**: Include multiple related questions in one request
    - **Specific Questions**: More specific questions yield better answers
    - **Document Quality**: Text-based documents work better than scanned images
    - **URL Accessibility**: Ensure document URLs are publicly accessible

    ### Error Handling
    The API provides detailed error responses with specific error codes for programmatic handling:
    - `VALIDATION_ERROR`: Input validation failed
    - `DOCUMENT_DOWNLOAD_ERROR`: Failed to download document
    - `DOCUMENT_PARSE_ERROR`: Failed to parse document content
    - `EMBEDDING_ERROR`: Failed to generate embeddings
    - `VECTOR_STORE_ERROR`: Vector database operation failed
    - `LLM_ERROR`: Language model API call failed
    """
)
async def process_query(
    request: QueryRequest,
    _: bool = Depends(verify_token)
) -> QueryResponse:
    """
    Process document and answer natural language questions.
    
    This endpoint implements the main functionality according to requirements:
    - 1.1: Accept JSON payload with documents URL and questions array
    - 1.2: Return JSON response with answers array
    - 7.1: Structured JSON format
    - 7.2: Answers correspond to input questions order
    
    Args:
        request: Query request containing document URL and questions
        
    Returns:
        QueryResponse: Answers corresponding to input questions
        
    Raises:
        HTTPException: For various error conditions
    """
    logger.info(f"Processing query with {len(request.questions)} questions for document: {request.documents}")
    
    try:
        # Initialize query controller
        controller = QueryController()
        
        # Process the request
        response = await controller.process_query_request(request)
        
        logger.info(f"Successfully processed query with {len(response.answers)} answers")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info"
    )