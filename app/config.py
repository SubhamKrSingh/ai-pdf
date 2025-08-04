"""
Comprehensive environment configuration management with validation for all required API keys and settings.
Implements requirements 9.1, 9.2, 9.3, 9.4 with enhanced deployment and logging configuration.
"""

from typing import Optional, Dict, Any, Literal
from pydantic import Field, field_validator, model_validator, ConfigDict
from pydantic_settings import BaseSettings
import os
import logging
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()
print("AUTH_TOKEN:", os.getenv("AUTH_TOKEN"))


class Settings(BaseSettings):
    """
    Comprehensive application settings with validation for all required API keys and configuration.
    Enhanced for production deployment with logging, security, and monitoring configuration.
    """
    
    # Environment Configuration
    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Application environment"
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # API Configuration
    auth_token: str = Field(..., description="Bearer token for API authentication")
    host: str = Field(default="0.0.0.0", description="Host to bind the server")
    port: int = Field(default=8000, description="Port to bind the server")
    
    # LLM Configuration
    gemini_api_key: str = Field(..., description="Google Gemini API key")
    gemini_model: str = Field(default="gemini-2.0-flash", description="Gemini model to use")
    
    # Embedding Configuration
    jina_api_key: str = Field(..., description="Jina embeddings API key")
    jina_model: str = Field(default="jina-embeddings-v4", description="Jina embedding model")
    
    # Vector Database Configuration
    pinecone_api_key: str = Field(..., description="Pinecone API key")
    pinecone_environment: str = Field(..., description="Pinecone environment")
    pinecone_index_name: str = Field(default="document-embeddings", description="Pinecone index name")
    
    # PostgreSQL Database Configuration
    database_url: str = Field(..., description="PostgreSQL database connection URL")
    database_pool_size: int = Field(default=10, description="Database connection pool size")
    database_max_overflow: int = Field(default=20, description="Database connection pool max overflow")
    
    # Document Processing Configuration
    max_chunk_size: int = Field(default=1000, description="Maximum size of text chunks")
    chunk_overlap: int = Field(default=200, description="Overlap between text chunks")
    max_document_size_mb: int = Field(default=50, description="Maximum document size in MB")
    
    # Request timeout settings
    request_timeout: int = Field(default=60, description="HTTP request timeout in seconds")  # Increased for embedding operations
    llm_timeout: int = Field(default=60, description="LLM API timeout in seconds")
    
    # Retry configuration
    max_retries: int = Field(default=3, description="Maximum number of retries for API calls")
    retry_delay: float = Field(default=1.0, description="Initial delay between retries in seconds")
    
    # Performance Configuration
    max_concurrent_requests: int = Field(
        default=100, description="Maximum concurrent requests"
    )
    worker_processes: int = Field(
        default=1, description="Number of worker processes for production"
    )
    
    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    log_format: Literal["json", "text"] = Field(
        default="json", description="Log format (json for production, text for development)"
    )
    log_file: Optional[str] = Field(
        default=None, description="Log file path (optional, logs to stdout if not set)"
    )
    enable_access_logs: bool = Field(
        default=True, description="Enable HTTP access logging"
    )
    enable_sql_logs: bool = Field(
        default=False, description="Enable SQL query logging (debug only)"
    )
    
    # Security Configuration
    allowed_hosts: list[str] = Field(
        default=["*"], description="Allowed hosts for TrustedHostMiddleware"
    )
    cors_origins: list[str] = Field(
        default=["*"], description="Allowed CORS origins"
    )
    cors_allow_credentials: bool = Field(
        default=True, description="Allow credentials in CORS requests"
    )
    enable_https_redirect: bool = Field(
        default=False, description="Enable HTTPS redirect middleware"
    )
    
    # Health Check Configuration
    enable_detailed_health: bool = Field(
        default=False, description="Enable detailed health check with dependency status"
    )
    health_check_timeout: int = Field(
        default=5, description="Timeout for health check operations in seconds"
    )
    
    
    # Caching Configuration
    document_cache_ttl_hours: int = Field(default=24, description="Document cache TTL in hours")
    embedding_cache_ttl_hours: int = Field(default=168, description="Embedding cache TTL in hours")
    
    # HTTP Client Optimization
    max_http_connections: int = Field(default=50, description="Maximum HTTP connections")
    max_keepalive_connections: int = Field(default=25, description="Maximum keepalive connections")
    keepalive_expiry: int = Field(default=30, description="Keepalive expiry in seconds")
    
    # Vector Search Optimization
    default_top_k: int = Field(default=5, description="Default top-k for vector search")
    min_similarity_threshold: float = Field(default=0.4, description="Minimum similarity threshold")
    max_context_chunks: int = Field(default=3, description="Maximum context chunks for LLM")
    
    # Optional Redis Configuration
    redis_url: Optional[str] = Field(default=None, description="Redis connection URL")
    enable_redis_cache: bool = Field(default=False, description="Enable Redis caching")

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
        
    @field_validator("port")
    @classmethod
    def validate_port(cls, v):
        """Validate port number is within valid range."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @field_validator("max_chunk_size")
    @classmethod
    def validate_chunk_size(cls, v):
        """Validate chunk size is reasonable."""
        if v < 100 or v > 10000:
            raise ValueError("Chunk size must be between 100 and 10000 characters")
        return v
    
    @field_validator("max_document_size_mb")
    @classmethod
    def validate_document_size(cls, v):
        """Validate document size limit is reasonable."""
        if v < 1 or v > 1000:
            raise ValueError("Document size limit must be between 1 and 1000 MB")
        return v
    
    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v):
        """Validate database URL format."""
        if not v.startswith(("postgresql://", "postgres://")):
            raise ValueError("Database URL must start with postgresql:// or postgres://")
        return v
    
    @field_validator("auth_token", "gemini_api_key", "jina_api_key", "pinecone_api_key")
    @classmethod
    def validate_api_keys(cls, v):
        """Validate API keys are not empty."""
        if not v or len(v.strip()) == 0:
            raise ValueError("API key cannot be empty")
        return v.strip()
    
    @field_validator("database_pool_size", "database_max_overflow")
    @classmethod
    def validate_pool_settings(cls, v):
        """Validate database pool settings."""
        if v < 1 or v > 100:
            raise ValueError("Database pool settings must be between 1 and 100")
        return v
    
    @field_validator("max_concurrent_requests")
    @classmethod
    def validate_concurrent_requests(cls, v):
        """Validate concurrent request limit."""
        if v < 1 or v > 1000:
            raise ValueError("Max concurrent requests must be between 1 and 1000")
        return v
    
    @field_validator("worker_processes")
    @classmethod
    def validate_worker_processes(cls, v):
        """Validate worker process count."""
        if v < 1 or v > 32:
            raise ValueError("Worker processes must be between 1 and 32")
        return v
    
    @field_validator("allowed_hosts", "cors_origins", mode="before")
    @classmethod
    def parse_list_fields(cls, v):
        """Parse list fields from string representation."""
        if isinstance(v, str):
            # Handle JSON-like string format
            if v.startswith('[') and v.endswith(']'):
                import json
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    # Fallback to simple parsing
                    return [item.strip().strip('"\'') for item in v.strip('[]').split(',')]
            # Handle comma-separated values
            return [item.strip() for item in v.split(',')]
        return v
    
    @model_validator(mode='after')
    def validate_settings(self):
        """Validate interdependent settings."""
        # Validate chunk overlap is less than chunk size
        if self.chunk_overlap >= self.max_chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        
        # Validate database pool settings
        if self.database_max_overflow < self.database_pool_size:
            raise ValueError("Database max overflow must be >= pool size")
        
        # Production environment validations
        if self.environment == "production":
            if self.debug:
                raise ValueError("Debug mode should not be enabled in production")
            if "*" in self.allowed_hosts:
                raise ValueError("Wildcard hosts not allowed in production")
            if "*" in self.cors_origins:
                raise ValueError("Wildcard CORS origins not allowed in production")
        
        return self


def setup_logging(settings: Settings) -> None:
    """
    Configure structured logging based on settings.
    
    Args:
        settings: Application settings containing logging configuration
    """
    import json
    from datetime import datetime
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set log level
    log_level = getattr(logging, settings.log_level.upper())
    root_logger.setLevel(log_level)
    
    # Create formatter based on format preference
    if settings.log_format == "json":
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                }
                
                if record.exc_info:
                    log_entry["exception"] = self.formatException(record.exc_info)
                
                # Add extra fields if present
                if hasattr(record, "extra_fields"):
                    log_entry.update(record.extra_fields)
                
                return json.dumps(log_entry)
        
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    # Configure handlers
    if settings.log_file:
        # File handler
        file_handler = logging.FileHandler(settings.log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    else:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Configure specific loggers
    if not settings.enable_access_logs:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    if settings.enable_sql_logs and settings.debug:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
    else:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_settings() -> Settings:
    """
    Get application settings with validation and logging setup.
    
    Returns:
        Settings: Validated application settings
        
    Raises:
        ValueError: If any required configuration is missing or invalid
    """
    try:
        settings = Settings()
        
        # Setup logging with the new settings
        setup_logging(settings)
        
        return settings
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {str(e)}")


def validate_environment() -> Dict[str, Any]:
    """
    Comprehensive environment validation with detailed reporting.
    
    Returns:
        Dict[str, Any]: Validation results with status and details
        
    Raises:
        ValueError: If any required environment variable is missing or invalid
    """
    validation_results = {
        "status": "valid",
        "errors": [],
        "warnings": [],
        "config_summary": {}
    }
    
    try:
        settings = get_settings()
        
        # Check required environment variables
        required_vars = [
            "AUTH_TOKEN",
            "GEMINI_API_KEY", 
            "JINA_API_KEY",
            "PINECONE_API_KEY",
            "PINECONE_ENVIRONMENT",
            "DATABASE_URL"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            validation_results["status"] = "invalid"
            validation_results["errors"].append(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )
        
        # Add configuration summary (without sensitive data)
        validation_results["config_summary"] = {
            "environment": settings.environment,
            "debug": settings.debug,
            "host": settings.host,
            "port": settings.port,
            "log_level": settings.log_level,
            "log_format": settings.log_format,
            "max_chunk_size": settings.max_chunk_size,
            "max_document_size_mb": settings.max_document_size_mb,
            "database_pool_size": settings.database_pool_size,
            "max_concurrent_requests": settings.max_concurrent_requests,
        }
        
        # Add warnings for development configurations in production
        if settings.environment == "production":
            if settings.debug:
                validation_results["warnings"].append("Debug mode enabled in production")
            if "*" in settings.cors_origins:
                validation_results["warnings"].append("Wildcard CORS origins in production")
        
        if validation_results["errors"]:
            validation_results["status"] = "invalid"
            raise ValueError(f"Environment validation failed: {'; '.join(validation_results['errors'])}")
        
        return validation_results
        
    except Exception as e:
        validation_results["status"] = "invalid"
        validation_results["errors"].append(str(e))
        raise ValueError(f"Environment validation failed: {str(e)}")


def get_health_check_info(settings: Settings) -> Dict[str, Any]:
    """
    Get comprehensive health check information.
    
    Args:
        settings: Application settings
        
    Returns:
        Dict[str, Any]: Health check information
    """
    health_info = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "1.0.0",
        "environment": settings.environment,
        "services": {}
    }
    
    if settings.enable_detailed_health:
        # Add detailed service status (implement actual checks as needed)
        health_info["services"] = {
            "database": {"status": "unknown", "message": "Health check not implemented"},
            "vector_store": {"status": "unknown", "message": "Health check not implemented"},
            "llm_service": {"status": "unknown", "message": "Health check not implemented"},
            "embedding_service": {"status": "unknown", "message": "Health check not implemented"}
        }
    
    return health_info


# Global settings instance - will be initialized when needed
_settings_instance: Optional[Settings] = None


def get_cached_settings() -> Settings:
    """
    Get cached settings instance to avoid repeated validation.
    
    Returns:
        Settings: Cached application settings
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = get_settings()
    return _settings_instance