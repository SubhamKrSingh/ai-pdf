"""
Comprehensive logging configuration for the LLM Query Retrieval System.
Provides structured logging with JSON format for production and readable format for development.
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from pythonjsonlogger import jsonlogger

from app.config import Settings


class StructuredFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter for structured logging with additional context.
    """
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        """
        Add custom fields to the log record.
        
        Args:
            log_record: The log record dictionary to modify
            record: The original logging record
            message_dict: Additional message fields
        """
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp in ISO format
        log_record['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        
        # Add service information
        log_record['service'] = 'llm-query-retrieval-system'
        log_record['version'] = '1.0.0'
        
        # Add request context if available
        if hasattr(record, 'request_id'):
            log_record['request_id'] = record.request_id
        
        if hasattr(record, 'user_id'):
            log_record['user_id'] = record.user_id
        
        # Add performance metrics if available
        if hasattr(record, 'duration_ms'):
            log_record['duration_ms'] = record.duration_ms
        
        # Add error context if available
        if hasattr(record, 'error_code'):
            log_record['error_code'] = record.error_code
        
        # Ensure level is always present
        if 'level' not in log_record:
            log_record['level'] = record.levelname


class ColoredFormatter(logging.Formatter):
    """
    Colored formatter for development logging with better readability.
    """
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with colors.
        
        Args:
            record: The logging record to format
            
        Returns:
            str: Formatted log message with colors
        """
        # Add color to level name
        level_color = self.COLORS.get(record.levelname, '')
        reset_color = self.COLORS['RESET']
        
        # Create colored level name
        colored_level = f"{level_color}{record.levelname:<8}{reset_color}"
        
        # Format the message
        formatted_time = self.formatTime(record, self.datefmt)
        
        # Build the log message
        log_message = (
            f"{formatted_time} | {colored_level} | "
            f"{record.name:<30} | {record.getMessage()}"
        )
        
        # Add exception info if present
        if record.exc_info:
            log_message += f"\n{self.formatException(record.exc_info)}"
        
        return log_message


def setup_logging(settings: Settings) -> None:
    """
    Set up comprehensive logging configuration based on settings.
    
    Args:
        settings: Application settings containing logging configuration
    """
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set log level
    log_level = getattr(logging, settings.logging.log_level.upper())
    root_logger.setLevel(log_level)
    
    # Create formatter based on format preference
    if settings.logging.log_format == "json":
        formatter = StructuredFormatter(
            fmt='%(timestamp)s %(level)s %(name)s %(message)s'
        )
    else:
        formatter = ColoredFormatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Configure handlers
    handlers = []
    
    if settings.logging.log_file:
        # Ensure log directory exists
        log_file_path = Path(settings.logging.log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            filename=settings.logging.log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        handlers.append(file_handler)
        
        # Also add console handler for important messages
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.WARNING)  # Only warnings and above to console
        handlers.append(console_handler)
    else:
        # Console handler only
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        handlers.append(console_handler)
    
    # Add handlers to root logger
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Configure specific loggers
    configure_third_party_loggers(settings)
    
    # Log the logging configuration
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging configured: level={settings.logging.log_level}, "
        f"format={settings.logging.log_format}, "
        f"file={settings.logging.log_file or 'console'}"
    )


def configure_third_party_loggers(settings: Settings) -> None:
    """
    Configure third-party library loggers.
    
    Args:
        settings: Application settings
    """
    # Uvicorn access logs
    if not settings.logging.enable_access_logs:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    else:
        logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    
    # Uvicorn error logs
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    
    # FastAPI logs
    logging.getLogger("fastapi").setLevel(logging.INFO)
    
    # SQLAlchemy logs
    if settings.logging.enable_sql_logs and settings.debug:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
        logging.getLogger("sqlalchemy.pool").setLevel(logging.DEBUG)
    else:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
    
    # HTTP client logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Pinecone logs
    logging.getLogger("pinecone").setLevel(logging.INFO)
    
    # Reduce noise from other libraries
    logging.getLogger("multipart").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


class RequestContextFilter(logging.Filter):
    """
    Filter to add request context to log records.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add request context to the log record if available.
        
        Args:
            record: The logging record to filter
            
        Returns:
            bool: Always True (don't filter out records)
        """
        # Add request context from contextvars if available
        try:
            from contextvars import ContextVar
            
            # Define context variables (these would be set in middleware)
            request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
            user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
            
            # Add context to record
            request_id = request_id_var.get()
            if request_id:
                record.request_id = request_id
            
            user_id = user_id_var.get()
            if user_id:
                record.user_id = user_id
                
        except ImportError:
            # contextvars not available in older Python versions
            pass
        
        return True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name and add request context filter.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    
    # Add request context filter if not already present
    if not any(isinstance(f, RequestContextFilter) for f in logger.filters):
        logger.addFilter(RequestContextFilter())
    
    return logger


def log_performance(logger: logging.Logger, operation: str, duration_ms: float, **kwargs) -> None:
    """
    Log performance metrics in a structured way.
    
    Args:
        logger: Logger instance
        operation: Operation name
        duration_ms: Duration in milliseconds
        **kwargs: Additional context
    """
    extra_fields = {
        'operation': operation,
        'duration_ms': duration_ms,
        'performance_metric': True,
        **kwargs
    }
    
    # Add extra fields to the log record
    logger.info(
        f"Performance: {operation} completed in {duration_ms:.2f}ms",
        extra={'extra_fields': extra_fields}
    )


def log_error(logger: logging.Logger, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log errors in a structured way with context.
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context information
    """
    extra_fields = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'error_context': context or {},
    }
    
    logger.error(
        f"Error occurred: {type(error).__name__}: {str(error)}",
        exc_info=True,
        extra={'extra_fields': extra_fields}
    )