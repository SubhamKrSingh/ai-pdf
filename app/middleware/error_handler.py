"""
Global error handler middleware for the LLM Query Retrieval System.

This module implements comprehensive error handling middleware with structured
error responses, logging, and monitoring integration according to requirements 8.1, 8.2, 8.3, 8.4, 8.5.
"""

import logging
import traceback
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from app.exceptions import (
    BaseSystemError,
    ErrorCategory,
    ClientError,
    ServerError,
    is_recoverable_error,
    get_error_category,
    create_error_context
)
from app.models.schemas import ErrorResponse

# Configure logger for error handling
logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Global error handler middleware that catches all exceptions and provides
    structured error responses with proper logging and monitoring.
    """
    
    def __init__(self, app, enable_debug: bool = False):
        super().__init__(app)
        self.enable_debug = enable_debug
        self.error_stats = {
            "total_errors": 0,
            "client_errors": 0,
            "server_errors": 0,
            "recoverable_errors": 0
        }
    
    async def dispatch(self, request: Request, call_next):
        """
        Process request and handle any exceptions that occur.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain
            
        Returns:
            Response with proper error handling
        """
        start_time = time.time()
        
        try:
            # Process the request
            response = await call_next(request)
            return response
            
        except Exception as exc:
            # Handle the exception
            return await self._handle_exception(request, exc, start_time)
    
    async def _handle_exception(
        self,
        request: Request,
        exc: Exception,
        start_time: float
    ) -> JSONResponse:
        """
        Handle an exception and return appropriate error response.
        
        Args:
            request: HTTP request that caused the exception
            exc: Exception that occurred
            start_time: Request start time for timing
            
        Returns:
            JSONResponse with structured error information
        """
        processing_time = time.time() - start_time
        
        # Update error statistics
        self.error_stats["total_errors"] += 1
        
        # Handle different exception types
        if isinstance(exc, BaseSystemError):
            return await self._handle_system_error(request, exc, processing_time)
        elif isinstance(exc, HTTPException):
            return await self._handle_http_exception(request, exc, processing_time)
        elif isinstance(exc, RequestValidationError):
            return await self._handle_validation_error(request, exc, processing_time)
        elif isinstance(exc, ValidationError):
            return await self._handle_pydantic_validation_error(request, exc, processing_time)
        else:
            return await self._handle_unexpected_error(request, exc, processing_time)
    
    async def _handle_system_error(
        self,
        request: Request,
        exc: BaseSystemError,
        processing_time: float
    ) -> JSONResponse:
        """Handle custom system errors."""
        
        # Update statistics
        if exc.category == ErrorCategory.CLIENT_ERROR:
            self.error_stats["client_errors"] += 1
        else:
            self.error_stats["server_errors"] += 1
            
        if exc.recoverable:
            self.error_stats["recoverable_errors"] += 1
        
        # Log the error
        log_level = logging.WARNING if exc.category == ErrorCategory.CLIENT_ERROR else logging.ERROR
        logger.log(
            log_level,
            f"System error in {request.method} {request.url.path}: {exc.message}",
            extra={
                "error_code": exc.error_code,
                "category": exc.category.value,
                "recoverable": exc.recoverable,
                "details": exc.details,
                "processing_time_ms": processing_time * 1000,
                "request_id": getattr(request.state, "request_id", None)
            }
        )
        
        # Create error response
        error_response = ErrorResponse(
            error=exc.message,
            error_code=exc.error_code,
            details={
                **exc.details,
                "category": exc.category.value,
                "recoverable": exc.recoverable,
                "processing_time_ms": round(processing_time * 1000, 2)
            },
            timestamp=datetime.now(timezone.utc)
        )
        
        return JSONResponse(
            status_code=exc.http_status_code,
            content=error_response.model_dump(mode='json'),
            headers={"X-Error-Category": exc.category.value}
        )
    
    async def _handle_http_exception(
        self,
        request: Request,
        exc: HTTPException,
        processing_time: float
    ) -> JSONResponse:
        """Handle FastAPI HTTP exceptions."""
        
        self.error_stats["client_errors"] += 1
        
        logger.warning(
            f"HTTP {exc.status_code} in {request.method} {request.url.path}: {exc.detail}",
            extra={
                "status_code": exc.status_code,
                "processing_time_ms": processing_time * 1000,
                "request_id": getattr(request.state, "request_id", None)
            }
        )
        
        error_response = ErrorResponse(
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            details={
                "status_code": exc.status_code,
                "processing_time_ms": round(processing_time * 1000, 2)
            },
            timestamp=datetime.now(timezone.utc)
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump(mode='json')
        )
    
    async def _handle_validation_error(
        self,
        request: Request,
        exc: RequestValidationError,
        processing_time: float
    ) -> JSONResponse:
        """Handle FastAPI request validation errors."""
        
        self.error_stats["client_errors"] += 1
        
        logger.warning(
            f"Validation error in {request.method} {request.url.path}: {len(exc.errors())} errors",
            extra={
                "validation_errors": exc.errors(),
                "processing_time_ms": processing_time * 1000,
                "request_id": getattr(request.state, "request_id", None)
            }
        )
        
        error_response = ErrorResponse(
            error="Request validation failed",
            error_code="VALIDATION_ERROR",
            details={
                "validation_errors": exc.errors(),
                "body": str(exc.body) if exc.body else None,
                "processing_time_ms": round(processing_time * 1000, 2)
            },
            timestamp=datetime.now(timezone.utc)
        )
        
        return JSONResponse(
            status_code=422,
            content=error_response.model_dump(mode='json')
        )
    
    async def _handle_pydantic_validation_error(
        self,
        request: Request,
        exc: ValidationError,
        processing_time: float
    ) -> JSONResponse:
        """Handle Pydantic validation errors."""
        
        self.error_stats["client_errors"] += 1
        
        logger.warning(
            f"Pydantic validation error in {request.method} {request.url.path}: {len(exc.errors())} errors",
            extra={
                "validation_errors": exc.errors(),
                "processing_time_ms": processing_time * 1000,
                "request_id": getattr(request.state, "request_id", None)
            }
        )
        
        error_response = ErrorResponse(
            error="Data validation failed",
            error_code="PYDANTIC_VALIDATION_ERROR",
            details={
                "validation_errors": exc.errors(),
                "processing_time_ms": round(processing_time * 1000, 2)
            },
            timestamp=datetime.now(timezone.utc)
        )
        
        return JSONResponse(
            status_code=422,
            content=error_response.model_dump(mode='json')
        )
    
    async def _handle_unexpected_error(
        self,
        request: Request,
        exc: Exception,
        processing_time: float
    ) -> JSONResponse:
        """Handle unexpected errors that weren't caught by specific handlers."""
        
        self.error_stats["server_errors"] += 1
        
        # Log the full traceback for unexpected errors
        logger.error(
            f"Unexpected error in {request.method} {request.url.path}: {type(exc).__name__}: {str(exc)}",
            extra={
                "exception_type": type(exc).__name__,
                "traceback": traceback.format_exc() if self.enable_debug else None,
                "processing_time_ms": processing_time * 1000,
                "request_id": getattr(request.state, "request_id", None)
            },
            exc_info=True
        )
        
        # Don't expose internal error details in production
        error_message = "Internal server error"
        error_details = {
            "processing_time_ms": round(processing_time * 1000, 2)
        }
        
        if self.enable_debug:
            error_message = f"{type(exc).__name__}: {str(exc)}"
            error_details["exception_type"] = type(exc).__name__
            error_details["traceback"] = traceback.format_exc()
        
        error_response = ErrorResponse(
            error=error_message,
            error_code="INTERNAL_SERVER_ERROR",
            details=error_details,
            timestamp=datetime.now(timezone.utc)
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(mode='json'),
            headers={"X-Error-Category": ErrorCategory.SERVER_ERROR.value}
        )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics for monitoring.
        
        Returns:
            Dict containing error statistics
        """
        return {
            **self.error_stats,
            "error_rate": (
                self.error_stats["total_errors"] / max(1, self.error_stats.get("total_requests", 1))
            ) if "total_requests" in self.error_stats else 0,
            "client_error_rate": (
                self.error_stats["client_errors"] / max(1, self.error_stats["total_errors"])
            ) if self.error_stats["total_errors"] > 0 else 0,
            "server_error_rate": (
                self.error_stats["server_errors"] / max(1, self.error_stats["total_errors"])
            ) if self.error_stats["total_errors"] > 0 else 0,
            "recovery_rate": (
                self.error_stats["recoverable_errors"] / max(1, self.error_stats["total_errors"])
            ) if self.error_stats["total_errors"] > 0 else 0
        }


class ErrorLogger:
    """
    Centralized error logging utility with structured logging and monitoring integration.
    """
    
    def __init__(self, logger_name: str = __name__):
        self.logger = logging.getLogger(logger_name)
    
    def log_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        level: int = logging.ERROR
    ) -> None:
        """
        Log an error with structured context information.
        
        Args:
            error: Exception that occurred
            context: Context information about the error
            level: Logging level to use
        """
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "category": get_error_category(error).value,
            "recoverable": is_recoverable_error(error),
            **context
        }
        
        if isinstance(error, BaseSystemError):
            error_info.update({
                "error_code": error.error_code,
                "details": error.details
            })
        
        self.logger.log(
            level,
            f"Error in {context.get('operation', 'unknown operation')}: {str(error)}",
            extra=error_info,
            exc_info=level >= logging.ERROR
        )
    
    def log_recovery_attempt(
        self,
        error: Exception,
        attempt: int,
        max_attempts: int,
        context: Dict[str, Any]
    ) -> None:
        """
        Log a recovery attempt for a recoverable error.
        
        Args:
            error: Exception being recovered from
            attempt: Current attempt number
            max_attempts: Maximum number of attempts
            context: Context information
        """
        self.logger.warning(
            f"Recovery attempt {attempt}/{max_attempts} for {type(error).__name__}: {str(error)}",
            extra={
                "error_type": type(error).__name__,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "recoverable": is_recoverable_error(error),
                **context
            }
        )
    
    def log_recovery_success(
        self,
        error: Exception,
        attempts_used: int,
        context: Dict[str, Any]
    ) -> None:
        """
        Log successful recovery from an error.
        
        Args:
            error: Exception that was recovered from
            attempts_used: Number of attempts used for recovery
            context: Context information
        """
        self.logger.info(
            f"Successfully recovered from {type(error).__name__} after {attempts_used} attempts",
            extra={
                "error_type": type(error).__name__,
                "attempts_used": attempts_used,
                "recovered": True,
                **context
            }
        )
    
    def log_recovery_failure(
        self,
        error: Exception,
        attempts_used: int,
        context: Dict[str, Any]
    ) -> None:
        """
        Log failed recovery from an error.
        
        Args:
            error: Exception that could not be recovered from
            attempts_used: Number of attempts used
            context: Context information
        """
        self.logger.error(
            f"Failed to recover from {type(error).__name__} after {attempts_used} attempts",
            extra={
                "error_type": type(error).__name__,
                "attempts_used": attempts_used,
                "recovered": False,
                "final_error": str(error),
                **context
            },
            exc_info=True
        )


# Global error logger instance
error_logger = ErrorLogger()


def setup_error_handling(app, enable_debug: bool = False) -> ErrorHandlerMiddleware:
    """
    Set up comprehensive error handling for the FastAPI application.
    
    Args:
        app: FastAPI application instance
        enable_debug: Whether to enable debug mode with detailed error information
        
    Returns:
        ErrorHandlerMiddleware instance
    """
    # Add the error handler middleware
    error_middleware = ErrorHandlerMiddleware(app, enable_debug=enable_debug)
    app.add_middleware(ErrorHandlerMiddleware, enable_debug=enable_debug)
    
    # Add error statistics endpoint for monitoring
    @app.get("/api/v1/errors/stats", tags=["Monitoring"])
    async def get_error_statistics():
        """Get error statistics for monitoring."""
        return error_middleware.get_error_statistics()
    
    return error_middleware