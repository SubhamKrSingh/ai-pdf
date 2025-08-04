"""
Security configuration including CORS and security headers for the FastAPI application.
Implements security requirements from task 3.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all responses.
    """
    
    async def dispatch(self, request: Request, call_next):
        """
        Add security headers to the response.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            Response with security headers added
        """
        response = await call_next(request)
        
        # Add security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


def configure_cors(
    app: FastAPI,
    allowed_origins: Optional[List[str]] = None,
    allowed_methods: Optional[List[str]] = None,
    allowed_headers: Optional[List[str]] = None,
    allow_credentials: bool = False
) -> None:
    """
    Configure CORS (Cross-Origin Resource Sharing) for the FastAPI application.
    
    Args:
        app: FastAPI application instance
        allowed_origins: List of allowed origins. Defaults to ["*"] for development
        allowed_methods: List of allowed HTTP methods
        allowed_headers: List of allowed headers
        allow_credentials: Whether to allow credentials in CORS requests
    """
    if allowed_origins is None:
        # Default to restrictive CORS policy
        # In production, this should be configured with specific domains
        allowed_origins = ["http://localhost:3000", "http://localhost:8080"]
    
    if allowed_methods is None:
        allowed_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    
    if allowed_headers is None:
        allowed_headers = [
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With"
        ]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=allowed_methods,
        allow_headers=allowed_headers,
        expose_headers=["X-Total-Count", "X-Request-ID"]
    )
    
    logger.info(f"CORS configured with origins: {allowed_origins}")


def configure_security_middleware(
    app: FastAPI,
    trusted_hosts: Optional[List[str]] = None
) -> None:
    """
    Configure security middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
        trusted_hosts: List of trusted host names. If None, allows all hosts
    """
    # Add security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Add trusted host middleware if hosts are specified
    if trusted_hosts:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=trusted_hosts
        )
        logger.info(f"Trusted hosts configured: {trusted_hosts}")
    else:
        logger.warning("No trusted hosts configured - allowing all hosts")
    
    logger.info("Security middleware configured")


def setup_security(
    app: FastAPI,
    cors_origins: Optional[List[str]] = None,
    trusted_hosts: Optional[List[str]] = None,
    allow_credentials: bool = False
) -> None:
    """
    Setup all security configurations for the FastAPI application.
    
    Args:
        app: FastAPI application instance
        cors_origins: List of allowed CORS origins
        trusted_hosts: List of trusted host names
        allow_credentials: Whether to allow credentials in CORS requests
    """
    try:
        # Configure CORS
        configure_cors(
            app=app,
            allowed_origins=cors_origins,
            allow_credentials=allow_credentials
        )
        
        # Configure security middleware
        configure_security_middleware(
            app=app,
            trusted_hosts=trusted_hosts
        )
        
        logger.info("Security configuration completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to configure security: {str(e)}")
        raise


# Security configuration presets
class SecurityConfig:
    """Predefined security configurations for different environments."""
    
    @staticmethod
    def development() -> dict:
        """Development security configuration - more permissive."""
        return {
            "cors_origins": ["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000"],
            "trusted_hosts": None,  # Allow all hosts in development
            "allow_credentials": True
        }
    
    @staticmethod
    def production() -> dict:
        """Production security configuration - restrictive."""
        return {
            "cors_origins": [],  # Should be configured with actual production domains
            "trusted_hosts": [],  # Should be configured with actual production hosts
            "allow_credentials": False
        }
    
    @staticmethod
    def testing() -> dict:
        """Testing security configuration."""
        return {
            "cors_origins": ["http://testserver"],
            "trusted_hosts": ["testserver"],
            "allow_credentials": False
        }