"""
Bearer token authentication middleware using FastAPI dependencies.
Implements requirements 1.3, 9.1
"""

from typing import Optional
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import get_settings, Settings
import logging

logger = logging.getLogger(__name__)

# Initialize the HTTPBearer security scheme
security = HTTPBearer()


class AuthenticationError(Exception):
    """Custom exception for authentication errors."""
    
    def __init__(self, message: str, status_code: int = status.HTTP_401_UNAUTHORIZED):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    settings: Settings = Depends(get_settings)
) -> bool:
    """
    Verify Bearer token authentication.
    
    Args:
        credentials: HTTP authorization credentials from request header
        settings: Application settings containing the valid auth token
        
    Returns:
        bool: True if token is valid
        
    Raises:
        HTTPException: If token is invalid or missing
    """
    try:
        if not credentials:
            logger.warning("Authentication attempt with missing credentials")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header is required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not credentials.credentials:
            logger.warning("Authentication attempt with empty token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Bearer token is required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Compare the provided token with the configured auth token
        try:
            configured_token = settings.auth_token
        except Exception as e:
            logger.error(f"Failed to access auth token from settings: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication service error"
            )
        
        if credentials.credentials != configured_token:
            logger.warning(f"Authentication failed for token: {credentials.credentials[:10]}...")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        logger.info("Authentication successful")
        return True
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during authentication: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )


async def get_current_user(
    token_valid: bool = Depends(verify_token)
) -> dict:
    """
    Get current authenticated user information.
    
    Args:
        token_valid: Result from token verification
        
    Returns:
        dict: User information (simplified for this implementation)
    """
    if not token_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication"
        )
    
    # For this implementation, we return a simple user object
    # In a real system, this would decode the token and return user details
    return {
        "authenticated": True,
        "user_type": "api_client"
    }


def create_auth_dependency():
    """
    Create an authentication dependency that can be used in FastAPI routes.
    
    Returns:
        Dependency function for FastAPI routes
    """
    return Depends(verify_token)