"""
Unit tests for authentication middleware and security functionality.
Tests requirements 1.3, 9.1
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from app.auth import verify_token, get_current_user, AuthenticationError, create_auth_dependency
from app.config import Settings


class TestVerifyToken:
    """Test cases for the verify_token function."""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings with a test auth token."""
        settings = Mock(spec=Settings)
        settings.auth_token = "test_valid_token_123"
        return settings
    
    @pytest.fixture
    def valid_credentials(self):
        """Valid HTTP authorization credentials."""
        return HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="test_valid_token_123"
        )
    
    @pytest.fixture
    def invalid_credentials(self):
        """Invalid HTTP authorization credentials."""
        return HTTPAuthorizationCredentials(
            scheme="Bearer", 
            credentials="invalid_token"
        )
    
    @pytest.fixture
    def empty_credentials(self):
        """Empty HTTP authorization credentials."""
        return HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=""
        )
    
    @pytest.mark.asyncio
    async def test_verify_token_success(self, valid_credentials, mock_settings):
        """Test successful token verification."""
        result = await verify_token(valid_credentials, mock_settings)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_verify_token_invalid_token(self, invalid_credentials, mock_settings):
        """Test token verification with invalid token."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_token(invalid_credentials, mock_settings)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid authentication token" in exc_info.value.detail
        assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}
    
    @pytest.mark.asyncio
    async def test_verify_token_empty_token(self, empty_credentials, mock_settings):
        """Test token verification with empty token."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_token(empty_credentials, mock_settings)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Bearer token is required" in exc_info.value.detail
        assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}
    
    @pytest.mark.asyncio
    async def test_verify_token_no_credentials(self, mock_settings):
        """Test token verification with no credentials."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_token(None, mock_settings)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Authorization header is required" in exc_info.value.detail
        assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}
    
    @pytest.mark.asyncio
    async def test_verify_token_settings_exception(self, valid_credentials):
        """Test token verification when settings access fails."""
        # Create a mock that raises an exception when auth_token is accessed
        class FailingSettings:
            @property
            def auth_token(self):
                raise Exception("Settings error")
        
        mock_settings = FailingSettings()
        
        with pytest.raises(HTTPException) as exc_info:
            await verify_token(valid_credentials, mock_settings)
        
        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Authentication service error" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_verify_token_case_sensitive(self, mock_settings):
        """Test that token verification is case sensitive."""
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="TEST_VALID_TOKEN_123"  # Different case
        )
        
        with pytest.raises(HTTPException) as exc_info:
            await verify_token(credentials, mock_settings)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED


class TestGetCurrentUser:
    """Test cases for the get_current_user function."""
    
    @pytest.mark.asyncio
    async def test_get_current_user_success(self):
        """Test successful user retrieval with valid token."""
        result = await get_current_user(token_valid=True)
        
        assert result == {
            "authenticated": True,
            "user_type": "api_client"
        }
    
    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self):
        """Test user retrieval with invalid token."""
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(token_valid=False)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid authentication" in exc_info.value.detail


class TestAuthenticationError:
    """Test cases for the AuthenticationError exception."""
    
    def test_authentication_error_default_status(self):
        """Test AuthenticationError with default status code."""
        error = AuthenticationError("Test error message")
        
        assert error.message == "Test error message"
        assert error.status_code == status.HTTP_401_UNAUTHORIZED
        assert str(error) == "Test error message"
    
    def test_authentication_error_custom_status(self):
        """Test AuthenticationError with custom status code."""
        error = AuthenticationError("Test error", status.HTTP_403_FORBIDDEN)
        
        assert error.message == "Test error"
        assert error.status_code == status.HTTP_403_FORBIDDEN


class TestCreateAuthDependency:
    """Test cases for the create_auth_dependency function."""
    
    def test_create_auth_dependency(self):
        """Test that create_auth_dependency returns a proper dependency."""
        dependency = create_auth_dependency()
        
        # The dependency should be a Depends object
        assert hasattr(dependency, 'dependency')
        assert dependency.dependency == verify_token


class TestAuthIntegration:
    """Integration tests for authentication components."""
    
    @pytest.mark.asyncio
    async def test_full_auth_flow_success(self):
        """Test complete authentication flow with valid credentials."""
        # Mock settings
        settings = Mock(spec=Settings)
        settings.auth_token = "integration_test_token"
        
        # Create valid credentials
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="integration_test_token"
        )
        
        # Test token verification
        token_valid = await verify_token(credentials, settings)
        assert token_valid is True
        
        # Test user retrieval
        user = await get_current_user(token_valid)
        assert user["authenticated"] is True
        assert user["user_type"] == "api_client"
    
    @pytest.mark.asyncio
    async def test_full_auth_flow_failure(self):
        """Test complete authentication flow with invalid credentials."""
        # Mock settings
        settings = Mock(spec=Settings)
        settings.auth_token = "valid_token"
        
        # Create invalid credentials
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="invalid_token"
        )
        
        # Test token verification fails
        with pytest.raises(HTTPException):
            await verify_token(credentials, settings)


# Fixtures for testing with FastAPI app
@pytest.fixture
def mock_auth_settings():
    """Mock settings for authentication testing."""
    settings = Mock(spec=Settings)
    settings.auth_token = "test_auth_token_12345"
    return settings


@pytest.fixture
def auth_headers():
    """Valid authorization headers for testing."""
    return {"Authorization": "Bearer test_auth_token_12345"}


@pytest.fixture
def invalid_auth_headers():
    """Invalid authorization headers for testing."""
    return {"Authorization": "Bearer invalid_token"}