"""
Unit tests for security configuration and middleware.
Tests security headers and CORS configuration functionality.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import Response
from app.security import (
    SecurityHeadersMiddleware,
    configure_cors,
    configure_security_middleware,
    setup_security,
    SecurityConfig
)


class TestSecurityHeadersMiddleware:
    """Test cases for SecurityHeadersMiddleware."""
    
    @pytest.fixture
    def middleware(self):
        """Create SecurityHeadersMiddleware instance."""
        app = Mock()
        return SecurityHeadersMiddleware(app)
    
    @pytest.mark.asyncio
    async def test_security_headers_added(self, middleware):
        """Test that security headers are added to responses."""
        # Mock request and response
        request = Mock(spec=Request)
        response = Mock(spec=Response)
        response.headers = {}
        
        # Mock call_next to return the response
        async def mock_call_next(req):
            return response
        
        # Call the middleware
        result = await middleware.dispatch(request, mock_call_next)
        
        # Verify security headers were added
        expected_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY", 
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
        
        for header, value in expected_headers.items():
            assert result.headers[header] == value
    
    @pytest.mark.asyncio
    async def test_middleware_preserves_existing_headers(self, middleware):
        """Test that middleware preserves existing response headers."""
        request = Mock(spec=Request)
        response = Mock(spec=Response)
        response.headers = {"Custom-Header": "custom-value"}
        
        async def mock_call_next(req):
            return response
        
        result = await middleware.dispatch(request, mock_call_next)
        
        # Verify custom header is preserved
        assert result.headers["Custom-Header"] == "custom-value"
        # Verify security headers are also added
        assert "X-Content-Type-Options" in result.headers


class TestConfigureCors:
    """Test cases for CORS configuration."""
    
    @pytest.fixture
    def mock_app(self):
        """Mock FastAPI application."""
        app = Mock(spec=FastAPI)
        app.add_middleware = Mock()
        return app
    
    def test_configure_cors_default_settings(self, mock_app):
        """Test CORS configuration with default settings."""
        configure_cors(mock_app)
        
        # Verify add_middleware was called
        mock_app.add_middleware.assert_called_once()
        
        # Get the call arguments
        call_args = mock_app.add_middleware.call_args
        middleware_class = call_args[0][0]
        kwargs = call_args[1]
        
        # Verify correct middleware class
        from fastapi.middleware.cors import CORSMiddleware
        assert middleware_class == CORSMiddleware
        
        # Verify default origins
        expected_origins = ["http://localhost:3000", "http://localhost:8080"]
        assert kwargs["allow_origins"] == expected_origins
        assert kwargs["allow_credentials"] is False
    
    def test_configure_cors_custom_settings(self, mock_app):
        """Test CORS configuration with custom settings."""
        custom_origins = ["https://example.com", "https://api.example.com"]
        custom_methods = ["GET", "POST"]
        custom_headers = ["Authorization", "Content-Type"]
        
        configure_cors(
            app=mock_app,
            allowed_origins=custom_origins,
            allowed_methods=custom_methods,
            allowed_headers=custom_headers,
            allow_credentials=True
        )
        
        call_args = mock_app.add_middleware.call_args
        kwargs = call_args[1]
        
        assert kwargs["allow_origins"] == custom_origins
        assert kwargs["allow_methods"] == custom_methods
        assert kwargs["allow_headers"] == custom_headers
        assert kwargs["allow_credentials"] is True


class TestConfigureSecurityMiddleware:
    """Test cases for security middleware configuration."""
    
    @pytest.fixture
    def mock_app(self):
        """Mock FastAPI application."""
        app = Mock(spec=FastAPI)
        app.add_middleware = Mock()
        return app
    
    def test_configure_security_middleware_no_trusted_hosts(self, mock_app):
        """Test security middleware configuration without trusted hosts."""
        configure_security_middleware(mock_app)
        
        # Should add SecurityHeadersMiddleware only
        assert mock_app.add_middleware.call_count == 1
        call_args = mock_app.add_middleware.call_args_list[0]
        assert call_args[0][0] == SecurityHeadersMiddleware
    
    def test_configure_security_middleware_with_trusted_hosts(self, mock_app):
        """Test security middleware configuration with trusted hosts."""
        trusted_hosts = ["example.com", "api.example.com"]
        
        configure_security_middleware(mock_app, trusted_hosts)
        
        # Should add both SecurityHeadersMiddleware and TrustedHostMiddleware
        assert mock_app.add_middleware.call_count == 2
        
        # Check SecurityHeadersMiddleware was added
        first_call = mock_app.add_middleware.call_args_list[0]
        assert first_call[0][0] == SecurityHeadersMiddleware
        
        # Check TrustedHostMiddleware was added
        second_call = mock_app.add_middleware.call_args_list[1]
        from fastapi.middleware.trustedhost import TrustedHostMiddleware
        assert second_call[0][0] == TrustedHostMiddleware
        assert second_call[1]["allowed_hosts"] == trusted_hosts


class TestSetupSecurity:
    """Test cases for the main setup_security function."""
    
    @pytest.fixture
    def mock_app(self):
        """Mock FastAPI application."""
        app = Mock(spec=FastAPI)
        app.add_middleware = Mock()
        return app
    
    @patch('app.security.configure_cors')
    @patch('app.security.configure_security_middleware')
    def test_setup_security_success(self, mock_configure_security, mock_configure_cors, mock_app):
        """Test successful security setup."""
        cors_origins = ["https://example.com"]
        trusted_hosts = ["example.com"]
        
        setup_security(
            app=mock_app,
            cors_origins=cors_origins,
            trusted_hosts=trusted_hosts,
            allow_credentials=True
        )
        
        # Verify both configuration functions were called
        mock_configure_cors.assert_called_once_with(
            app=mock_app,
            allowed_origins=cors_origins,
            allow_credentials=True
        )
        mock_configure_security.assert_called_once_with(
            app=mock_app,
            trusted_hosts=trusted_hosts
        )
    
    @patch('app.security.configure_cors')
    def test_setup_security_exception_handling(self, mock_configure_cors, mock_app):
        """Test security setup exception handling."""
        mock_configure_cors.side_effect = Exception("CORS configuration failed")
        
        with pytest.raises(Exception) as exc_info:
            setup_security(mock_app)
        
        assert "CORS configuration failed" in str(exc_info.value)


class TestSecurityConfig:
    """Test cases for SecurityConfig presets."""
    
    def test_development_config(self):
        """Test development security configuration."""
        config = SecurityConfig.development()
        
        assert "http://localhost:3000" in config["cors_origins"]
        assert "http://localhost:8080" in config["cors_origins"]
        assert "http://127.0.0.1:3000" in config["cors_origins"]
        assert config["trusted_hosts"] is None
        assert config["allow_credentials"] is True
    
    def test_production_config(self):
        """Test production security configuration."""
        config = SecurityConfig.production()
        
        assert config["cors_origins"] == []
        assert config["trusted_hosts"] == []
        assert config["allow_credentials"] is False
    
    def test_testing_config(self):
        """Test testing security configuration."""
        config = SecurityConfig.testing()
        
        assert config["cors_origins"] == ["http://testserver"]
        assert config["trusted_hosts"] == ["testserver"]
        assert config["allow_credentials"] is False


class TestSecurityIntegration:
    """Integration tests for security components."""
    
    def test_full_security_setup_integration(self):
        """Test complete security setup with real FastAPI app."""
        app = FastAPI()
        
        # Apply security configuration
        setup_security(
            app=app,
            cors_origins=["http://localhost:3000"],
            trusted_hosts=["localhost"],
            allow_credentials=False
        )
        
        # Verify middleware was added (check middleware stack)
        middleware_names = [middleware.cls.__name__ for middleware in app.user_middleware]
        
        # Should have CORS and Security middleware
        assert "CORSMiddleware" in middleware_names
        assert "TrustedHostMiddleware" in middleware_names
        assert "SecurityHeadersMiddleware" in middleware_names
    
    def test_security_headers_in_response(self):
        """Test that security headers appear in actual HTTP responses."""
        app = FastAPI()
        
        # Add security middleware
        configure_security_middleware(app)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        client = TestClient(app)
        response = client.get("/test")
        
        # Verify security headers are present
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        assert "Strict-Transport-Security" in response.headers