"""
Unit tests for configuration management.
Tests the environment configuration validation functionality.
"""

import pytest
import os
from unittest.mock import patch
from app.config import Settings, get_settings, validate_environment


class TestSettings:
    """Test cases for Settings configuration class."""
    
    def test_settings_with_valid_environment(self):
        """Test that settings load correctly with valid environment variables."""
        with patch.dict(os.environ, {
            'AUTH_TOKEN': 'test_token',
            'GEMINI_API_KEY': 'test_gemini_key',
            'JINA_API_KEY': 'test_jina_key',
            'PINECONE_API_KEY': 'test_pinecone_key',
            'PINECONE_ENVIRONMENT': 'test_env',
            'DATABASE_URL': 'postgresql://user:pass@localhost:5432/test'
        }):
            settings = Settings()
            assert settings.auth_token == 'test_token'
            assert settings.gemini_api_key == 'test_gemini_key'
            assert settings.jina_api_key == 'test_jina_key'
            assert settings.pinecone_api_key == 'test_pinecone_key'
            assert settings.pinecone_environment == 'test_env'
            assert settings.database_url == 'postgresql://user:pass@localhost:5432/test'
    
    def test_settings_with_missing_required_fields(self):
        """Test that settings raise validation error when required fields are missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(Exception):  # Should raise validation error
                Settings()
    
    def test_port_validation(self):
        """Test port number validation."""
        with patch.dict(os.environ, {
            'AUTH_TOKEN': 'test_token',
            'GEMINI_API_KEY': 'test_gemini_key',
            'JINA_API_KEY': 'test_jina_key',
            'PINECONE_API_KEY': 'test_pinecone_key',
            'PINECONE_ENVIRONMENT': 'test_env',
            'DATABASE_URL': 'postgresql://user:pass@localhost:5432/test',
            'PORT': '70000'  # Invalid port
        }):
            with pytest.raises(Exception):  # Should raise validation error
                Settings()
    
    def test_database_url_validation(self):
        """Test database URL format validation."""
        with patch.dict(os.environ, {
            'AUTH_TOKEN': 'test_token',
            'GEMINI_API_KEY': 'test_gemini_key',
            'JINA_API_KEY': 'test_jina_key',
            'PINECONE_API_KEY': 'test_pinecone_key',
            'PINECONE_ENVIRONMENT': 'test_env',
            'DATABASE_URL': 'invalid_url'  # Invalid database URL
        }):
            with pytest.raises(Exception):  # Should raise validation error
                Settings()


class TestConfigurationFunctions:
    """Test cases for configuration utility functions."""
    
    def test_validate_environment_success(self):
        """Test successful environment validation."""
        with patch.dict(os.environ, {
            'AUTH_TOKEN': 'test_token',
            'GEMINI_API_KEY': 'test_gemini_key',
            'JINA_API_KEY': 'test_jina_key',
            'PINECONE_API_KEY': 'test_pinecone_key',
            'PINECONE_ENVIRONMENT': 'test_env',
            'DATABASE_URL': 'postgresql://user:pass@localhost:5432/test'
        }):
            assert validate_environment() is True
    
    def test_validate_environment_missing_vars(self):
        """Test environment validation with missing variables."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                validate_environment()
            assert "Environment validation failed" in str(exc_info.value)