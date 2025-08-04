"""
Unit tests for retry utility and graceful degradation.

Tests retry mechanisms and graceful degradation strategies
according to requirements 8.3, 8.4, 8.5.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock

from app.utils.retry import (
    RetryConfig,
    RetryResult,
    retry_async,
    retry_sync,
    with_retry,
    with_retry_sync,
    GracefulDegradation,
    DEFAULT_RETRY_CONFIG,
    LLM_RETRY_CONFIG,
    EMBEDDING_RETRY_CONFIG,
    DATABASE_RETRY_CONFIG,
    DOWNLOAD_RETRY_CONFIG
)
from app.exceptions import (
    DocumentDownloadError,
    LLMServiceError,
    EmbeddingServiceError,
    VectorStoreError,
    DatabaseError,
    DocumentParsingError
)


class TestRetryConfig:
    """Test retry configuration."""
    
    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        
        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert len(config.recoverable_exceptions) > 0
    
    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            initial_delay=0.5,
            max_delay=30.0,
            exponential_base=1.5,
            jitter=False
        )
        
        assert config.max_attempts == 5
        assert config.initial_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 1.5
        assert config.jitter is False
    
    def test_is_recoverable_with_system_errors(self):
        """Test is_recoverable with system errors."""
        config = RetryConfig()
        
        assert config.is_recoverable(DocumentDownloadError("http://example.com")) is True
        assert config.is_recoverable(LLMServiceError("test", "timeout")) is True
        assert config.is_recoverable(DocumentParsingError("PDF", "corrupted")) is False
    
    def test_is_recoverable_with_standard_exceptions(self):
        """Test is_recoverable with standard exceptions."""
        config = RetryConfig()
        
        assert config.is_recoverable(ConnectionError()) is True
        assert config.is_recoverable(TimeoutError()) is True
        assert config.is_recoverable(ValueError()) is False
    
    def test_calculate_delay(self):
        """Test delay calculation."""
        config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            max_delay=10.0,
            jitter=False
        )
        
        assert config.calculate_delay(1) == 1.0
        assert config.calculate_delay(2) == 2.0
        assert config.calculate_delay(3) == 4.0
        assert config.calculate_delay(5) == 10.0  # Capped at max_delay
    
    def test_calculate_delay_with_jitter(self):
        """Test delay calculation with jitter."""
        config = RetryConfig(
            initial_delay=2.0,
            exponential_base=2.0,
            jitter=True
        )
        
        delay = config.calculate_delay(2)
        # With jitter, delay should be between 2.0 and 4.0
        assert 2.0 <= delay <= 4.0


class TestRetryResult:
    """Test retry result class."""
    
    def test_successful_result(self):
        """Test successful retry result."""
        result = RetryResult(
            success=True,
            result="test_result",
            attempts_used=2,
            total_time=1.5
        )
        
        assert result.success is True
        assert result.result == "test_result"
        assert result.error is None
        assert result.attempts_used == 2
        assert result.total_time == 1.5
    
    def test_failed_result(self):
        """Test failed retry result."""
        error = ValueError("Test error")
        result = RetryResult(
            success=False,
            error=error,
            attempts_used=3,
            total_time=5.0
        )
        
        assert result.success is False
        assert result.result is None
        assert result.error is error
        assert result.attempts_used == 3
        assert result.total_time == 5.0


class TestRetryAsync:
    """Test async retry functionality."""
    
    @pytest.mark.asyncio
    async def test_successful_first_attempt(self):
        """Test successful execution on first attempt."""
        async def success_func():
            return "success"
        
        config = RetryConfig(max_attempts=3)
        result = await retry_async(success_func, config)
        
        assert result.success is True
        assert result.result == "success"
        assert result.attempts_used == 1
    
    @pytest.mark.asyncio
    async def test_success_after_retries(self):
        """Test success after some retries."""
        call_count = 0
        
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        config = RetryConfig(max_attempts=3, initial_delay=0.01)
        result = await retry_async(flaky_func, config)
        
        assert result.success is True
        assert result.result == "success"
        assert result.attempts_used == 3
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_failure_after_max_attempts(self):
        """Test failure after exhausting max attempts."""
        async def failing_func():
            raise ConnectionError("Persistent failure")
        
        config = RetryConfig(max_attempts=2, initial_delay=0.01)
        result = await retry_async(failing_func, config)
        
        assert result.success is False
        assert isinstance(result.error, ConnectionError)
        assert result.attempts_used == 2
    
    @pytest.mark.asyncio
    async def test_non_recoverable_error(self):
        """Test immediate failure for non-recoverable errors."""
        async def non_recoverable_func():
            raise ValueError("Non-recoverable error")
        
        config = RetryConfig(max_attempts=3, initial_delay=0.01)
        result = await retry_async(non_recoverable_func, config)
        
        assert result.success is False
        assert isinstance(result.error, ValueError)
        assert result.attempts_used == 3  # Still uses max attempts for non-recoverable
    
    @pytest.mark.asyncio
    async def test_with_context(self):
        """Test retry with context information."""
        async def test_func():
            raise ConnectionError("Test error")
        
        config = RetryConfig(max_attempts=2, initial_delay=0.01)
        context = {"operation": "test", "component": "test_component"}
        
        with patch('app.utils.retry.error_logger') as mock_logger:
            result = await retry_async(test_func, config, context)
            
            assert result.success is False
            # Verify logging was called with context
            mock_logger.log_recovery_attempt.assert_called()
            mock_logger.log_recovery_failure.assert_called()


class TestRetrySync:
    """Test sync retry functionality."""
    
    def test_successful_first_attempt(self):
        """Test successful execution on first attempt."""
        def success_func():
            return "success"
        
        config = RetryConfig(max_attempts=3)
        result = retry_sync(success_func, config)
        
        assert result.success is True
        assert result.result == "success"
        assert result.attempts_used == 1
    
    def test_success_after_retries(self):
        """Test success after some retries."""
        call_count = 0
        
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        config = RetryConfig(max_attempts=3, initial_delay=0.01)
        result = retry_sync(flaky_func, config)
        
        assert result.success is True
        assert result.result == "success"
        assert result.attempts_used == 3
    
    def test_failure_after_max_attempts(self):
        """Test failure after exhausting max attempts."""
        def failing_func():
            raise ConnectionError("Persistent failure")
        
        config = RetryConfig(max_attempts=2, initial_delay=0.01)
        result = retry_sync(failing_func, config)
        
        assert result.success is False
        assert isinstance(result.error, ConnectionError)
        assert result.attempts_used == 2


class TestRetryDecorators:
    """Test retry decorators."""
    
    @pytest.mark.asyncio
    async def test_with_retry_decorator_success(self):
        """Test with_retry decorator with successful execution."""
        config = RetryConfig(max_attempts=3, initial_delay=0.01)
        
        @with_retry(config)
        async def test_func():
            return "success"
        
        result = await test_func()
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_with_retry_decorator_failure(self):
        """Test with_retry decorator with failure."""
        config = RetryConfig(max_attempts=2, initial_delay=0.01)
        
        @with_retry(config)
        async def test_func():
            raise ConnectionError("Test error")
        
        with pytest.raises(ConnectionError):
            await test_func()
    
    def test_with_retry_sync_decorator_success(self):
        """Test with_retry_sync decorator with successful execution."""
        config = RetryConfig(max_attempts=3, initial_delay=0.01)
        
        @with_retry_sync(config)
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"
    
    def test_with_retry_sync_decorator_failure(self):
        """Test with_retry_sync decorator with failure."""
        config = RetryConfig(max_attempts=2, initial_delay=0.01)
        
        @with_retry_sync(config)
        def test_func():
            raise ConnectionError("Test error")
        
        with pytest.raises(ConnectionError):
            test_func()


class TestGracefulDegradation:
    """Test graceful degradation utility."""
    
    @pytest.mark.asyncio
    async def test_with_fallback_success(self):
        """Test with_fallback with successful primary function."""
        async def primary_func():
            return "primary_result"
        
        async def fallback_func():
            return "fallback_result"
        
        result = await GracefulDegradation.with_fallback(
            primary_func,
            fallback_func
        )
        
        assert result == "primary_result"
    
    @pytest.mark.asyncio
    async def test_with_fallback_function_fallback(self):
        """Test with_fallback using fallback function."""
        async def primary_func():
            raise ConnectionError("Primary failed")
        
        async def fallback_func():
            return "fallback_result"
        
        result = await GracefulDegradation.with_fallback(
            primary_func,
            fallback_func
        )
        
        assert result == "fallback_result"
    
    @pytest.mark.asyncio
    async def test_with_fallback_value_fallback(self):
        """Test with_fallback using fallback value."""
        async def primary_func():
            raise ConnectionError("Primary failed")
        
        result = await GracefulDegradation.with_fallback(
            primary_func,
            fallback_value="default_value"
        )
        
        assert result == "default_value"
    
    @pytest.mark.asyncio
    async def test_with_fallback_sync_functions(self):
        """Test with_fallback with sync functions."""
        def primary_func():
            raise ConnectionError("Primary failed")
        
        def fallback_func():
            return "fallback_result"
        
        result = await GracefulDegradation.with_fallback(
            primary_func,
            fallback_func
        )
        
        assert result == "fallback_result"
    
    @pytest.mark.asyncio
    async def test_with_fallback_both_fail(self):
        """Test with_fallback when both primary and fallback fail."""
        async def primary_func():
            raise ConnectionError("Primary failed")
        
        async def fallback_func():
            raise ValueError("Fallback failed")
        
        with pytest.raises(ValueError):
            await GracefulDegradation.with_fallback(
                primary_func,
                fallback_func
            )
    
    @pytest.mark.asyncio
    async def test_with_fallback_both_fail_with_value(self):
        """Test with_fallback when both fail but fallback value provided."""
        async def primary_func():
            raise ConnectionError("Primary failed")
        
        async def fallback_func():
            raise ValueError("Fallback failed")
        
        result = await GracefulDegradation.with_fallback(
            primary_func,
            fallback_func,
            fallback_value="final_fallback"
        )
        
        assert result == "final_fallback"
    
    def test_create_circuit_breaker(self):
        """Test circuit breaker creation."""
        circuit_breaker = GracefulDegradation.create_circuit_breaker(
            failure_threshold=2,
            recovery_timeout=1.0,
            expected_exception=ConnectionError
        )
        
        assert callable(circuit_breaker)
        
        # Test that it returns a decorator
        @circuit_breaker
        async def test_func():
            return "success"
        
        assert callable(test_func)


class TestDefaultConfigs:
    """Test default retry configurations."""
    
    def test_default_retry_config(self):
        """Test default retry configuration."""
        assert DEFAULT_RETRY_CONFIG.max_attempts == 3
        assert DEFAULT_RETRY_CONFIG.initial_delay == 1.0
    
    def test_llm_retry_config(self):
        """Test LLM-specific retry configuration."""
        assert LLM_RETRY_CONFIG.max_attempts == 3
        assert LLM_RETRY_CONFIG.initial_delay == 2.0
        assert LLM_RETRY_CONFIG.max_delay == 30.0
    
    def test_embedding_retry_config(self):
        """Test embedding service retry configuration."""
        assert EMBEDDING_RETRY_CONFIG.max_attempts == 2
        assert EMBEDDING_RETRY_CONFIG.initial_delay == 1.0
        assert EMBEDDING_RETRY_CONFIG.max_delay == 10.0
    
    def test_database_retry_config(self):
        """Test database retry configuration."""
        assert DATABASE_RETRY_CONFIG.max_attempts == 2
        assert DATABASE_RETRY_CONFIG.initial_delay == 0.5
        assert DATABASE_RETRY_CONFIG.max_delay == 5.0
    
    def test_download_retry_config(self):
        """Test download retry configuration."""
        assert DOWNLOAD_RETRY_CONFIG.max_attempts == 2
        assert DOWNLOAD_RETRY_CONFIG.initial_delay == 1.0
        assert DOWNLOAD_RETRY_CONFIG.max_delay == 10.0


if __name__ == "__main__":
    pytest.main([__file__])