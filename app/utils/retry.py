"""
Retry utility for handling recoverable errors with graceful degradation.

This module provides retry mechanisms and graceful degradation strategies
for external service failures according to requirements 8.3, 8.4, 8.5.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Optional, Type, Union, List
from functools import wraps

from app.exceptions import (
    BaseSystemError,
    is_recoverable_error,
    EmbeddingServiceError,
    LLMServiceError,
    VectorStoreError,
    DatabaseError,
    DocumentDownloadError
)
from app.middleware.error_handler import error_logger

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        recoverable_exceptions: Optional[List[Type[Exception]]] = None
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.recoverable_exceptions = recoverable_exceptions or [
            EmbeddingServiceError,
            LLMServiceError,
            VectorStoreError,
            DatabaseError,
            DocumentDownloadError,
            ConnectionError,
            TimeoutError,
            OSError
        ]
    
    def is_recoverable(self, error: Exception) -> bool:
        """Check if an error is recoverable based on configuration."""
        if is_recoverable_error(error):
            return True
        
        return any(isinstance(error, exc_type) for exc_type in self.recoverable_exceptions)
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number."""
        delay = self.initial_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add jitter to prevent thundering herd
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        return delay


class RetryResult:
    """Result of a retry operation."""
    
    def __init__(
        self,
        success: bool,
        result: Any = None,
        error: Optional[Exception] = None,
        attempts_used: int = 0,
        total_time: float = 0.0
    ):
        self.success = success
        self.result = result
        self.error = error
        self.attempts_used = attempts_used
        self.total_time = total_time


async def retry_async(
    func: Callable,
    config: RetryConfig,
    context: Optional[dict] = None,
    *args,
    **kwargs
) -> RetryResult:
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry
        config: Retry configuration
        context: Context information for logging
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        RetryResult with success status and result/error
    """
    start_time = time.time()
    last_error = None
    context = context or {}
    
    for attempt in range(1, config.max_attempts + 1):
        try:
            # Log retry attempt
            if attempt > 1:
                error_logger.log_recovery_attempt(
                    last_error,
                    attempt,
                    config.max_attempts,
                    {**context, "function": func.__name__}
                )
            
            # Execute the function
            result = await func(*args, **kwargs)
            
            # Log successful recovery if this wasn't the first attempt
            if attempt > 1:
                error_logger.log_recovery_success(
                    last_error,
                    attempt,
                    {**context, "function": func.__name__}
                )
            
            return RetryResult(
                success=True,
                result=result,
                attempts_used=attempt,
                total_time=time.time() - start_time
            )
            
        except Exception as error:
            last_error = error
            
            # Check if error is recoverable
            if not config.is_recoverable(error):
                logger.error(
                    f"Non-recoverable error in {func.__name__}: {str(error)}",
                    extra={**context, "attempt": attempt}
                )
                break
            
            # If this is the last attempt, don't wait
            if attempt >= config.max_attempts:
                break
            
            # Calculate delay and wait
            delay = config.calculate_delay(attempt)
            logger.warning(
                f"Attempt {attempt} failed for {func.__name__}, retrying in {delay:.2f}s: {str(error)}",
                extra={**context, "delay": delay}
            )
            await asyncio.sleep(delay)
    
    # Log final failure
    error_logger.log_recovery_failure(
        last_error,
        config.max_attempts,
        {**context, "function": func.__name__}
    )
    
    return RetryResult(
        success=False,
        error=last_error,
        attempts_used=config.max_attempts,
        total_time=time.time() - start_time
    )


def retry_sync(
    func: Callable,
    config: RetryConfig,
    context: Optional[dict] = None,
    *args,
    **kwargs
) -> RetryResult:
    """
    Retry a sync function with exponential backoff.
    
    Args:
        func: Sync function to retry
        config: Retry configuration
        context: Context information for logging
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        RetryResult with success status and result/error
    """
    start_time = time.time()
    last_error = None
    context = context or {}
    
    for attempt in range(1, config.max_attempts + 1):
        try:
            # Log retry attempt
            if attempt > 1:
                error_logger.log_recovery_attempt(
                    last_error,
                    attempt,
                    config.max_attempts,
                    {**context, "function": func.__name__}
                )
            
            # Execute the function
            result = func(*args, **kwargs)
            
            # Log successful recovery if this wasn't the first attempt
            if attempt > 1:
                error_logger.log_recovery_success(
                    last_error,
                    attempt,
                    {**context, "function": func.__name__}
                )
            
            return RetryResult(
                success=True,
                result=result,
                attempts_used=attempt,
                total_time=time.time() - start_time
            )
            
        except Exception as error:
            last_error = error
            
            # Check if error is recoverable
            if not config.is_recoverable(error):
                logger.error(
                    f"Non-recoverable error in {func.__name__}: {str(error)}",
                    extra={**context, "attempt": attempt}
                )
                break
            
            # If this is the last attempt, don't wait
            if attempt >= config.max_attempts:
                break
            
            # Calculate delay and wait
            delay = config.calculate_delay(attempt)
            logger.warning(
                f"Attempt {attempt} failed for {func.__name__}, retrying in {delay:.2f}s: {str(error)}",
                extra={**context, "delay": delay}
            )
            time.sleep(delay)
    
    # Log final failure
    error_logger.log_recovery_failure(
        last_error,
        config.max_attempts,
        {**context, "function": func.__name__}
    )
    
    return RetryResult(
        success=False,
        error=last_error,
        attempts_used=config.max_attempts,
        total_time=time.time() - start_time
    )


def with_retry(config: Optional[RetryConfig] = None, context: Optional[dict] = None):
    """
    Decorator for adding retry behavior to async functions.
    
    Args:
        config: Retry configuration (uses default if None)
        context: Context information for logging
        
    Returns:
        Decorated function with retry behavior
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await retry_async(func, config, context, *args, **kwargs)
            if result.success:
                return result.result
            else:
                raise result.error
        
        return wrapper
    
    return decorator


def with_retry_sync(config: Optional[RetryConfig] = None, context: Optional[dict] = None):
    """
    Decorator for adding retry behavior to sync functions.
    
    Args:
        config: Retry configuration (uses default if None)
        context: Context information for logging
        
    Returns:
        Decorated function with retry behavior
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = retry_sync(func, config, context, *args, **kwargs)
            if result.success:
                return result.result
            else:
                raise result.error
        
        return wrapper
    
    return decorator


class GracefulDegradation:
    """
    Utility for implementing graceful degradation strategies.
    
    Provides fallback mechanisms when services are unavailable.
    """
    
    @staticmethod
    async def with_fallback(
        primary_func: Callable,
        fallback_func: Optional[Callable] = None,
        fallback_value: Any = None,
        context: Optional[dict] = None
    ) -> Any:
        """
        Execute primary function with fallback on failure.
        
        Args:
            primary_func: Primary function to execute
            fallback_func: Fallback function to execute on failure
            fallback_value: Static fallback value if no fallback function
            context: Context information for logging
            
        Returns:
            Result from primary function or fallback
        """
        context = context or {}
        
        try:
            if asyncio.iscoroutinefunction(primary_func):
                return await primary_func()
            else:
                return primary_func()
                
        except Exception as error:
            logger.warning(
                f"Primary function {primary_func.__name__} failed, using fallback: {str(error)}",
                extra={**context, "error_type": type(error).__name__}
            )
            
            if fallback_func:
                try:
                    if asyncio.iscoroutinefunction(fallback_func):
                        return await fallback_func()
                    else:
                        return fallback_func()
                except Exception as fallback_error:
                    logger.error(
                        f"Fallback function also failed: {str(fallback_error)}",
                        extra={**context, "fallback_error_type": type(fallback_error).__name__}
                    )
                    if fallback_value is not None:
                        return fallback_value
                    raise fallback_error
            elif fallback_value is not None:
                return fallback_value
            else:
                raise error
    
    @staticmethod
    def create_circuit_breaker(
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Create a circuit breaker for preventing cascading failures.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
            expected_exception: Exception type that triggers circuit breaker
            
        Returns:
            Circuit breaker decorator
        """
        class CircuitBreaker:
            def __init__(self):
                self.failure_count = 0
                self.last_failure_time = None
                self.state = "closed"  # closed, open, half-open
            
            def __call__(self, func):
                @wraps(func)
                async def wrapper(*args, **kwargs):
                    # Check circuit state
                    if self.state == "open":
                        if (time.time() - self.last_failure_time) > recovery_timeout:
                            self.state = "half-open"
                            logger.info(f"Circuit breaker for {func.__name__} entering half-open state")
                        else:
                            raise Exception(f"Circuit breaker open for {func.__name__}")
                    
                    try:
                        if asyncio.iscoroutinefunction(func):
                            result = await func(*args, **kwargs)
                        else:
                            result = func(*args, **kwargs)
                        
                        # Reset on success
                        if self.state == "half-open":
                            self.state = "closed"
                            self.failure_count = 0
                            logger.info(f"Circuit breaker for {func.__name__} closed after recovery")
                        
                        return result
                        
                    except expected_exception as error:
                        self.failure_count += 1
                        self.last_failure_time = time.time()
                        
                        if self.failure_count >= failure_threshold:
                            self.state = "open"
                            logger.error(
                                f"Circuit breaker opened for {func.__name__} after {self.failure_count} failures"
                            )
                        
                        raise error
                
                return wrapper
        
        return CircuitBreaker()


# Default retry configurations for different service types
DEFAULT_RETRY_CONFIG = RetryConfig(max_attempts=3, initial_delay=1.0)
LLM_RETRY_CONFIG = RetryConfig(max_attempts=3, initial_delay=2.0, max_delay=30.0)
EMBEDDING_RETRY_CONFIG = RetryConfig(max_attempts=3, initial_delay=2.0, max_delay=30.0)  # More resilient for timeouts
DATABASE_RETRY_CONFIG = RetryConfig(max_attempts=2, initial_delay=0.5, max_delay=5.0)
DOWNLOAD_RETRY_CONFIG = RetryConfig(max_attempts=2, initial_delay=1.0, max_delay=10.0)