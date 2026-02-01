"""
Retry Utility Module
Provides configurable retry logic with exponential backoff for LLM API calls.

Features:
- Exponential backoff with configurable base delay
- Maximum delay cap to prevent excessive waits
- Jitter to prevent thundering herd problem
- Configurable retryable exceptions
- Both sync and async support
- Detailed logging of retry attempts
"""

import asyncio
import functools
import logging
import random
import time
from typing import (
    Any,
    Callable,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

# Type variable for generic return types
T = TypeVar("T")


# Common HTTP status codes that indicate transient failures
RETRYABLE_STATUS_CODES: Set[int] = {
    408,  # Request Timeout
    429,  # Too Many Requests (Rate Limit)
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
    520,  # Cloudflare Unknown Error
    522,  # Cloudflare Connection Timed Out
    524,  # Cloudflare Timeout
}


class RetryConfig:
    """
    Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds before first retry (default: 1.0)
        max_delay: Maximum delay in seconds between retries (default: 60.0)
        exponential_base: Base for exponential backoff calculation (default: 2)
        jitter: Whether to add random jitter to delays (default: True)
        jitter_factor: Maximum jitter as fraction of delay (default: 0.1)
        retryable_exceptions: Tuple of exception types to retry on
        retryable_status_codes: Set of HTTP status codes to retry on
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2,
        jitter: bool = True,
        jitter_factor: float = 0.1,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
        retryable_status_codes: Optional[Set[int]] = None,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.jitter_factor = jitter_factor

        # Default retryable exceptions for LLM API calls
        self.retryable_exceptions = retryable_exceptions or (
            ConnectionError,
            TimeoutError,
            OSError,
        )

        self.retryable_status_codes = retryable_status_codes or RETRYABLE_STATUS_CODES

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given retry attempt using exponential backoff.

        Args:
            attempt: The current attempt number (0-indexed)

        Returns:
            Delay in seconds

        Example:
            >>> config = RetryConfig(base_delay=1.0, exponential_base=2)
            >>> config.calculate_delay(0)  # ~1 second
            >>> config.calculate_delay(1)  # ~2 seconds
            >>> config.calculate_delay(2)  # ~4 seconds
        """
        # Exponential backoff: base_delay * (exponential_base ^ attempt)
        delay = self.base_delay * (self.exponential_base ** attempt)

        # Cap at max_delay
        delay = min(delay, self.max_delay)

        # Add jitter to prevent thundering herd
        if self.jitter:
            jitter_amount = delay * self.jitter_factor * random.random()
            delay += jitter_amount

        return delay


# Default configuration for LLM API calls
DEFAULT_LLM_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0,
    exponential_base=2,
    jitter=True,
)


def is_retryable_exception(
    exception: Exception,
    config: RetryConfig
) -> bool:
    """
    Check if an exception should trigger a retry.

    Args:
        exception: The exception to check
        config: Retry configuration

    Returns:
        True if the exception is retryable, False otherwise
    """
    # Check if it's a directly retryable exception type
    if isinstance(exception, config.retryable_exceptions):
        return True

    # Check for HTTP status code in exception
    # Different HTTP libraries store status codes differently
    status_code = None

    # httpx style
    if hasattr(exception, 'response') and hasattr(exception.response, 'status_code'):
        status_code = exception.response.status_code

    # requests style
    elif hasattr(exception, 'status_code'):
        status_code = exception.status_code

    # aiohttp style
    elif hasattr(exception, 'status'):
        status_code = exception.status

    # anthropic/openai SDK style - check message for rate limit
    exception_str = str(exception).lower()
    if 'rate limit' in exception_str or 'rate_limit' in exception_str:
        return True
    if 'overloaded' in exception_str:
        return True
    if 'temporarily unavailable' in exception_str:
        return True
    if 'connection' in exception_str and 'error' in exception_str:
        return True
    if 'timeout' in exception_str:
        return True

    if status_code and status_code in config.retryable_status_codes:
        return True

    return False


def retry_sync(
    func: Callable[..., T],
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> T:
    """
    Execute a synchronous function with retry logic.

    Args:
        func: The function to execute (should be a zero-argument callable)
        config: Retry configuration (uses DEFAULT_LLM_RETRY_CONFIG if None)
        on_retry: Optional callback called on each retry with (attempt, exception, delay)

    Returns:
        The result of the function

    Raises:
        The last exception if all retries are exhausted

    Example:
        >>> def call_api():
        ...     return requests.get("https://api.example.com/data")
        >>> result = retry_sync(call_api)
    """
    config = config or DEFAULT_LLM_RETRY_CONFIG
    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            return func()

        except Exception as e:
            last_exception = e

            # Check if we should retry
            if attempt >= config.max_retries:
                logger.error(
                    f"All {config.max_retries + 1} attempts failed. "
                    f"Last error: {type(e).__name__}: {e}"
                )
                raise

            if not is_retryable_exception(e, config):
                logger.error(f"Non-retryable exception: {type(e).__name__}: {e}")
                raise

            # Calculate delay and log
            delay = config.calculate_delay(attempt)
            logger.warning(
                f"Attempt {attempt + 1}/{config.max_retries + 1} failed: "
                f"{type(e).__name__}: {e}. "
                f"Retrying in {delay:.2f}s..."
            )

            # Call retry callback if provided
            if on_retry:
                on_retry(attempt, e, delay)

            # Sleep before retry
            time.sleep(delay)

    # This shouldn't be reached, but just in case
    raise last_exception


async def retry_async(
    func: Callable[..., T],
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> T:
    """
    Execute an async function with retry logic.

    Args:
        func: The async function to execute (should be a zero-argument callable)
        config: Retry configuration (uses DEFAULT_LLM_RETRY_CONFIG if None)
        on_retry: Optional callback called on each retry with (attempt, exception, delay)

    Returns:
        The result of the function

    Raises:
        The last exception if all retries are exhausted

    Example:
        >>> async def call_api():
        ...     async with httpx.AsyncClient() as client:
        ...         return await client.get("https://api.example.com/data")
        >>> result = await retry_async(call_api)
    """
    config = config or DEFAULT_LLM_RETRY_CONFIG
    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func()

        except Exception as e:
            last_exception = e

            # Check if we should retry
            if attempt >= config.max_retries:
                logger.error(
                    f"All {config.max_retries + 1} attempts failed. "
                    f"Last error: {type(e).__name__}: {e}"
                )
                raise

            if not is_retryable_exception(e, config):
                logger.error(f"Non-retryable exception: {type(e).__name__}: {e}")
                raise

            # Calculate delay and log
            delay = config.calculate_delay(attempt)
            logger.warning(
                f"Attempt {attempt + 1}/{config.max_retries + 1} failed: "
                f"{type(e).__name__}: {e}. "
                f"Retrying in {delay:.2f}s..."
            )

            # Call retry callback if provided
            if on_retry:
                on_retry(attempt, e, delay)

            # Sleep before retry
            await asyncio.sleep(delay)

    # This shouldn't be reached, but just in case
    raise last_exception


def with_retry(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
):
    """
    Decorator to add retry logic to a function.

    Automatically detects if the function is async or sync.

    Args:
        config: Retry configuration (uses DEFAULT_LLM_RETRY_CONFIG if None)
        on_retry: Optional callback called on each retry

    Returns:
        Decorated function with retry logic

    Example:
        >>> @with_retry(config=RetryConfig(max_retries=5))
        ... async def call_llm(prompt: str):
        ...     return await client.complete(prompt)

        >>> @with_retry()
        ... def call_api():
        ...     return requests.get("https://api.example.com")
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                return await retry_async(
                    lambda: func(*args, **kwargs),
                    config=config,
                    on_retry=on_retry,
                )
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                return retry_sync(
                    lambda: func(*args, **kwargs),
                    config=config,
                    on_retry=on_retry,
                )
            return sync_wrapper

    return decorator


class RetryContext:
    """
    Context manager for retry logic with detailed tracking.

    Useful when you need more control over the retry process or
    want to track retry statistics.

    Attributes:
        config: Retry configuration
        attempts: Number of attempts made
        total_delay: Total time spent waiting between retries
        exceptions: List of exceptions encountered

    Example:
        >>> async with RetryContext(config=RetryConfig(max_retries=3)) as ctx:
        ...     result = await ctx.execute(lambda: call_llm(prompt))
        >>> print(f"Succeeded after {ctx.attempts} attempts")
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or DEFAULT_LLM_RETRY_CONFIG
        self.attempts = 0
        self.total_delay = 0.0
        self.exceptions: list[Exception] = []
        self._success = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    async def execute_async(self, func: Callable[..., T]) -> T:
        """Execute an async function with retry tracking."""
        def on_retry(attempt: int, exception: Exception, delay: float):
            self.attempts = attempt + 1
            self.total_delay += delay
            self.exceptions.append(exception)

        result = await retry_async(func, config=self.config, on_retry=on_retry)
        self.attempts += 1
        self._success = True
        return result

    def execute_sync(self, func: Callable[..., T]) -> T:
        """Execute a sync function with retry tracking."""
        def on_retry(attempt: int, exception: Exception, delay: float):
            self.attempts = attempt + 1
            self.total_delay += delay
            self.exceptions.append(exception)

        result = retry_sync(func, config=self.config, on_retry=on_retry)
        self.attempts += 1
        self._success = True
        return result

    @property
    def succeeded(self) -> bool:
        """Whether the operation eventually succeeded."""
        return self._success

    @property
    def had_retries(self) -> bool:
        """Whether any retries were needed."""
        return len(self.exceptions) > 0


# Pre-configured retry configs for common use cases
AGGRESSIVE_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    base_delay=0.5,
    max_delay=30.0,
    exponential_base=2,
)

CONSERVATIVE_RETRY_CONFIG = RetryConfig(
    max_retries=2,
    base_delay=2.0,
    max_delay=10.0,
    exponential_base=2,
)

NO_RETRY_CONFIG = RetryConfig(
    max_retries=0,
)
