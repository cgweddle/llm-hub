"""
Unit tests for retry utility module.

Tests cover:
- RetryConfig configuration and delay calculation
- Exponential backoff behavior
- Jitter application
- Sync and async retry functions
- Exception detection and handling
- Retry decorator
- RetryContext tracking

Run with: pytest tests/test_retry.py -v
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from utils.retry import (
        RetryConfig,
        RetryContext,
        retry_sync,
        retry_async,
        with_retry,
        is_retryable_exception,
        DEFAULT_LLM_RETRY_CONFIG,
        AGGRESSIVE_RETRY_CONFIG,
        CONSERVATIVE_RETRY_CONFIG,
        NO_RETRY_CONFIG,
        RETRYABLE_STATUS_CODES,
    )
    RETRY_AVAILABLE = True
except ImportError:
    RETRY_AVAILABLE = False


# ============================================================================
# Test: RetryConfig
# ============================================================================

@pytest.mark.skipif(not RETRY_AVAILABLE, reason="Retry module not available")
class TestRetryConfig:
    """Tests for RetryConfig configuration"""

    def test_default_config_values(self):
        """Test default configuration values"""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2
        assert config.jitter is True
        assert config.jitter_factor == 0.1

    def test_custom_config_values(self):
        """Test custom configuration values"""
        config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=3,
            jitter=False,
        )

        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 3
        assert config.jitter is False

    def test_calculate_delay_exponential_backoff(self):
        """Test exponential backoff delay calculation"""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2,
            jitter=False,  # Disable jitter for predictable testing
            max_delay=100.0,
        )

        # Delay should double each attempt
        assert config.calculate_delay(0) == 1.0   # 1 * 2^0 = 1
        assert config.calculate_delay(1) == 2.0   # 1 * 2^1 = 2
        assert config.calculate_delay(2) == 4.0   # 1 * 2^2 = 4
        assert config.calculate_delay(3) == 8.0   # 1 * 2^3 = 8
        assert config.calculate_delay(4) == 16.0  # 1 * 2^4 = 16

    def test_calculate_delay_respects_max_delay(self):
        """Test that delay is capped at max_delay"""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2,
            max_delay=5.0,
            jitter=False,
        )

        # After a few attempts, should be capped at max_delay
        assert config.calculate_delay(0) == 1.0
        assert config.calculate_delay(1) == 2.0
        assert config.calculate_delay(2) == 4.0
        assert config.calculate_delay(3) == 5.0  # Capped at max_delay
        assert config.calculate_delay(10) == 5.0  # Still capped

    def test_calculate_delay_with_jitter(self):
        """Test that jitter adds randomness to delay"""
        config = RetryConfig(
            base_delay=10.0,
            jitter=True,
            jitter_factor=0.1,
            max_delay=100.0,
        )

        # With jitter, delays should vary but be within expected range
        delays = [config.calculate_delay(0) for _ in range(100)]

        # All delays should be between base and base * (1 + jitter_factor)
        for delay in delays:
            assert 10.0 <= delay <= 11.0  # 10 + (10 * 0.1) = 11

        # With randomness, not all delays should be exactly the same
        assert len(set(delays)) > 1


# ============================================================================
# Test: Predefined Configs
# ============================================================================

@pytest.mark.skipif(not RETRY_AVAILABLE, reason="Retry module not available")
class TestPredefinedConfigs:
    """Tests for predefined retry configurations"""

    def test_default_llm_retry_config(self):
        """Test DEFAULT_LLM_RETRY_CONFIG values"""
        assert DEFAULT_LLM_RETRY_CONFIG.max_retries == 3
        assert DEFAULT_LLM_RETRY_CONFIG.base_delay == 1.0

    def test_aggressive_retry_config(self):
        """Test AGGRESSIVE_RETRY_CONFIG values"""
        assert AGGRESSIVE_RETRY_CONFIG.max_retries == 5
        assert AGGRESSIVE_RETRY_CONFIG.base_delay == 0.5

    def test_conservative_retry_config(self):
        """Test CONSERVATIVE_RETRY_CONFIG values"""
        assert CONSERVATIVE_RETRY_CONFIG.max_retries == 2
        assert CONSERVATIVE_RETRY_CONFIG.base_delay == 2.0

    def test_no_retry_config(self):
        """Test NO_RETRY_CONFIG values"""
        assert NO_RETRY_CONFIG.max_retries == 0


# ============================================================================
# Test: Exception Detection
# ============================================================================

@pytest.mark.skipif(not RETRY_AVAILABLE, reason="Retry module not available")
class TestExceptionDetection:
    """Tests for retryable exception detection"""

    def test_connection_error_is_retryable(self):
        """Test ConnectionError is retryable"""
        config = RetryConfig()
        assert is_retryable_exception(ConnectionError("Connection failed"), config)

    def test_timeout_error_is_retryable(self):
        """Test TimeoutError is retryable"""
        config = RetryConfig()
        assert is_retryable_exception(TimeoutError("Request timed out"), config)

    def test_os_error_is_retryable(self):
        """Test OSError is retryable"""
        config = RetryConfig()
        assert is_retryable_exception(OSError("Network error"), config)

    def test_value_error_not_retryable(self):
        """Test ValueError is not retryable"""
        config = RetryConfig()
        assert not is_retryable_exception(ValueError("Invalid input"), config)

    def test_key_error_not_retryable(self):
        """Test KeyError is not retryable"""
        config = RetryConfig()
        assert not is_retryable_exception(KeyError("key"), config)

    def test_rate_limit_message_is_retryable(self):
        """Test exception with 'rate limit' message is retryable"""
        config = RetryConfig()
        assert is_retryable_exception(Exception("Rate limit exceeded"), config)
        assert is_retryable_exception(Exception("rate_limit_error"), config)

    def test_overloaded_message_is_retryable(self):
        """Test exception with 'overloaded' message is retryable"""
        config = RetryConfig()
        assert is_retryable_exception(Exception("Server is overloaded"), config)

    def test_temporarily_unavailable_is_retryable(self):
        """Test 'temporarily unavailable' exception is retryable"""
        config = RetryConfig()
        assert is_retryable_exception(Exception("Service temporarily unavailable"), config)


# ============================================================================
# Test: Sync Retry
# ============================================================================

@pytest.mark.skipif(not RETRY_AVAILABLE, reason="Retry module not available")
class TestSyncRetry:
    """Tests for synchronous retry function"""

    def test_success_on_first_attempt(self):
        """Test function succeeds on first attempt"""
        mock_func = Mock(return_value="success")

        result = retry_sync(mock_func, config=RetryConfig(max_retries=3))

        assert result == "success"
        assert mock_func.call_count == 1

    def test_success_after_retry(self):
        """Test function succeeds after retry"""
        mock_func = Mock(side_effect=[ConnectionError("fail"), "success"])

        config = RetryConfig(max_retries=3, base_delay=0.01, jitter=False)
        result = retry_sync(mock_func, config=config)

        assert result == "success"
        assert mock_func.call_count == 2

    def test_exhausts_retries(self):
        """Test function exhausts all retries"""
        mock_func = Mock(side_effect=ConnectionError("always fail"))

        config = RetryConfig(max_retries=2, base_delay=0.01, jitter=False)

        with pytest.raises(ConnectionError):
            retry_sync(mock_func, config=config)

        assert mock_func.call_count == 3  # Initial + 2 retries

    def test_non_retryable_exception_not_retried(self):
        """Test non-retryable exceptions are not retried"""
        mock_func = Mock(side_effect=ValueError("invalid"))

        config = RetryConfig(max_retries=3, base_delay=0.01)

        with pytest.raises(ValueError):
            retry_sync(mock_func, config=config)

        assert mock_func.call_count == 1  # No retry

    def test_on_retry_callback_called(self):
        """Test on_retry callback is called on each retry"""
        mock_func = Mock(side_effect=[ConnectionError("fail1"), ConnectionError("fail2"), "success"])
        on_retry = Mock()

        config = RetryConfig(max_retries=3, base_delay=0.01, jitter=False)
        result = retry_sync(mock_func, config=config, on_retry=on_retry)

        assert result == "success"
        assert on_retry.call_count == 2

    def test_delay_between_retries(self):
        """Test there is a delay between retries"""
        mock_func = Mock(side_effect=[ConnectionError("fail"), "success"])

        config = RetryConfig(max_retries=1, base_delay=0.1, jitter=False)

        start_time = time.time()
        retry_sync(mock_func, config=config)
        elapsed = time.time() - start_time

        assert elapsed >= 0.1  # At least base_delay elapsed


# ============================================================================
# Test: Async Retry
# ============================================================================

@pytest.mark.skipif(not RETRY_AVAILABLE, reason="Retry module not available")
class TestAsyncRetry:
    """Tests for asynchronous retry function"""

    @pytest.mark.asyncio
    async def test_async_success_on_first_attempt(self):
        """Test async function succeeds on first attempt"""
        mock_func = AsyncMock(return_value="success")

        result = await retry_async(mock_func, config=RetryConfig(max_retries=3))

        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_async_success_after_retry(self):
        """Test async function succeeds after retry"""
        mock_func = AsyncMock(side_effect=[ConnectionError("fail"), "success"])

        config = RetryConfig(max_retries=3, base_delay=0.01, jitter=False)
        result = await retry_async(mock_func, config=config)

        assert result == "success"
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_async_exhausts_retries(self):
        """Test async function exhausts all retries"""
        mock_func = AsyncMock(side_effect=ConnectionError("always fail"))

        config = RetryConfig(max_retries=2, base_delay=0.01, jitter=False)

        with pytest.raises(ConnectionError):
            await retry_async(mock_func, config=config)

        assert mock_func.call_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_async_non_retryable_exception(self):
        """Test async non-retryable exceptions are not retried"""
        mock_func = AsyncMock(side_effect=ValueError("invalid"))

        config = RetryConfig(max_retries=3, base_delay=0.01)

        with pytest.raises(ValueError):
            await retry_async(mock_func, config=config)

        assert mock_func.call_count == 1


# ============================================================================
# Test: Decorator
# ============================================================================

@pytest.mark.skipif(not RETRY_AVAILABLE, reason="Retry module not available")
class TestRetryDecorator:
    """Tests for @with_retry decorator"""

    def test_sync_decorator(self):
        """Test decorator works with sync functions"""
        call_count = 0

        @with_retry(config=RetryConfig(max_retries=2, base_delay=0.01, jitter=False))
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("fail")
            return "success"

        result = failing_func()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_decorator(self):
        """Test decorator works with async functions"""
        call_count = 0

        @with_retry(config=RetryConfig(max_retries=2, base_delay=0.01, jitter=False))
        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("fail")
            return "success"

        result = await failing_func()

        assert result == "success"
        assert call_count == 2

    def test_decorator_preserves_function_name(self):
        """Test decorator preserves function name"""
        @with_retry()
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_decorator_with_arguments(self):
        """Test decorated function works with arguments"""
        @with_retry(config=RetryConfig(max_retries=1, base_delay=0.01))
        def add(a, b):
            return a + b

        result = add(2, 3)
        assert result == 5


# ============================================================================
# Test: RetryContext
# ============================================================================

@pytest.mark.skipif(not RETRY_AVAILABLE, reason="Retry module not available")
class TestRetryContext:
    """Tests for RetryContext context manager"""

    def test_sync_context_tracks_attempts(self):
        """Test sync context tracks attempts"""
        call_count = 0

        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("fail")
            return "success"

        config = RetryConfig(max_retries=3, base_delay=0.01, jitter=False)

        with RetryContext(config=config) as ctx:
            result = ctx.execute_sync(failing_func)

        assert result == "success"
        assert ctx.attempts == 3
        assert ctx.succeeded is True
        assert ctx.had_retries is True
        assert len(ctx.exceptions) == 2

    @pytest.mark.asyncio
    async def test_async_context_tracks_attempts(self):
        """Test async context tracks attempts"""
        call_count = 0

        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("fail")
            return "success"

        config = RetryConfig(max_retries=3, base_delay=0.01, jitter=False)

        async with RetryContext(config=config) as ctx:
            result = await ctx.execute_async(failing_func)

        assert result == "success"
        assert ctx.attempts == 2
        assert ctx.succeeded is True

    def test_context_tracks_total_delay(self):
        """Test context tracks total delay"""
        def failing_func():
            raise ConnectionError("fail")

        config = RetryConfig(max_retries=2, base_delay=0.1, jitter=False)

        with RetryContext(config=config) as ctx:
            try:
                ctx.execute_sync(failing_func)
            except ConnectionError:
                pass

        # Should have waited: 0.1 (first retry) + 0.2 (second retry) = 0.3
        assert ctx.total_delay >= 0.3


# ============================================================================
# Test: Status Codes
# ============================================================================

@pytest.mark.skipif(not RETRY_AVAILABLE, reason="Retry module not available")
class TestRetryableStatusCodes:
    """Tests for retryable status codes"""

    @pytest.mark.parametrize("status_code", [
        408, 429, 500, 502, 503, 504, 520, 522, 524
    ])
    def test_retryable_status_codes(self, status_code):
        """Test all defined retryable status codes"""
        assert status_code in RETRYABLE_STATUS_CODES

    @pytest.mark.parametrize("status_code", [
        200, 201, 400, 401, 403, 404
    ])
    def test_non_retryable_status_codes(self, status_code):
        """Test common non-retryable status codes"""
        assert status_code not in RETRYABLE_STATUS_CODES


# ============================================================================
# Test: Integration Scenarios
# ============================================================================

@pytest.mark.skipif(not RETRY_AVAILABLE, reason="Retry module not available")
class TestIntegrationScenarios:
    """Integration tests for retry scenarios"""

    @pytest.mark.asyncio
    async def test_llm_rate_limit_scenario(self):
        """Test handling of LLM rate limit scenario"""
        call_count = 0

        async def mock_llm_call():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Rate limit exceeded. Please retry after 1 second.")
            return {"response": "Hello!"}

        config = RetryConfig(max_retries=3, base_delay=0.01, jitter=False)
        result = await retry_async(mock_llm_call, config=config)

        assert result == {"response": "Hello!"}
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_transient_network_error_scenario(self):
        """Test handling of transient network errors"""
        call_count = 0

        async def mock_api_call():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Connection reset by peer")
            return {"status": "ok"}

        config = RetryConfig(max_retries=2, base_delay=0.01)
        result = await retry_async(mock_api_call, config=config)

        assert result == {"status": "ok"}
        assert call_count == 2

    def test_no_retry_with_zero_max_retries(self):
        """Test that NO_RETRY_CONFIG doesn't retry"""
        mock_func = Mock(side_effect=ConnectionError("fail"))

        with pytest.raises(ConnectionError):
            retry_sync(mock_func, config=NO_RETRY_CONFIG)

        assert mock_func.call_count == 1


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
