# Testing Patterns

**Analysis Date:** 2026-03-15

## Test Framework

**Runner:**
- pytest (Python test runner)
- Config: No explicit `pytest.ini` file; relies on pytest defaults
- Located at: `/Users/chris/Documents/repos/llm-hub/tests/` directory

**Assertion Library:**
- Python `assert` statements
- pytest fixtures for setup/teardown

**Run Commands:**
```bash
pytest tests/                           # Run all tests
pytest tests/test_retry.py              # Run single test file
pytest tests/test_retry.py::TestRetryConfig::test_default_config_values -v  # Run specific test
pytest tests/ -v --tb=short             # Verbose with short traceback
```

## Test File Organization

**Location:**
- Python tests: co-located in dedicated `tests/` directory at project root
- Path structure: `/Users/chris/Documents/repos/llm-hub/tests/test_*.py`
- No frontend tests currently present

**Naming:**
- Test files: `test_*.py` (e.g., `test_retry.py`, `test_pydanticai_components.py`)
- Test classes: `Test*` (e.g., `TestRetryConfig`, `TestAsyncRetry`)
- Test methods: `test_*` (e.g., `test_default_config_values`)

**Structure:**
```
tests/
├── conftest.py                    # Shared pytest fixtures
├── test_retry.py                  # Retry utility tests
├── test_pydanticai_components.py  # PydanticAI integration tests
├── test_python_script_tool_factory.py  # Tool factory tests
├── test_python_agent_tools.py     # Agent tools tests
└── test_google_adk.py             # Google ADK tests
```

## Test Structure

**Suite Organization:**

```python
# tests/test_retry.py - Typical test class structure
@pytest.mark.skipif(not RETRY_AVAILABLE, reason="Retry module not available")
class TestRetryConfig:
    """Tests for RetryConfig configuration"""

    def test_default_config_values(self):
        """Test default configuration values"""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
```

**Patterns:**

- **Setup/Teardown:** Not used; test isolation via fresh fixture instances
- **Fixtures:** Defined in `conftest.py` for shared setup
  ```python
  # tests/conftest.py
  @pytest.fixture
  def mock_session():
      """Provide a mock database session for tests"""
      class MockSession:
          def __init__(self):
              self.added = []
              self.committed = False

          def add(self, obj):
              self.added.append(obj)

          def commit(self):
              self.committed = True

      return MockSession()
  ```

- **Test isolation:** Each test method is independent; no shared state between tests
- **Assertions:** Plain Python assertions with descriptive failure messages
- **Async tests:** Marked with `@pytest.mark.asyncio` decorator
  ```python
  @pytest.mark.asyncio
  async def test_async_success_on_first_attempt(self):
      """Test async function succeeds on first attempt"""
      mock_func = AsyncMock(return_value="success")

      result = await retry_async(mock_func, config=RetryConfig(max_retries=3))

      assert result == "success"
      assert mock_func.call_count == 1
  ```

## Mocking

**Framework:** unittest.mock (standard library)

**Patterns:**

```python
# tests/test_retry.py - Mocking patterns
from unittest.mock import Mock, AsyncMock, patch

# Simple mock return value
mock_func = Mock(return_value="success")

# Mock with side effects (sequential return values)
mock_func = Mock(side_effect=[ConnectionError("fail"), "success"])

# Async mock
mock_func = AsyncMock(return_value="success")

# Patch external module
@patch('module.function')
def test_with_patch(mock_function):
    # test code
    pass

# Access call counts and arguments
assert mock_func.call_count == 2
assert on_retry.call_count == 2  # on_retry callback called twice
```

**What to Mock:**
- External API calls and network requests
- Database operations (use `mock_session` fixture from `conftest.py`)
- LLM provider calls
- File system operations
- Time-dependent operations (use `base_delay=0.01` for fast testing)

**What NOT to Mock:**
- Core business logic (e.g., exponential backoff calculation, validation functions)
- Internal module functions (test integration instead)
- Configuration objects
- Data structure operations

## Fixtures and Factories

**Test Data:**

Mock objects defined inline for tests:
```python
# tests/test_pydanticai_components.py
class MockTool:
    """Mock Tool database object"""
    def __init__(
        self,
        id: int = 1,
        name: str = "test_tool",
        description: str = "A test tool",
        tool_type: str = "function",
        input_schema: Dict = None,
        output_schema: Dict = None,
        function_code: str = None,
        main_function: str = "test_func",
        helper_functions: Dict = None
    ):
        self.id = id
        self.name = name
        # ...

class MockAgent:
    """Mock Agent database object"""
    def __init__(
        self,
        id: int = 1,
        name: str = "test_agent",
        # ...
    ):
        # ...
```

**Location:**
- Fixtures: `tests/conftest.py` for shared fixtures (e.g., `mock_session`)
- Mock classes: Defined at top of test files (e.g., `MockTool`, `MockAgent` in `test_pydanticai_components.py`)
- Test data: Inline within test methods or as class attributes

## Coverage

**Requirements:** No explicit coverage target enforced; coverage measurement available via pytest-cov if installed

**View Coverage:**
```bash
pytest tests/ --cov=src --cov-report=html
# View coverage report in htmlcov/index.html
```

## Test Types

**Unit Tests:**
- Scope: Single function or small module behavior
- Approach: Mock external dependencies, test isolated logic
- Examples:
  - `TestRetryConfig`: Tests `RetryConfig.calculate_delay()` with various inputs
  - `TestSyncRetry`: Tests `retry_sync()` function with mock functions
  - `test_default_config_values`: Verifies configuration initialization

**Integration Tests:**
- Scope: Multiple components working together
- Approach: Mix of real and mocked components; test workflows
- Examples:
  - `TestIntegrationScenarios`: Tests retry logic with simulated LLM rate limits and network errors
  - PydanticAI component tests: Test converter → factory → executor chain

**E2E Tests:**
- Framework: Not formally structured; some ad-hoc integration tests present
- Example: `/Users/chris/Documents/repos/llm-hub/test_pydanticai_integration.py` (root level) tests end-to-end PydanticAI agent execution

## Common Patterns

**Async Testing:**

```python
# tests/test_retry.py
@pytest.mark.asyncio
async def test_async_success_on_first_attempt(self):
    """Test async function succeeds on first attempt"""
    mock_func = AsyncMock(return_value="success")

    result = await retry_async(mock_func, config=RetryConfig(max_retries=3))

    assert result == "success"
    assert mock_func.call_count == 1
```

**Error Testing:**

```python
# tests/test_retry.py
def test_exhausts_retries(self):
    """Test function exhausts all retries"""
    mock_func = Mock(side_effect=ConnectionError("always fail"))

    config = RetryConfig(max_retries=2, base_delay=0.01, jitter=False)

    with pytest.raises(ConnectionError):
        retry_sync(mock_func, config=config)

    assert mock_func.call_count == 3  # Initial + 2 retries
```

**Parametrized Testing:**

```python
# tests/test_retry.py
@pytest.mark.parametrize("status_code", [
    408, 429, 500, 502, 503, 504, 520, 522, 524
])
def test_retryable_status_codes(self, status_code):
    """Test all defined retryable status codes"""
    assert status_code in RETRYABLE_STATUS_CODES
```

**Configuration Testing:**

```python
# tests/test_retry.py
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
```

## Test Organization by Concern

**RetryConfig Tests** (`tests/test_retry.py`):
- Configuration validation and defaults
- Exponential backoff calculation
- Jitter application
- Max delay capping

**Sync/Async Retry Tests**:
- Success on first attempt
- Success after retry
- Retry exhaustion
- Non-retryable exception handling
- Callback execution
- Delay between retries

**Decorator Tests**:
- Works with sync functions
- Works with async functions
- Preserves function name
- Works with function arguments

**Context Manager Tests** (`RetryContext`):
- Tracks attempt count
- Tracks total delay
- Tracks exception list
- Success and retry flags

**Integration Scenarios**:
- LLM rate limit handling
- Transient network error handling
- No-retry configuration behavior

## Pytest Configuration

**conftest.py** (`tests/conftest.py`):
- Configures `pytest-asyncio` plugin for async test support
- Registers custom markers (e.g., `@pytest.mark.asyncio`)
- Provides `mock_session` fixture for database mocking

**Import Setup Pattern:**
Tests handle import failures gracefully with conditional test skipping:
```python
# tests/test_retry.py
try:
    from utils.retry import (
        RetryConfig,
        retry_async,
        # ...
    )
    RETRY_AVAILABLE = True
except ImportError:
    RETRY_AVAILABLE = False

@pytest.mark.skipif(not RETRY_AVAILABLE, reason="Retry module not available")
class TestRetryConfig:
    # tests...
```

This ensures tests are skipped if dependencies aren't available, rather than failing hard.

---

*Testing analysis: 2026-03-15*
