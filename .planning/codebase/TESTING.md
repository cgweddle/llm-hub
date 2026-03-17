# Testing Patterns

**Analysis Date:** 2026-03-15

## Test Framework

**Runner:**
- pytest (Python testing framework)
- Config: `tests/conftest.py` — minimal configuration, enables pytest-asyncio plugin
- No formal config file (pytest.ini or setup.cfg) detected; uses defaults

**Assertion Library:**
- pytest built-in assertions (no external assertion library)

**Run Commands:**
```bash
pytest tests/                              # Run all tests
pytest tests/test_pydanticai_components.py -v  # Run specific test file with verbose output
pytest tests/test_retry.py::TestRetryDecorator::test_successful_first_attempt -v  # Run specific test
pytest tests/ -k "test_agent"              # Run tests matching pattern
```

## Test File Organization

**Location:**
- Tests are co-located in `tests/` directory at project root (separate from source)
- Python tests: `tests/test_*.py`
- No frontend tests detected (no Jest, Vitest, or similar configured in `frontend/package.json`)

**Naming:**
- Test files: `test_<module>.py` (e.g., `test_pydanticai_components.py`, `test_retry.py`, `test_google_adk.py`)
- Test classes: `Test<Feature>` (e.g., `TestRetryConfig`, `TestPydanticAIToolConverter`, `TestAgentExecutor`)
- Test methods: `test_<behavior>` (e.g., `test_default_config_values`, `test_json_schema_to_pydantic_simple`)

**Structure:**
```
tests/
├── conftest.py                      # Shared pytest configuration and fixtures
├── test_pydanticai_components.py    # ~1005 lines — comprehensive integration tests
├── test_retry.py                    # Retry utility tests
├── test_google_adk.py               # Google ADK agent creation tests
└── test_python_script_tool_factory.py # Tool factory tests
```

## Test Structure

**Suite Organization (from `tests/test_pydanticai_components.py`):**
```python
"""
Unit tests for PydanticAI integration components.

Tests cover:
- API models (AgentCreate, AgentResponse with output_schema)
- Database functions (create_agent with output_schema)
- PydanticAI Tool Converter (JSON schema to Pydantic model conversion)
- PydanticAI Agent Factory (agent creation and validation)
- Agent Executor (execution routing)

Run with: pytest tests/test_pydanticai_components.py -v
"""

@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class TestAPIModels:
    """Tests for API Pydantic models with output_schema field."""

    def test_agent_create_with_output_schema(self):
        """Test AgentCreate model accepts output_schema"""
        # Arrange: create model definition
        AgentCreate = self._create_agent_create_model()
        output_schema = {...}

        # Act: instantiate model
        agent_data = AgentCreate(...)

        # Assert: verify behavior
        assert agent_data.output_schema == output_schema
```

**Patterns:**
- Setup: Helper methods like `_create_agent_create_model()` for reusable test fixtures
- Arrange-Act-Assert: Clear separation of test phases with comments
- Async tests: `@pytest.mark.asyncio` marker on async test methods (e.g., in `test_pydanticai_components.py`)

## Mocking

**Framework:** `unittest.mock` (Python standard library)

**Patterns (from `tests/test_pydanticai_components.py`):**

Mock classes created inline for common database objects:
```python
class MockSession:
    """Mock database session for testing"""
    def __init__(self):
        self.added = []
        self.committed = False

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.committed = True

    def refresh(self, obj):
        if not hasattr(obj, 'id'):
            obj.id = 1

    def close(self):
        pass

    def query(self, model):
        return MockQuery()
```

Patch decorator for function mocking:
```python
@patch('factories.pydanticai_agent_factory.get_agent_by_id')
@patch('factories.pydanticai_agent_factory.get_llm_config_by_name')
def test_validate_agent_config_valid(self, mock_get_llm_config, mock_get_agent):
    mock_agent = MockAgent(id=1, agent_type="pydanticai", llm_config={"model_name": "test_model"})
    mock_get_agent.return_value = mock_agent
    mock_get_llm_config.return_value = {"provider": "openai", "model": "gpt-4"}

    factory = PydanticAIAgentFactory(session=MockSession())
    validation = factory.validate_agent_config(1)

    assert validation["valid"] is True
```

Async mocks:
```python
@pytest.mark.asyncio
@patch('executors.agent_executor.get_agent_by_id')
async def test_execute_agent_not_found(self, mock_get_agent):
    mock_get_agent.return_value = None
    # Test async execution...
```

**What to Mock:**
- External dependencies (LLM APIs, database sessions)
- Factory method results for isolated unit testing
- Functions with side effects (HTTP calls, file I/O)

**What NOT to Mock:**
- Core business logic (Pydantic schema conversion, retry logic)
- Data model creation and validation
- Simple utility functions

## Fixtures and Factories

**Test Data (from `tests/conftest.py` and inline in test files):**

Shared fixture in conftest:
```python
@pytest.fixture
def mock_session():
    """Provide a mock database session for tests"""
    class MockSession:
        def __init__(self):
            self.added = []
            self.committed = False
        # ... methods ...
    return MockSession()
```

Inline test data factories (from `test_pydanticai_components.py`):
```python
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
        # ... initialization ...

class MockAgent:
    """Mock Agent database object"""
    def __init__(self, id: int = 1, name: str = "test_agent", ...):
        # ... initialization ...
```

**Location:**
- Shared fixtures: `tests/conftest.py`
- Test-specific mocks: Defined inline at top of test class (e.g., `MockSession`, `MockTool`)
- Schema fixtures: Embedded in test methods as dicts (e.g., input/output schemas)

## Coverage

**Requirements:** No enforced coverage requirement detected (no `.coveragerc` or pytest config)

**View Coverage:**
```bash
pytest tests/ --cov=src --cov-report=html  # Generate HTML coverage report
pytest tests/ --cov=src --cov-report=term  # Show coverage in terminal
```

## Test Types

**Unit Tests:**
- Scope: Individual functions, classes, and methods
- Approach: Isolated testing with mocks for external dependencies
- Examples: `test_default_config_values`, `test_calculate_delay_exponential_backoff`, `test_json_schema_to_pydantic_simple`
- Located in: `tests/test_*.py` files

**Integration Tests:**
- Scope: Multiple components working together
- Approach: Real object construction without mocks where possible
- Examples: `TestIntegrationScenarios` class in `test_pydanticai_components.py`
  - `test_tool_conversion_and_execution` — tool conversion → execution flow
  - `test_structured_output_schema_conversion` — schema conversion for agent responses
  - `test_tool_with_array_input` — tool handling array inputs

**E2E Tests:**
- Framework: Not used
- No end-to-end tests detected (no Cypress, Playwright, or similar in frontend)

## Common Patterns

**Async Testing (from `test_pydanticai_components.py`):**
```python
@pytest.mark.asyncio
@patch('executors.agent_executor.get_agent_by_id')
async def test_execute_agent_not_found(self, mock_get_agent):
    """Test execution fails when agent not found"""
    mock_get_agent.return_value = None

    session = MockSession()
    executor = AgentExecutor(session)

    with pytest.raises(ValueError, match="not found"):
        await executor.execute_agent(
            agent_id=999,
            user_id=1,
            input_data="Test"
        )
```

**Error Testing (from `test_pydanticai_components.py`):**
```python
def test_json_schema_to_pydantic_required_fields(self):
    """Test that required fields are enforced"""
    converter = PydanticAIToolConverter()

    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"}
        },
        "required": ["query"]
    }

    Model = converter.json_schema_to_pydantic(schema, "SearchInput")

    # Required field must be provided
    with pytest.raises(ValidationError):
        Model(limit=10)  # Missing required 'query'

    # Works with required field
    instance = Model(query="test")
    assert instance.query == "test"
```

**Parametrized Testing (from `test_pydanticai_components.py`):**
```python
@pytest.mark.parametrize("json_type,expected_python_type", [
    ("string", str),
    ("integer", int),
    ("number", float),
    ("boolean", bool),
])
def test_type_mapping(self, json_type, expected_python_type):
    """Test JSON type to Python type mapping for various types"""
    converter = PydanticAIToolConverter()
    result = converter._json_type_to_python_type({"type": json_type})
    assert result == expected_python_type
```

**Conditional Test Skipping (from all test files):**
```python
# At module level, detect if dependencies are available
try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

# Skip entire test class if dependency missing
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class TestAPIModels:
    def test_agent_create_with_output_schema(self):
        # ...
```

## Test Organization by Layer

**Database Tests:**
- Mocking: `MockSession` with `add()`, `commit()`, `refresh()`, `query()` methods
- Fixtures: In `tests/conftest.py` and inline
- Files: `tests/test_pydanticai_components.py::TestDatabaseFunctions`

**Factory Tests:**
- Mocking: Mock agent/tool configs, mock database lookups
- Patterns: Patch external imports with `@patch` decorator
- Files: `tests/test_pydanticai_components.py::TestPydanticAIAgentFactory`, `tests/test_python_script_tool_factory.py::TestPythonScriptToolFactory`

**Converter Tests:**
- Mocking: Minimal (converters are pure functions when possible)
- Patterns: Direct instantiation of converter with test data
- Files: `tests/test_pydanticai_components.py::TestPydanticAIToolConverter`

**Executor Tests:**
- Mocking: Mock database queries, mock agent/tool retrieval
- Async support: `@pytest.mark.asyncio` for async executor tests
- Files: `tests/test_pydanticai_components.py::TestAgentExecutor`

**Utility Tests:**
- Approach: Pure function testing, no mocks needed
- Files: `tests/test_retry.py::TestRetryConfig`, `tests/test_retry.py::TestRetryDecorator`

## Test Coverage Analysis

**Well-tested areas:**
- `src/converters/pydanticai_tool_converter.py` — Comprehensive coverage in `tests/test_pydanticai_components.py`
- `src/utils/retry.py` — Dedicated test file `tests/test_retry.py` with config and behavior tests
- `src/factories/pydanticai_agent_factory.py` — Validation and creation tested in `tests/test_pydanticai_components.py`
- `src/factories/python_script_tool_factory.py` — Parser and schema generation tested in `tests/test_python_script_tool_factory.py`

**Under-tested areas:**
- `src/executors/flow_executor.py` — No dedicated test file detected
- `src/api/backend.py` — No API endpoint tests detected (no FastAPI TestClient usage)
- `src/tools/` — Tool-specific modules lack dedicated tests
- `src/validate/tool_compatibility.py` — No dedicated test file
- Frontend: No test files in `frontend/` directory

---

*Testing analysis: 2026-03-15*
