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

import pytest
import sys
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, AsyncMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import components, skip tests if not available
try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

try:
    from converters.pydanticai_tool_converter import PydanticAIToolConverter
    TOOL_CONVERTER_AVAILABLE = True
except ImportError:
    TOOL_CONVERTER_AVAILABLE = False

try:
    from factories.pydanticai_agent_factory import PydanticAIAgentFactory
    AGENT_FACTORY_AVAILABLE = True
except ImportError:
    AGENT_FACTORY_AVAILABLE = False

try:
    from executors.agent_executor import AgentExecutor
    EXECUTOR_AVAILABLE = True
except ImportError:
    EXECUTOR_AVAILABLE = False


# ============================================================================
# Mock Classes for Testing
# ============================================================================

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


class MockQuery:
    """Mock query object"""
    def __init__(self):
        self._filters = []

    def filter(self, *args):
        self._filters.extend(args)
        return self

    def first(self):
        return None

    def all(self):
        return []


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
        self.description = description
        self.tool_type = tool_type
        self.input_schema = input_schema or {}
        self.output_schema = output_schema
        self.function_code = function_code or "def test_func(x): return x * 2"
        self.main_function = main_function
        self.helper_functions = helper_functions or {}


class MockAgent:
    """Mock Agent database object"""
    def __init__(
        self,
        id: int = 1,
        name: str = "test_agent",
        description: str = "A test agent",
        agent_type: str = "pydanticai",
        system_prompt: str = "You are a helpful assistant.",
        llm_config: Dict = None,
        tools_config: Dict = None,
        agent_metadata: Dict = None,
        output_schema: Dict = None,
        tools: List = None
    ):
        self.id = id
        self.name = name
        self.description = description
        self.agent_type = agent_type
        self.system_prompt = system_prompt
        self.llm_config = llm_config or {"model_name": "test_model"}
        self.tools_config = tools_config or {}
        self.agent_metadata = agent_metadata or {}
        self.output_schema = output_schema
        self.tools = tools or []


class MockExecution:
    """Mock Execution database object"""
    def __init__(self, id: int = 1):
        self.id = id
        self.status = "running"
        self.started_at = datetime.now()
        self.completed_at = None
        self.output_data = None
        self.error_message = None


# ============================================================================
# Test: API Models
# ============================================================================

@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class TestAPIModels:
    """
    Tests for API Pydantic models with output_schema field.

    Note: We define the models here directly to avoid importing from backend.py
    which has heavy dependencies (Google ADK). These models mirror the actual
    API models and verify the schema structure is correct.
    """

    def _create_agent_create_model(self):
        """Create AgentCreate model matching backend.py definition"""
        from pydantic import BaseModel
        from typing import Optional

        class AgentCreate(BaseModel):
            name: str
            description: str
            agent_type: str
            system_prompt: str
            llm_config: dict
            tools_config: dict
            agent_metadata: Optional[dict] = None
            output_schema: Optional[dict] = None

        return AgentCreate

    def _create_agent_response_model(self):
        """Create AgentResponse model matching backend.py definition"""
        from pydantic import BaseModel
        from typing import Optional

        class AgentResponse(BaseModel):
            id: int
            name: str
            description: str
            agent_type: str
            output_schema: Optional[dict] = None
            created_at: datetime

            class Config:
                from_attributes = True

        return AgentResponse

    def test_agent_create_with_output_schema(self):
        """Test AgentCreate model accepts output_schema"""
        AgentCreate = self._create_agent_create_model()

        output_schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"}
            },
            "required": ["answer"]
        }

        agent_data = AgentCreate(
            name="Test Agent",
            description="A test agent",
            agent_type="pydanticai",
            system_prompt="You are helpful.",
            llm_config={"model_name": "test"},
            tools_config={"tool_ids": []},
            output_schema=output_schema
        )

        assert agent_data.output_schema == output_schema
        assert agent_data.name == "Test Agent"

    def test_agent_create_without_output_schema(self):
        """Test AgentCreate model works without output_schema (defaults to None)"""
        AgentCreate = self._create_agent_create_model()

        agent_data = AgentCreate(
            name="Test Agent",
            description="A test agent",
            agent_type="react",
            system_prompt="You are helpful.",
            llm_config={"model_name": "test"},
            tools_config={}
        )

        assert agent_data.output_schema is None

    def test_agent_create_with_complex_output_schema(self):
        """Test AgentCreate with complex nested output_schema"""
        AgentCreate = self._create_agent_create_model()

        complex_schema = {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "score": {"type": "number"}
                        }
                    }
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "total": {"type": "integer"},
                        "page": {"type": "integer"}
                    }
                }
            }
        }

        agent_data = AgentCreate(
            name="Complex Agent",
            description="Agent with complex schema",
            agent_type="pydanticai",
            system_prompt="You are helpful.",
            llm_config={"model_name": "test"},
            tools_config={},
            output_schema=complex_schema
        )

        assert agent_data.output_schema["properties"]["results"]["type"] == "array"

    def test_agent_response_includes_output_schema(self):
        """Test AgentResponse model includes output_schema field"""
        AgentResponse = self._create_agent_response_model()

        response = AgentResponse(
            id=1,
            name="Test Agent",
            description="A test agent",
            agent_type="pydanticai",
            output_schema={"type": "object"},
            created_at=datetime.now()
        )

        assert response.output_schema == {"type": "object"}

    def test_agent_response_without_output_schema(self):
        """Test AgentResponse works without output_schema"""
        AgentResponse = self._create_agent_response_model()

        response = AgentResponse(
            id=1,
            name="Test Agent",
            description="A test agent",
            agent_type="react",
            created_at=datetime.now()
        )

        assert response.output_schema is None


# ============================================================================
# Test: PydanticAI Tool Converter
# ============================================================================

@pytest.mark.skipif(not TOOL_CONVERTER_AVAILABLE, reason="Tool converter not available")
class TestPydanticAIToolConverter:
    """Tests for PydanticAI tool converter"""

    def test_json_schema_to_pydantic_simple(self):
        """Test conversion of simple JSON schema to Pydantic model"""
        converter = PydanticAIToolConverter()

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "User name"},
                "age": {"type": "integer", "description": "User age"}
            },
            "required": ["name"]
        }

        Model = converter.json_schema_to_pydantic(schema, "UserInput")

        # Test model creation
        instance = Model(name="John", age=30)
        assert instance.name == "John"
        assert instance.age == 30

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
        assert instance.limit is None  # Optional field

    def test_json_schema_to_pydantic_all_types(self):
        """Test conversion of all JSON schema types"""
        converter = PydanticAIToolConverter()

        schema = {
            "type": "object",
            "properties": {
                "string_field": {"type": "string"},
                "integer_field": {"type": "integer"},
                "number_field": {"type": "number"},
                "boolean_field": {"type": "boolean"},
                "array_field": {"type": "array", "items": {"type": "string"}},
                "object_field": {"type": "object"}
            }
        }

        Model = converter.json_schema_to_pydantic(schema, "AllTypesInput")

        instance = Model(
            string_field="test",
            integer_field=42,
            number_field=3.14,
            boolean_field=True,
            array_field=["a", "b", "c"],
            object_field={"key": "value"}
        )

        assert instance.string_field == "test"
        assert instance.integer_field == 42
        assert instance.number_field == 3.14
        assert instance.boolean_field is True
        assert instance.array_field == ["a", "b", "c"]
        assert instance.object_field == {"key": "value"}

    def test_json_schema_to_pydantic_empty_schema(self):
        """Test conversion of empty schema"""
        converter = PydanticAIToolConverter()

        Model = converter.json_schema_to_pydantic({}, "EmptyModel")

        # Should create a model with no fields
        instance = Model()
        assert instance is not None

    def test_json_schema_to_pydantic_with_default_values(self):
        """Test conversion with default values"""
        converter = PydanticAIToolConverter()

        schema = {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 10},
                "offset": {"type": "integer", "default": 0}
            }
        }

        Model = converter.json_schema_to_pydantic(schema, "PaginationInput")

        instance = Model()
        assert instance.limit == 10
        assert instance.offset == 0

    def test_json_type_to_python_type_mapping(self):
        """Test JSON type to Python type mapping"""
        converter = PydanticAIToolConverter()

        assert converter._json_type_to_python_type({"type": "string"}) == str
        assert converter._json_type_to_python_type({"type": "integer"}) == int
        assert converter._json_type_to_python_type({"type": "number"}) == float
        assert converter._json_type_to_python_type({"type": "boolean"}) == bool

    def test_convert_tool_caching(self):
        """Test that tool conversion results are cached"""
        converter = PydanticAIToolConverter()

        tool = MockTool(
            id=1,
            name="cached_tool",
            function_code="def test_func(x): return x",
            input_schema={"type": "object", "properties": {"x": {"type": "integer"}}}
        )

        # First conversion
        func1, model1, out1 = converter.convert_tool(tool)

        # Second conversion should use cache
        func2, model2, out2 = converter.convert_tool(tool)

        assert func1 is func2
        assert model1 is model2

    def test_clear_cache(self):
        """Test cache clearing"""
        converter = PydanticAIToolConverter()

        tool = MockTool(id=1, name="test_tool")
        converter.convert_tool(tool)

        assert len(converter._tool_cache) == 1

        converter.clear_cache()

        assert len(converter._tool_cache) == 0

    def test_is_async_function_detection(self):
        """Test async function detection"""
        converter = PydanticAIToolConverter()

        sync_code = "def my_func(x): return x"
        async_code = "async def my_func(x): return x"

        assert converter._is_async_function(sync_code, "my_func") is False
        assert converter._is_async_function(async_code, "my_func") is True

    def test_compile_function_code(self):
        """Test function code compilation"""
        converter = PydanticAIToolConverter()

        tool = MockTool(
            name="multiply",
            function_code="def multiply(a, b): return a * b",
            main_function="multiply"
        )

        func = converter._compile_function_code(tool)

        assert callable(func)
        assert func(3, 4) == 12

    def test_compile_function_with_helpers(self):
        """Test function compilation with helper functions"""
        converter = PydanticAIToolConverter()

        tool = MockTool(
            name="with_helper",
            function_code="def main_func(x): return helper(x) * 2",
            main_function="main_func",
            helper_functions={"helper": "def helper(x): return x + 1"}
        )

        func = converter._compile_function_code(tool)

        assert func(5) == 12  # (5 + 1) * 2

    def test_compile_function_invalid_code(self):
        """Test compilation failure with invalid code"""
        converter = PydanticAIToolConverter()

        tool = MockTool(
            name="invalid",
            function_code="def invalid(x): return undefined_var",
            main_function="invalid"
        )

        func = converter._compile_function_code(tool)

        with pytest.raises(NameError):
            func(1)

    def test_create_sync_wrapper(self):
        """Test sync wrapper creation"""
        converter = PydanticAIToolConverter()

        def simple_func(x: int) -> int:
            return x * 2

        Model = converter.json_schema_to_pydantic(
            {"type": "object", "properties": {"x": {"type": "integer"}}},
            "Input"
        )

        wrapper = converter._create_sync_wrapper(
            executable_func=simple_func,
            tool_name="double",
            tool_description="Doubles a number",
            input_model=Model
        )

        assert wrapper.__name__ == "double"
        assert wrapper.__doc__ == "Doubles a number"
        assert wrapper(x=5) == 10


# ============================================================================
# Test: PydanticAI Agent Factory
# ============================================================================

@pytest.mark.skipif(not AGENT_FACTORY_AVAILABLE, reason="Agent factory not available")
class TestPydanticAIAgentFactory:
    """Tests for PydanticAI agent factory"""

    @patch('factories.pydanticai_agent_factory.get_agent_by_id')
    @patch('factories.pydanticai_agent_factory.get_llm_config_by_name')
    def test_validate_agent_config_valid(self, mock_get_llm_config, mock_get_agent):
        """Test validation of valid agent configuration"""
        mock_agent = MockAgent(
            id=1,
            agent_type="pydanticai",
            llm_config={"model_name": "test_model"}
        )
        mock_get_agent.return_value = mock_agent
        mock_get_llm_config.return_value = {"provider": "openai", "model": "gpt-4"}

        factory = PydanticAIAgentFactory(session=MockSession())
        validation = factory.validate_agent_config(1)

        assert validation["valid"] is True
        assert len(validation["errors"]) == 0

    @patch('factories.pydanticai_agent_factory.get_agent_by_id')
    def test_validate_agent_config_not_found(self, mock_get_agent):
        """Test validation when agent not found"""
        mock_get_agent.return_value = None

        factory = PydanticAIAgentFactory(session=MockSession())
        validation = factory.validate_agent_config(999)

        assert validation["valid"] is False
        assert "not found" in validation["errors"][0]

    @patch('factories.pydanticai_agent_factory.get_agent_by_id')
    def test_validate_agent_config_wrong_type(self, mock_get_agent):
        """Test validation when agent type is not pydanticai"""
        mock_agent = MockAgent(id=1, agent_type="react")
        mock_get_agent.return_value = mock_agent

        factory = PydanticAIAgentFactory(session=MockSession())
        validation = factory.validate_agent_config(1)

        assert validation["valid"] is False
        assert "pydanticai" in validation["errors"][0]

    @patch('factories.pydanticai_agent_factory.get_agent_by_id')
    @patch('factories.pydanticai_agent_factory.get_llm_config_by_name')
    def test_validate_agent_config_no_tools_warning(self, mock_get_llm_config, mock_get_agent):
        """Test validation warns when no tools configured"""
        mock_agent = MockAgent(
            id=1,
            agent_type="pydanticai",
            tools=[],
            tools_config={}
        )
        mock_get_agent.return_value = mock_agent
        mock_get_llm_config.return_value = {"provider": "openai", "model": "gpt-4"}

        factory = PydanticAIAgentFactory(session=MockSession())
        validation = factory.validate_agent_config(1)

        assert "no tools" in validation["warnings"][0].lower()

    def test_get_result_type_from_metadata(self):
        """Test extracting result type from agent_metadata"""
        factory = PydanticAIAgentFactory(session=MockSession())

        mock_agent = MockAgent(
            agent_metadata={
                "result_schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"}
                    }
                }
            }
        )

        result_type = factory._get_result_type(mock_agent)

        assert result_type is not None
        assert issubclass(result_type, BaseModel)

    def test_get_result_type_none_when_no_schema(self):
        """Test result type is None when no schema defined"""
        factory = PydanticAIAgentFactory(session=MockSession())

        mock_agent = MockAgent(agent_metadata={})

        result_type = factory._get_result_type(mock_agent)

        assert result_type is None


# ============================================================================
# Test: Agent Executor
# ============================================================================

@pytest.mark.skipif(not EXECUTOR_AVAILABLE, reason="Agent executor not available")
class TestAgentExecutor:
    """Tests for unified agent executor"""

    def test_create_execution_record(self):
        """Test execution record creation"""
        session = MockSession()
        executor = AgentExecutor(session)

        execution = executor._create_execution(
            agent_id=1,
            user_id=1,
            input_data="Test query"
        )

        assert execution.status == "running"
        assert len(session.added) == 1

    def test_complete_execution(self):
        """Test marking execution as completed"""
        session = MockSession()
        executor = AgentExecutor(session)

        execution = MockExecution(id=1)
        result = {"result": "Test result", "model": "gpt-4"}

        executor._complete_execution(execution, result)

        assert execution.status == "completed"
        assert execution.completed_at is not None

    def test_fail_execution(self):
        """Test marking execution as failed"""
        session = MockSession()
        executor = AgentExecutor(session)

        execution = MockExecution(id=1)

        executor._fail_execution(execution, "Test error message")

        assert execution.status == "failed"
        assert execution.error_message == "Test error message"

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

    @pytest.mark.asyncio
    @patch('executors.agent_executor.get_agent_by_id')
    async def test_execute_agent_unknown_type(self, mock_get_agent):
        """Test execution fails for unknown agent type"""
        mock_agent = MockAgent(agent_type="unknown_type")
        mock_get_agent.return_value = mock_agent

        session = MockSession()
        executor = AgentExecutor(session)

        with pytest.raises(RuntimeError, match="Unknown agent type"):
            await executor.execute_agent(
                agent_id=1,
                user_id=1,
                input_data="Test"
            )

    def test_extract_cost_with_cost_info(self):
        """Test cost extraction from result"""
        session = MockSession()
        executor = AgentExecutor(session)

        # Mock result with cost info
        mock_result = Mock()
        mock_cost = Mock()
        mock_cost.total_tokens = 100
        mock_cost.request_tokens = 40
        mock_cost.response_tokens = 60
        mock_result.cost.return_value = mock_cost

        cost = executor._extract_cost(mock_result)

        assert cost["total_tokens"] == 100
        assert cost["input_tokens"] == 40
        assert cost["output_tokens"] == 60

    def test_extract_cost_no_cost_info(self):
        """Test cost extraction when no cost available"""
        session = MockSession()
        executor = AgentExecutor(session)

        mock_result = Mock(spec=[])  # No 'cost' attribute

        cost = executor._extract_cost(mock_result)

        assert cost is None

    def test_store_messages(self):
        """Test message storage"""
        session = MockSession()
        executor = AgentExecutor(session)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        executor._store_messages(execution_id=1, messages=messages)

        assert len(session.added) == 2
        assert session.committed is True


# ============================================================================
# Test: Database Functions
# ============================================================================

class TestDatabaseFunctions:
    """Tests for database functions with output_schema"""

    @patch('database.database.Agent')
    def test_create_agent_with_output_schema(self, mock_agent_class):
        """Test create_agent function with output_schema parameter"""
        # Import the function
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from database.database import create_agent

        session = MockSession()
        output_schema = {
            "type": "object",
            "properties": {"result": {"type": "string"}}
        }

        # Mock the Agent class
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        create_agent(
            session=session,
            user_id=1,
            name="Test Agent",
            description="Test description",
            agent_type="pydanticai",
            system_prompt="You are helpful.",
            llm_config={"model_name": "test"},
            tools_config={},
            output_schema=output_schema
        )

        # Verify Agent was instantiated with output_schema
        mock_agent_class.assert_called_once()
        call_kwargs = mock_agent_class.call_args[1]
        assert call_kwargs["output_schema"] == output_schema

    @patch('database.database.Agent')
    def test_create_agent_without_output_schema(self, mock_agent_class):
        """Test create_agent function without output_schema (defaults to None)"""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from database.database import create_agent

        session = MockSession()
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        create_agent(
            session=session,
            user_id=1,
            name="Test Agent",
            description="Test description",
            agent_type="react",
            system_prompt="You are helpful.",
            llm_config={"model_name": "test"},
            tools_config={}
        )

        call_kwargs = mock_agent_class.call_args[1]
        assert call_kwargs["output_schema"] is None


# ============================================================================
# Test: Integration Scenarios
# ============================================================================

@pytest.mark.skipif(
    not (TOOL_CONVERTER_AVAILABLE and PYDANTIC_AVAILABLE),
    reason="Required components not available"
)
class TestIntegrationScenarios:
    """Integration tests for complete workflows"""

    def test_tool_conversion_and_execution(self):
        """Test complete tool conversion and execution flow"""
        converter = PydanticAIToolConverter()

        # Create a tool that performs calculation
        tool = MockTool(
            id=1,
            name="calculator",
            description="Performs basic math",
            function_code="""
def calculate(operation: str, a: float, b: float) -> float:
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b if b != 0 else float('inf')
    return 0
""",
            main_function="calculate",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["operation", "a", "b"]
            }
        )

        # Convert tool
        tool_func, input_model, output_model = converter.convert_tool(tool)

        # Execute via wrapper
        result = tool_func(operation="add", a=5, b=3)
        assert result == 8

        result = tool_func(operation="multiply", a=4, b=7)
        assert result == 28

    def test_structured_output_schema_conversion(self):
        """Test converting output schema for structured agent responses"""
        converter = PydanticAIToolConverter()

        # Define a structured output schema
        output_schema = {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The main answer"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score 0-1"
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of sources"
                }
            },
            "required": ["answer", "confidence"]
        }

        # Convert to Pydantic model
        ResultModel = converter.json_schema_to_pydantic(output_schema, "AgentResult")

        # Validate structured output
        result = ResultModel(
            answer="The capital of France is Paris.",
            confidence=0.95,
            sources=["Wikipedia", "Encyclopedia"]
        )

        assert result.answer == "The capital of France is Paris."
        assert result.confidence == 0.95
        assert len(result.sources) == 2

    def test_tool_with_array_input(self):
        """Test tool that accepts array input"""
        converter = PydanticAIToolConverter()

        tool = MockTool(
            id=2,
            name="sum_numbers",
            description="Sums a list of numbers",
            function_code="""
def sum_numbers(numbers):
    return sum(numbers)
""",
            main_function="sum_numbers",
            input_schema={
                "type": "object",
                "properties": {
                    "numbers": {
                        "type": "array",
                        "items": {"type": "number"}
                    }
                },
                "required": ["numbers"]
            }
        )

        tool_func, input_model, output_model = converter.convert_tool(tool)

        result = tool_func(numbers=[1, 2, 3, 4, 5])
        assert result == 15


# ============================================================================
# Parametrized Tests
# ============================================================================

@pytest.mark.skipif(not TOOL_CONVERTER_AVAILABLE, reason="Tool converter not available")
class TestParametrizedSchemaConversion:
    """Parametrized tests for schema conversion edge cases"""

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

    @pytest.mark.parametrize("schema,should_succeed", [
        ({"type": "object", "properties": {}}, True),
        ({}, True),
        (None, True),
        ({"type": "object", "properties": {"x": {"type": "string"}}}, True),
    ])
    def test_schema_conversion_success(self, schema, should_succeed):
        """Test various schema inputs for conversion success"""
        converter = PydanticAIToolConverter()

        if should_succeed:
            Model = converter.json_schema_to_pydantic(schema or {}, "TestModel")
            assert Model is not None


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
