"""
Test suite for Google ADK Agent Creator
Pytest-compliant tests for the Google ADK implementation
"""

import pytest
import sys
import os

# Add the scripts directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'tools'))

# Mock the google.adk imports for testing without installation
try:
    import google.adk
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False

    # Create mock classes for testing structure
    class MockAgent:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        @classmethod
        def builder(cls):
            return MockAgentBuilder()

    class MockAgentBuilder:
        def name(self, name):
            self._name = name
            return self

        def model(self, model):
            self._model = model
            return self

        def description(self, description):
            self._description = description
            return self

        def instruction(self, instruction):
            self._instruction = instruction
            return self

        def tools(self, *tools):
            self._tools = tools
            return self

        def build(self):
            return MockAgent(
                name=getattr(self, '_name', 'test'),
                model=getattr(self, '_model', 'test-model'),
                description=getattr(self, '_description', 'test description'),
                instruction=getattr(self, '_instruction', 'test instruction'),
                tools=getattr(self, '_tools', [])
            )

    class MockFunctionTool:
        @classmethod
        def create(cls, func, name=None):
            return cls(func=func, name=name)

        def __init__(self, func=None, name=None):
            self.func = func
            self.name = name

    class MockSession:
        @classmethod
        def create(cls):
            return cls()

    # Mock the google.adk module
    sys.modules['google'] = type('module', (), {})()
    sys.modules['google.adk'] = type('module', (), {})()
    sys.modules['google.adk.agents'] = type('module', (), {
        'LlmAgent': MockAgent,
        'BaseAgent': MockAgent,
        'SequentialAgent': MockAgent,
        'ParallelAgent': MockAgent
    })()
    sys.modules['google.adk.tools'] = type('module', (), {
        'BaseTool': object,
        'FunctionTool': MockFunctionTool,
        'AgentTool': object
    })()
    sys.modules['google.adk.agents.session'] = type('module', (), {
        'Session': MockSession
    })()
    sys.modules['google.adk.agents.memory'] = type('module', (), {
        'MemoryService': object
    })()

# Now import our module
from google_adk_agent_creator import (
    ADKTool,
    ADKAgentNode,
    ADKCustomAgent,
    ADKAgentOrchestrator,
    create_adk_tool,
    create_adk_agent,
    create_example_tools,
    create_example_agents,
    create_agent_only_use_tools,
    create_supervisor_node
)


class TestADKTool:
    """Test cases for ADKTool class"""

    def test_tool_creation(self):
        """Test creating an ADK tool"""
        def sample_function(x: int, y: int) -> int:
            return x * y

        tool = ADKTool("multiply", "Multiply two numbers", sample_function)

        assert tool.name == "multiply"
        assert tool.description == "Multiply two numbers"
        assert tool.func == sample_function

    def test_tool_invoke(self):
        """Test tool execution"""
        def add_function(a: int, b: int) -> int:
            return a + b

        tool = ADKTool("add", "Add two numbers", add_function)
        result = tool.invoke(a=5, b=3)

        assert result == 8

    def test_tool_invoke_with_error(self):
        """Test tool execution with error handling"""
        def error_function():
            raise ValueError("Test error")

        tool = ADKTool("error", "Error function", error_function)
        result = tool.invoke()

        assert isinstance(result, dict)
        assert "error" in result
        assert result["status"] == "failed"

    def test_create_function_tool(self):
        """Test creating ADK function tool"""
        def test_function():
            return "test"

        tool = ADKTool("test", "Test function", test_function)
        function_tool = tool.create_function_tool()

        assert function_tool is not None
        assert hasattr(function_tool, 'func')


class TestADKAgentNode:
    """Test cases for ADKAgentNode class"""

    def test_agent_node_creation(self):
        """Test creating an ADK agent node"""
        node = ADKAgentNode(
            name="test_agent",
            system_prompt="You are a test agent",
            description="Test agent description"
        )

        assert node.name == "test_agent"
        assert node.system_prompt == "You are a test agent"
        assert node.description == "Test agent description"
        assert node.model == "gemini-2.0-flash"  # default

    def test_agent_node_with_tools(self):
        """Test creating an agent node with tools"""
        def test_tool():
            return "test result"

        tool = ADKTool("test", "Test tool", test_tool)
        node = ADKAgentNode(
            name="tool_agent",
            tools=[tool]
        )

        assert len(node.tools) == 1
        assert node.tools[0].name == "test"


class TestADKCustomAgent:
    """Test cases for ADKCustomAgent class"""

    def test_custom_agent_creation(self):
        """Test creating a custom ADK agent"""
        agent = ADKCustomAgent(
            name="custom_agent",
            description="A custom test agent"
        )

        assert agent.name == "custom_agent"
        assert agent.description == "A custom test agent"
        assert agent.model == "gemini-2.0-flash"
        assert agent.agent_type == "llm"

    def test_custom_agent_with_tools(self):
        """Test creating agent with tools"""
        def math_tool(a: int, b: int) -> int:
            return a + b

        tool = ADKTool("math", "Math tool", math_tool)
        agent = ADKCustomAgent(
            name="math_agent",
            description="Math agent",
            tools=[tool]
        )

        assert len(agent.tools) == 1
        assert agent.tools[0].name == "math"

    def test_create_instruction(self):
        """Test instruction generation"""
        def test_tool():
            return "test"

        tool = ADKTool("test", "Test tool", test_tool)
        agent = ADKCustomAgent(
            name="test_agent",
            description="test agent",
            tools=[tool]
        )

        instruction = agent.create_instruction()

        assert "test agent" in instruction
        assert "test: Test tool" in instruction

    def test_agent_types(self):
        """Test different agent types"""
        for agent_type in ["llm", "sequential", "parallel"]:
            agent = ADKCustomAgent(
                name=f"{agent_type}_agent",
                description=f"A {agent_type} agent",
                agent_type=agent_type
            )
            assert agent.agent_type == agent_type


class TestADKAgentOrchestrator:
    """Test cases for ADKAgentOrchestrator class"""

    def test_orchestrator_creation(self):
        """Test creating an orchestrator"""
        agent1 = ADKCustomAgent("agent1", "First agent")
        agent2 = ADKCustomAgent("agent2", "Second agent")

        orchestrator = ADKAgentOrchestrator([agent1, agent2])

        assert len(orchestrator.agents) == 2
        assert "agent1" in orchestrator.agents
        assert "agent2" in orchestrator.agents

    def test_empty_orchestrator(self):
        """Test creating orchestrator with no agents"""
        orchestrator = ADKAgentOrchestrator([])

        assert len(orchestrator.agents) == 0


class TestUtilityFunctions:
    """Test cases for utility functions"""

    def test_create_adk_tool(self):
        """Test create_adk_tool function"""
        def sample_func(x: int) -> int:
            return x * 2

        tool = create_adk_tool("double", "Double a number", sample_func)

        assert isinstance(tool, ADKTool)
        assert tool.name == "double"
        assert tool.description == "Double a number"
        assert tool.func == sample_func

    def test_create_adk_agent(self):
        """Test create_adk_agent function"""
        agent = create_adk_agent(
            "test_agent",
            "Test agent description",
            model="test-model"
        )

        assert isinstance(agent, ADKCustomAgent)
        assert agent.name == "test_agent"
        assert agent.description == "Test agent description"
        assert agent.model == "test-model"

    def test_create_agent_only_use_tools(self):
        """Test compatibility function"""
        tools = [create_adk_tool("test", "Test", lambda: None)]
        agent = create_agent_only_use_tools("test-model", tools)

        assert isinstance(agent, ADKCustomAgent)
        assert agent.name == "tool_executor"
        assert len(agent.tools) == 1

    def test_create_supervisor_node(self):
        """Test supervisor creation"""
        agents = [
            create_adk_agent("agent1", "First agent"),
            create_adk_agent("agent2", "Second agent")
        ]
        supervisor = create_supervisor_node("test-model", agents)

        assert isinstance(supervisor, ADKAgentOrchestrator)
        assert len(supervisor.agents) == 2


class TestExampleFunctions:
    """Test cases for example functions"""

    def test_create_example_tools(self):
        """Test example tools creation"""
        tools = create_example_tools()

        assert len(tools) == 3
        assert all(isinstance(tool, ADKTool) for tool in tools)

        # Test tool names
        tool_names = [tool.name for tool in tools]
        assert "add_numbers" in tool_names
        assert "get_weather" in tool_names
        assert "search_web" in tool_names

    def test_example_tool_execution(self):
        """Test executing example tools"""
        tools = create_example_tools()

        # Test math tool
        math_tool = next(t for t in tools if t.name == "add_numbers")
        result = math_tool.invoke(a=5, b=3)
        assert result["result"] == 8

        # Test weather tool
        weather_tool = next(t for t in tools if t.name == "get_weather")
        result = weather_tool.invoke(city="New York")
        assert result["city"] == "New York"
        assert "weather" in result

        # Test search tool
        search_tool = next(t for t in tools if t.name == "search_web")
        result = search_tool.invoke(query="test")
        assert result["query"] == "test"
        assert "results" in result

    def test_create_example_agents(self):
        """Test example agents creation"""
        agents = create_example_agents()

        assert len(agents) == 3
        assert all(isinstance(agent, ADKCustomAgent) for agent in agents)

        # Test agent names
        agent_names = [agent.name for agent in agents]
        assert "math_agent" in agent_names
        assert "weather_agent" in agent_names
        assert "search_agent" in agent_names

        # Test that each agent has tools
        for agent in agents:
            assert len(agent.tools) == 1


class TestErrorHandling:
    """Test error handling scenarios"""

    def test_tool_with_invalid_function(self):
        """Test tool creation with problematic function"""
        def problematic_function():
            raise RuntimeError("Something went wrong")

        tool = ADKTool("problem", "Problematic tool", problematic_function)
        result = tool.invoke()

        assert isinstance(result, dict)
        assert "error" in result
        assert "Something went wrong" in result["error"]

    def test_agent_with_invalid_type(self):
        """Test agent creation with invalid type"""
        with pytest.raises(ValueError):
            ADKCustomAgent(
                name="invalid_agent",
                description="Invalid agent",
                agent_type="invalid_type"
            )


@pytest.mark.skipif(not ADK_AVAILABLE, reason="Google ADK not installed")
class TestRealADKIntegration:
    """Integration tests that require actual ADK installation"""

    def test_real_adk_import(self):
        """Test that real ADK components can be imported"""
        from google.adk.agents import LlmAgent
        from google.adk.tools import FunctionTool

        assert LlmAgent is not None
        assert FunctionTool is not None

    @pytest.mark.asyncio
    async def test_async_agent_execution(self):
        """Test async agent execution with real ADK"""
        # This would test actual async execution
        # Requires proper ADK setup and credentials
        pass


# Fixtures for common test data
@pytest.fixture
def sample_tools():
    """Fixture providing sample tools"""
    def add_func(a: int, b: int) -> int:
        return a + b

    def multiply_func(a: int, b: int) -> int:
        return a * b

    return [
        ADKTool("add", "Add numbers", add_func),
        ADKTool("multiply", "Multiply numbers", multiply_func)
    ]


@pytest.fixture
def sample_agents(sample_tools):
    """Fixture providing sample agents"""
    return [
        ADKCustomAgent("math_agent", "Math operations", tools=sample_tools),
        ADKCustomAgent("helper_agent", "General helper")
    ]


# Parametrized tests
@pytest.mark.parametrize("agent_type", ["llm", "sequential", "parallel"])
def test_agent_type_creation(agent_type):
    """Test creating agents of different types"""
    agent = ADKCustomAgent(
        name=f"test_{agent_type}",
        description=f"Test {agent_type} agent",
        agent_type=agent_type
    )
    assert agent.agent_type == agent_type


@pytest.mark.parametrize("model_name", [
    "gemini-2.0-flash",
    "gemini-pro",
    "custom-model"
])
def test_different_models(model_name):
    """Test agent creation with different models"""
    agent = ADKCustomAgent(
        name="test_agent",
        description="Test agent",
        model=model_name
    )
    assert agent.model == model_name


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])