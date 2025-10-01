"""
Google ADK Agent Creator
A replacement for tool_and_agent_creator.py using Google's Agent Development Kit (ADK)
"""

from google.adk.agents import LlmAgent, BaseAgent, SequentialAgent, ParallelAgent
from google.adk.tools import BaseTool, FunctionTool, AgentTool
from google.adk.agents.session import Session
from google.adk.agents.memory import MemoryService
from typing import Callable, List, Dict, Any, Optional, Literal, get_type_hints, Union
from pydantic import BaseModel, Field
import inspect
import json
import logging
import asyncio
import os
from functools import wraps

# Setup logging
logging.basicConfig(filename='../logs/google_adk_agent_creator.log',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

class ADKAgentState(BaseModel):
    """Enhanced state for ADK agents"""
    session_id: str
    context: Dict[str, Any] = Field(default_factory=dict)
    memory: Dict[str, Any] = Field(default_factory=dict)
    agent_history: List[str] = Field(default_factory=list)

class ADKTool:
    """Wrapper for ADK-compatible tools"""

    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func
        self.function_tool = None

    def create_function_tool(self) -> FunctionTool:
        """Create ADK FunctionTool from the wrapped function"""
        if self.function_tool is None:
            self.function_tool = FunctionTool.create(self.func, name=self.name)
        return self.function_tool

    def invoke(self, **kwargs) -> Any:
        """Execute the tool with given arguments"""
        try:
            logger.debug(f"Executing tool {self.name} with args: {kwargs}")
            result = self.func(**kwargs)
            logger.debug(f"Tool {self.name} result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error executing tool {self.name}: {e}")
            return {"error": str(e), "status": "failed"}

class ADKAgentNode:
    """Agent node using Google ADK"""

    def __init__(self,
                 name: str,
                 model: str = "gemini-2.0-flash",
                 system_prompt: str = "",
                 description: str = "",
                 tools: List[ADKTool] = None):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.description = description
        self.tools = tools or []
        self.agent = self._create_agent()

    def _create_agent(self) -> LlmAgent:
        """Create the underlying ADK agent"""
        try:
            # Convert tools to ADK format
            adk_tools = [tool.create_function_tool() for tool in self.tools]

            agent = LlmAgent.builder() \
                .name(self.name) \
                .model(self.model) \
                .description(self.description) \
                .instruction(self.system_prompt) \
                .tools(*adk_tools) \
                .build()

            logger.debug(f"Created ADK agent: {self.name}")
            return agent

        except Exception as e:
            logger.error(f"Error creating ADK agent {self.name}: {e}")
            raise

    async def invoke_async(self, user_prompt: str, session: Optional[Session] = None) -> Any:
        """Asynchronously invoke the agent"""
        try:
            if session is None:
                # Create a new session if none provided
                session = Session.create()

            result = await self.agent.run_async(user_prompt, session=session)
            return result

        except Exception as e:
            logger.error(f"Error in async agent invocation for {self.name}: {e}")
            return {"error": str(e), "status": "failed"}

    def invoke(self, user_prompt: str, session: Optional[Session] = None) -> Any:
        """Synchronously invoke the agent"""
        try:
            # Run async method in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.invoke_async(user_prompt, session))
                return result
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"Error in agent invocation for {self.name}: {e}")
            return {"error": str(e), "status": "failed"}

class ADKCustomAgent:
    """Custom agent implementation using Google ADK"""

    def __init__(self,
                 name: str,
                 description: str,
                 model: str = "gemini-2.0-flash",
                 tools: List[ADKTool] = None,
                 agent_type: Literal["llm", "sequential", "parallel"] = "llm",
                 **kwargs):
        self.name = name
        self.description = description
        self.model = model
        self.tools = tools or []
        self.agent_type = agent_type
        self.kwargs = kwargs
        self.agent = self._create_agent()

    def _create_agent(self) -> BaseAgent:
        """Create the appropriate ADK agent based on type"""
        try:
            # Convert tools to ADK format
            adk_tools = [tool.create_function_tool() for tool in self.tools]

            instruction = self.create_instruction()

            if self.agent_type == "llm":
                agent = LlmAgent.builder() \
                    .name(self.name) \
                    .model(self.model) \
                    .description(self.description) \
                    .instruction(instruction) \
                    .tools(*adk_tools) \
                    .build()

            elif self.agent_type == "sequential":
                # For sequential agents, we might need sub-agents
                agent = SequentialAgent.builder() \
                    .name(self.name) \
                    .description(self.description) \
                    .build()

            elif self.agent_type == "parallel":
                # For parallel agents, we might need sub-agents
                agent = ParallelAgent.builder() \
                    .name(self.name) \
                    .description(self.description) \
                    .build()

            else:
                raise ValueError(f"Unknown agent type: {self.agent_type}")

            logger.debug(f"Created ADK custom agent: {self.name} ({self.agent_type})")
            return agent

        except Exception as e:
            logger.error(f"Error creating ADK custom agent {self.name}: {e}")
            raise

    def create_instruction(self) -> str:
        """Generate instruction prompt for the agent"""
        tools_str = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])

        instruction = f"You are an AI agent tasked with {self.description}. "

        if self.tools:
            instruction += f"You have access to the following tools:\n{tools_str}\n"
            instruction += ("Use the appropriate tools to complete tasks. "
                          "Call tools when needed and provide clear, actionable results.")

        instruction += ("Focus on the specific task at hand. "
                       "Only address requests within your capabilities. "
                       "If a task is outside your scope, clearly indicate this limitation.")

        return instruction

    async def invoke_async(self, user_prompt: str, session: Optional[Session] = None) -> Any:
        """Asynchronously invoke the agent"""
        try:
            logger.debug(f'Invoking ADK agent {self.name} async')

            if session is None:
                session = Session.create()

            result = await self.agent.run_async(user_prompt, session=session)

            logger.debug(f'ADK agent {self.name} completed with result type: {type(result)}')
            return result

        except Exception as e:
            logger.error(f"Error in async invocation for {self.name}: {e}")
            return {"error": str(e), "status": "failed"}

    def invoke(self, user_prompt: str, session: Optional[Session] = None) -> Any:
        """Synchronously invoke the agent"""
        try:
            # Handle event loop properly
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an event loop, create a task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self.invoke_async(user_prompt, session))
                        return future.result()
                else:
                    return loop.run_until_complete(self.invoke_async(user_prompt, session))
            except RuntimeError:
                # No event loop exists, create one
                return asyncio.run(self.invoke_async(user_prompt, session))

        except Exception as e:
            logger.error(f"Error in sync invocation for {self.name}: {e}")
            return {"error": str(e), "status": "failed"}

class ADKAgentOrchestrator:
    """Orchestrator for managing multiple ADK agents"""

    def __init__(self, agents: List[ADKCustomAgent], supervisor_model: str = "gemini-2.0-flash"):
        self.agents = {agent.name: agent for agent in agents}
        self.supervisor_model = supervisor_model
        self.supervisor_agent = self._create_supervisor()

    def _create_supervisor(self) -> LlmAgent:
        """Create a supervisor agent for routing"""
        agent_descriptions = "\n".join([
            f"- {name}: {agent.description}"
            for name, agent in self.agents.items()
        ])

        instruction = f"""You are a supervisor managing specialized agents:
{agent_descriptions}

Analyze user requests and respond with ONLY the most appropriate agent name.
Available agents: {', '.join(self.agents.keys())}

Rules:
- Respond with just the agent name
- Choose the agent best suited for the specific task
- If no agent is suitable, respond with "NONE"
"""

        supervisor = LlmAgent.builder() \
            .name("supervisor") \
            .model(self.supervisor_model) \
            .description("Routes requests to appropriate agents") \
            .instruction(instruction) \
            .build()

        return supervisor

    async def route_request_async(self, request: str, session: Optional[Session] = None) -> Optional[str]:
        """Asynchronously route a request to appropriate agent"""
        try:
            if session is None:
                session = Session.create()

            routing_prompt = f"User Request: {request}"
            response = await self.supervisor_agent.run_async(routing_prompt, session=session)

            # Extract agent name from response
            if hasattr(response, 'content'):
                agent_name = response.content.strip()
            else:
                agent_name = str(response).strip()

            if agent_name in self.agents:
                return agent_name
            elif agent_name == "NONE":
                return None
            else:
                # Fallback to first agent
                return list(self.agents.keys())[0] if self.agents else None

        except Exception as e:
            logger.error(f"Error in request routing: {e}")
            return None

    def route_request(self, request: str, session: Optional[Session] = None) -> Optional[str]:
        """Synchronously route a request"""
        try:
            return asyncio.run(self.route_request_async(request, session))
        except Exception as e:
            logger.error(f"Error in sync request routing: {e}")
            return None

    async def process_request_async(self, request: str, session: Optional[Session] = None) -> Dict[str, Any]:
        """Asynchronously process request with appropriate agent"""
        if session is None:
            session = Session.create()

        # Route to appropriate agent
        selected_agent_name = await self.route_request_async(request, session)

        if not selected_agent_name or selected_agent_name not in self.agents:
            return {
                "result": "No suitable agent found for this request",
                "agent_used": None,
                "status": "no_agent"
            }

        # Execute with selected agent
        selected_agent = self.agents[selected_agent_name]
        result = await selected_agent.invoke_async(request, session)

        return {
            "result": result,
            "agent_used": selected_agent_name,
            "status": "success"
        }

    def process_request(self, request: str, session: Optional[Session] = None) -> Dict[str, Any]:
        """Synchronously process request"""
        try:
            return asyncio.run(self.process_request_async(request, session))
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                "result": f"Error: {str(e)}",
                "agent_used": None,
                "status": "error"
            }

# Utility functions for compatibility with existing code

def create_adk_tool(name: str, description: str, func: Callable) -> ADKTool:
    """
    Create an ADK-compatible tool

    Args:
        name: Tool name
        description: Tool description
        func: Function to wrap as tool

    Returns:
        ADKTool instance
    """
    return ADKTool(name=name, description=description, func=func)

def create_adk_agent(name: str,
                    description: str,
                    model: str = "gemini-2.0-flash",
                    tools: List[ADKTool] = None,
                    agent_type: str = "llm",
                    **kwargs) -> ADKCustomAgent:
    """
    Create an ADK-based agent

    Args:
        name: Agent name
        description: Agent description
        model: Model to use
        tools: List of tools
        agent_type: Type of agent ("llm", "sequential", "parallel")
        **kwargs: Additional parameters

    Returns:
        ADKCustomAgent instance
    """
    return ADKCustomAgent(
        name=name,
        description=description,
        model=model,
        tools=tools or [],
        agent_type=agent_type,
        **kwargs
    )

def create_agent_only_use_tools(model: str, tools: List[ADKTool]) -> ADKCustomAgent:
    """
    Create an agent that only uses tools (compatibility function)
    """
    return create_adk_agent(
        name="tool_executor",
        description="Execute tools based on user requests",
        model=model,
        tools=tools,
        agent_type="llm"
    )

def create_supervisor_node(model: str, agents: List[ADKCustomAgent]) -> ADKAgentOrchestrator:
    """
    Create a supervisor node (compatibility function)
    """
    return ADKAgentOrchestrator(agents=agents, supervisor_model=model)

# Example usage functions

def create_example_tools() -> List[ADKTool]:
    """Create example tools for testing"""

    def add_numbers(a: int, b: int) -> dict:
        """Add two numbers together"""
        result = a + b
        return {"result": result, "operation": f"{a} + {b} = {result}"}

    def get_weather(city: str) -> dict:
        """Get weather information for a city"""
        # Mock weather data
        weather_data = {
            "new york": "Sunny, 75°F",
            "london": "Cloudy, 60°F",
            "tokyo": "Rainy, 68°F"
        }

        weather = weather_data.get(city.lower(), f"Weather data not available for {city}")
        return {
            "city": city,
            "weather": weather,
            "status": "success" if city.lower() in weather_data else "not_found"
        }

    def search_web(query: str) -> dict:
        """Search the web for information"""
        return {
            "query": query,
            "results": f"Mock search results for '{query}'",
            "count": 5,
            "status": "success"
        }

    return [
        create_adk_tool("add_numbers", "Add two numbers", add_numbers),
        create_adk_tool("get_weather", "Get weather for a city", get_weather),
        create_adk_tool("search_web", "Search the web", search_web)
    ]

def create_example_agents() -> List[ADKCustomAgent]:
    """Create example agents for testing"""

    tools = create_example_tools()

    agents = [
        create_adk_agent(
            name="math_agent",
            description="Perform mathematical calculations and operations",
            tools=[tools[0]]  # add_numbers
        ),
        create_adk_agent(
            name="weather_agent",
            description="Provide weather information and forecasts",
            tools=[tools[1]]  # get_weather
        ),
        create_adk_agent(
            name="search_agent",
            description="Search for information on the web",
            tools=[tools[2]]  # search_web
        )
    ]

    return agents

# Main execution example
if __name__ == "__main__":
    # Example usage
    try:
        # Create tools and agents
        agents = create_example_agents()

        # Create orchestrator
        orchestrator = ADKAgentOrchestrator(agents)

        # Process sample requests
        test_requests = [
            "What's 15 + 27?",
            "What's the weather like in New York?",
            "Search for information about Python programming"
        ]

        for request in test_requests:
            print(f"\nProcessing: {request}")
            result = orchestrator.process_request(request)
            print(f"Agent used: {result['agent_used']}")
            print(f"Result: {result['result']}")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"Error: {e}")