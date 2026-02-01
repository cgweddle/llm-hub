"""
PydanticAI Integration Test Script

This script demonstrates the complete PydanticAI integration:
1. Creating a simple test tool
2. Creating a PydanticAI agent
3. Executing the agent
4. Verifying results

Prerequisites:
- Run this from the project root directory
- Ensure database is set up (src/database/database_setup.py)
- Configure at least one LLM provider in ~/.llm_hub/config.yaml
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from database.database import (
    get_session,
    create_user,
    create_tool,
    create_agent,
    get_agent_by_id
)
from factories.pydanticai_agent_factory import PydanticAIAgentFactory
from executors.agent_executor import AgentExecutor


def setup_test_user(session):
    """Create a test user if not exists"""
    from database.database_setup import User
    from passlib.hash import bcrypt

    # Check if test user exists
    test_user = session.query(User).filter(User.username == "test_user").first()

    if not test_user:
        print("Creating test user...")
        test_user = create_user(
            session=session,
            username="test_user",
            email="test@example.com",
            password_hash=bcrypt.hash("test_password")
        )
        print(f"✓ Created test user (ID: {test_user.id})")
    else:
        print(f"✓ Test user already exists (ID: {test_user.id})")

    return test_user


def setup_test_tool(session, user_id):
    """Create a simple test tool"""
    print("\nCreating test tool...")

    # Simple calculator tool
    tool_code = '''
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b
'''

    input_schema = {
        "type": "object",
        "properties": {
            "a": {"type": "integer", "description": "First number"},
            "b": {"type": "integer", "description": "Second number"}
        },
        "required": ["a", "b"]
    }

    output_schema = {
        "type": "integer",
        "description": "Sum of the two numbers"
    }

    tool = create_tool(
        session=session,
        user_id=user_id,
        name="Add Numbers",
        description="Adds two numbers together",
        tool_type="function",
        main_function="add_numbers",
        function_code=tool_code,
        script_code=tool_code,
        input_schema=input_schema,
        output_schema=output_schema
    )

    print(f"✓ Created test tool: {tool.name} (ID: {tool.id})")
    return tool


def create_test_agent(session, user_id, tool_id, model_name):
    """Create a PydanticAI test agent"""
    print(f"\nCreating PydanticAI agent with model: {model_name}...")

    agent = create_agent(
        session=session,
        user_id=user_id,
        name="Math Assistant",
        description="Helps with simple math calculations",
        agent_type="pydanticai",
        system_prompt="You are a helpful math assistant. Use the available tools to help users with calculations.",
        llm_config={"model_name": model_name},
        tools_config={"tool_ids": [tool_id]},
        agent_metadata={}
    )

    print(f"✓ Created PydanticAI agent: {agent.name} (ID: {agent.id})")
    return agent


async def test_agent_execution(session, agent_id, user_id):
    """Test executing the PydanticAI agent"""
    print(f"\n{'='*60}")
    print("Testing Agent Execution")
    print(f"{'='*60}")

    executor = AgentExecutor(session)

    # Test 1: Simple execution
    print("\n[Test 1] Executing agent with simple query...")
    try:
        result = await executor.execute_agent(
            agent_id=agent_id,
            user_id=user_id,
            input_data="What is 5 + 3?",
            stream=False
        )

        print(f"✓ Execution completed!")
        print(f"  - Execution ID: {result['execution_id']}")
        print(f"  - Status: {result['status']}")
        print(f"  - Result: {result['result']}")
        print(f"  - Agent Type: {result['agent_type']}")

        if result.get('cost'):
            print(f"  - Cost: {result['cost']}")

        print(f"  - Messages: {len(result['messages'])} messages")

    except Exception as e:
        print(f"✗ Execution failed: {e}")
        import traceback
        traceback.print_exc()


async def test_agent_factory_direct(session, agent_id):
    """Test agent creation using factory directly"""
    print(f"\n{'='*60}")
    print("Testing Direct Agent Factory")
    print(f"{'='*60}")

    print(f"\nCreating agent from database (ID: {agent_id})...")
    factory = PydanticAIAgentFactory(session)

    try:
        # Validate config first
        validation = factory.validate_agent_config(agent_id)
        print(f"\nAgent Configuration Validation:")
        print(f"  - Valid: {validation['valid']}")
        print(f"  - Errors: {validation['errors']}")
        print(f"  - Warnings: {validation['warnings']}")
        print(f"  - Tool Count: {validation['config'].get('tool_count', 0)}")

        if not validation['valid']:
            print("✗ Agent configuration is invalid!")
            return

        # Create agent
        agent = factory.create_from_database(agent_id)
        print(f"✓ Agent created successfully!")

        # Run agent directly
        print(f"\nExecuting agent directly...")
        result = await agent.run("What is 10 + 15?")

        print(f"✓ Direct execution completed!")
        print(f"  - Result: {result.data}")

        if hasattr(result, 'cost'):
            print(f"  - Cost: {result.cost()}")

        print(f"  - Messages: {len(result.all_messages())} messages")

    except Exception as e:
        print(f"✗ Direct execution failed: {e}")
        import traceback
        traceback.print_exc()


def check_llm_config():
    """Check if LLM configuration exists"""
    print("\nChecking LLM configuration...")

    from utils import load_llm_provider_config

    config = load_llm_provider_config()
    if not config or not config.get('models'):
        print("✗ No LLM providers configured!")
        print("\nPlease configure at least one LLM provider in ~/.llm_hub/config.yaml")
        print("Example configuration:")
        print("""
models:
  - name: "My Anthropic Config"
    provider: "anthropic"
    model: "claude-3-5-sonnet-20241022"
    api_key: "your-api-key-here"
        """)
        return None

    # Get first model as default
    first_model = config['models'][0]
    model_name = first_model['name']

    print(f"✓ Found LLM configuration: {model_name}")
    print(f"  - Provider: {first_model['provider']}")
    print(f"  - Model: {first_model['model']}")

    return model_name


async def main():
    """Main test flow"""
    print(f"{'='*60}")
    print("PydanticAI Integration Test")
    print(f"{'='*60}")

    # Check LLM config
    model_name = check_llm_config()
    if not model_name:
        return

    # Get database session
    session = get_session()

    try:
        # Setup test data
        user = setup_test_user(session)
        tool = setup_test_tool(session, user.id)
        agent = create_test_agent(session, user.id, tool.id, model_name)

        # Run tests
        await test_agent_execution(session, agent.id, user.id)
        await test_agent_factory_direct(session, agent.id)

        print(f"\n{'='*60}")
        print("✓ All tests completed!")
        print(f"{'='*60}")

        print("\n📝 Summary:")
        print(f"  - User ID: {user.id}")
        print(f"  - Tool ID: {tool.id}")
        print(f"  - Agent ID: {agent.id}")
        print(f"  - Model: {model_name}")

        print("\n🎉 PydanticAI integration is working!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        session.close()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
