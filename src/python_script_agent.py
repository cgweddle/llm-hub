"""
Python Script Writing and Execution Agent
Uses Google ADK to create an agent that can write and execute Python scripts
"""

import sys
import os
import subprocess
import tempfile
import json
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

# Add paths for our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.tools.google_adk_agent_creator import (
        create_adk_tool,
        create_adk_agent,
        ADKTool,
        ADKCustomAgent
    )
    ADK_AVAILABLE = True
except ImportError as e:
    print(f"Google ADK not available: {e}")
    print("Creating mock classes for testing...")
    ADK_AVAILABLE = False

    # Create mock classes for testing
    class ADKTool:
        def __init__(self, name, description, func):
            self.name = name
            self.description = description
            self.func = func

    class ADKCustomAgent:
        def __init__(self, name, description, tools=None, model="gemini-2.0-flash", agent_type="llm", **kwargs):
            self.name = name
            self.description = description
            self.tools = tools or []
            self.model = model
            self.agent_type = agent_type
            self.kwargs = kwargs

        def create_instruction(self):
            return f"You are {self.description}"

    def create_adk_tool(name, description, func):
        return ADKTool(name, description, func)

    def create_adk_agent(name, description, tools=None, **kwargs):
        return ADKCustomAgent(name, description, tools, **kwargs)

try:
    from scripts.database.database import (
        get_session,
        create_agent as db_create_agent,
        create_tool as db_create_tool
    )
    DATABASE_AVAILABLE = True
except ImportError as e:
    print(f"Database module not available: {e}")
    DATABASE_AVAILABLE = False

    def get_session():
        return None

    def db_create_agent(**kwargs):
        class MockAgent:
            def __init__(self):
                self.id = 1
                self.name = kwargs.get('name', 'mock')
        return MockAgent()

    def db_create_tool(**kwargs):
        class MockTool:
            def __init__(self):
                self.id = 1
                self.name = kwargs.get('name', 'mock')
        return MockTool()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PythonScriptTools:
    """Collection of tools for Python script operations"""

    @staticmethod
    def write_python_script(filename: str, code: str) -> Dict[str, Any]:
        """
        Write Python code to a file

        Args:
            filename: Name of the Python file to create
            code: Python code content

        Returns:
            Dict with operation result
        """
        try:
            # Ensure filename has .py extension
            if not filename.endswith('.py'):
                filename += '.py'

            # Create scripts directory if it doesn't exist
            scripts_dir = os.path.join(os.path.dirname(__file__), '..', 'generated_scripts')
            os.makedirs(scripts_dir, exist_ok=True)

            # Full path for the script
            script_path = os.path.join(scripts_dir, filename)

            # Write the code to file
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(code)

            logger.info(f"Created Python script: {script_path}")

            return {
                "status": "success",
                "message": f"Successfully wrote Python script to {filename}",
                "file_path": script_path,
                "lines_written": len(code.split('\n'))
            }

        except Exception as e:
            logger.error(f"Error writing Python script: {e}")
            return {
                "status": "error",
                "message": f"Failed to write script: {str(e)}",
                "file_path": None
            }

    @staticmethod
    def execute_python_script(script_path_or_code: str, is_code: bool = False) -> Dict[str, Any]:
        """
        Execute a Python script or code

        Args:
            script_path_or_code: Either file path to script or Python code
            is_code: True if input is code, False if it's a file path

        Returns:
            Dict with execution result
        """
        try:
            if is_code:
                # Create temporary file for code execution
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                    temp_file.write(script_path_or_code)
                    script_path = temp_file.name
            else:
                script_path = script_path_or_code

                # Check if file exists
                if not os.path.exists(script_path):
                    return {
                        "status": "error",
                        "message": f"Script file not found: {script_path}",
                        "output": "",
                        "error": "File not found"
                    }

            # Execute the script
            logger.info(f"Executing Python script: {script_path}")

            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )

            # Clean up temporary file if created
            if is_code:
                try:
                    os.unlink(script_path)
                except:
                    pass

            return {
                "status": "success" if result.returncode == 0 else "error",
                "message": f"Script executed with return code {result.returncode}",
                "output": result.stdout,
                "error": result.stderr,
                "return_code": result.returncode
            }

        except subprocess.TimeoutExpired:
            logger.error("Script execution timed out")
            return {
                "status": "error",
                "message": "Script execution timed out (30s limit)",
                "output": "",
                "error": "Timeout"
            }
        except Exception as e:
            logger.error(f"Error executing Python script: {e}")
            return {
                "status": "error",
                "message": f"Execution failed: {str(e)}",
                "output": "",
                "error": str(e)
            }

    @staticmethod
    def validate_python_syntax(code: str) -> Dict[str, Any]:
        """
        Validate Python code syntax without execution

        Args:
            code: Python code to validate

        Returns:
            Dict with validation result
        """
        try:
            # Try to compile the code
            compile(code, '<string>', 'exec')

            return {
                "status": "success",
                "message": "Python syntax is valid",
                "valid": True,
                "error": None
            }

        except SyntaxError as e:
            return {
                "status": "error",
                "message": f"Syntax error: {str(e)}",
                "valid": False,
                "error": {
                    "line": e.lineno,
                    "offset": e.offset,
                    "text": e.text,
                    "message": e.msg
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Validation error: {str(e)}",
                "valid": False,
                "error": str(e)
            }

    @staticmethod
    def list_generated_scripts() -> Dict[str, Any]:
        """
        List all generated Python scripts

        Returns:
            Dict with list of scripts
        """
        try:
            scripts_dir = os.path.join(os.path.dirname(__file__), '..', 'generated_scripts')

            if not os.path.exists(scripts_dir):
                return {
                    "status": "success",
                    "message": "No generated scripts directory found",
                    "scripts": []
                }

            scripts = []
            for filename in os.listdir(scripts_dir):
                if filename.endswith('.py'):
                    file_path = os.path.join(scripts_dir, filename)
                    stat = os.stat(file_path)

                    scripts.append({
                        "filename": filename,
                        "path": file_path,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })

            return {
                "status": "success",
                "message": f"Found {len(scripts)} Python scripts",
                "scripts": scripts
            }

        except Exception as e:
            logger.error(f"Error listing scripts: {e}")
            return {
                "status": "error",
                "message": f"Failed to list scripts: {str(e)}",
                "scripts": []
            }

    @staticmethod
    def install_package(package_name: str) -> Dict[str, Any]:
        """
        Install a Python package using pip

        Args:
            package_name: Name of the package to install

        Returns:
            Dict with installation result
        """
        try:
            logger.info(f"Installing Python package: {package_name}")

            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', package_name],
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout for package installation
            )

            return {
                "status": "success" if result.returncode == 0 else "error",
                "message": f"Package installation completed with return code {result.returncode}",
                "package": package_name,
                "output": result.stdout,
                "error": result.stderr,
                "return_code": result.returncode
            }

        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "message": "Package installation timed out",
                "package": package_name,
                "output": "",
                "error": "Timeout"
            }
        except Exception as e:
            logger.error(f"Error installing package: {e}")
            return {
                "status": "error",
                "message": f"Installation failed: {str(e)}",
                "package": package_name,
                "output": "",
                "error": str(e)
            }

def create_python_script_tools() -> List[ADKTool]:
    """Create ADK tools for Python script operations"""

    tools = [
        create_adk_tool(
            name="write_python_script",
            description="Write Python code to a file. Provide filename and code content.",
            func=PythonScriptTools.write_python_script
        ),
        create_adk_tool(
            name="execute_python_script",
            description="Execute a Python script from file path or execute Python code directly.",
            func=PythonScriptTools.execute_python_script
        ),
        create_adk_tool(
            name="validate_python_syntax",
            description="Validate Python code syntax without executing it.",
            func=PythonScriptTools.validate_python_syntax
        ),
        create_adk_tool(
            name="list_generated_scripts",
            description="List all generated Python scripts with metadata.",
            func=PythonScriptTools.list_generated_scripts
        ),
        create_adk_tool(
            name="install_package",
            description="Install a Python package using pip.",
            func=PythonScriptTools.install_package
        )
    ]

    return tools

def create_python_script_agent() -> ADKCustomAgent:
    """Create the Python script writing and execution agent"""

    tools = create_python_script_tools()

    agent = create_adk_agent(
        name="python_script_agent",
        description="An intelligent agent that can write, validate, and execute Python scripts. It can create Python files, run them, check syntax, manage packages, and list generated scripts.",
        model="gemini-2.0-flash",
        tools=tools,
        agent_type="llm"
    )

    return agent

def upload_tools_to_database(tools: List[ADKTool], user_id: int = 1) -> List[Dict[str, Any]]:
    """Upload tools to the database"""

    if not DATABASE_AVAILABLE:
        print("Database not available - using mock upload")
        uploaded_tools = []
        for tool in tools:
            uploaded_tools.append({
                "tool_id": len(uploaded_tools) + 1,
                "name": tool.name,
                "status": "mock_uploaded"
            })
        return uploaded_tools

    uploaded_tools = []
    session = get_session()

    try:
        import inspect
        for tool in tools:
            # Prepare tool data for database
            tool_data = {
                "name": tool.name,
                "description": tool.description,
                "tool_type": "function",
                "function_name": tool.func.__name__,
                "function_code": inspect.getsource(tool.func) if hasattr(tool, 'func') else "",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                "is_public": True
            }

            # Create tool in database
            db_tool = db_create_tool(
                session=session,
                user_id=user_id,
                **tool_data
            )

            uploaded_tools.append({
                "tool_id": db_tool.id,
                "name": db_tool.name,
                "status": "uploaded"
            })

            logger.info(f"Uploaded tool to database: {tool.name} (ID: {db_tool.id})")

    except Exception as e:
        logger.error(f"Error uploading tools to database: {e}")
        return []

    finally:
        if session:
            session.close()

    return uploaded_tools

def upload_agent_to_database(agent: ADKCustomAgent, user_id: int = 1) -> Optional[Dict[str, Any]]:
    """Upload agent to the database"""

    if not DATABASE_AVAILABLE:
        print("Database not available - using mock upload")
        return {
            "agent_id": 1,
            "name": agent.name,
            "status": "mock_uploaded",
            "tools_count": len(agent.tools)
        }

    session = get_session()

    try:
        # Prepare agent data for database
        agent_data = {
            "name": agent.name,
            "description": agent.description,
            "agent_type": "google_adk_" + getattr(agent, 'agent_type', 'llm'),
            "system_prompt": agent.create_instruction(),
            "llm_config": {
                "model": getattr(agent, 'model', 'gemini-2.0-flash'),
                "temperature": 0.7,
                "max_tokens": 4096
            },
            "tools_config": {
                "tools": [tool.name for tool in agent.tools],
                "tool_descriptions": {tool.name: tool.description for tool in agent.tools}
            },
            "agent_metadata": {
                "created_with": "google_adk_agent_creator",
                "agent_type": getattr(agent, 'agent_type', 'llm'),
                "model": getattr(agent, 'model', 'gemini-2.0-flash'),
                "tools_count": len(agent.tools),
                "created_at": datetime.now().isoformat()
            }
        }

        # Create agent in database
        db_agent = db_create_agent(
            session=session,
            user_id=user_id,
            name=agent_data["name"],
            description=agent_data["description"],
            agent_type=agent_data["agent_type"],
            system_prompt=agent_data["system_prompt"],
            llm_config=agent_data["llm_config"],
            tools_config=agent_data["tools_config"],
            metadata=agent_data["agent_metadata"]
        )

        logger.info(f"Uploaded agent to database: {agent.name} (ID: {db_agent.id})")

        return {
            "agent_id": db_agent.id,
            "name": db_agent.name,
            "status": "uploaded",
            "tools_count": len(agent.tools)
        }

    except Exception as e:
        logger.error(f"Error uploading agent to database: {e}")
        return None

    finally:
        if session:
            session.close()

def main():
    """Main function to create and upload the Python script agent"""

    print("Creating Python Script Agent with Google ADK...")

    # Create the agent
    agent = create_python_script_agent()
    print(f"✓ Created agent: {agent.name}")
    print(f"  Description: {agent.description}")
    print(f"  Model: {agent.model}")
    print(f"  Tools: {len(agent.tools)}")

    # List the tools
    print("\nTools created:")
    for tool in agent.tools:
        print(f"  - {tool.name}: {tool.description}")

    # Upload tools to database
    print("\nUploading tools to database...")
    try:
        uploaded_tools = upload_tools_to_database(agent.tools)

        if uploaded_tools:
            print(f"✓ Successfully uploaded {len(uploaded_tools)} tools")
            for tool_info in uploaded_tools:
                print(f"  - {tool_info['name']} (ID: {tool_info['tool_id']})")
        else:
            print("✗ Failed to upload tools - using mock data for demonstration")
            # Create mock data to continue demonstration
            uploaded_tools = [
                {"tool_id": i+1, "name": tool.name, "status": "mock_uploaded"}
                for i, tool in enumerate(agent.tools)
            ]
    except Exception as e:
        print(f"⚠️ Database upload failed: {e}")
        print("Using mock data for demonstration...")
        uploaded_tools = [
            {"tool_id": i+1, "name": tool.name, "status": "mock_uploaded"}
            for i, tool in enumerate(agent.tools)
        ]

    # Upload agent to database
    print("\nUploading agent to database...")
    try:
        uploaded_agent = upload_agent_to_database(agent)

        if uploaded_agent:
            print(f"✓ Successfully uploaded agent: {uploaded_agent['name']} (ID: {uploaded_agent['agent_id']})")
            print(f"  Tools linked: {uploaded_agent['tools_count']}")
        else:
            print("✗ Failed to upload agent - using mock data")
            uploaded_agent = {
                "agent_id": 1,
                "name": agent.name,
                "status": "mock_uploaded",
                "tools_count": len(agent.tools)
            }
    except Exception as e:
        print(f"⚠️ Agent upload failed: {e}")
        print("Using mock data for demonstration...")
        uploaded_agent = {
            "agent_id": 1,
            "name": agent.name,
            "status": "mock_uploaded",
            "tools_count": len(agent.tools)
        }

    print("\n" + "="*50)
    print("Python Script Agent Creation Complete!")
    print("="*50)
    print(f"Agent ID: {uploaded_agent['agent_id']}")
    print(f"Tools uploaded: {len(uploaded_tools)}")
    print("\nThe agent can now:")
    print("- Write Python scripts to files")
    print("- Execute Python scripts and code")
    print("- Validate Python syntax")
    print("- Install Python packages")
    print("- List and manage generated scripts")

if __name__ == "__main__":
    import inspect
    main()