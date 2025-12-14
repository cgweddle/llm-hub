"""
Tool Executor
Executes database tools by converting them to callable Python functions
"""

import sys
import os
import json
import logging
import subprocess
import tempfile
from typing import Dict, Any, Callable, Optional
import pickle

logger = logging.getLogger(__name__)


def parse_imports_and_classes(script_code: str) -> Dict[str, Any]:
    """
    Parse imports and class definitions from full Python script using AST
    Returns a namespace dict that can be used to resolve type strings

    Args:
        script_code: Full Python script code

    Returns:
        Dict mapping names to imported modules/classes
    """
    import ast

    namespace = {}

    try:
        tree = ast.parse(script_code)

        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Store module name mapping (e.g., 'os' -> 'os')
                    module_name = alias.asname if alias.asname else alias.name
                    try:
                        namespace[module_name] = __import__(alias.name)
                    except ImportError:
                        logger.warning(f"Could not import {alias.name}")

            elif isinstance(node, ast.ImportFrom):
                module = node.module
                for alias in node.names:
                    # Store imported name (e.g., 'DataFrame' from pandas)
                    name = alias.asname if alias.asname else alias.name
                    try:
                        imported_module = __import__(module, fromlist=[alias.name])
                        namespace[name] = getattr(imported_module, alias.name)
                    except (ImportError, AttributeError):
                        logger.warning(f"Could not import {alias.name} from {module}")

            elif isinstance(node, ast.ClassDef):
                # For custom classes defined in the script, use Any as placeholder
                # since we can't instantiate them without execution
                from typing import Any
                namespace[node.name] = Any

    except SyntaxError as e:
        logger.error(f"Syntax error parsing code: {e}")

    return namespace


def eval_type_string(type_str: str, namespace: Dict[str, Any] = None):
    """
    Evaluate a Python type string to get the actual type object
    Uses the namespace containing imports and custom classes from the tool's code

    Args:
        type_str: Type string to evaluate (e.g., "str", "List[int]", "DataFrame")
        namespace: Namespace with imported types and classes

    Returns:
        Actual type object

    Examples:
        "str" -> str
        "List[int]" -> List[int]
        "DataFrame" -> DataFrame (from namespace)
    """
    from typing import List, Dict, Optional, Any

    # Build a namespace with common built-in types
    type_namespace = {
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
        'List': List,
        'Dict': Dict,
        'Optional': Optional,
        'Any': Any,
    }

    # Merge in the namespace from the tool's code
    if namespace:
        type_namespace.update(namespace)

    try:
        # Safely evaluate the type string
        return eval(type_str, {"__builtins__": {}}, type_namespace)
    except:
        logger.warning(f"Could not evaluate type string '{type_str}', using Any")
        from typing import Any
        return Any


def create_conda_executable_function(tool, conda_env: str) -> Callable:
    """
    Create an executable function that runs a subprocess with a conda environment
    Args:
        tool: Tool with function_code and function_name
        conda_env: Path to the conda environment to use

    Returns: 
        Callable function that executes tool in subprocess
    """

    def subprocess_wrapper(**kwargs):
        script_template = '''
import pickle
import sys
input_data = pickle.loads(sys.stdin.buffer.read())

{script_code}

# Execute main function
try:
    result = {function_name}(**input_data)
    output = {{"status": "success", "result": result}}
except Exception as e:
    import traceback
    output = {{
        "status": "error",
        "error": str(e),
        "error_type": type(e).__name__,
        "traceback": traceback.format_exc()    
    }}

# Output result as pickle to stdout
sys.stdout.buffer.write(pickle.dumps(output))
        '''

        script = script_template.format(
            script_code = tool.script_code,
            function_name = tool.function_name
        )

        # Write script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix = '.py', delete=False) as f:
            script_path = f.name
            f.write(script)
        try:
            # Pickle the input arguments
            input_pickle = pickle.dumps(kwargs)

            # Execute with conda environment 
            cmd = ['conda', 'run', '-p', conda_env, 'python', script_path]

            logger.info(f"Executing tool {tool.function_name} in conda env: {conda_env}")

            result = subprocess.run(
                cmd,
                input=input_pickle,
                capture_output=True,
                timeout=300 # 5 minute timeout
            )

            # If the subprocess crashes
            if result.returncode != 0:
                stderr = result.stderr.decode() if result.stderr else "No error output"
                logger.error(f"Tool execution failed: {stderr}")
                raise RuntimeError(f"Tool execution failed: {stderr}")

            # Unpickle the output
            try:
                output = pickle.loads(result.stdout)
            except (pickle.UnpicklingError, EOFError) as e:
                stdout = result.stdout.decode() if result.stdout else "No Output"
                logger.error(f"Failed to unpickle tool output. stdout: {stdout}")
                raise RuntimeError(f"Failed to unpickle tool output: {e}")

            # Check execution status
            if output["status"] == "error":
                error_msg = f"{output['error_type']}: {output['error']}"
                if 'traceback' in output:
                    error_msg += f"\n{output['traceback']}"
                raise RuntimeError(error_msg)
            
            return output["result"]

        finally:
            # Clean up temp script file
            try:
                os.unlink(script_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp script {script_path}: {e}")
            
    return subprocess_wrapper



def create_executable_function(tool) -> Callable:
    """
    Create an executable function from database tool

    Args:
        tool: Database Tool object with function_code, function_name, and helper_functions

    Returns:
        Callable function that can be invoked

    Raises:
        ValueError: If function cannot be created
    """
    # Prepare the execution environment
    exec_globals = {
        '__builtins__': __builtins__,
        'json': json,
        'os': os,
        'sys': sys
    }

    # Add helper functions to the execution environment
    if hasattr(tool, 'helper_functions') and tool.helper_functions:
        for helper_name, helper_code in tool.helper_functions.items():
            try:
                exec(helper_code, exec_globals)
                logger.debug(f"Added helper function: {helper_name}")
            except Exception as e:
                logger.error(f"Failed to execute helper function {helper_name}: {e}")
                raise

    # Execute the main function code
    try:
        exec(tool.function_code, exec_globals)
        main_function = exec_globals.get(tool.function_name)

        if not main_function:
            raise ValueError(f"Main function '{tool.function_name}' not found after execution")

        logger.info(f"Successfully created executable function: {tool.function_name}")
        return main_function

    except Exception as e:
        logger.error(f"Failed to create executable function: {e}")
        raise


def execute_tool(tool, **kwargs) -> Any:
    """
    Execute a database tool with given arguments

    Args:
        tool: Database Tool object
        **kwargs: Arguments to pass to the tool function

    Returns:
        Result from tool execution
    """
    func = create_executable_function(tool)
    return func(**kwargs)



if __name__ == "__main__":
    # Example usage
    print("Tool Executor - Example Usage")

    # Mock tool for demonstration
    class MockTool:
        def __init__(self):
            self.function_name = "add_numbers"
            self.function_code = """
def add_numbers(a, b):
    '''Add two numbers together'''
    return a + b
"""
            self.helper_functions = {}

    # Create executable function
    mock_tool = MockTool()
    func = create_executable_function(mock_tool)

    # Execute
    result = func(5, 3)
    print(f"Result: {result}")  # Should print 8
    print("✓ Example completed successfully")
