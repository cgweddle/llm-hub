"""
Python Script to Tool Factory
Converts Python scripts with helper functions into database tools with proper type validation
"""

import ast
import inspect
import re
import sys
import os
from typing import Dict, Any, List, Optional, Union, get_type_hints, get_origin, get_args
import logging

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    class BaseModel:
        pass

# Add database path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from database.database import create_tool as db_create_tool, get_session




logger = logging.getLogger(__name__)

class FunctionInfo:
    """Information about a parsed function"""
    def __init__(self, name: str, code: str, docstring: Optional[str],
                 args: List[str], type_hints: Dict[str, Any],
                 return_type: Optional[str], is_main: bool = False):
        self.name = name
        self.code = code
        self.docstring = docstring
        self.args = args
        self.type_hints = type_hints
        self.return_type = return_type
        self.is_main = is_main

class PydanticModelInfo:
    """Information about a detected Pydantic model"""
    def __init__(self, name: str, code: str, schema: Dict[str, Any],
                 fields: Dict[str, Any], docstring: Optional[str] = None):
        self.name = name
        self.code = code
        self.schema = schema
        self.fields = fields
        self.docstring = docstring

class PythonScriptAnalyzer:
    """Analyzes Python scripts to extract functions and type information"""

    def __init__(self, main_function: str=None):
        self.functions = {}
        self.main_function = main_function
        self.pydantic_models = {}

    def parse_script(self, script_code: str) -> Dict[str, FunctionInfo]:
        """Parse a Python script and extract all functions"""
        try:
            tree = ast.parse(script_code)
            self.functions = {}

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = self._extract_function_info(node, script_code)
                    self.functions[func_info.name] = func_info


            return self.functions

        except SyntaxError as e:
            logger.error(f"Syntax error in script: {e}")
            raise ValueError(f"Invalid Python syntax: {e}")

    def _extract_function_info(self, node: ast.FunctionDef, script_code: str) -> FunctionInfo:
        """Extract detailed information about a function"""
        # Get function code
        lines = script_code.split('\n')
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else len(lines)
        func_code = '\n'.join(lines[start_line:end_line])

        # Get docstring
        docstring = ast.get_docstring(node)

        # Get arguments
        args = [arg.arg for arg in node.args.args]

        # Extract type hints from annotations
        type_hints = {}
        for arg in node.args.args:
            if arg.annotation:
                type_hints[arg.arg] = ast.unparse(arg.annotation)

        # Get return type
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns)

        return FunctionInfo(
            name=node.name,
            code=func_code,
            docstring=docstring,
            args=args,
            type_hints=type_hints,
            return_type=return_type
        )





class TypeSchemaGenerator:
    """Generates JSON schemas from Python type hints"""


    def generate_input_schema(self, func_info: FunctionInfo) -> Dict[str, Any]:
        """Generate JSON schema for function inputs"""
        parameters = {}
        for arg in func_info.args:
            if arg == 'self': 
                continue
            arg_type = func_info.type_hints.get(arg, 'Any')
            if arg_type.startswith('Optional['):
                base_type = arg_type[9:-1]
                is_optional = True
            else:
                base_type = arg_type
                is_optional = False
            parameters[arg] = {"type": base_type, "optional": is_optional}
        return parameters


    def generate_output_schema(self, func_info: FunctionInfo) -> Dict[str, Any]:
        #TODO: Handle List, Dict, Tuple and other multi-return types
        """
        Generate JSON schema for function output
        Return 
        {"type": type}
        Or, if multiple return types,
        {"type": Tuple[type1, type2]}
        """
        return {
            "type": func_info.return_type if func_info.return_type else "None"
        }


class PythonScriptToolFactory:
    """Factory for creating database tools from Python scripts"""

    def __init__(self, main_function: str=None):
        self.analyzer = PythonScriptAnalyzer(main_function=main_function)
        self.schema_generator = TypeSchemaGenerator()

    def create_tool_from_script(self, script_code: str, tool_name: str,
                              tool_description: str, user_id: int = 1) -> int:
        """
        Create a tool in the database from a Python script

        Args:
            script_code: Complete Python script with functions
            tool_name: Name for the tool
            tool_description: Description of what the tool does
            user_id: User ID to associate with the tool

        Returns:
            ID of created tool
        """
        try:
            # Parse the script
            functions = self.analyzer.parse_script(script_code)

            if not functions:
                raise ValueError("No functions found in script")

            if not self.analyzer.main_function:
                raise ValueError("Could not identify main function")

            main_func = functions[self.analyzer.main_function]

            # Generate schemas
            input_schema = self.schema_generator.generate_input_schema(main_func)
            output_schema = self.schema_generator.generate_output_schema(main_func)

            # Create tool in database
            session = get_session()
            try:
                tool = db_create_tool(
                    session=session,
                    user_id=user_id,
                    name=tool_name,
                    description=tool_description,
                    tool_type="python_script",
                    function_name=main_func.name,
                    function_code=main_func.code,
                    script_code=script_code,
                    input_schema=input_schema,
                    output_schema=output_schema,
                    is_public=True
                )

                logger.info(f"Created tool '{tool_name}' with ID {tool.id}")
                return tool.id

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Failed to create tool from script: {e}")
            raise


# Example usage
if __name__ == "__main__":
    from typing import List, Dict

    # Example script with functions
    example_script = '''
def calculate_stats(numbers: List[int]) -> Dict[str, float]:
    """Calculate basic statistics for a list of numbers"""
    return compute_statistics(numbers)

def compute_statistics(data: List[int]) -> Dict[str, float]:
    """Helper function to compute mean, median, std dev"""
    if not data:
        return {"error": "Empty dataset"}

    mean = sum(data) / len(data)
    sorted_data = sorted(data)
    n = len(sorted_data)
    median = sorted_data[n//2] if n % 2 == 1 else (sorted_data[n//2-1] + sorted_data[n//2]) / 2

    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std_dev = variance ** 0.5

    return {
        "mean": mean,
        "median": median,
        "std_dev": std_dev,
        "count": len(data),
        "min": min(data),
        "max": max(data)
    }
'''

    factory = PythonScriptToolFactory(main_function="calculate_stats")

    try:
        tool_id = factory.create_tool_from_script(
            script_code=example_script,
            tool_name="statistics_calculator",
            tool_description="Calculate statistical measures for numeric data",
            user_id=1
        )
        print(f"✓ Created example tool with ID: {tool_id}")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()