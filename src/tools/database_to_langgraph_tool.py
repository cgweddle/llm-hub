"""
Database Tool to LangGraph Tool Transformer
Converts database tools created by python_script_tool_factory into LangGraph-compatible tools
"""

import sys
import os
from typing import Dict, Any, Callable, Optional, List
import json
import inspect
import logging

# Add database path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from database.database import get_session
    from database.database_setup import Tool
    from langgraph.prebuilt import create_react_agent
    from langchain_core.tools import tool
    from pydantic import BaseModel, Field, create_model
    DATABASE_AVAILABLE = True
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    DATABASE_AVAILABLE = False
    LANGGRAPH_AVAILABLE = False
    print(f"Import error: {e}")

logger = logging.getLogger(__name__)

class LangGraphToolInfo(BaseModel):
    """Information about a converted LangGraph tool"""
    name: str
    description: str
    function: Callable
    input_model: Optional[BaseModel] = None
    original_tool_id: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True

class DatabaseToLangGraphTransformer:
    """Transforms database tools into LangGraph-compatible tools"""

    def __init__(self):
        self.transformed_tools = {}
        self.session = None

    def get_database_tool(self, tool_id: int) -> Optional[Tool]:
        """Retrieve a tool from the database"""
        if not DATABASE_AVAILABLE:
            raise RuntimeError("Database not available")

        if not self.session:
            self.session = get_session()

        return self.session.query(Tool).filter(Tool.id == tool_id).first()

    def create_pydantic_model_from_schema(self, schema: Dict[str, Any], model_name: str) -> BaseModel:
        """Create a Pydantic model from JSON schema"""
        if not schema or not schema.get('properties'):
            return None

        fields = {}
        properties = schema.get('properties', {})
        required_fields = schema.get('required', [])

        for field_name, field_schema in properties.items():
            field_type = self._json_type_to_python_type(field_schema.get('type', 'string'))
            field_description = field_schema.get('description', f'{field_name} parameter')

            if field_name in required_fields:
                fields[field_name] = (field_type, Field(..., description=field_description))
            else:
                fields[field_name] = (field_type, Field(None, description=field_description))

        # Create model with arbitrary types allowed for custom types
        model = create_model(model_name, **fields)
        model.__config__.arbitrary_types_allowed = True
        return model

    def _json_type_to_python_type(self, json_type: str):
        """Convert JSON schema type to Python type"""
        type_mapping = {
            'string': str,
            'integer': int,
            'number': float,
            'boolean': bool,
            'array': List,
            'object': Dict,
            'null': type(None)
        }

        # For standard JSON types, return the mapped Python type
        if json_type in type_mapping:
            return type_mapping[json_type]

        # For custom types, return Any - Pydantic will handle validation at runtime
        # with arbitrary_types_allowed = True
        return Any

    def create_executable_function(self, tool: Tool) -> Callable:
        """Create an executable function from database tool"""
        # Prepare the execution environment
        exec_globals = {
            '__builtins__': __builtins__,
            'json': json,
            'os': os,
            'sys': sys
        }

        # Add helper functions to the execution environment
        if tool.helper_functions:
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

    def transform_tool(self, tool_id: int) -> LangGraphToolInfo:
        """Transform a database tool into a LangGraph tool"""
        try:
            # Get tool from database
            db_tool = self.get_database_tool(tool_id)
            if not db_tool:
                raise ValueError(f"Tool with ID {tool_id} not found")

            logger.info(f"Transforming tool: {db_tool.name} (ID: {tool_id})")

            # Create executable function
            executable_func = self.create_executable_function(db_tool)

            # Create Pydantic input model if schema exists
            input_model = None
            if db_tool.input_schema:
                model_name = f"{db_tool.function_name.title()}Input"
                input_model = self.create_pydantic_model_from_schema(
                    db_tool.input_schema,
                    model_name
                )

            # Create LangGraph tool wrapper
            if input_model:
                # With input validation
                @tool(name=db_tool.name, description=db_tool.description)
                def langgraph_tool(input_data: input_model) -> Any:
                    """LangGraph tool wrapper with input validation"""
                    # Convert Pydantic model to kwargs
                    kwargs = input_data.dict()
                    return executable_func(**kwargs)
            else:
                # Without input validation (fallback)
                @tool(name=db_tool.name, description=db_tool.description)
                def langgraph_tool(**kwargs) -> Any:
                    """LangGraph tool wrapper without input validation"""
                    return executable_func(**kwargs)

            # Store the transformed tool
            tool_info = LangGraphToolInfo(
                name=db_tool.name,
                description=db_tool.description,
                function=langgraph_tool,
                input_model=input_model,
                original_tool_id=tool_id
            )

            self.transformed_tools[tool_id] = tool_info
            logger.info(f"Successfully transformed tool '{db_tool.name}' to LangGraph tool")

            return tool_info

        except Exception as e:
            logger.error(f"Failed to transform tool {tool_id}: {e}")
            raise

    def get_langgraph_tools(self, tool_ids: List[int]) -> List[Callable]:
        """Get a list of LangGraph tools for the given tool IDs"""
        tools = []

        for tool_id in tool_ids:
            if tool_id not in self.transformed_tools:
                self.transform_tool(tool_id)

            tools.append(self.transformed_tools[tool_id].function)

        return tools

    def create_react_agent_with_tools(self, tool_ids: List[int], llm, system_message: str = None):
        """Create a ReAct agent with the transformed tools"""
        if not LANGGRAPH_AVAILABLE:
            raise RuntimeError("LangGraph not available")

        tools = self.get_langgraph_tools(tool_ids)

        if system_message:
            return create_react_agent(llm, tools, state_modifier=system_message)
        else:
            return create_react_agent(llm, tools)

    def validate_tool_compatibility(self, tool_ids: List[int]) -> Dict[str, Any]:
        """Validate that tools can work together in a workflow"""
        compatibility_report = {
            "compatible": True,
            "issues": [],
            "tool_chain": []
        }

        for i, tool_id in enumerate(tool_ids):
            if tool_id not in self.transformed_tools:
                self.transform_tool(tool_id)

            tool_info = self.transformed_tools[tool_id]
            db_tool = self.get_database_tool(tool_id)

            compatibility_report["tool_chain"].append({
                "position": i,
                "tool_id": tool_id,
                "name": tool_info.name,
                "input_schema": db_tool.input_schema,
                "output_schema": db_tool.output_schema
            })

            # Check compatibility with next tool
            if i < len(tool_ids) - 1:
                next_tool_id = tool_ids[i + 1]
                next_db_tool = self.get_database_tool(next_tool_id)

                if db_tool.output_schema and next_db_tool.input_schema:
                    # Basic type compatibility check
                    output_type = db_tool.output_schema.get('type')
                    next_input_props = next_db_tool.input_schema.get('properties', {})

                    # Check if any input property can accept the output
                    compatible_inputs = []
                    for prop_name, prop_schema in next_input_props.items():
                        if prop_schema.get('type') == output_type:
                            compatible_inputs.append(prop_name)

                    if not compatible_inputs:
                        compatibility_report["compatible"] = False
                        compatibility_report["issues"].append(
                            f"Tool {tool_id} output type '{output_type}' "
                            f"not compatible with tool {next_tool_id} inputs"
                        )

        return compatibility_report

    def close(self):
        """Close database session"""
        if self.session:
            self.session.close()

# Example usage and testing
def create_example_tools():
    """Create example tools for testing"""
    if not DATABASE_AVAILABLE:
        print("Database not available - cannot create example tools")
        return []

    from python_script_tool_factory import PythonScriptToolFactory

    # Example script 1: Text processor
    text_script = '''
def process_text(text: str, uppercase: bool = False) -> dict:
    """Process text with optional formatting"""
    return format_text_result(text, uppercase)

def format_text_result(text: str, uppercase: bool) -> dict:
    """Helper to format text processing result"""
    processed = text.upper() if uppercase else text.lower()
    return {
        "original": text,
        "processed": processed,
        "length": len(text),
        "word_count": len(text.split())
    }
'''

    # Example script 2: Number processor
    number_script = '''
def process_numbers(numbers: list) -> dict:
    """Process a list of numbers"""
    return calculate_statistics(numbers)

def calculate_statistics(data: list) -> dict:
    """Calculate basic statistics"""
    if not data:
        return {"error": "Empty dataset"}

    return {
        "sum": sum(data),
        "mean": sum(data) / len(data),
        "count": len(data),
        "min": min(data),
        "max": max(data)
    }
'''

    factory = PythonScriptToolFactory()

    try:
        tool1_id = factory.create_tool_from_script(
            main_function="process_text",
            script_code=text_script,
            tool_name="text_processor",
            tool_description="Process and format text",
            user_id=1
        )

        tool2_id = factory.create_tool_from_script(
            main_function="process_numbers",
            script_code=number_script,
            tool_name="number_processor",
            tool_description="Calculate statistics for numbers",
            user_id=1
        )

        return [tool1_id, tool2_id]

    except Exception as e:
        print(f"Failed to create example tools: {e}")
        return []

if __name__ == "__main__":
    # Test the transformer
    print("Testing Database to LangGraph Tool Transformer...")

    if not DATABASE_AVAILABLE or not LANGGRAPH_AVAILABLE:
        print("❌ Required dependencies not available")
        sys.exit(1)

    try:
        # Create example tools
        tool_ids = create_example_tools()
        if not tool_ids:
            print("❌ Failed to create example tools")
            sys.exit(1)

        print(f"✓ Created example tools: {tool_ids}")

        # Test transformation
        transformer = DatabaseToLangGraphTransformer()

        for tool_id in tool_ids:
            tool_info = transformer.transform_tool(tool_id)
            print(f"✓ Transformed tool: {tool_info.name}")

        # Test compatibility validation
        compatibility = transformer.validate_tool_compatibility(tool_ids)
        print(f"✓ Compatibility check: {compatibility['compatible']}")

        # Test getting LangGraph tools
        langgraph_tools = transformer.get_langgraph_tools(tool_ids)
        print(f"✓ Created {len(langgraph_tools)} LangGraph tools")

        transformer.close()
        print("✓ All tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()