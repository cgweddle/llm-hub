"""
PydanticAI Tool Converter
Converts database Tool records into PydanticAI-compatible tool functions

Supports two schema formats:
1. Python type strings (from python_script_tool_factory.py):
   {"arg_name": {"type": "List[int]", "optional": False}}

2. JSON Schema (OpenAPI-style):
   {"type": "object", "properties": {"arg_name": {"type": "array", "items": {"type": "integer"}}}}
"""

import inspect
import json
import logging
import os
import sys
from typing import Dict, Any, Callable, Type, Optional, Tuple, List

from pydantic import BaseModel, Field, create_model

from src.runners.tool_runner import eval_type_string

logger = logging.getLogger(__name__)


class PydanticAIToolConverter:
    """
    Converts database Tool records into PydanticAI tool functions.

    Handles:
    - Python type string → Pydantic model conversion
    - JSON Schema → Pydantic model conversion
    - Function code compilation and execution
    - Input validation and output validation
    - Helper function integration
    - Sync/async function detection
    """

    def __init__(self, session=None):
        self.session = session
        self._tool_cache: Dict[int, Tuple[Callable, Type[BaseModel], Optional[Type[BaseModel]]]] = {}

    def convert_tool(self, tool) -> Tuple[Callable, Type[BaseModel], Optional[Type[BaseModel]]]:
        """
        Convert a database Tool record into a PydanticAI-compatible tool function.

        Args:
            tool: Database Tool object with attributes:
                - id: int
                - name: str
                - description: str
                - input_schema: dict (Python type strings or JSON Schema)
                - output_schema: dict (Python type strings or JSON Schema)
                - function_code: str (Python code)
                - main_function: str (function name)
                - helper_functions: dict (name -> code mapping)

        Returns:
            Tuple of (tool_function, input_model, output_model)
            - tool_function: Callable that can be used with @agent.tool()
            - input_model: Pydantic model for tool input validation
            - output_model: Pydantic model for tool output validation (None if no output_schema)
        """
        # Check cache first
        if hasattr(tool, 'id') and tool.id in self._tool_cache:
            logger.debug(f"Using cached tool conversion for: {tool.name}")
            return self._tool_cache[tool.id]

        logger.info(f"Converting tool: {tool.name} (type: {tool.tool_type})")

        safe_name = tool.name.replace(' ', '_').replace('-', '_')

        # Create input model from input_schema
        input_model = self.schema_to_pydantic(
            schema=tool.input_schema or {},
            model_name=f"{safe_name}_Input"
        )

        # Create output model from output_schema
        output_schema = getattr(tool, 'output_schema', None) or {}
        output_model = self.schema_to_pydantic(
            schema=output_schema,
            model_name=f"{safe_name}_Output"
        ) if output_schema else None

        # Create executable tool function
        tool_function = self.create_tool_function(
            tool=tool,
            input_model=input_model,
            output_model=output_model
        )

        # Cache the result
        if hasattr(tool, 'id'):
            self._tool_cache[tool.id] = (tool_function, input_model, output_model)

        logger.info(f"✓ Successfully converted tool: {tool.name}")
        return tool_function, input_model, output_model

    def _is_python_type_format(self, schema: dict) -> bool:
        """
        Detect if schema is in Python type string format.

        Python type format: {"arg_name": {"type": "List[int]", "optional": False}}
        JSON Schema format: {"type": "object", "properties": {...}}
        """
        if not schema or not isinstance(schema, dict):
            return False

        if schema.get("type") == "object" and "properties" in schema:
            return False

        for key, value in schema.items():
            if isinstance(value, dict) and "type" in value:
                type_val = value.get("type", "")
                if isinstance(type_val, str):
                    json_schema_types = {"string", "integer", "number", "boolean", "array", "object", "null"}
                    if type_val not in json_schema_types:
                        return True
                    if "optional" in value:
                        return True

        return False

    def schema_to_pydantic(
        self,
        schema: dict,
        model_name: str = "DynamicModel"
    ) -> Type[BaseModel]:
        """
        Convert a schema (either Python type format or JSON Schema) to a Pydantic model.
        Automatically detects the format and delegates to the appropriate converter.

        Args:
            schema: Schema dict (Python type strings or JSON Schema format)
            model_name: Name for the generated Pydantic model

        Returns:
            Dynamically created Pydantic model class
        """
        if not schema or not isinstance(schema, dict):
            return create_model(model_name)

        if self._is_python_type_format(schema):
            logger.debug(f"Detected Python type string format for {model_name}")
            return self._python_types_to_pydantic(schema, model_name)
        else:
            logger.debug(f"Detected JSON Schema format for {model_name}")
            return self.json_schema_to_pydantic(schema, model_name)

    def _python_types_to_pydantic(
        self,
        schema: dict,
        model_name: str
    ) -> Type[BaseModel]:
        """
        Convert Python type string schema to Pydantic model.
        Uses eval_type_string from tool_runner.py for type resolution.

        Schema format:
        {
            "arg_name": {"type": "List[int]", "optional": False},
            "other_arg": {"type": "str", "optional": True}
        }
        """
        field_definitions = {}

        for field_name, field_info in schema.items():
            if not isinstance(field_info, dict):
                continue

            type_str = field_info.get("type", "Any")
            is_optional = field_info.get("optional", False)

            # Evaluate the Python type string to get the actual type
            field_type = eval_type_string(type_str)

            if is_optional:
                field_definitions[field_name] = (
                    Optional[field_type],
                    Field(default=None)
                )
            else:
                field_definitions[field_name] = (
                    field_type,
                    Field(...)
                )

            logger.debug(f"Field '{field_name}': {type_str} -> {field_type} (optional={is_optional})")

        try:
            model = create_model(model_name, **field_definitions)
            logger.debug(f"Created Pydantic model: {model_name} with fields: {list(field_definitions.keys())}")
            return model
        except Exception as e:
            logger.error(f"Failed to create Pydantic model '{model_name}': {e}")
            return create_model(model_name)

    def json_schema_to_pydantic(
        self,
        schema: dict,
        model_name: str = "DynamicModel"
    ) -> Type[BaseModel]:
        """
        Convert JSON Schema to Pydantic model dynamically.

        Args:
            schema: JSON Schema dict with 'properties', 'required', etc.
            model_name: Name for the generated Pydantic model

        Returns:
            Dynamically created Pydantic model class
        """
        if not schema or not isinstance(schema, dict):
            return create_model(model_name)

        properties = schema.get("properties", {})
        required_fields = schema.get("required", [])

        field_definitions = {}

        for field_name, field_schema in properties.items():
            field_type = self._json_type_to_python_type(field_schema)
            field_description = field_schema.get("description", "")
            is_required = field_name in required_fields

            if is_required:
                field_definitions[field_name] = (
                    field_type,
                    Field(..., description=field_description)
                )
            else:
                default_value = field_schema.get("default", None)
                field_definitions[field_name] = (
                    Optional[field_type],
                    Field(default=default_value, description=field_description)
                )

        try:
            model = create_model(model_name, **field_definitions)
            logger.debug(f"Created Pydantic model: {model_name} with fields: {list(field_definitions.keys())}")
            return model
        except Exception as e:
            logger.error(f"Failed to create Pydantic model '{model_name}': {e}")
            return create_model(model_name)

    def _json_type_to_python_type(self, field_schema: dict) -> Type:
        """Convert JSON Schema type to Python type annotation."""
        from typing import List, Dict, Any

        json_type = field_schema.get("type", "string")

        if json_type == "array":
            items_schema = field_schema.get("items", {})
            if items_schema:
                item_type = self._json_type_to_python_type(items_schema)
                return List[item_type]
            return List[Any]

        if json_type == "object":
            return Dict[str, Any]

        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "null": type(None),
        }

        return type_mapping.get(json_type, Any)

    def create_tool_function(
        self,
        tool,
        input_model: Type[BaseModel],
        output_model: Optional[Type[BaseModel]] = None
    ) -> Callable:
        """
        Create an executable tool function from database tool.

        Args:
            tool: Database Tool object
            input_model: Pydantic model for input validation
            output_model: Optional Pydantic model for output validation
        """
        executable_func = self._compile_function_code(tool)
        is_async = self._is_async_function(tool.function_code, tool.main_function)

        if is_async:
            return self._create_async_wrapper(
                executable_func=executable_func,
                tool_name=tool.name,
                tool_description=tool.description,
                input_model=input_model,
                output_model=output_model
            )
        else:
            return self._create_sync_wrapper(
                executable_func=executable_func,
                tool_name=tool.name,
                tool_description=tool.description,
                input_model=input_model,
                output_model=output_model
            )

    def _compile_function_code(self, tool) -> Callable:
        """
        Compile tool's function code into an executable function.
        Similar to compile_tool() in tool_runner.py
        """
        exec_globals = {
            '__builtins__': __builtins__,
            'json': json,
            'os': os,
            'sys': sys,
        }

        try:
            exec(tool.script_code, exec_globals)
            main_function = exec_globals.get(tool.main_function)

            if not main_function:
                raise ValueError(f"Main function '{tool.main_function}' not found after execution")

            if not callable(main_function):
                raise ValueError(f"'{tool.main_function}' is not callable")

            logger.debug(f"Compiled function: {tool.main_function}")
            return main_function

        except Exception as e:
            logger.error(f"Failed to compile function code for tool '{tool.name}': {e}")
            raise ValueError(f"Function compilation failed: {e}")

    def _is_async_function(self, function_code: str, function_name: str) -> bool:
        """Detect if a function is async by checking the code."""
        try:
            return f"async def {function_name}" in function_code
        except Exception:
            return False

    def _validate_output(self, result: Any, output_model: Optional[Type[BaseModel]], tool_name: str) -> Any:
        """
        Validate tool output against the output model.

        Args:
            result: Raw result from tool execution
            output_model: Pydantic model to validate against (None to skip)
            tool_name: Tool name for logging

        Returns:
            The result, validated if output_model is provided
        """
        if not output_model:
            return result

        try:
            if isinstance(result, dict):
                validated = output_model(**result)
                return validated.model_dump()
            else:
                # Non-dict results (str, int, list, etc.) can't be validated
                # against a Pydantic model — return as-is
                return result
        except Exception as e:
            logger.warning(f"Tool '{tool_name}' output validation failed: {e}. Returning raw result.")
            return result

    def _build_signature(self, input_model: Type[BaseModel]) -> inspect.Signature:
        """
        Build an inspect.Signature from a Pydantic input model so that
        PydanticAI can introspect the wrapper's typed parameters.
        """
        params = []
        for field_name, field_info in input_model.model_fields.items():
            default = (
                field_info.default
                if field_info.is_required() is False
                else inspect.Parameter.empty
            )
            params.append(inspect.Parameter(
                field_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=field_info.annotation,
            ))
        return inspect.Signature(params, return_annotation=Any)

    def _apply_signature(
        self,
        func: Callable,
        input_model: Type[BaseModel],
        tool_name: str,
        tool_description: str,
    ) -> None:
        """Apply name, docstring, signature, and annotations to a wrapper."""
        func.__name__ = tool_name.replace(" ", "_").replace("-", "_")
        func.__doc__ = tool_description or f"Execute {tool_name}"
        func.__signature__ = self._build_signature(input_model)
        func.__annotations__ = {
            name: info.annotation
            for name, info in input_model.model_fields.items()
        }
        func.__annotations__["return"] = Any

    def _create_sync_wrapper(
        self,
        executable_func: Callable,
        tool_name: str,
        tool_description: str,
        input_model: Type[BaseModel],
        output_model: Optional[Type[BaseModel]] = None
    ) -> Callable:
        """Create synchronous wrapper function for PydanticAI"""
        validate_output = self._validate_output

        def tool_wrapper(**kwargs) -> Any:
            try:
                if kwargs:
                    validated_input = input_model(**kwargs)
                    result = executable_func(**validated_input.model_dump())
                else:
                    result = executable_func()

                return validate_output(result, output_model, tool_name)

            except Exception as e:
                logger.error(f"Tool '{tool_name}' execution failed: {e}")
                raise

        self._apply_signature(tool_wrapper, input_model, tool_name, tool_description)

        return tool_wrapper

    def _create_async_wrapper(
        self,
        executable_func: Callable,
        tool_name: str,
        tool_description: str,
        input_model: Type[BaseModel],
        output_model: Optional[Type[BaseModel]] = None
    ) -> Callable:
        """Create asynchronous wrapper function for PydanticAI"""
        validate_output = self._validate_output

        async def tool_wrapper(**kwargs) -> Any:
            try:
                if kwargs:
                    validated_input = input_model(**kwargs)
                    result = await executable_func(**validated_input.model_dump())
                else:
                    result = await executable_func()

                return validate_output(result, output_model, tool_name)

            except Exception as e:
                logger.error(f"Tool '{tool_name}' execution failed: {e}")
                raise

        self._apply_signature(tool_wrapper, input_model, tool_name, tool_description)

        return tool_wrapper

    def clear_cache(self):
        """Clear the tool conversion cache"""
        self._tool_cache.clear()
        logger.debug("Tool conversion cache cleared")


# Convenience function for direct usage
def convert_database_tool_to_pydanticai(tool) -> Tuple[Callable, Type[BaseModel], Optional[Type[BaseModel]]]:
    """
    Convenience function to convert a database tool to PydanticAI format.

    Args:
        tool: Database Tool object

    Returns:
        Tuple of (tool_function, input_model, output_model)
    """
    converter = PydanticAIToolConverter()
    return converter.convert_tool(tool)
