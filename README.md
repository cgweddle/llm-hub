# llm-hub

## Python Script Tool Factory

The Python Script Tool Factory converts Python scripts with helper functions into database tools with proper type validation and flow compatibility checking.

### Features

- **AST-based Function Parsing**: Extracts functions, type hints, and docstrings from Python scripts
- **Type Schema Generation**: Automatically generates JSON schemas from Python type annotations
- **Helper Function Support**: Separates main functions from helper functions for clean tool organization
- **Flow Validation**: Creates input/output schemas to validate tool compatibility in workflows
- **User-defined Main Functions**: Explicitly specify which function serves as the tool's entry point

### Usage

```python
from src.tools.python_script_tool_factory import PythonScriptToolFactory

# Create factory with specified main function
factory = PythonScriptToolFactory(main_function="process_data")

# Create tool from script
tool_id = factory.create_tool_from_script(
    script_code=your_python_script,
    tool_name="data_processor",
    tool_description="Process and transform data",
    user_id=1
)

# Validate tool compatibility for workflows
compatibility = factory.validate_tool_compatibility(tool1_id, tool2_id)
```

### Supported Type Annotations

- Basic types: `str`, `int`, `float`, `bool`, `list`, `dict`
- Generic types: `List[T]`, `Dict[K,V]`, `Optional[T]`, `Union[T]`
- Custom types with automatic fallback to string description

### Database Integration

Tools created through the factory include:
- Main function code and metadata
- Helper functions stored as JSON
- Input schema for parameter validation
- Output schema for flow compatibility checking