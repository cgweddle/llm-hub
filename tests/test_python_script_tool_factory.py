"""
Test suite for the enhanced Python Script Tool Factory
"""

import pytest
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.tools.python_script_tool_factory import (
        PythonScriptAnalyzer,
        TypeSchemaGenerator,
        PythonScriptToolFactory,
        FunctionInfo
    )
    FACTORY_AVAILABLE = True
except ImportError as e:
    print(f"Factory not available: {e}")
    FACTORY_AVAILABLE = False

@pytest.mark.skipif(not FACTORY_AVAILABLE, reason="Factory not available")
class TestPythonScriptAnalyzer:
    """Test the Python script analyzer"""

    def test_parse_simple_script(self):
        """Test parsing a simple script"""
        script = '''
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

def multiply(x: float, y: float) -> float:
    """Multiply two numbers"""
    return x * y
'''
        analyzer = PythonScriptAnalyzer()
        functions = analyzer.parse_script(script)

        assert len(functions) == 2
        assert "add_numbers" in functions
        assert "multiply" in functions

        add_func = functions["add_numbers"]
        assert add_func.name == "add_numbers"
        assert add_func.docstring == "Add two numbers together"
        assert add_func.args == ["a", "b"]
        assert add_func.type_hints == {"a": "int", "b": "int"}
        assert add_func.return_type == "int"

    def test_identify_main_function(self):
        """Test main function identification"""
        script = '''
def helper_function(x):
    return x * 2

def main_processing_function(data: list) -> dict:
    """This is the main function"""
    result = []
    for item in data:
        result.append(helper_function(item))
    return {"processed": result, "count": len(result)}
'''
        analyzer = PythonScriptAnalyzer(main_function="main_processing_function")
        functions = analyzer.parse_script(script)

        assert analyzer.main_function == "main_processing_function"
        assert functions["main_processing_function"].is_main



@pytest.mark.skipif(not FACTORY_AVAILABLE, reason="Factory not available")
class TestTypeSchemaGenerator:
    """Test the type schema generator"""

    def test_generate_input_schema(self):
        """Test input schema generation"""
        func_info = FunctionInfo(
            name="test_func",
            code="def test_func(a: int, b: str, c: Optional[float]) -> dict: pass",
            docstring="Test function",
            args=["a", "b", "c"],
            type_hints={"a": "int", "b": "str", "c": "Optional[float]"},
            return_type="dict"
        )

        generator = TypeSchemaGenerator()
        schema = generator.generate_input_schema(func_info)

        expected = {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "string"},
                "c": {"type": "number", "nullable": True}
            },
            "required": ["a", "b", "c"]
        }

        assert schema == expected

    def test_generate_output_schema(self):
        """Test output schema generation"""
        func_info = FunctionInfo(
            name="test_func",
            code="",
            docstring="",
            args=[],
            type_hints={},
            return_type="Dict[str, int]"
        )

        generator = TypeSchemaGenerator()
        schema = generator.generate_output_schema(func_info)

        assert schema == {"type": "object"}

    def test_complex_types(self):
        """Test handling of complex types"""
        generator = TypeSchemaGenerator()

        # Test List type
        list_schema = generator._type_to_schema("List[str]")
        assert list_schema == {"type": "array", "items": {"type": "string"}}

        # Test Optional type
        optional_schema = generator._type_to_schema("Optional[int]")
        assert optional_schema == {"type": "integer", "nullable": True}

        # Test Union type
        union_schema = generator._type_to_schema("Union[str, int]")
        assert union_schema == {"type": "string"}  # Takes first type

@pytest.mark.skipif(not FACTORY_AVAILABLE, reason="Factory not available")
class TestPythonScriptToolFactory:
    """Test the tool factory"""

    def test_create_tool_from_script(self):
        """Test creating a tool from a script"""
        script = '''
def process_text(text: str, uppercase: bool = False) -> dict:
    """Process text with optional uppercase conversion"""
    return format_result(text, uppercase)

def format_result(text: str, uppercase: bool) -> dict:
    """Helper function to format the result"""
    if uppercase:
        text = text.upper()

    return {
        "original": text if not uppercase else text.lower(),
        "processed": text,
        "length": len(text),
        "uppercase": uppercase
    }
'''

        # Mock the database parts for testing
        original_get_session = None
        original_db_create_tool = None

        try:
            from src.tools.python_script_tool_factory import get_session
            from src.database.database import create_tool as db_create_tool

            # Mock database functions
            class MockSession:
                def close(self):
                    pass

            class MockTool:
                def __init__(self):
                    self.id = 123

            def mock_get_session():
                return MockSession()

            def mock_db_create_tool(**kwargs):
                return MockTool()

            # Replace functions
            import src.tools.python_script_tool_factory as factory_module
            factory_module.get_session = mock_get_session
            factory_module.db_create_tool = mock_db_create_tool

            # Test the factory
            factory = PythonScriptToolFactory()
            tool_id = factory.create_tool_from_script(
                script_code=script,
                tool_name="text_processor",
                tool_description="Process text with formatting options",
                user_id=1
            )

            assert tool_id == 123

        except ImportError:
            # Skip test if database modules not available
            pytest.skip("Database modules not available")

    def test_validate_script_syntax(self):
        """Test script syntax validation"""
        factory = PythonScriptToolFactory()

        # Valid script
        valid_script = '''
def valid_function(x: int) -> int:
    return x * 2
'''

        # Should not raise exception
        functions = factory.analyzer.parse_script(valid_script)
        assert "valid_function" in functions

        # Invalid script
        invalid_script = '''
def invalid_function(x: int) -> int
    return x * 2  # Missing colon
'''

        with pytest.raises(ValueError, match="Invalid Python syntax"):
            factory.analyzer.parse_script(invalid_script)

def test_example_usage():
    """Test that the example script works"""
    if not FACTORY_AVAILABLE:
        pytest.skip("Factory not available")

    # Test the example tool creation logic without database
    script = '''
def calculate_stats(numbers: list) -> dict:
    """Calculate basic statistics for a list of numbers"""
    return compute_statistics(numbers)

def compute_statistics(data: list) -> dict:
    """Helper function to compute mean, median, std dev"""
    if not data:
        return {"error": "Empty dataset"}

    mean = sum(data) / len(data)
    sorted_data = sorted(data)
    n = len(sorted_data)
    median = sorted_data[n//2] if n % 2 == 1 else (sorted_data[n//2-1] + sorted_data[n//2]) / 2

    return {
        "mean": mean,
        "median": median,
        "count": len(data)
    }
'''

    analyzer = PythonScriptAnalyzer()
    functions = analyzer.parse_script(script)

    assert len(functions) == 2
    assert "calculate_stats" in functions
    assert "compute_statistics" in functions
    assert analyzer.main_function == "calculate_stats"

    # Test schema generation
    main_func = functions["calculate_stats"]
    generator = TypeSchemaGenerator()

    input_schema = generator.generate_input_schema(main_func)
    output_schema = generator.generate_output_schema(main_func)

    # Verify schemas
    assert input_schema["type"] == "object"
    assert "numbers" in input_schema["properties"]
    assert output_schema["type"] == "object"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])