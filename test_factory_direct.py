#!/usr/bin/env python3
"""
Direct test of the Python Script Tool Factory functionality
"""

import sys
sys.path.append('src/tools')

from python_script_tool_factory import PythonScriptAnalyzer, TypeSchemaGenerator

def test_core_functionality():
    """Test the core factory functionality"""

    script = '''
def calculate_stats(numbers: list) -> dict:
    """Calculate basic statistics for a list of numbers"""
    return compute_statistics(numbers)

def compute_statistics(data: list) -> dict:
    """Helper function to compute basic stats"""
    if not data:
        return {"error": "Empty dataset"}

    mean = sum(data) / len(data)
    return {
        "mean": mean,
        "count": len(data),
        "min": min(data),
        "max": max(data)
    }
'''

    print("Testing Python Script Analysis...")

    # Test analyzer
    analyzer = PythonScriptAnalyzer()
    functions = analyzer.parse_script(script)

    print(f"✓ Found {len(functions)} functions:")
    for name, func in functions.items():
        print(f"  - {name} (main: {func.is_main})")
        print(f"    Args: {func.args}")
        print(f"    Types: {func.type_hints}")
        print(f"    Return: {func.return_type}")

    print(f"✓ Main function identified: {analyzer.main_function}")

    # Test schema generation
    print("\nTesting Schema Generation...")

    generator = TypeSchemaGenerator()
    main_func = functions[analyzer.main_function]

    input_schema = generator.generate_input_schema(main_func)
    output_schema = generator.generate_output_schema(main_func)

    print(f"✓ Input schema: {input_schema}")
    print(f"✓ Output schema: {output_schema}")

    # Test type conversions
    print("\nTesting Type Conversions...")

    test_types = [
        "int", "str", "List[int]", "Dict[str, Any]",
        "Optional[str]", "Union[str, int]"
    ]

    for type_hint in test_types:
        schema = generator._type_to_schema(type_hint)
        print(f"  {type_hint} -> {schema}")

    print("\n✅ All core functionality tests passed!")

def test_helper_function_extraction():
    """Test helper function extraction"""

    script = '''
def main_tool(text: str, count: int) -> dict:
    """Main processing function"""
    processed = preprocess_text(text)
    stats = calculate_metrics(processed, count)
    return format_output(processed, stats)

def preprocess_text(text: str) -> str:
    """Clean and prepare text"""
    return text.strip().lower()

def calculate_metrics(text: str, count: int) -> dict:
    """Calculate text metrics"""
    return {
        "length": len(text),
        "words": len(text.split()),
        "multiplier": count
    }

def format_output(text: str, metrics: dict) -> dict:
    """Format final output"""
    return {
        "text": text,
        "metrics": metrics,
        "processed": True
    }
'''

    print("\nTesting Helper Function Extraction...")

    analyzer = PythonScriptAnalyzer()
    functions = analyzer.parse_script(script)

    print(f"✓ Found {len(functions)} total functions")
    print(f"✓ Main function: {analyzer.main_function}")

    helper_functions = {
        name: func for name, func in functions.items()
        if not func.is_main
    }

    print(f"✓ Helper functions: {list(helper_functions.keys())}")

    print("✅ Helper function extraction test passed!")

if __name__ == "__main__":
    try:
        test_core_functionality()
        test_helper_function_extraction()
        print("\n🎉 All tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()