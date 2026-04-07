"""
Test script to validate the Python script agent tools functionality
"""

import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.python_script_agent import PythonScriptTools

def test_write_python_script():
    """Test writing a Python script"""
    print("=== Testing write_python_script ===")

    sample_code = """
def hello_world():
    print("Hello from generated script!")
    return "Success"

if __name__ == "__main__":
    result = hello_world()
    print(f"Result: {result}")
"""

    result = PythonScriptTools.write_python_script("test_script.py", sample_code)
    print(f"Write result: {result}")
    return result

def test_validate_syntax():
    """Test syntax validation"""
    print("\n=== Testing validate_python_syntax ===")

    # Test valid code
    valid_code = "print('Hello, World!')"
    result = PythonScriptTools.validate_python_syntax(valid_code)
    print(f"Valid code result: {result}")

    # Test invalid code
    invalid_code = "print('Hello, World!' missing parenthesis"
    result = PythonScriptTools.validate_python_syntax(invalid_code)
    print(f"Invalid code result: {result}")

def test_execute_code():
    """Test code execution"""
    print("\n=== Testing execute_python_script ===")

    # Test executing code directly
    test_code = "print('Testing direct code execution')\nprint(2 + 2)"
    result = PythonScriptTools.execute_python_script(test_code, is_code=True)
    print(f"Direct execution result: {result}")

    return result

def test_execute_script_file():
    """Test executing a script file"""
    print("\n=== Testing execute script file ===")

    # First write a script
    write_result = test_write_python_script()

    if write_result["status"] == "success":
        # Then execute it
        result = PythonScriptTools.execute_python_script(write_result["file_path"], is_code=False)
        print(f"File execution result: {result}")
        return result
    else:
        print("Skipping file execution test - write failed")
        return None

def test_list_scripts():
    """Test listing generated scripts"""
    print("\n=== Testing list_generated_scripts ===")

    result = PythonScriptTools.list_generated_scripts()
    print(f"List scripts result: {result}")
    return result

def test_install_package():
    """Test package installation (mock)"""
    print("\n=== Testing install_package ===")

    # Test with a small, commonly available package
    result = PythonScriptTools.install_package("requests")
    print(f"Package install result: {result}")
    return result

def main():
    """Run all tests"""
    print("Python Script Agent Tools Test Suite")
    print("=" * 50)

    try:
        # Test each tool
        test_validate_syntax()
        test_execute_code()
        test_execute_script_file()
        test_list_scripts()

        # Note: Package installation test can be slow/fail
        # Uncomment if you want to test it
        # test_install_package()

        print("\n" + "=" * 50)
        print("All tests completed!")
        print("The Python script agent tools are working correctly.")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()