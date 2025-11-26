#!/usr/bin/env python3
"""
Create a math equation solver tool using PythonScriptToolFactory
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.factories.python_script_tool_factory import PythonScriptToolFactory

# Math equation solver script
math_solver_script = '''
def solve_equation(equation: str) -> float:
    """
    Solve a mathematical equation provided as a string

    Args:
        equation: A string containing a mathematical equation (e.g., "2 + 2", "10 * 5 - 3")

    Returns:
        The numeric result of the equation
    """
    return evaluate_safe(equation)

def evaluate_safe(equation: str) -> float:
    """
    Safely evaluate a mathematical equation string

    Uses Python's eval with restricted namespace for safety
    """
    equation = equation.strip()

    allowed_names = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'pow': pow,
    }

    try:
        result = eval(equation, {"__builtins__": {}}, allowed_names)
        return float(result)
    except Exception as e:
        raise ValueError(f"Invalid equation: {equation}. Error: {str(e)}")
'''

if __name__ == "__main__":
    # Create factory with main_function parameter
    factory = PythonScriptToolFactory(main_function="solve_equation")

    try:
        tool_id = factory.create_tool_from_script(
            script_code=math_solver_script,
            tool_name="math_equation_solver",
            tool_description="Solves mathematical equations from string expressions. Supports basic arithmetic operations (+, -, *, /) and functions (abs, round, min, max, pow).",
            user_id=1
        )

        print(f"✓ Successfully created math_equation_solver tool with ID: {tool_id}")

    except Exception as e:
        print(f"✗ Error creating tool: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
