#!/usr/bin/env python3
"""
Generic tool creation script - creates a tool from a Python file

Usage:
    python create_tool.py <python_file> <tool_name> <tool_description> [main_function] [user_id]

Example:
    python create_tool.py statistics_calculator.py "Statistics Calculator" "Calculate stats from numbers" calculate_statistics 1
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.factories.python_script_tool_factory import PythonScriptToolFactory


def create_tool_from_file(python_file: str, tool_name: str, tool_description: str,
                          main_function: str = None, user_id: int = 1):
    """
    Create a tool from a Python file

    Args:
        python_file: Path to the Python file containing the tool code
        tool_name: Name for the tool
        tool_description: Description of what the tool does
        main_function: Name of the main function (optional - will auto-detect)
        user_id: User ID to associate with the tool (default: 1)

    Returns:
        Tool ID if successful
    """
    # Read the Python file
    if not os.path.exists(python_file):
        raise FileNotFoundError(f"Python file not found: {python_file}")

    with open(python_file, 'r') as f:
        script_code = f.read()

    # Create factory
    factory = PythonScriptToolFactory(main_function=main_function)

    # Create tool in database
    tool_id = factory.create_tool_from_script(
        script_code=script_code,
        tool_name=tool_name,
        tool_description=tool_description,
        user_id=user_id
    )

    return tool_id


def main():
    parser = argparse.ArgumentParser(
        description="Create a tool from a Python file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_tool.py statistics.py "Stats Calculator" "Calculate statistics"
  python create_tool.py solver.py "Math Solver" "Solve equations" solve_equation
  python create_tool.py tool.py "My Tool" "Does stuff" my_function 2
        """
    )

    parser.add_argument('python_file', help='Path to Python file containing tool code')
    parser.add_argument('tool_name', help='Name for the tool')
    parser.add_argument('tool_description', help='Description of what the tool does')
    parser.add_argument('main_function', nargs='?', default=None,
                       help='Name of main function (optional - will auto-detect)')
    parser.add_argument('user_id', nargs='?', type=int, default=1,
                       help='User ID (default: 1)')

    args = parser.parse_args()

    try:
        tool_id = create_tool_from_file(
            python_file=args.python_file,
            tool_name=args.tool_name,
            tool_description=args.tool_description,
            main_function=args.main_function,
            user_id=args.user_id
        )

        print(f"✓ Successfully created tool '{args.tool_name}' with ID: {tool_id}")

    except Exception as e:
        print(f"✗ Error creating tool: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
