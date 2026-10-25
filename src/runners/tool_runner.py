"""
Tool Runner

In-process tool compilation for flow execution. A tool's full script_code is
exec'd into a fresh namespace and its main_function returned as a plain
callable — inputs and outputs are native Python objects, no serialization.

This module is only ever imported by flow-running processes (the local flow
child, the hosted flow-runner container, pytest) — never by the backend API
process, which must not execute user tool code.
"""

import logging
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


class ToolCompileError(Exception):
    """Tool script failed to exec or its main function is missing/uncallable."""


def _compile(script_code: str, main_function: str) -> Callable:
    # __name__ is set so `if __name__ == "__main__"` blocks stay inert;
    # exec fills in __builtins__ automatically.
    namespace: Dict[str, Any] = {"__name__": "__tool_execution__"}
    try:
        exec(script_code, namespace)
    except Exception as e:
        raise ToolCompileError(
            f"Tool script failed to load: {type(e).__name__}: {e}"
        ) from e
    func = namespace.get(main_function)
    if not callable(func):
        raise ToolCompileError(
            f"Main function '{main_function}' not found or not callable in tool script"
        )
    return func


def compile_tool(tool) -> Callable:
    """Compile a database Tool's script_code and return its main function.

    Each call execs into fresh globals, so edited tools recompile cleanly and
    tools never share module-level state.
    """
    return _compile(tool.script_code, tool.main_function)


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
