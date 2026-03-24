# Math equation solver script

## Main function
def solve_equation(equation: str) -> float:
    """
    Solve a mathematical equation provided as a string

    Args:
        equation: A string containing a mathematical equation (e.g., "2 + 2", "10 * 5 - 3")

    Returns:
        The numeric result of the equation
    """
    return evaluate_safe(equation)

## Helper function
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


