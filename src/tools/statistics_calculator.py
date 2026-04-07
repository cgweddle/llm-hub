"""
Statistics Calculator Tool
Calculates mean, median, min, max, and count from a list of numbers
"""

class Statistics:
    mean: float
    median: float
    min_value: int
    max_value: int
    count: int

def calculate_statistics(numbers: list) -> Statistics:
    """Calculate various statistics from a list of numbers"""
    sorted_nums = sorted(numbers)
    n = len(sorted_nums)

    # Calculate mean
    mean = sum(sorted_nums) / n

    # Calculate median
    if n % 2 == 0:
        median = (sorted_nums[n//2 - 1] + sorted_nums[n//2]) / 2
    else:
        median = sorted_nums[n//2]

    return {
        "mean": mean,
        "median": median,
        "min_value": min(sorted_nums),
        "max_value": max(sorted_nums),
        "count": n
    }
