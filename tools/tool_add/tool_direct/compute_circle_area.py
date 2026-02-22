import math

def compute_circle_area(radius: float) -> float:
    """Calculate the area of a circle given its radius."""
    if radius < 0:
        raise ValueError("Radius must be a non-negative number")
    return math.pi * radius * radius