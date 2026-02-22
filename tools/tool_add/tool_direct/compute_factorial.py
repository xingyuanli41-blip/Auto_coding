def compute_factorial(N: int) -> int:
    """Calculate the factorial of a non-negative integer N."""
    if N < 0:
        raise ValueError("N must be a non-negative integer")
    if N == 0 or N == 1:
        return 1
    factorial = 1
    for i in range(2, N + 1):
        factorial *= i
    return factorial