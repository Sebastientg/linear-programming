import numpy as np
from itertools import combinations


def compute_basic_solutions(A, b):
    """
    Compute all basic solutions for the given matrix A and vector b.
    Supports both square and non-square submatrices.

    Args:
        A (np.ndarray): The input m x n matrix.
        b (np.ndarray): The input vector of length m.

    Returns:
        List of tuples, where each tuple contains the column indices and the full solution vector.
    """
    m, n = A.shape
    basic_solutions = []

    # Iterate over all possible combinations of min(m, n) columns
    for cols in combinations(range(n), min(m, n)):
        # Extract the submatrix corresponding to the chosen columns
        A_sub = A[:, cols]
        try:
            if A_sub.shape[0] == A_sub.shape[1]:  # Square matrix
                # Solve the system A_sub * x = b
                x_sub = np.linalg.solve(A_sub, b)
            else:  # Non-square matrix
                # Solve using least squares
                x_sub, _, _, _ = np.linalg.lstsq(A_sub, b, rcond=None)

            # Create a full solution vector
            x_full = np.zeros(n)
            x_full[list(cols)] = x_sub

            # Store the solution
            basic_solutions.append((cols, x_full))
        except np.linalg.LinAlgError:
            # Skip if the submatrix is singular
            continue

    return basic_solutions


def main():
    # Input m x n matrix A
    rows = int(input("Enter the number of rows (m): "))
    cols = int(input("Enter the number of columns (n): "))
    print(f"Enter the {rows}x{cols} matrix A row by row:")
    A = np.array([list(map(float, input().split())) for _ in range(rows)])

    # Input vector b
    print(f"Enter the vector b with {rows} elements:")
    b = np.array(list(map(float, input().split())))

    # Validate dimensions
    if A.shape[0] != b.shape[0]:
        print("Error: The number of rows in A must match the length of b.")
        return

    # Compute basic solutions
    solutions = compute_basic_solutions(A, b)

    # Display results
    if solutions:
        print("\nBasic Solutions:")
        for cols, solution in solutions:
            print(f"Columns: {cols}, Solution: {solution}")
    else:
        print("\nNo basic solutions found.")


if __name__ == "__main__":
    main()
