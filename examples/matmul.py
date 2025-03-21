import random

def generate_matrix(size):
    """Generates a square matrix of given size with random float values."""
    return [[random.random() for _ in range(size)] for _ in range(size)]

def matrix_multiplication(A, B):
    """Performs naive matrix multiplication (O(n^3)) without optimizations."""
    size = len(A)
    result = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            for k in range(size):
                result[i][j] += A[i][k] * B[k][j]
    return result

def expensive_operation(matrix):
    """Performs matrix multiplication followed by an element-wise operation."""
    result = matrix_multiplication(matrix, matrix)  # Matrix multiplication
    for i in range(len(result)):
        for j in range(len(result[i])):
            result[i][j] = (result[i][j] ** 0.5) * 0.5  # Arbitrary expensive operation
    return result

def main():
    """Main function to run the computation."""
    num_matrices = 10
    matrix_size = 100  # Keeping it reasonable for pure Python execution

    matrices = [generate_matrix(matrix_size) for _ in range(num_matrices)]

    results = [expensive_operation(matrix) for matrix in matrices]  # Sequential execution
