import numpy as np
import torch


def generate_matrices_torch(num_matrices, size, device):
    # Generate a batch of random matrices directly on GPU using torch
    return torch.rand(num_matrices, size, size, device=device)


def expensive_operation_torch(matrices):
    # Perform batch matrix multiplication and element-wise sqrt and scaling
    result = torch.bmm(matrices, matrices)
    result = torch.sqrt(result) * 0.5
    return result


def generate_matrices_numpy(num_matrices, size):
    # Generate a batch of random matrices using numpy
    return np.random.rand(num_matrices, size, size).astype(np.float32)


def expensive_operation_numpy(matrices):
    # Use numpy's batch matrix multiplication (np.matmul uses optimized BLAS routines)
    result = np.matmul(matrices, matrices)
    result = np.sqrt(result) * 0.5
    return result


def main():
    num_matrices = 10
    matrix_size = 100
    
    if torch.cuda.is_available():
        device = 'cuda'
        matrices = generate_matrices_torch(num_matrices, matrix_size, device)
        # Use batched operation on GPU; this leverages data parallelism and GPU offloading
        result = expensive_operation_torch(matrices)
    else:
        # For CPU, use numpy's vectorized operations which internally use BLAS for further optimization
        matrices = generate_matrices_numpy(num_matrices, matrix_size)
        result = expensive_operation_numpy(matrices)
    
    return result


if __name__ == '__main__':
    main()