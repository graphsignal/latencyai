import torch


def main():
    """Main function to run the computation using batched operations for data parallelism and GPU acceleration."""

    # Choose device: GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_matrices = 10
    matrix_size = 100  
    
    # Generate a batch of matrices directly on the target device
    # This generates a tensor of shape (num_matrices, matrix_size, matrix_size) with random values
    matrices = torch.rand((num_matrices, matrix_size, matrix_size), device=device)

    # Batched matrix multiplication: square each matrix using torch.bmm
    # This call is data parallel and leverages optimized GPU BLAS routines if available
    multiplied = torch.bmm(matrices, matrices)

    # Apply an element-wise operation: square root and scaling
    # Leveraging GPU parallelism, this applies the operation to each element concurrently
    result = torch.sqrt(multiplied) * 0.5

    # Ensure GPU operations are complete before returning
    if device.type == 'cuda':
        torch.cuda.synchronize()

    return result


if __name__ == '__main__':
    # Run main for benchmarking multiple times externally if needed
    main()