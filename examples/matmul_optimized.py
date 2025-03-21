import torch
import concurrent.futures


# Check if CUDA is available
if not torch.cuda.is_available():
    raise EnvironmentError('CUDA is not available on this system.')

def generate_matrix(size):
    """Generates a square matrix on GPU of given size with random float values using PyTorch."""
    # Generate matrix directly on the GPU
    return torch.rand(size, size, device='cuda')


def expensive_operation(matrix, stream):
    """Performs matrix multiplication followed by an element-wise sqrt and scaling, using a dedicated CUDA stream."""
    # Use the provided CUDA stream for asynchronous execution
    with torch.cuda.stream(stream):
        # Matrix multiplication using torch.mm which utilizes GPU BLAS libraries
        result = torch.mm(matrix, matrix)
        # Perform an element-wise operation (sqrt then scale by 0.5)
        result = result.sqrt() * 0.5
        # Optionally, we can synchronize the stream here to ensure the operations finish
        # before returning the result. This helps in latency hiding by deferring wait until needed.
        stream.synchronize()
    return result


def main():
    """Main function to run the computation in a parallel and optimized manner."""
    # Synchronize GPU to ensure a clean start
    torch.cuda.synchronize()

    num_matrices = 10
    matrix_size = 100  # Keeping size reasonable for demonstration

    # Create a separate CUDA stream for each matrix to enable concurrent execution
    streams = [torch.cuda.Stream() for _ in range(num_matrices)]

    # Generate matrices directly on the GPU
    matrices = [generate_matrix(matrix_size) for _ in range(num_matrices)]

    # Use ThreadPoolExecutor to submit tasks concurrently. GPU operations are asynchronous
    # and releasing the GIL allows for effective parallelism even in threads.
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_matrices) as executor:
        # Launch expensive_operation for each matrix on its dedicated CUDA stream
        futures = [executor.submit(expensive_operation, matrix, stream) for matrix, stream in zip(matrices, streams)]
        # Collect results as they complete. as_completed will not block unnecessarily.
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    # Final GPU synchronization to ensure all operations are complete before finishing the main function
    torch.cuda.synchronize()

    # Optionally, you can move results back to CPU if needed:
    # cpu_results = [result.cpu() for result in results]


if __name__ == '__main__':
    main()