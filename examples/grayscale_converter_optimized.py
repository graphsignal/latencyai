import torch
import numpy as np

# Optimized conversion using CUDA streams for GPU parallelism, data parallelism, and latency hiding

def parallel_rgb_to_grayscale(image, num_chunks=4):
    """
    Convert an RGB image to grayscale using parallel CUDA streams if available.
    The image tensor should be of shape (H, W, 3) and be on the appropriate device.
    """
    if image.device.type == 'cuda':
        # Create CUDA streams
        streams = [torch.cuda.Stream() for _ in range(num_chunks)]
        # Split the image into chunks along the height dimension
        chunks = torch.chunk(image, num_chunks, dim=0)
        results = [None] * num_chunks
        # Preallocate weights on the same device
        weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype)
        # Process each chunk asynchronously on its own stream
        for i, (chunk, stream) in enumerate(zip(chunks, streams)):
            with torch.cuda.stream(stream):
                # Perform the grayscale conversion in a vectorized manner
                results[i] = (chunk * weights).sum(dim=-1)
        # Synchronize all streams
        torch.cuda.synchronize()
        # Concatenate the chunks back into a full image
        gray_image = torch.cat(results, dim=0)
    else:
        # For CPU, use the vectorized operation directly with torch
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=image.dtype)
        gray_image = (image * weights).sum(dim=-1)
    return gray_image


def main():
    """
    Main function to be benchmarked. It creates a dummy image and converts it to grayscale.
    Uses GPU acceleration and parallel streams if available.
    """
    # Determine device. If CUDA is available, we'll use GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Create a dummy image of size 100x100 with 3 channels using torch.full.
    # This avoids explicit clone and redundant memory operations.
    image = torch.full((100, 100, 3), 0.0, device=device, dtype=torch.float32)
    # Set the RGB values by assigning a constant vector to each pixel
    image[..., 0] = 100.0  # R channel
    image[..., 1] = 150.0  # G channel
    image[..., 2] = 200.0  # B channel

    # Disable gradient calculations as we are just doing inference
    with torch.no_grad():
        gray = parallel_rgb_to_grayscale(image, num_chunks=4)

    # Asynchronously move the result to CPU to hide latency (if GPU is used)
    gray_cpu = gray.to('cpu', non_blocking=True)

    # If running on GPU, ensure all operations are complete
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Optionally return the result for further verification
    return gray_cpu


if __name__ == '__main__':
    result = main()
    print('Grayscale image shape:', result.shape)