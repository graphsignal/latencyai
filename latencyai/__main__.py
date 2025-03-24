import logging
import asyncio
import os
import argparse
from pathlib import Path
import torch
from latencyai.openai_chat import OpenAIChat
from latencyai.benchmark_runner import run_benchmark

# configure logger
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


async def optimize_script(script_path, num_runs=2):
    logger.debug(f"Optimizing script: {script_path}, num_runs: {num_runs}")

    chat = OpenAIChat("You are a helpful assistant who optimizes Python code.")

    with Path(script_path).open('r', encoding='utf-8') as file:
        original_code = file.read()  # Read entire file content as a single string

    # Benchmark the original code
    try:
        original_result, stderr = await run_benchmark(original_code)
        if stderr:
            logger.debug(f"Benchmark error: {stderr}")
            return
        logger.debug(f"Benchmark result for original code: {original_result.avg_time_per_op_ns} ns/op")
    except Exception as e:
        logger.debug(f"Benchmark error: {e}", exc_info=True)
        return
    results = []
    for attempt in range(1, num_runs + 1):
        try:
            logger.debug(f"Attempt {attempt}.")

            if attempt == 1:
                torch.cuda.is_available()
                chat.user(f"Convert provided Python code to utilize GPU offloading, data parallel, task parallel, latency hiding, and other techniques to optimize performance. Use numpy and torch modules. Provide a runnable python code. The code will be benchmarked by calling main() multiple times.")
                chat.user(original_code)
                chat.user(f"Benchmark result for original code: {original_result.avg_time_per_op_ns} ns/op")
                if torch.cuda.is_available():
                    chat.user(f'CUDA is available on the target system. Number of GPUs: {torch.cuda.device_count()}')
            else:
                chat.user("Optimize the original code further based on previous optimization attempts. Generate runnable Python code.")

            optimized_code = await chat.generate_code()
            chat.assistant(optimized_code)

            result, stderr = await run_benchmark(optimized_code)
            if stderr:
                logger.debug(f"Error benchmarking code: {stderr}")
                chat.user(f"Error benchmarking code: {stderr}")
                continue

            results.append(result)
            chat.user(f'Benchmark result for attempt #{attempt}: {result.avg_time_per_op_ns} ns/op. Speedup: {original_result.avg_time_per_op_ns / result.avg_time_per_op_ns:.2f}')
            logger.debug(f"Benchmark result for attempt #{attempt}: {result.avg_time_per_op_ns} ns/op. Speedup: {original_result.avg_time_per_op_ns / result.avg_time_per_op_ns:.2f}")

            if result.profile_str.strip():
                chat.user(f"PyTorch CPU profile: {result.profile_str}")
                logger.debug(f"PyTorch CPU profile: {result.profile_str}")
        except Exception as e:
            logger.debug(f"Error: {e}", exc_info=True)

    best_result = None
    for result in results:
        if result.avg_time_per_op_ns < original_result.avg_time_per_op_ns:
            best_result = result

    if best_result:
        if best_result.avg_time_per_op_ns >= original_result.avg_time_per_op_ns:
            logger.debug(f"Code was not optimized.")
        else:
            logger.debug(f"Code was optimized. Optimized time: {best_result.avg_time_per_op_ns} ns/op. Speedup: {original_result.avg_time_per_op_ns / best_result.avg_time_per_op_ns:.2f}")

            dir_name, base_name = os.path.split(script_path)
            name, ext = os.path.splitext(base_name)
            optimized_name = f"{name}_optimized{ext}"
            optimized_script_path = os.path.join(dir_name, optimized_name)
            with open(optimized_script_path, 'w', encoding='utf-8') as f:
                f.write(best_result.code)
            logger.debug(f"Optimized script saved to {optimized_script_path}")
    else:
        logger.debug(f"No optimization results.")

    await chat.close()

async def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="LatencyAI code optimized for CUDA")
    parser.add_argument("script", help="The Python script to load")
    parser.add_argument("--runs", type=int, default=2, help="Number of optimization attempts to make")
    args = parser.parse_args()

    await optimize_script(args.script, num_runs=args.runs)

if __name__ == "__main__":
    asyncio.run(main())