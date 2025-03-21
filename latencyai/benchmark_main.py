import argparse
import importlib.util
import sys
import os
import time
import json
import torch

def exit_with_error(message):
    print(message, file=sys.stderr)
    sys.exit(1)

# Argument parser setup
parser = argparse.ArgumentParser(description="Benchmark runner")
parser.add_argument("code_path", help="The Python script to load")
parser.add_argument("result_path", help="The result JSON file path")
parser.add_argument("profile_path", help="The Pytorch profile file path")
args = parser.parse_args()

result_path = args.result_path

try:
    # Load module dynamically
    if not os.path.exists(args.code_path):
        exit_with_error(f"Error: Script '{args.code_path}' not found.")

    module_name = os.path.splitext(os.path.basename(args.code_path))[0]  # Extract module name from filename
    print(f"Loading script '{args.code_path}' as module '{module_name}'")
    spec = importlib.util.spec_from_file_location(module_name, args.code_path)
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)  # Execute the script (once)

    # Test the script
    if hasattr(module, "main") and callable(module.main):
        module.main()
    else:
        exit_with_error(f"Error: No 'main()' function found in {module.__name__}.")

    # Benchmarking script execution time
    print("Benchmarking execution time")
    iterations = 1
    threshold_seconds = 2.0

    while True:
        # Measure total execution time for multiple iterations
        start_time = time.time()
        
        for _ in range(iterations):
            module.main()

        elapsed_time = time.time() - start_time
        elapsed_ns = elapsed_time * 1e9  # Convert seconds to nanoseconds
        avg_time_per_op = int(elapsed_ns / iterations)  # Average time per operation in ns

        if elapsed_time > threshold_seconds:
            result = {
                "iterations": iterations,
                "total_time_sec": elapsed_time,
                "avg_time_per_op_ns": avg_time_per_op
            }
            with open(args.result_path, "w") as f:
                json.dump(result, f, indent=4)
            break

        iterations *= 10  # Increase iterations exponentially

    # Pytorch profiling
    print("Profiling with PyTorch")

    def _schedule_func(step):
        return torch.profiler.ProfilerAction.RECORD
    with torch.profiler.profile(
        schedule=_schedule_func,
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
        with_flops=True) as prof:

        for _ in range(iterations):
            module.main()

    profile_output = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)
    with open(args.profile_path, "w") as f:
        f.write(profile_output)

except Exception as e:
    exit_with_error(f"Error while benchmarking: {e}")
