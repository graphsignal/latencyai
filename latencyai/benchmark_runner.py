import logging
import asyncio
import os
import sys
import time
import tempfile
import json
from pydantic import BaseModel

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class BenchmarkResult(BaseModel):
    code: str = ""
    avg_time_per_op_ns: float = 0.0
    profile_str: str = ""

async def run_benchmark(code: str, timeout=60) -> tuple[BenchmarkResult, str]:
    logger.debug("Benchmarking provided code.")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    main_path = os.path.join(current_dir, "benchmark_main.py")

    with tempfile.NamedTemporaryFile('w+', suffix='.py', delete=False) as tmp_file:
        tmp_file.write(code)
        tmp_code_path = tmp_file.name

    with tempfile.NamedTemporaryFile('w+', suffix='.json', delete=False) as tmp_file:
        tmp_result_path = tmp_file.name
    
    with tempfile.NamedTemporaryFile('w+', suffix='.txt', delete=False) as tmp_file:
        tmp_profile_path = tmp_file.name

    process = await asyncio.create_subprocess_exec(
        sys.executable, main_path, tmp_code_path, tmp_result_path, tmp_profile_path,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout
        )

        logger.debug(f"Benchmark stdout: {stdout.decode()}")
        logger.debug(f"Benchmark stderr: {stderr.decode()}")

        if process.returncode == 0:
            with open(tmp_result_path, "r") as f:
                result_dict = json.load(f)
            avg_time_per_op_ns = result_dict["avg_time_per_op_ns"]
            with open(tmp_profile_path, "r") as f:
                profile_str = f.read()
            benchmark_result = BenchmarkResult(
                code=code, 
                avg_time_per_op_ns=avg_time_per_op_ns,
                profile_str=profile_str)
            return (benchmark_result, "")
        elif stderr:
            return (None, stderr.decode())
        else:
            raise Exception(f"Benchmarking failed with return code {process.returncode}")
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise Exception(f"Benchmarking timed out. Timeout {timeout} seconds.")
    finally:
        os.remove(tmp_code_path)
        os.remove(tmp_result_path)
        os.remove(tmp_profile_path)
        logger.debug("Benchmarking completed.")
