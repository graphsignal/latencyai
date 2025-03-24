# LatencyAI - AI Performance Engineer

## Introduction

LatencyAI is an AI agent that optimizes any Python code for best performance using reasoning LLMs. It iteratively profiles, optimizes, and benchmarks the code. The goal is to optimize code by GPU offloading, using data/task parallel, latency hiding and other techniques.

**Note: this is an experimental library.**

## Installation

* (Optional) Deploy a CUDA-enabled GPU instance
* `pip install --upgrade latencyai`

## Usage

* Set the OPEANAI_API_KEY environment variable
* Run `python -m latencyai --runs=3 script-to-optimize.py`. Optionally set `--runs`, which is the number of optimization attempts, i.e. optimize-benchmark-profile iterations. The default is 2.

The provided script should have a `main` function. The benchmark runner calls it multiple times, depending on it's execution time.

If optimization is successful, a file named `<original-script>_optimized.py` is be written to original script directory.

## Tracking optimizations

After integrating optimized code into your application, you can verify and track end-to-end performance improvements in deployed applications using [Graphsignal](https://graphsignal.com/).