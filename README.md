# LatencyAI - AI Performance Engineer


## Introduction

LatencyAI is an AI agent that optimizes any Python code for best performance using reasoning LLMs. It iteratively profiles, optimizes, and benchmarks the code. The goal is to optimize code by GPU offloading, using data/task parallel, latency hiding and other techniques.


## Installation

* Deploy a CUDA-enabled GPU instance
* Clone this repo
* Run `pip install -r requirements.txt`


## Usage

* `python -m latencyai script-to-optimize.py`

The script should have a `main` function. The benchmark runner will call it multiple times to benchmark performance and record a CPU/GPU profile.


## Notes

* This library reuses some modules/code from `https://github.com/graphsignal/solver-demo` repo, which is a simple agent to solve and verify coding problems.