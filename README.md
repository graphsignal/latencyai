# LatencyAI - AI Performance Engineer

## Introduction

LatencyAI is an AI agent that optimizes any Python code for best performance using reasoning LLMs. It iteratively profiles, optimizes, and benchmarks the code. The goal is to optimize code by GPU offloading, using data/task parallel, latency hiding and other techniques.


## Installation

* (Optional) Deploy a CUDA-enabled GPU instance
* Clone this repo
* Run `pip install -r requirements.txt`


## Usage

* set the OPEANAI_API_KEY environment variable
* `python -m latencyai script-to-optimize.py`

The script should have a `main` function. The benchmark runner will call it multiple times to benchmark performance and record a CPU/GPU profile.


## Tracking optimizations

After integrating optimized code into your application, you can verify and track end-to-end performance improvements in deployed applications using [Graphsignal](https://graphsignal.com/).