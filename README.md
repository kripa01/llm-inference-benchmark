# llm-inference-benchmark

# Lightweight LLM Inference Benchmarking Framework

A lightweight framework for measuring local LLM inference performance.

## Features

- Model loading benchmarks
- Inference latency measurement
- Tokens per second calculation
- Memory usage tracking
- CSV result storage

## Supported Models

Currently tested:
- distilgpt2

## Run

Install dependencies:
pip install -r requirements.txt

Run benchmark:
python src/benchmark.py

Results are saved in:
results/benchmark_results.csv
