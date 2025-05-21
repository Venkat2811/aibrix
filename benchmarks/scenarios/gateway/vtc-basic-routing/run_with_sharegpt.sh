#!/bin/bash
set -e

# Set the working directory to the root of the project
cd "$(dirname "$0")/../../../"

# Download the ShareGPT dataset
echo "Downloading ShareGPT dataset if needed..."
python3 benchmarks/scenarios/gateway/vtc-basic-routing/download_sharegpt.py

# Run the benchmark
echo "Running VTC-basic routing benchmark with ShareGPT data..."
python3 benchmarks/scenarios/gateway/vtc-basic-routing/run_benchmark.py

echo "Benchmark completed! Check the results in benchmarks/scenarios/gateway/vtc-basic-routing/results/" 