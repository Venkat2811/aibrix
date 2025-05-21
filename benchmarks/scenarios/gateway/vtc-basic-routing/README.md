# VTC-Basic Routing Benchmark with TinyLlama

This benchmark compares the performance of VTC-basic routing with random routing using the TinyLlama model.

## Overview

This benchmark tests how VTC-basic routing performs compared to random routing on several key metrics:

- Latency (P50, P95, P99)
- Time to First Token (TTFT)
- Success rate
- Throughput
- User fairness across different token usage patterns

## Prerequisites

1. A Kubernetes cluster with AIBrix installed
2. The TinyLlama model deployed (or another model configured in the `config.yaml`)
3. VTC-basic routing enabled in the gateway plugin (Redis for token tracking)

## Deploying TinyLlama

Deploy the TinyLlama model using the provided deployment file:

```bash
kubectl apply -f benchmarks/scenarios/gateway/vtc-basic-routing/tinyllama-deployment.yaml
```

## Running the Benchmark

Set up the necessary environment variables:

```bash
export api_key="your-api-key"
export api_endpoint="http://localhost:8888"  # Your gateway endpoint
```

Run the benchmark:

```bash
cd /path/to/aibrix
python benchmarks/scenarios/gateway/vtc-basic-routing/run_benchmark.py
```

### Using ShareGPT Data (Recommended)

By default, the benchmark now uses ShareGPT data instead of synthetic data for more realistic testing:

```bash
# Run the benchmark with ShareGPT data
./benchmarks/scenarios/gateway/vtc-basic-routing/run_with_sharegpt.sh
```

This script automatically downloads the ShareGPT dataset (if not already present) and runs the benchmark. The ShareGPT dataset provides real-world conversation patterns which better simulate actual usage.

## Customizing the Benchmark

The benchmark can be customized via the `config.yaml` file. Key parameters include:

- `client_pool_size`: Number of concurrent clients
- `output_token_limit`: Maximum output token length
- `duration_ms`: Duration of the benchmark in milliseconds
- `target_qps`: Target queries per second
- `prompt_type`: Set to "sharegpt" for real-world data or "synthetic_multiturn" for synthetic data

User distribution is defined in the `add_user_categories` function in `run_benchmark.py`. The default distribution is:

- 30% low token users (1x token scale)
- 40% medium token users (2x token scale)
- 20% high token users (4x token scale)
- 10% extreme token users (8x token scale)

## Results

Results are saved in the `benchmarks/scenarios/gateway/vtc-basic-routing/results/` directory, with separate subdirectories for each routing strategy. A comparison report is generated in `comparison_report.json`.

The script automatically compares key metrics between the two routing strategies and calculates improvement percentages.

## VTC Configuration

The VTC-basic algorithm's performance can be influenced by these environment variables in the gateway plugin:

- `AIBRIX_ROUTER_VTC_TOKEN_TRACKER_WINDOW_SIZE`: Window size for token tracking
- `AIBRIX_ROUTER_VTC_TOKEN_TRACKER_MIN_TOKENS`: Minimum token threshold
- `AIBRIX_ROUTER_VTC_FAIRNESS_WEIGHT`: Weight for fairness component
- `AIBRIX_ROUTER_VTC_UTILIZATION_WEIGHT`: Weight for utilization component

Adjust these variables to fine-tune VTC behavior for your workload.
