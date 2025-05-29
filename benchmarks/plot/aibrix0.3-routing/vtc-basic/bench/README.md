# VTC (Virtual Token Clustering) Benchmarking Suite

This benchmarking suite tests the effectiveness of VTC routing in ensuring fairness between users with different token consumption patterns in AIBrix.

## Overview

The VTC routing algorithm aims to prevent high-token users from overwhelming low-token users by clustering requests based on their token requirements. This benchmark suite validates that goal by comparing VTC-basic routing against random routing.

## Files Structure

```
.
├── bench_and_collect_stats.py  # Main benchmarking script
├── analyze.py                  # General analysis functions (fairness, pod distribution)
├── analyze_vtc_config.py       # VTC-specific analysis (bucket sizes, stability)
├── run_analysis.py             # CLI tool for analyzing results
├── prepare_dataset.py          # Dataset preparation utilities
├── constants.py                # Configuration constants
└── README.md                   # This file
```

## Prerequisites

1. AIBrix deployment with:

   - 3+ TinyLlama pods running
   - Envoy gateway with VTC routing plugin
   - Prometheus metrics collection
   - Redis for user data storage

2. Port forwarding setup:

   ```bash
   # From AIBrix project root
   make dev-port-forward
   ```

3. Python dependencies:
   ```bash
   pip install redis requests tiktoken numpy
   ```

## Quick Start

### 1. Prepare Dataset (if needed)

```bash
python prepare_dataset.py
```

### 2. Run Benchmark

```bash
# Default: 5 requests with balanced traffic pattern, tests both algorithms
python bench_and_collect_stats.py

# Run with 100 requests at 5 QPS
python bench_and_collect_stats.py --requests 100 --qps 5

# Test only VTC-basic algorithm (useful for config tuning)
python bench_and_collect_stats.py --requests 50 --algorithms vtc-basic

# Test only random algorithm
python bench_and_collect_stats.py --requests 50 --algorithms random

# Test specific traffic pattern
python bench_and_collect_stats.py --requests 100 --pattern high_usage

# Full example with all options
python bench_and_collect_stats.py --requests 100 --pattern balanced --qps 10 --algorithms random vtc-basic

# Strict mode - any failure invalidates comparison
python bench_and_collect_stats.py --requests 20 --qps 2 --strict
```

**Command-line options:**

- `--requests`: Number of requests to send (default: 5)
- `--pattern`: Traffic pattern - "balanced", "high_usage", or "bursty" (default: balanced)
- `--qps`: Target queries per second, 0 = sequential (default: 0)
- `--algorithms`: Space-separated list of algorithms to test (default: both random and vtc-basic)
- `--strict`: Strict validation mode - any failure invalidates comparison (default: 2% threshold)
- `--stream`: Enable streaming mode (not implemented yet)

### 3. Analyze Results

```bash
# Full analysis
python run_analysis.py /tmp/aibrix-benchmark/run_results/run_YYYYMMDD_HHMMSS

# VTC configuration analysis only
python run_analysis.py /path/to/results --vtc-only

# Fairness analysis only
python run_analysis.py /path/to/results --fairness-only

# Save report to file
python run_analysis.py /path/to/results --output report.json
```

## User Categories

The benchmark defines three user categories based on token consumption:

- **Small Users**: 10-100 tokens per request
- **Medium Users**: 100-300 tokens per request
- **High Users**: 300-800 tokens per request

## Traffic Patterns

1. **Balanced**: Equal distribution across categories (33% each)
2. **High Usage**: More high-token users (50% high, 30% medium, 20% small)
3. **Bursty**: Variable burstiness levels across categories

## Metrics Collected

### Application Metrics

- Request latency
- Success/failure rates
- Pod assignment per request
- Token counts (prompt/completion/total)

### System Metrics (via Prometheus)

- Pod CPU and memory usage
- VLLM metrics (request counts, token rates, TTFT, etc.)
- Envoy gateway metrics
- **VTC bucket sizes** (key metric for configuration validation)

### Analysis Outputs

1. **VTC Configuration Analysis**:

   - Bucket size stability (stable/oscillating)
   - Configuration health status
   - Recommendations for tuning

2. **Fairness Analysis**:

   - Average latency per user category
   - Comparison between routing algorithms
   - Overall fairness assessment

3. **Pod Distribution**:
   - Request distribution across pods
   - Load balancing effectiveness

## Success Rate Validation

The benchmark includes automatic validation of request success rates:

- **✅ Valid Comparison**: ≤2% failure rate across all algorithms
- **❌ Invalid Comparison**: >2% failure rate prevents fairness analysis
- **Detailed Logging**: All failed requests are logged with error details
- **Connection Issues**: Detects common problems like "Connection refused" errors

If validation fails, the benchmark will:

1. Log detailed error information for each failed request
2. Skip fairness comparison analysis
3. Save raw results for debugging
4. Suggest checking system connectivity

## Key Insights

The benchmark validates that VTC routing:

- ✅ Prevents high-token users from overwhelming low-token users
- ✅ Maintains stable bucket sizes (no oscillations)
- ✅ Provides fairness improvements over random routing
- ✅ Ensures reliable comparisons with success rate validation

## Configuration Tuning

If VTC shows instability or poor fairness:

1. Check `vtc_bucket_size_active` metric oscillations
2. Adjust environment variables:
   - `AIBRIX_ROUTER_VTC_MIN_BUCKET_SIZE`
   - `AIBRIX_ROUTER_VTC_ADJUSTMENT_WINDOW`
3. Re-run benchmark to validate improvements

## Example Results

With 5 requests in balanced mode:

```
✅ VTC ROUTING SHOWS FAIRNESS IMPROVEMENT: +1.4% average
   High token users are NOT overwhelming low token users!

VTC Configuration Status: good
VTC Stability: stable
```

For production validation, scale up to 100-1000 requests for statistically significant results.
