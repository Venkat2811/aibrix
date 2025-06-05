# VTC Benchmarking Workflow

This directory contains a streamlined 4-script workflow for VTC (Virtual Token Clustering) benchmarking and analysis.

## The 4 Core Scripts

### 1. `prepare_dataset.py` - Data Preparation

Prepares VTC benchmark dataset from ShareGPT data with proper user categorization and token distribution.

```bash
# Generate standard dataset
python prepare_dataset.py

# Generate CPU-optimized dataset (smaller token ranges)
python prepare_dataset.py --cpu-only-users

# Custom configuration
python prepare_dataset.py --min-requests 100 --interval-ms 1000
```

### 2. `vtc-basic_accuracy_benchmark.py` - VTC Accuracy Testing

Tests VTC fairness algorithm with concurrent users to validate proper routing behavior.

```bash
# Quick accuracy test (5 users, 20 requests each)
python vtc-basic_accuracy_benchmark.py

# Custom test configuration
python vtc-basic_accuracy_benchmark.py --users 10 --requests-per-user 30 --qps 2
```

### 3. `bench_and_collect_stats.py` - Main Benchmarking

Comprehensive benchmarking tool that tests multiple routing algorithms and collects detailed metrics.

```bash
# Quick benchmark (5 requests, sequential)
python bench_and_collect_stats.py --requests 5

# Production-like benchmark with QPS control
python bench_and_collect_stats.py --requests 50 --qps 2 --pattern balanced

# Test specific algorithms
python bench_and_collect_stats.py --requests 20 --algorithms vtc-basic random
```

### 4. `comprehensive_analysis.py` - Complete Analysis

One-stop analysis tool that runs all analysis types on `bench_and_collect_stats.py` results.

```bash
# Full analysis (recommended)
python comprehensive_analysis.py /tmp/aibrix-benchmark/run_results/run_YYYYMMDD_HHMMSS

# Specific analysis only
python comprehensive_analysis.py RESULT_DIR --analysis-only pod
python comprehensive_analysis.py RESULT_DIR --analysis-only fairness
python comprehensive_analysis.py RESULT_DIR --analysis-only comprehensive
```

## Complete Workflow Example

```bash
# 1. Prepare dataset
python prepare_dataset.py --cpu-only-users

# 2. Test VTC accuracy
python vtc-basic_accuracy_benchmark.py

# 3. Run main benchmark
python bench_and_collect_stats.py --requests 20 --qps 1 --pattern balanced

# 4. Analyze results (use the directory path from step 3 output)
python comprehensive_analysis.py /tmp/aibrix-benchmark/run_results/run_20250127_143022
```

## Output Structure

### After `bench_and_collect_stats.py`:

```
/tmp/aibrix-benchmark/run_results/run_YYYYMMDD_HHMMSS/
├── comprehensive_benchmark_stats.json    # Main stats file
├── fairness_analysis.json               # Fairness metrics
├── system_metrics_*.json                # Prometheus metrics
└── request_*.json                       # Individual request results
```

### After `comprehensive_analysis.py`:

```
run_YYYYMMDD_HHMMSS/
├── enhanced_analysis/
│   ├── detailed_analysis_report.json    # Comprehensive report
│   └── visualizations/
│       ├── vtc_bucket_evolution.png     # VTC bucket size plots
│       └── fairness_comparison.png      # Fairness comparison charts
└── [existing files...]
```

## Key Features

- **Concise & Professional**: No emojis, minimal redundant comments
- **High Quality Code**: Proper error handling, type hints, logging
- **Reusable Functions**: Shared utilities across scripts
- **Comprehensive Analysis**: All analysis types in one tool
- **Flexible Configuration**: Extensive command-line options
- **Clear Output**: Professional logging and structured results

## Analysis Types Included

1. **Pod Performance Analysis**: Per-pod latency comparison between algorithms
2. **VTC Configuration Analysis**: VTC bucket size evolution and adaptation
3. **Fairness Analysis**: User category fairness comparison (VTC vs Random)
4. **Comprehensive VTC Analysis**: Full analysis with visualizations and recommendations

## Dependencies

- Standard Python libraries (json, logging, argparse, etc.)
- External: numpy, matplotlib, redis, requests, tiktoken
- Local modules: analyze.py, analyze_vtc_config.py, constants.py

## Notes

- Results are saved to `/tmp/aibrix-benchmark/run_results/` with timestamp
- Prometheus metrics require Prometheus running on localhost:9090
- Redis setup is handled automatically in benchmarking scripts
- All scripts include comprehensive help: `python SCRIPT.py --help`
