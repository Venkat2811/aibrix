#!/usr/bin/env python3
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Add parent directories to path to use existing framework
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)
from benchmark import BenchmarkRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def add_user_categories(workload_file, output_file=None):
    """
    Add specific user categories to the workload for VTC routing.
    Uses exactly 6 users: 2 small, 2 medium, and 2 high users.
    """
    if output_file is None:
        output_file = workload_file.replace(".jsonl", "_with_users.jsonl")

    # Define 6 specific users with their categories and token scales
    users = [
        {"id": "user-small-1", "category": "small", "token_scale": 1.0},
        {"id": "user-small-2", "category": "small", "token_scale": 1.0},
        {"id": "user-med-1", "category": "medium", "token_scale": 3.0},
        {"id": "user-med-2", "category": "medium", "token_scale": 3.0},
        {"id": "user-high-1", "category": "high", "token_scale": 6.0},
        {"id": "user-high-2", "category": "high", "token_scale": 6.0},
    ]

    # Process workload file
    modified_workload = []
    total_requests = 0
    user_request_counts = {user["id"]: 0 for user in users}

    with open(workload_file, "r") as f:
        for line in f:
            entry = json.loads(line)

            if "requests" in entry:
                # Distribute requests evenly across the 6 users in a round-robin fashion
                for req_index, req in enumerate(entry["requests"]):
                    # Select user in round-robin fashion
                    user = users[req_index % len(users)]

                    # Set user header
                    req["user"] = user["id"]
                    req["user_category"] = user["category"]

                    # Scale output length based on category
                    if "output_length" in req and req["output_length"] is not None:
                        req["output_length"] = int(
                            req["output_length"] * user["token_scale"]
                        )

                    user_request_counts[user["id"]] += 1
                    total_requests += 1

            modified_workload.append(entry)

    # Write modified workload to file
    with open(output_file, "w") as f:
        for entry in modified_workload:
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Added user categories to workload. Total requests: {total_requests}")
    logger.info("Request distribution by user:")
    for user_id, count in user_request_counts.items():
        logger.info(f"  {user_id}: {count} requests")
    logger.info(f"Modified workload saved to {output_file}")

    return output_file


def compare_routing_strategies(config_file):
    """Run benchmark with different routing strategies and compare results"""
    strategies = ["random", "vtc-basic"]
    results = {}

    for strategy in strategies:
        logger.info(f"===== Running benchmark with {strategy} routing =====")

        # Create strategy-specific output directories
        strategy_output = f"vtc-routing-{strategy}"

        # Load the base config
        with open(config_file, "r") as f:
            config_content = f.read()

        # Modify config for this strategy
        config_content = config_content.replace(
            'client_output: "benchmarks/scenarios/gateway/vtc-basic-routing/results/client"',
            f'client_output: "benchmarks/scenarios/gateway/vtc-basic-routing/results/{strategy}/client"',
        ).replace(
            'trace_output: "benchmarks/scenarios/gateway/vtc-basic-routing/results/trace"',
            f'trace_output: "benchmarks/scenarios/gateway/vtc-basic-routing/results/{strategy}/trace"',
        )

        # Add routing strategy
        config_content += f'\nrouting_strategy: "{strategy}"\n'

        # Save modified config
        strategy_config = config_file.replace(".yaml", f"_{strategy}.yaml")
        with open(strategy_config, "w") as f:
            f.write(config_content)

        # Run the benchmark
        runner = BenchmarkRunner(config_base=strategy_config)

        # Generate dataset and workload (only needed for first run)
        if strategy == strategies[0]:
            runner.generate_dataset()
            runner.generate_workload()

            # Add user categories to the workload
            workload_file = runner.config["workload_file"]
            processed_workload = add_user_categories(workload_file)
            runner.config["workload_file"] = processed_workload
        else:
            # Use the same processed workload
            runner.config["workload_file"] = processed_workload

        # Record start time
        start_time = time.time()

        # Run client
        runner.run_client()

        # Record end time
        end_time = time.time()
        logger.info(f"Benchmark took {end_time - start_time:.2f} seconds")

        # Run analysis
        runner.run_analysis()

        # Store results
        results[strategy] = {
            "client_output": runner.config["client_output"],
            "trace_output": runner.config["trace_output"],
            "duration": end_time - start_time,
        }

    # Compare results
    compare_results(results)

    return results


def compare_results(results):
    """Compare benchmark results between routing strategies"""
    logger.info("\n===== Comparing Results =====")

    # Extract metrics from each strategy's results
    metrics = {}
    for strategy, paths in results.items():
        client_output = paths["client_output"]
        trace_output = paths["trace_output"]

        # Load trace output summary
        summary_file = os.path.join(trace_output, "summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, "r") as f:
                metrics[strategy] = json.load(f)
                # Add benchmark duration
                metrics[strategy]["benchmark_duration"] = paths["duration"]
        else:
            logger.warning(f"Summary file not found for {strategy}")

    # Print comparison table
    if len(metrics) > 1:
        logger.info("\nPerformance Comparison:")
        logger.info("-" * 80)
        logger.info(
            f"{'Metric':<30} | {'Random':<20} | {'VTC-Basic':<20} | {'Improvement':<15}"
        )
        logger.info("-" * 80)

        # Common metrics to compare
        compare_metrics = [
            "avg_latency",
            "p50_latency",
            "p95_latency",
            "p99_latency",
            "avg_ttft",
            "p50_ttft",
            "p95_ttft",
            "p99_ttft",
            "success_rate",
            "goodput",
            "throughput",
        ]

        for metric in compare_metrics:
            if metric in metrics.get("random", {}) and metric in metrics.get(
                "vtc-basic", {}
            ):
                random_val = metrics["random"][metric]
                vtc_val = metrics["vtc-basic"][metric]

                # Format values as strings
                if isinstance(random_val, float):
                    random_str = f"{random_val:.3f}"
                else:
                    random_str = str(random_val)

                if isinstance(vtc_val, float):
                    vtc_str = f"{vtc_val:.3f}"
                else:
                    vtc_str = str(vtc_val)

                # Calculate improvement
                if (
                    isinstance(random_val, (int, float))
                    and isinstance(vtc_val, (int, float))
                    and random_val > 0
                ):
                    # For latency metrics, lower is better
                    if metric.endswith("latency") or metric.endswith("ttft"):
                        improvement = ((random_val - vtc_val) / random_val) * 100
                        improvement_str = f"{improvement:.2f}%"
                    # For other metrics, higher is better
                    else:
                        improvement = ((vtc_val - random_val) / random_val) * 100
                        improvement_str = f"{improvement:.2f}%"
                else:
                    improvement_str = "N/A"

                logger.info(
                    f"{metric:<30} | {random_str:<20} | {vtc_str:<20} | {improvement_str:<15}"
                )

        logger.info("-" * 80)

    # Create comparison report
    report_dir = os.path.dirname(results["random"]["client_output"])
    os.makedirs(report_dir, exist_ok=True)

    report_file = os.path.join(report_dir, "../comparison_report.json")
    with open(report_file, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Comparison report saved to {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Run VTC-basic routing benchmark")
    parser.add_argument(
        "--config",
        default="benchmarks/scenarios/gateway/vtc-basic-routing/config.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()

    # Ensure directories exist
    base_dir = os.path.dirname(args.config)
    os.makedirs(os.path.join(base_dir, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "workload"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)

    # Run benchmark with different routing strategies
    results = compare_routing_strategies(args.config)

    logger.info("\nBenchmark completed successfully!")


if __name__ == "__main__":
    main()
