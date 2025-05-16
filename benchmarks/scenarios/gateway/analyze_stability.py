#!/usr/bin/env python3
"""
Analyze gateway benchmark results with a focus on VTC stability metrics.

This script processes benchmark results and stability reports to generate
visualizations and insights about VTC algorithm stability, TTFT performance,
and workload pattern impacts.
"""

import json
import argparse
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


def load_benchmark_data(input_file: str) -> List[Dict[str, Any]]:
    """
    Load benchmark data from a JSONL file.
    
    Args:
        input_file: Path to the JSONL file
        
    Returns:
        List of benchmark data entries
    """
    data = []
    with open(input_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line: {line[:50]}...")
    return data


def load_stability_report(report_file: str) -> Dict[str, Any]:
    """
    Load stability report from a JSON file.
    
    Args:
        report_file: Path to the JSON file
        
    Returns:
        Stability report data
    """
    try:
        with open(report_file, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not load stability report: {e}")
        return {}


def extract_metrics(data: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """
    Extract metrics from benchmark data.
    
    Args:
        data: List of benchmark data entries
        
    Returns:
        Dictionary of extracted metrics
    """
    metrics = {
        "prompt_tokens": [],
        "output_tokens": [],
        "total_tokens": [],
        "latencies": [],
        "throughputs": [],
        "ttft": [],
        "timestamps": [],
        "workload_rates": [],
        "workload_tiers": []
    }
    
    for i, item in enumerate(data):
        metrics["timestamps"].append(item.get("timestamp", i))
        metrics["prompt_tokens"].append(item.get("prompt_tokens", 0))
        metrics["output_tokens"].append(item.get("output_tokens", 0))
        metrics["total_tokens"].append(item.get("total_tokens", 0))
        metrics["latencies"].append(item.get("latency", 0))
        metrics["throughputs"].append(item.get("throughput", 0))
        metrics["ttft"].append(item.get("ttft", None))
        metrics["workload_rates"].append(item.get("workload_rate", None))
        metrics["workload_tiers"].append(item.get("workload_tier", None))
    
    return metrics


def plot_ttft_analysis(metrics: Dict[str, List[Any]], 
                      stability_report: Dict[str, Any], 
                      output_dir: str,
                      title_prefix: str = ""):
    """
    Generate TTFT analysis plots.
    
    Args:
        metrics: Dictionary of extracted metrics
        stability_report: Stability report data
        output_dir: Directory to save plots
        title_prefix: Prefix for plot titles
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out None values
    ttft_values = [t for t in metrics["ttft"] if t is not None]
    timestamps = metrics["timestamps"]
    
    if not ttft_values:
        print("No TTFT values available for plotting")
        return
    
    # 1. TTFT over time
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(ttft_values)), ttft_values, alpha=0.5)
    plt.axhline(y=np.median(ttft_values), color='r', linestyle='-', label=f'Median: {np.median(ttft_values):.3f}s')
    plt.axhline(y=np.percentile(ttft_values, 90), color='orange', linestyle='--', label=f'P90: {np.percentile(ttft_values, 90):.3f}s')
    plt.axhline(y=np.percentile(ttft_values, 99), color='red', linestyle='--', label=f'P99: {np.percentile(ttft_values, 99):.3f}s')
    plt.title(f"{title_prefix}TTFT Over Time")
    plt.xlabel("Request Number")
    plt.ylabel("TTFT (seconds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "ttft_over_time.png"))
    
    # 2. TTFT histogram
    plt.figure(figsize=(10, 6))
    plt.hist(ttft_values, bins=30, alpha=0.7)
    plt.axvline(x=np.median(ttft_values), color='r', linestyle='-', label=f'Median: {np.median(ttft_values):.3f}s')
    plt.axvline(x=np.percentile(ttft_values, 90), color='orange', linestyle='--', label=f'P90: {np.percentile(ttft_values, 90):.3f}s')
    plt.axvline(x=np.percentile(ttft_values, 99), color='red', linestyle='--', label=f'P99: {np.percentile(ttft_values, 99):.3f}s')
    plt.title(f"{title_prefix}TTFT Distribution")
    plt.xlabel("TTFT (seconds)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "ttft_histogram.png"))
    
    # 3. TTFT vs Throughput
    plt.figure(figsize=(10, 6))
    throughputs = [t for t, ttft in zip(metrics["throughputs"], metrics["ttft"]) if ttft is not None]
    if len(throughputs) == len(ttft_values):
        plt.scatter(ttft_values, throughputs, alpha=0.5)
        plt.title(f"{title_prefix}TTFT vs Throughput")
        plt.xlabel("TTFT (seconds)")
        plt.ylabel("Throughput (tokens/second)")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "ttft_vs_throughput.png"))
    
    # 4. Goodput analysis (if available in stability report)
    goodput_metrics = {k: v for k, v in stability_report.items() if k.startswith("ttft_goodput_")}
    if goodput_metrics:
        thresholds = []
        goodputs = []
        for k, v in sorted(goodput_metrics.items()):
            match = re.search(r'ttft_goodput_(.+)s', k)
            if match:
                threshold = float(match.group(1))
                thresholds.append(threshold)
                goodputs.append(v)
        
        plt.figure(figsize=(10, 6))
        plt.bar(thresholds, goodputs, alpha=0.7)
        plt.title(f"{title_prefix}TTFT Goodput (% requests meeting threshold)")
        plt.xlabel("TTFT Threshold (seconds)")
        plt.ylabel("Goodput (%)")
        plt.ylim(0, 1.0)
        for i, v in enumerate(goodputs):
            plt.text(thresholds[i], v + 0.02, f"{v:.2%}", ha='center')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "ttft_goodput.png"))


def plot_bucket_size_analysis(stability_report: Dict[str, Any], 
                             output_dir: str,
                             title_prefix: str = ""):
    """
    Generate bucket size analysis plots.
    
    Args:
        stability_report: Stability report data
        output_dir: Directory to save plots
        title_prefix: Prefix for plot titles
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract bucket size metrics
    bucket_metrics = {}
    for k, v in stability_report.items():
        if k.startswith("bucket_size_"):
            parts = k.split("_")
            metric_type = "_".join(parts[0:2])  # bucket_size
            metric_name = "_".join(parts[2:-1])  # volatility, max, min
            pod = parts[-1]  # pod name
            
            if pod not in bucket_metrics:
                bucket_metrics[pod] = {}
            bucket_metrics[pod][metric_name] = v
    
    if not bucket_metrics:
        print("No bucket size metrics available for plotting")
        return
    
    # Plot bucket size metrics by pod
    plt.figure(figsize=(12, 6))
    pods = list(bucket_metrics.keys())
    
    # Volatility comparison
    volatilities = [bucket_metrics[pod].get("volatility", 0) for pod in pods]
    plt.bar(pods, volatilities, alpha=0.7)
    plt.title(f"{title_prefix}VTC Bucket Size Volatility by Pod")
    plt.xlabel("Pod")
    plt.ylabel("Volatility (std dev of changes)")
    for i, v in enumerate(volatilities):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "bucket_size_volatility.png"))
    
    # Min/Max range comparison
    plt.figure(figsize=(12, 6))
    ranges = []
    labels = []
    for pod in pods:
        if "max" in bucket_metrics[pod] and "min" in bucket_metrics[pod]:
            min_val = bucket_metrics[pod]["min"]
            max_val = bucket_metrics[pod]["max"]
            ranges.append((min_val, max_val - min_val))
            labels.append(pod)
    
    if ranges:
        plt.bar(labels, [r[1] for r in ranges], bottom=[r[0] for r in ranges], alpha=0.7)
        plt.title(f"{title_prefix}VTC Bucket Size Range by Pod")
        plt.xlabel("Pod")
        plt.ylabel("Bucket Size")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "bucket_size_range.png"))


def plot_workload_impact(metrics: Dict[str, List[Any]], 
                        output_dir: str,
                        title_prefix: str = ""):
    """
    Generate plots showing workload pattern impact on performance.
    
    Args:
        metrics: Dictionary of extracted metrics
        output_dir: Directory to save plots
        title_prefix: Prefix for plot titles
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have workload data
    if all(r is None for r in metrics["workload_rates"]):
        print("No workload rate data available for plotting")
        return
    
    # Filter out None values
    valid_indices = [i for i, (r, t) in enumerate(zip(metrics["workload_rates"], metrics["ttft"])) 
                    if r is not None and t is not None]
    
    if not valid_indices:
        return
        
    rates = [metrics["workload_rates"][i] for i in valid_indices]
    ttfts = [metrics["ttft"][i] for i in valid_indices]
    
    # TTFT vs Workload Rate
    plt.figure(figsize=(10, 6))
    plt.scatter(rates, ttfts, alpha=0.5)
    plt.title(f"{title_prefix}TTFT vs Workload Rate")
    plt.xlabel("Workload Rate (requests/second)")
    plt.ylabel("TTFT (seconds)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "ttft_vs_workload_rate.png"))
    
    # If we have tier data, plot by tier
    if any(t is not None for t in metrics["workload_tiers"]):
        tiers = [metrics["workload_tiers"][i] for i in valid_indices]
        unique_tiers = list(set(t for t in tiers if t is not None))
        
        if unique_tiers:
            plt.figure(figsize=(10, 6))
            for tier in unique_tiers:
                tier_indices = [i for i, t in enumerate(tiers) if t == tier]
                tier_ttfts = [ttfts[i] for i in tier_indices]
                plt.boxplot(tier_ttfts, positions=[unique_tiers.index(tier)], 
                           labels=[tier], widths=0.5)
            
            plt.title(f"{title_prefix}TTFT by Workload Tier")
            plt.xlabel("Workload Tier")
            plt.ylabel("TTFT (seconds)")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, "ttft_by_tier.png"))


def generate_summary_table(metrics: Dict[str, List[Any]], 
                          stability_report: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate a summary table of key metrics.
    
    Args:
        metrics: Dictionary of extracted metrics
        stability_report: Stability report data
        
    Returns:
        DataFrame containing summary statistics
    """
    summary = {}
    
    # TTFT statistics
    ttft_values = [t for t in metrics["ttft"] if t is not None]
    if ttft_values:
        summary["TTFT P50 (s)"] = np.percentile(ttft_values, 50)
        summary["TTFT P90 (s)"] = np.percentile(ttft_values, 90)
        summary["TTFT P99 (s)"] = np.percentile(ttft_values, 99)
        summary["TTFT Mean (s)"] = np.mean(ttft_values)
        summary["TTFT Std Dev (s)"] = np.std(ttft_values)
    
    # Latency statistics
    latencies = [l for l in metrics["latencies"] if l is not None]
    if latencies:
        summary["Latency P50 (s)"] = np.percentile(latencies, 50)
        summary["Latency P90 (s)"] = np.percentile(latencies, 90)
        summary["Latency P99 (s)"] = np.percentile(latencies, 99)
        summary["Latency Mean (s)"] = np.mean(latencies)
    
    # Throughput statistics
    throughputs = [t for t in metrics["throughputs"] if t is not None]
    if throughputs:
        summary["Throughput Mean (tokens/s)"] = np.mean(throughputs)
    
    # Bucket size statistics from stability report
    for k, v in stability_report.items():
        if k.startswith("bucket_size_volatility_"):
            pod = k.split("_")[-1]
            summary[f"Bucket Size Volatility ({pod})"] = v
    
    # Goodput statistics from stability report
    for k, v in stability_report.items():
        if k.startswith("ttft_goodput_"):
            threshold = k.split("_")[-1]
            summary[f"TTFT Goodput {threshold}"] = v
    
    # Convert to DataFrame for better formatting
    return pd.DataFrame([summary])


def main(args):
    """
    Main function to analyze benchmark results.
    
    Args:
        args: Command line arguments
    """
    # Load benchmark data
    data = load_benchmark_data(args.trace)
    print(f"Loaded {len(data)} benchmark entries from {args.trace}")
    
    # Determine stability report path if not provided
    if not args.stability_report:
        base_name = os.path.splitext(args.trace)[0]
        stability_report_path = f"{base_name}_stability_report.json"
    else:
        stability_report_path = args.stability_report
    
    # Load stability report
    stability_report = load_stability_report(stability_report_path)
    print(f"Loaded stability report from {stability_report_path}")
    
    # Extract metrics
    metrics = extract_metrics(data)
    
    # Create output directory
    output_dir = args.output if args.output else os.path.join(os.path.dirname(args.trace), "analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate title prefix from trace filename
    title_prefix = ""
    if args.title:
        title_prefix = f"{args.title} - "
    
    # Generate plots
    print("Generating TTFT analysis plots...")
    plot_ttft_analysis(metrics, stability_report, output_dir, title_prefix)
    
    print("Generating bucket size analysis plots...")
    plot_bucket_size_analysis(stability_report, output_dir, title_prefix)
    
    print("Generating workload impact plots...")
    plot_workload_impact(metrics, output_dir, title_prefix)
    
    # Generate summary table
    summary_df = generate_summary_table(metrics, stability_report)
    summary_path = os.path.join(output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Generated summary table at {summary_path}")
    
    # Print summary to console
    print("\n=== Summary Statistics ===")
    print(summary_df.to_string(index=False))
    
    print(f"\nAll analysis results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze gateway benchmark results with focus on VTC stability')
    parser.add_argument('--trace', type=str, required=True, help='Input trace containing benchmark results (JSONL)')
    parser.add_argument('--stability-report', type=str, help='Stability report JSON file (if not provided, will look for <trace>_stability_report.json)')
    parser.add_argument('--output', type=str, help='Output directory for analysis results (default: <trace_dir>/analysis)')
    parser.add_argument('--title', type=str, help='Title prefix for plots')
    
    args = parser.parse_args()
    main(args)
