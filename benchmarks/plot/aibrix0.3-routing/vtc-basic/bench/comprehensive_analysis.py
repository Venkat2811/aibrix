#!/usr/bin/env python3
"""Comprehensive VTC Analysis - One-stop analysis for bench_and_collect_stats results."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from analyze import analyze_fairness_metrics, log_fairness_summary
from analyze_vtc_benchmark import VTCBenchmarkAnalyzer
from analyze_vtc_config import (
    analyze_vtc_metrics,
    assess_vtc_configuration,
    log_vtc_analysis,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_benchmark_results(result_dir: str) -> Dict:
    """Load comprehensive benchmark results from directory."""
    result_path = Path(result_dir)

    # Load main stats file
    stats_file = result_path / "comprehensive_benchmark_stats.json"
    if not stats_file.exists():
        logger.error(f"Comprehensive stats file not found: {stats_file}")
        sys.exit(1)

    with open(stats_file) as f:
        stats_data = json.load(f)

    # Load individual request files
    request_files = sorted(result_path.glob("request_*.json"))
    requests = []

    for request_file in request_files:
        with open(request_file) as f:
            request_data = json.load(f)
            requests.append(request_data)

    return {
        "stats": stats_data,
        "requests": requests,
        "system_metrics": stats_data.get("system_metrics", {}),
    }


def run_pod_performance_analysis(stats_data: Dict):
    """Run pod performance analysis."""
    logger.info("Running pod performance analysis...")

    pod_performance = {}
    for req in stats_data["request_data"]:
        pod = req["target_pod"]
        algo = req["algorithm"]
        latency = req["latency"]

        if pod not in pod_performance:
            pod_performance[pod] = {"vtc-basic": [], "random": []}
        pod_performance[pod][algo].append(latency)

    print("\nPod Performance Analysis:")
    print("=" * 60)

    for pod in sorted(pod_performance.keys()):
        vtc_latencies = pod_performance[pod]["vtc-basic"]
        random_latencies = pod_performance[pod]["random"]

        print(f"{pod}:")
        if vtc_latencies:
            avg_vtc = sum(vtc_latencies) / len(vtc_latencies)
            print(
                f"  VTC-Basic: {len(vtc_latencies):2d} reqs, avg {avg_vtc:.2f}s, "
                f"range {min(vtc_latencies):.2f}-{max(vtc_latencies):.2f}s"
            )
        if random_latencies:
            avg_random = sum(random_latencies) / len(random_latencies)
            print(
                f"  Random:    {len(random_latencies):2d} reqs, avg {avg_random:.2f}s, "
                f"range {min(random_latencies):.2f}-{max(random_latencies):.2f}s"
            )

        if vtc_latencies and random_latencies:
            diff = avg_random - avg_vtc
            print(f"  Difference: Random is {diff:.2f}s slower than VTC")
        print()


def run_vtc_configuration_analysis(system_metrics: Dict):
    """Run VTC configuration analysis."""
    logger.info("Running VTC configuration analysis...")

    baseline_metrics = system_metrics.get("baseline", {})
    final_metrics = system_metrics.get("final", {})

    if not baseline_metrics or not final_metrics:
        logger.warning("Missing baseline or final metrics for VTC analysis")
        return

    baseline_analysis = analyze_vtc_metrics(baseline_metrics)
    final_analysis = analyze_vtc_metrics(final_metrics)

    log_vtc_analysis(baseline_analysis, "Baseline")
    log_vtc_analysis(final_analysis, "Final")

    assessment = assess_vtc_configuration(baseline_analysis, final_analysis)

    print(f"\nVTC Configuration Assessment:")
    print(f"Overall Status: {assessment['overall_status']}")
    print(f"Stability Trend: {assessment['stability_trend']}")

    if assessment["recommendations"]:
        print("\nVTC Recommendations:")
        for rec in assessment["recommendations"]:
            print(f"  - {rec}")

    if assessment["critical_issues"]:
        print("\nCritical Issues:")
        for issue in assessment["critical_issues"]:
            print(f"  - {issue}")


def run_fairness_analysis(stats_data: Dict):
    """Run fairness analysis using data from comprehensive stats."""
    logger.info("Running fairness analysis...")

    # Convert request data to the format expected by analyze_fairness_metrics
    requests_by_algorithm = {}

    for req in stats_data.get("request_data", []):
        algorithm = req.get("algorithm", "unknown")
        if algorithm not in requests_by_algorithm:
            requests_by_algorithm[algorithm] = []

        # Convert to expected format
        converted_req = {
            "user": req.get("user", "unknown"),
            "success": req.get("success", False),
            "latency": req.get("latency", 0),
            "status_code": req.get("status_code", 0),
            "target_pod": req.get("target_pod", "unknown"),
            "category": req.get("category", "unknown"),
            "routing_algorithm": algorithm,
        }
        requests_by_algorithm[algorithm].append(converted_req)

    if not requests_by_algorithm:
        logger.warning("No request data found for fairness analysis")
        print("⚠️  No request data available for fairness analysis")
        return None

    fairness_analysis = analyze_fairness_metrics(requests_by_algorithm)
    log_fairness_summary(fairness_analysis)

    return fairness_analysis


def run_comprehensive_vtc_analysis(result_dir: str, prometheus_url: str):
    """Run comprehensive VTC benchmark analysis with visualizations."""
    logger.info("Running comprehensive VTC benchmark analysis...")

    try:
        # Check if the required files exist
        result_path = Path(result_dir)
        required_files = [
            "comprehensive_benchmark_stats.json",
            "fairness_analysis.json",
        ]

        missing_files = []
        for file in required_files:
            if not (result_path / file).exists():
                missing_files.append(file)

        if missing_files:
            logger.warning(f"Missing required files for VTC analysis: {missing_files}")
            return None

        analyzer = VTCBenchmarkAnalyzer(result_dir, prometheus_url)
        report = analyzer.generate_report()

        # Display executive summary
        summary = report["summary"]
        print(f"\nVTC Benchmark Analysis Summary:")
        print(f"VTC Status: {summary['vtc_status']}")
        print(f"Fairness: {summary['fairness_status']}")
        print(f"Overall: {summary['overall_status']}")

        if report["tuning_recommendations"]:
            print(f"\nTuning Recommendations:")
            for i, rec in enumerate(report["tuning_recommendations"], 1):
                print(f"  {i}. {rec}")

        return report
    except IndexError as e:
        logger.warning(f"Comprehensive VTC analysis failed - data structure issue: {e}")
        logger.warning(
            "This may indicate incomplete benchmark data or configuration issues"
        )
        return None
    except Exception as e:
        logger.warning(f"Comprehensive VTC analysis failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive VTC Analysis - Analyze bench_and_collect_stats results"
    )
    parser.add_argument("result_dir", help="Path to benchmark results directory")
    parser.add_argument(
        "--prometheus-url", default="http://localhost:9090", help="Prometheus URL"
    )
    parser.add_argument(
        "--skip-vtc-comprehensive",
        action="store_true",
        help="Skip comprehensive VTC analysis",
    )
    parser.add_argument(
        "--analysis-only",
        choices=["pod", "vtc", "fairness", "comprehensive"],
        help="Run only specific analysis",
    )

    args = parser.parse_args()

    result_path = Path(args.result_dir)
    if not result_path.exists():
        logger.error(f"Result directory not found: {args.result_dir}")
        sys.exit(1)

    logger.info(f"Starting comprehensive analysis of: {args.result_dir}")

    # Load benchmark results
    results = load_benchmark_results(args.result_dir)

    print("\n" + "=" * 80)
    print("COMPREHENSIVE VTC BENCHMARK ANALYSIS")
    print("=" * 80)
    print(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results directory: {args.result_dir}")
    print(f"Total requests analyzed: {len(results['stats'].get('request_data', []))}")
    print("=" * 80)

    # Run analyses based on arguments
    if args.analysis_only == "pod":
        run_pod_performance_analysis(results["stats"])
    elif args.analysis_only == "vtc":
        run_vtc_configuration_analysis(results["system_metrics"])
    elif args.analysis_only == "fairness":
        run_fairness_analysis(results["stats"])
    elif args.analysis_only == "comprehensive":
        run_comprehensive_vtc_analysis(args.result_dir, args.prometheus_url)
    else:
        # Run all analyses
        run_pod_performance_analysis(results["stats"])
        print("\n" + "-" * 80)

        run_vtc_configuration_analysis(results["system_metrics"])
        print("\n" + "-" * 80)

        fairness_analysis = run_fairness_analysis(results["stats"])
        print("\n" + "-" * 80)

        if not args.skip_vtc_comprehensive:
            comprehensive_report = run_comprehensive_vtc_analysis(
                args.result_dir, args.prometheus_url
            )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")

    # Check for analysis outputs
    enhanced_dir = result_path / "enhanced_analysis"
    if enhanced_dir.exists():
        print(f"Enhanced analysis saved to: {enhanced_dir}")
        vis_dir = enhanced_dir / "visualizations"
        if vis_dir.exists():
            print(f"Visualizations available in: {vis_dir}")

    fairness_file = result_path / "fairness_analysis.json"
    if fairness_file.exists():
        print(f"Fairness analysis: {fairness_file}")

    print("=" * 80)


if __name__ == "__main__":
    main()
