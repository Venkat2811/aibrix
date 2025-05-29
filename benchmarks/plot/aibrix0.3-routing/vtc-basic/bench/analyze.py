#!/usr/bin/env python3
"""
General analysis functions for VTC benchmarking results.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def analyze_fairness_metrics(
    requests_by_algorithm: Dict, metrics_by_algorithm: Optional[Dict] = None
) -> Dict:
    """
    Analyze fairness between user categories to validate VTC effectiveness.

    The goal is to ensure high token users don't overwhelm low token users.

    Args:
        requests_by_algorithm: Dict of algorithm -> list of request results
        metrics_by_algorithm: Optional dict of algorithm -> metrics data

    Returns:
        Dict with fairness analysis including:
        - Per-algorithm statistics by user category
        - Fairness comparison between algorithms
        - Overall fairness assessment
    """
    if metrics_by_algorithm is None:
        metrics_by_algorithm = {}

    analysis = {"algorithms": {}, "fairness_comparison": {}}

    for algorithm, requests in requests_by_algorithm.items():
        # Group by user categories
        category_stats = {}
        for req in requests:
            user = req.get("user", "unknown")
            category = "unknown"

            # Determine category from user name
            if "small" in user:
                category = "small"
            elif "med" in user:
                category = "medium"
            elif "high" in user:
                category = "high"

            if category not in category_stats:
                category_stats[category] = {
                    "requests": [],
                    "total_latency": 0,
                    "success_count": 0,
                    "failure_count": 0,
                }

            category_stats[category]["requests"].append(req)
            if req.get("success", False):
                category_stats[category]["success_count"] += 1
                category_stats[category]["total_latency"] += req.get("latency", 0)
            else:
                category_stats[category]["failure_count"] += 1

        # Calculate fairness metrics
        algo_analysis = {}
        for category, stats in category_stats.items():
            total_requests = len(stats["requests"])
            avg_latency = stats["total_latency"] / max(stats["success_count"], 1)
            success_rate = stats["success_count"] / max(total_requests, 1)

            algo_analysis[category] = {
                "request_count": total_requests,
                "avg_latency": avg_latency,
                "success_rate": success_rate,
                "failure_count": stats["failure_count"],
            }

        analysis["algorithms"][algorithm] = algo_analysis

    # Compare fairness between algorithms
    if "random" in analysis["algorithms"] and "vtc-basic" in analysis["algorithms"]:
        random_stats = analysis["algorithms"]["random"]
        vtc_stats = analysis["algorithms"]["vtc-basic"]

        comparison = {}
        for category in ["small", "medium", "high"]:
            if category in random_stats and category in vtc_stats:
                random_latency = random_stats[category]["avg_latency"]
                vtc_latency = vtc_stats[category]["avg_latency"]

                latency_improvement = (
                    ((random_latency - vtc_latency) / random_latency) * 100
                    if random_latency > 0
                    else 0
                )

                comparison[category] = {
                    "random_avg_latency": random_latency,
                    "vtc_avg_latency": vtc_latency,
                    "latency_improvement_pct": latency_improvement,
                    "fairness_score": (
                        "improved" if latency_improvement > 0 else "degraded"
                    ),
                }

        analysis["fairness_comparison"] = comparison

    return analysis


def log_fairness_summary(fairness_analysis: Dict) -> None:
    """
    Log a comprehensive fairness summary.

    Args:
        fairness_analysis: Fairness analysis dictionary
    """
    logger.info("\n" + "=" * 60)
    logger.info("FAIRNESS ANALYSIS SUMMARY")
    logger.info("=" * 60)

    # Check validation status
    validation = fairness_analysis.get("validation", {})
    if not validation.get("valid_for_comparison", True):
        logger.error("⚠️  ANALYSIS MAY BE UNRELIABLE DUE TO HIGH FAILURE RATES")
        logger.error(f"   Reason: {validation.get('reason', 'Unknown')}")
        logger.info("")

    # Log per-algorithm statistics
    for algorithm, stats in fairness_analysis.get("algorithms", {}).items():
        logger.info(f"\n{algorithm.upper()} Routing Statistics:")
        for category in ["small", "medium", "high"]:
            if category in stats:
                cat_stats = stats[category]
                logger.info(
                    f"  {category}: avg_latency={cat_stats['avg_latency']:.3f}s, "
                    f"requests={cat_stats['request_count']}, "
                    f"success_rate={cat_stats['success_rate']:.1%}"
                )

    # Log fairness comparison
    comparison = fairness_analysis.get("fairness_comparison", {})
    if comparison:
        logger.info("\nFAIRNESS COMPARISON (VTC vs Random):")

        for category in ["small", "medium", "high"]:
            if category in comparison:
                comp = comparison[category]
                improvement = comp["latency_improvement_pct"]
                logger.info(f"\n  {category.upper()} token users:")
                logger.info(f"    Random routing: {comp['random_avg_latency']:.3f}s")
                logger.info(f"    VTC routing:    {comp['vtc_avg_latency']:.3f}s")
                logger.info(
                    f"    Improvement:    {improvement:+.1f}% ({comp['fairness_score']})"
                )

        # Overall fairness assessment
        improvements = [comp["latency_improvement_pct"] for comp in comparison.values()]
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0

        logger.info("\n" + "-" * 60)
        if avg_improvement > 0:
            logger.info(
                f"✅ VTC ROUTING SHOWS FAIRNESS IMPROVEMENT: {avg_improvement:+.1f}% average"
            )
            logger.info("   High token users are NOT overwhelming low token users!")
        else:
            logger.info(f"⚠️  VTC ROUTING NEEDS TUNING: {avg_improvement:+.1f}% average")
            logger.info("   Consider adjusting bucket sizes or clustering parameters")
    else:
        logger.info("⚠️  Not enough data for fairness comparison")

    logger.info("=" * 60)


def load_benchmark_results(result_dir: str) -> Dict:
    """
    Load all benchmark results from a directory.

    Args:
        result_dir: Path to the results directory

    Returns:
        Dict containing all loaded data
    """
    results = {
        "requests": [],
        "system_metrics": {},
        "fairness_analysis": None,
        "metadata": {"result_dir": result_dir, "load_time": datetime.now().isoformat()},
    }

    # Load request results
    request_files = list(Path(result_dir).glob("request_*.json"))
    for file in request_files:
        with open(file, "r") as f:
            results["requests"].append(json.load(f))

    # Load system metrics
    metric_files = list(Path(result_dir).glob("system_metrics_*.json"))
    for file in metric_files:
        metric_type = file.stem.replace("system_metrics_", "")
        with open(file, "r") as f:
            results["system_metrics"][metric_type] = json.load(f)

    # Load fairness analysis if exists
    fairness_file = Path(result_dir) / "fairness_analysis.json"
    if fairness_file.exists():
        with open(fairness_file, "r") as f:
            results["fairness_analysis"] = json.load(f)

    return results


def analyze_pod_distribution(requests: List[Dict]) -> Dict:
    """
    Analyze how requests were distributed across pods.

    Args:
        requests: List of request results

    Returns:
        Dict with pod distribution analysis
    """
    pod_stats = {}
    algorithm_pod_distribution = {}

    for req in requests:
        pod = req.get("target_pod", "unknown")
        algorithm = req.get("routing_algorithm", "unknown")

        # Overall pod stats
        if pod not in pod_stats:
            pod_stats[pod] = {
                "request_count": 0,
                "total_latency": 0,
                "success_count": 0,
                "user_categories": {},
            }

        pod_stats[pod]["request_count"] += 1
        if req.get("success", False):
            pod_stats[pod]["success_count"] += 1
            pod_stats[pod]["total_latency"] += req.get("latency", 0)

        # Track user categories per pod
        user = req.get("user", "")
        category = "unknown"
        if "small" in user:
            category = "small"
        elif "med" in user:
            category = "medium"
        elif "high" in user:
            category = "high"

        pod_stats[pod]["user_categories"][category] = (
            pod_stats[pod]["user_categories"].get(category, 0) + 1
        )

        # Algorithm-specific pod distribution
        if algorithm not in algorithm_pod_distribution:
            algorithm_pod_distribution[algorithm] = {}
        if pod not in algorithm_pod_distribution[algorithm]:
            algorithm_pod_distribution[algorithm][pod] = 0
        algorithm_pod_distribution[algorithm][pod] += 1

    # Calculate average latencies
    for pod, stats in pod_stats.items():
        if stats["success_count"] > 0:
            stats["avg_latency"] = stats["total_latency"] / stats["success_count"]
        else:
            stats["avg_latency"] = 0

    return {
        "pod_stats": pod_stats,
        "algorithm_distribution": algorithm_pod_distribution,
        "unique_pods": list(pod_stats.keys()),
        "pod_count": len(pod_stats),
    }


def generate_analysis_report(
    result_dir: str, output_file: Optional[str] = None
) -> None:
    """
    Generate a comprehensive analysis report from benchmark results.

    Args:
        result_dir: Path to the results directory
        output_file: Optional output file path for the report
    """
    # Load results
    results = load_benchmark_results(result_dir)

    # Perform analyses
    pod_analysis = analyze_pod_distribution(results["requests"])

    # Group requests by algorithm for fairness analysis
    requests_by_algorithm = {}
    for req in results["requests"]:
        algo = req.get("routing_algorithm", "unknown")
        if algo not in requests_by_algorithm:
            requests_by_algorithm[algo] = []
        requests_by_algorithm[algo].append(req)

    fairness_analysis = analyze_fairness_metrics(requests_by_algorithm)

    # Create report
    report = {
        "metadata": results["metadata"],
        "summary": {
            "total_requests": len(results["requests"]),
            "algorithms_tested": list(requests_by_algorithm.keys()),
            "pods_used": pod_analysis["pod_count"],
        },
        "pod_distribution": pod_analysis,
        "fairness_analysis": fairness_analysis,
        "recommendations": [],
    }

    # Add recommendations based on analysis
    if fairness_analysis.get("fairness_comparison"):
        improvements = [
            comp["latency_improvement_pct"]
            for comp in fairness_analysis["fairness_comparison"].values()
        ]
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0

        if avg_improvement < 0:
            report["recommendations"].append(
                "VTC routing shows degraded performance - consider tuning parameters"
            )
        elif avg_improvement < 5:
            report["recommendations"].append(
                "VTC routing shows minimal improvement - may need more traffic to demonstrate benefits"
            )
        else:
            report["recommendations"].append(
                "VTC routing successfully prevents high-token users from overwhelming the system"
            )

    # Save report
    if output_file:
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Analysis report saved to: {output_file}")

    return report
