#!/usr/bin/env python3
"""
VTC (Virtual Token Clustering) configuration and metrics analysis.
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def analyze_vtc_metrics(metrics: Dict) -> Dict:
    """
    Analyze VTC bucket size metrics for configuration validation.

    Args:
        metrics: Prometheus metrics data containing VTC information

    Returns:
        Dict with VTC analysis results including:
        - bucket_count: Number of bucket size data points
        - bucket_sizes: List of bucket sizes per pod
        - bucket_changes: List of bucket change rates
        - stability_assessment: Overall stability status
        - configuration_status: Configuration health status
        - static_bucket_warning: Warning if bucket sizes are static
        - bucket_adaptation_status: Whether buckets are adapting to workload
        - uniform_bucket_sizes: Whether all buckets are uniform
        - potential_issues: List of potential issues detected
    """
    vtc_data = metrics.get("vtc", {})
    bucket_size_data = vtc_data.get("vtc_bucket_size_active", [])
    bucket_changes_data = vtc_data.get("vtc_bucket_size_changes", [])

    analysis = {
        "bucket_count": len(bucket_size_data),
        "bucket_sizes": [],
        "bucket_changes": [],
        "stability_assessment": "unknown",
        "configuration_status": "unknown",
        "static_bucket_warning": False,
        "bucket_adaptation_status": "unknown",
        "uniform_bucket_sizes": False,
        "potential_issues": [],
    }

    if bucket_size_data:
        # Extract bucket sizes per pod
        for entry in bucket_size_data:
            pod = entry.get("metric", {}).get("exported_pod", "unknown")
            model = entry.get("metric", {}).get("model", "unknown")
            size = float(entry.get("value", [0, "0"])[1])
            analysis["bucket_sizes"].append({"pod": pod, "model": model, "size": size})

    if bucket_changes_data:
        # Extract bucket change rates
        for entry in bucket_changes_data:
            pod = entry.get("metric", {}).get("exported_pod", "unknown")
            change_rate = float(entry.get("value", [0, "0"])[1])
            analysis["bucket_changes"].append({"pod": pod, "change_rate": change_rate})

    # Check for static bucket sizes (major issue indicator)
    if analysis["bucket_changes"]:
        total_changes = sum(item["change_rate"] for item in analysis["bucket_changes"])
        if total_changes == 0:
            analysis["static_bucket_warning"] = True
            analysis["bucket_adaptation_status"] = "static_no_adaptation"
            analysis["potential_issues"].append(
                "CRITICAL: Bucket sizes are completely static - VTC not adapting to workload"
            )
        elif total_changes < 0.01:
            analysis["bucket_adaptation_status"] = "minimal_adaptation"
            analysis["potential_issues"].append(
                "WARNING: Very minimal bucket adaptation detected"
            )
        else:
            analysis["bucket_adaptation_status"] = "active_adaptation"

    # Check for uniform bucket sizes across pods
    if analysis["bucket_sizes"] and len(analysis["bucket_sizes"]) > 1:
        sizes = [item["size"] for item in analysis["bucket_sizes"]]
        unique_sizes = set(sizes)
        if len(unique_sizes) == 1:
            analysis["uniform_bucket_sizes"] = True
            uniform_size = sizes[0]
            analysis["potential_issues"].append(
                f"WARNING: All pods have identical bucket size ({uniform_size}) - may indicate configuration issue"
            )

    # Enhanced bucket size analysis
    if analysis["bucket_sizes"]:
        sizes = [item["size"] for item in analysis["bucket_sizes"]]
        min_size = min(sizes)
        max_size = max(sizes)
        avg_size = sum(sizes) / len(sizes)

        # Check for problematic bucket size patterns
        if min_size == max_size == 100:
            analysis["potential_issues"].append(
                "CRITICAL: All bucket sizes are exactly 100 (default) - VTC likely not working"
            )
            analysis["configuration_status"] = "default_values_not_adapting"
        elif min_size > 0 and max_size < 1000 and not analysis["uniform_bucket_sizes"]:
            analysis["configuration_status"] = "good"
        elif min_size == 0:
            analysis["configuration_status"] = "bucket_size_zero"
            analysis["potential_issues"].append("CRITICAL: Some bucket sizes are zero")
        elif max_size > 1000:
            analysis["configuration_status"] = "bucket_size_too_large"
            analysis["potential_issues"].append(
                f"WARNING: Large bucket size detected ({max_size})"
            )
        elif analysis["uniform_bucket_sizes"]:
            analysis["configuration_status"] = "uniform_buckets_suspicious"
        else:
            analysis["configuration_status"] = "needs_review"

        # Add bucket size statistics
        analysis["bucket_size_stats"] = {
            "min": min_size,
            "max": max_size,
            "avg": avg_size,
            "range": max_size - min_size,
        }

    # Assess stability based on change rates
    if analysis["bucket_changes"]:
        avg_change = sum(
            item["change_rate"] for item in analysis["bucket_changes"]
        ) / len(analysis["bucket_changes"])
        if avg_change < 0.1:
            analysis["stability_assessment"] = "stable"
        elif avg_change < 0.5:
            analysis["stability_assessment"] = "moderate_oscillation"
        else:
            analysis["stability_assessment"] = "high_oscillation"
            analysis["potential_issues"].append(
                f"WARNING: High bucket oscillation detected (avg change: {avg_change:.2f})"
            )

    return analysis


def analyze_pod_concentration(requests_data: List[Dict]) -> Dict:
    """
    Analyze pod concentration in VTC requests to detect routing issues.

    Args:
        requests_data: List of request dictionaries with pod information

    Returns:
        Dict with pod concentration analysis
    """
    if not requests_data:
        return {"status": "no_data", "concentration_warning": False}

    # Count requests per pod
    pod_counts = {}
    total_requests = 0

    for req in requests_data:
        if req.get("success", False):
            pod = req.get("target_pod", "unknown")
            pod_counts[pod] = pod_counts.get(pod, 0) + 1
            total_requests += 1

    if total_requests == 0:
        return {"status": "no_successful_requests", "concentration_warning": False}

    # Calculate concentration metrics
    pod_percentages = {
        pod: (count / total_requests) * 100 for pod, count in pod_counts.items()
    }

    max_concentration = max(pod_percentages.values()) if pod_percentages else 0
    num_pods_used = len(pod_counts)

    analysis = {
        "total_requests": total_requests,
        "pods_used": num_pods_used,
        "pod_distribution": pod_counts,
        "pod_percentages": pod_percentages,
        "max_concentration": max_concentration,
        "concentration_warning": False,
        "status": "analyzed",
        "issues": [],
    }

    # Detect concentration issues
    if max_concentration > 90:
        analysis["concentration_warning"] = True
        analysis["issues"].append(
            f"CRITICAL: {max_concentration:.1f}% of requests went to single pod - VTC over-clustering"
        )
    elif max_concentration > 70:
        analysis["concentration_warning"] = True
        analysis["issues"].append(
            f"WARNING: {max_concentration:.1f}% of requests went to single pod - high concentration"
        )

    if num_pods_used == 1 and total_requests > 10:
        analysis["issues"].append(
            "CRITICAL: All requests routed to single pod - VTC clustering too aggressive"
        )

    return analysis


def log_vtc_analysis(analysis: Dict, prefix: str = "") -> None:
    """
    Log VTC analysis results in a readable format.

    Args:
        analysis: VTC analysis dictionary
        prefix: Optional prefix for log messages
    """
    if prefix:
        prefix = f"{prefix} "

    logger.info(f"{prefix}VTC Configuration Status: {analysis['configuration_status']}")
    logger.info(f"{prefix}VTC Stability: {analysis['stability_assessment']}")
    logger.info(
        f"{prefix}VTC Bucket Adaptation: {analysis['bucket_adaptation_status']}"
    )

    # Log critical issues first
    if analysis.get("potential_issues"):
        logger.warning(f"{prefix}VTC Issues Detected:")
        for issue in analysis["potential_issues"]:
            logger.warning(f"  {issue}")

    if analysis["bucket_sizes"]:
        logger.info(f"{prefix}VTC Bucket Sizes:")
        for bucket in analysis["bucket_sizes"]:
            logger.info(f"  Pod {bucket['pod']}: bucket_size={bucket['size']}")

        # Log bucket statistics
        if "bucket_size_stats" in analysis:
            stats = analysis["bucket_size_stats"]
            logger.info(
                f"  Bucket Size Stats: min={stats['min']}, max={stats['max']}, avg={stats['avg']:.1f}, range={stats['range']}"
            )

    if analysis["bucket_changes"]:
        has_changes = any(b["change_rate"] > 0.01 for b in analysis["bucket_changes"])
        if has_changes:
            logger.info(f"{prefix}VTC Bucket Changes:")
            for change in analysis["bucket_changes"]:
                if change["change_rate"] > 0.01:
                    logger.info(
                        f"  Pod {change['pod']}: change_rate={change['change_rate']:.3f}"
                    )
        else:
            logger.warning(
                f"{prefix}No significant bucket changes detected (all rates < 0.01)"
            )


def assess_vtc_configuration(baseline_analysis: Dict, final_analysis: Dict) -> Dict:
    """
    Compare baseline and final VTC metrics to assess overall configuration health.

    Args:
        baseline_analysis: VTC analysis from baseline metrics
        final_analysis: VTC analysis from final metrics

    Returns:
        Dict with overall VTC assessment
    """
    assessment = {
        "overall_status": "unknown",
        "stability_trend": "unknown",
        "recommendations": [],
        "critical_issues": [],
    }

    # Check for static bucket issue
    if baseline_analysis.get("static_bucket_warning") or final_analysis.get(
        "static_bucket_warning"
    ):
        assessment["critical_issues"].append("Static bucket sizes detected")
        assessment["recommendations"].extend(
            [
                "Check VTC_TOKEN_TRACKER_MIN_TOKENS and MAX_TOKENS configuration",
                "Verify token accumulation is working properly",
                "Consider adjusting VTC_TOKEN_TRACKER_WINDOW_SIZE",
                "Check if user token usage varies enough to trigger bucket changes",
            ]
        )

    # Check for uniform bucket sizes
    if baseline_analysis.get("uniform_bucket_sizes") and final_analysis.get(
        "uniform_bucket_sizes"
    ):
        assessment["critical_issues"].append("Uniform bucket sizes across all pods")
        assessment["recommendations"].append(
            "All pods have identical bucket sizes - VTC may not be differentiating workloads"
        )

    # Compare stability
    baseline_stability = baseline_analysis.get("stability_assessment", "unknown")
    final_stability = final_analysis.get("stability_assessment", "unknown")

    if baseline_stability == "stable" and final_stability == "stable":
        if final_analysis.get("static_bucket_warning"):
            assessment["stability_trend"] = "stable_but_static"
        else:
            assessment["stability_trend"] = "consistent_stable"
    elif final_stability == "high_oscillation":
        assessment["stability_trend"] = "degraded"
        assessment["recommendations"].append(
            "Consider increasing minimum bucket size or lengthening adjustment window"
        )
    else:
        assessment["stability_trend"] = "acceptable"

    # Check configuration
    baseline_config = baseline_analysis.get("configuration_status", "unknown")
    final_config = final_analysis.get("configuration_status", "unknown")

    if baseline_config == "good" and final_config == "good":
        assessment["overall_status"] = "healthy"
    elif "default_values_not_adapting" in final_config:
        assessment["overall_status"] = "critical"
        assessment["critical_issues"].append(
            "VTC using default values and not adapting"
        )
        assessment["recommendations"].extend(
            [
                "VTC appears to be using default bucket size (100) without adaptation",
                "Check if token tracking is enabled and working",
                "Verify VTC configuration environment variables are set correctly",
            ]
        )
    elif "zero" in final_config:
        assessment["overall_status"] = "critical"
        assessment["recommendations"].append(
            "Bucket size reached zero - check VTC configuration immediately"
        )
    elif "too_large" in final_config:
        assessment["overall_status"] = "warning"
        assessment["recommendations"].append(
            "Bucket size is too large - may impact fairness"
        )
    else:
        assessment["overall_status"] = "needs_attention"

    return assessment
