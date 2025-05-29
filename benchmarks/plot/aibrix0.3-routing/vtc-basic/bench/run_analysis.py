#!/usr/bin/env python3
"""
CLI script to analyze existing benchmark results.

Usage:
    python run_analysis.py <result_dir> [--output report.json]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from analyze import (
    generate_analysis_report,
    load_benchmark_results,
    log_fairness_summary,
)
from analyze_vtc_config import (
    analyze_vtc_metrics,
    assess_vtc_configuration,
    log_vtc_analysis,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Analyze VTC benchmark results")
    parser.add_argument("result_dir", help="Path to the benchmark results directory")
    parser.add_argument(
        "--output", help="Output file for the analysis report (optional)", default=None
    )
    parser.add_argument(
        "--vtc-only", action="store_true", help="Only run VTC configuration analysis"
    )
    parser.add_argument(
        "--fairness-only", action="store_true", help="Only run fairness analysis"
    )

    args = parser.parse_args()

    # Validate result directory
    result_path = Path(args.result_dir)
    if not result_path.exists():
        logger.error(f"Result directory not found: {args.result_dir}")
        sys.exit(1)

    # Run full analysis if no specific flags
    if not args.vtc_only and not args.fairness_only:
        logger.info(f"Running full analysis on: {args.result_dir}")
        report = generate_analysis_report(args.result_dir, args.output)

        if args.output:
            logger.info(f"Analysis report saved to: {args.output}")
        else:
            print("\n" + "=" * 60)
            print("ANALYSIS REPORT")
            print("=" * 60)
            print(json.dumps(report, indent=2))

        return

    # Load results for specific analyses
    results = load_benchmark_results(args.result_dir)

    # VTC analysis
    if args.vtc_only:
        logger.info("Running VTC configuration analysis...")

        baseline_metrics = results["system_metrics"].get("baseline", {})
        final_metrics = results["system_metrics"].get("final", {})

        if baseline_metrics and final_metrics:
            baseline_analysis = analyze_vtc_metrics(baseline_metrics)
            final_analysis = analyze_vtc_metrics(final_metrics)

            log_vtc_analysis(baseline_analysis, "Baseline")
            log_vtc_analysis(final_analysis, "Final")

            assessment = assess_vtc_configuration(baseline_analysis, final_analysis)
            logger.info(f"\nOverall VTC Assessment: {assessment['overall_status']}")
            logger.info(f"Stability Trend: {assessment['stability_trend']}")

            if assessment["recommendations"]:
                logger.info("\nRecommendations:")
                for rec in assessment["recommendations"]:
                    logger.info(f"  - {rec}")
        else:
            logger.error("Missing baseline or final metrics for VTC analysis")

    # Fairness analysis
    if args.fairness_only:
        logger.info("Running fairness analysis...")

        # Group requests by algorithm
        requests_by_algorithm = {}
        for req in results["requests"]:
            algo = req.get("routing_algorithm", "unknown")
            if algo not in requests_by_algorithm:
                requests_by_algorithm[algo] = []
            requests_by_algorithm[algo].append(req)

        from analyze import analyze_fairness_metrics

        fairness_analysis = analyze_fairness_metrics(requests_by_algorithm)
        log_fairness_summary(fairness_analysis)


if __name__ == "__main__":
    main()
