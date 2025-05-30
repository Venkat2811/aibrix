#!/usr/bin/env python3
"""
VTC Benchmark Analysis Script
Analyzes VTC routing algorithm performance using raw benchmark data and Prometheus metrics.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns


class VTCBenchmarkAnalyzer:
    def __init__(self, results_dir: str, prometheus_url: str = "http://localhost:9090"):
        self.results_dir = Path(results_dir)
        self.prometheus_url = prometheus_url

        # Load basic analysis data
        self.fairness_data = self.load_json("fairness_analysis.json")
        self.comprehensive_stats = self.load_json("comprehensive_benchmark_stats.json")

        # Extract benchmark timeframe
        self.start_time, self.end_time = self._extract_timeframe()

        print(f"✅ Analyzing benchmark results from {self.results_dir}")
        print(f"📊 Timeframe: {self.start_time} to {self.end_time}")

    def load_json(self, filename: str) -> Dict:
        """Load JSON file from results directory"""
        try:
            with open(self.results_dir / filename) as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️  {filename} not found")
            return {}

    def _extract_timeframe(self) -> Tuple[datetime, datetime]:
        """Extract benchmark timeframe from request files"""
        request_files = sorted(self.results_dir.glob("request_*.json"))

        # Get first and last request timestamps - ensure proper ordering
        all_timestamps = []
        for request_file in request_files:
            with open(request_file) as f:
                data = json.load(f)
                timestamp = datetime.fromisoformat(
                    data["timestamp"].replace("Z", "+00:00")
                )
                all_timestamps.append(timestamp)

        all_timestamps.sort()
        start_time = all_timestamps[0]
        end_time = all_timestamps[-1]

        return start_time, end_time

    def query_prometheus(
        self,
        query: str,
        start_time: datetime = None,
        end_time: datetime = None,
        step: str = "30s",
    ) -> Dict:
        """Query Prometheus for metrics data"""
        if start_time is None:
            start_time = self.start_time
        if end_time is None:
            end_time = self.end_time

        start_epoch = int(start_time.timestamp())
        end_epoch = int(end_time.timestamp())

        params = {"query": query, "start": start_epoch, "end": end_epoch, "step": step}

        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query_range", params=params, timeout=30
            )
            response.raise_for_status()
            result = response.json()

            # Check if the response has the expected structure
            if "data" not in result:
                print(f"⚠️  Prometheus query returned unexpected format: {result}")
                return {"data": {"result": []}}

            return result
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Prometheus connection failed: {e}")
            return {"data": {"result": []}}
        except Exception as e:
            print(f"⚠️  Prometheus query failed: {e}")
            return {"data": {"result": []}}

    def analyze_vtc_bucket_evolution(self) -> Dict[str, Any]:
        """Analyze VTC bucket size evolution during benchmark"""
        print("\n🔄 Analyzing VTC bucket evolution...")

        query = "vtc_bucket_size_active"
        result = self.query_prometheus(query, step="30s")

        bucket_data = {}
        for series in result["data"]["result"]:
            pod = series["metric"]["exported_pod"]
            values = [float(v[1]) for v in series["values"]]
            timestamps = [datetime.fromtimestamp(float(v[0])) for v in series["values"]]

            bucket_data[pod] = {
                "timestamps": timestamps,
                "values": values,
                "min": min(values),
                "max": max(values),
                "final": values[-1] if values else 0,
                "initial": values[0] if values else 0,
                "range": max(values) - min(values) if values else 0,
                "std": np.std(values) if values else 0,
            }

        # Calculate adaptation metrics
        adaptation_analysis = self._analyze_bucket_adaptation(bucket_data)

        return {"bucket_data": bucket_data, "adaptation_analysis": adaptation_analysis}

    def _analyze_bucket_adaptation(self, bucket_data: Dict) -> Dict[str, Any]:
        """Analyze how well VTC adapted during the benchmark"""
        all_values = []
        for pod_data in bucket_data.values():
            all_values.extend(pod_data["values"])

        if not all_values:
            return {"status": "no_data", "recommendations": ["No VTC data available"]}

        total_range = max(all_values) - min(all_values)
        avg_std = np.mean([pod_data["std"] for pod_data in bucket_data.values()])

        # Monotonicity analysis
        monotonicity_scores = []
        for pod_data in bucket_data.values():
            values = pod_data["values"]
            if len(values) > 1:
                # Calculate how often bucket size changes in the expected direction
                diffs = np.diff(values)
                # During workload, we expect bucket sizes to generally increase then stabilize
                # Count significant changes (> 5% of initial value)
                significant_changes = sum(1 for d in diffs if abs(d) > values[0] * 0.05)
                monotonicity_scores.append(significant_changes / max(1, len(diffs)))

        avg_monotonicity = np.mean(monotonicity_scores) if monotonicity_scores else 0

        # Determine adaptation quality
        if total_range < 50:
            status = "static_poor"
            recommendations = [
                "VTC bucket sizes barely changed during benchmark",
                "Consider increasing VTC sensitivity parameters",
                "Check if VTC adaptation is enabled",
            ]
        elif avg_std < 20:
            status = "stable_good"
            recommendations = [
                "VTC bucket sizes adapted well and stabilized",
                "Good adaptation behavior observed",
            ]
        elif avg_std > 100:
            status = "oscillating_poor"
            recommendations = [
                "VTC bucket sizes oscillated significantly",
                "Consider reducing VTC adaptation rate",
                "Increase stabilization time between changes",
            ]
        else:
            status = "moderate_adaptation"
            recommendations = [
                "VTC showed moderate adaptation behavior",
                "Monitor for consistent performance patterns",
            ]

        return {
            "status": status,
            "total_range": total_range,
            "avg_std": avg_std,
            "avg_monotonicity": avg_monotonicity,
            "recommendations": recommendations,
        }

    def analyze_ttft_performance(self) -> Dict[str, Any]:
        """Analyze Time-to-First-Token performance"""
        print("\n⚡ Analyzing TTFT performance...")

        # Query P50, P90, P99 TTFT metrics
        metrics = {}
        for percentile in [50, 90, 99]:
            query = f"histogram_quantile(0.{percentile:02d}, sum by(le, model_name) (rate(vllm:time_to_first_token_seconds_bucket[1m])))"
            result = self.query_prometheus(query, step="60s")

            for series in result["data"]["result"]:
                model = series["metric"]["model_name"]
                if model not in metrics:
                    metrics[model] = {}
                values = [float(v[1]) for v in series["values"] if v[1] != "NaN"]
                if values:
                    metrics[model][f"p{percentile}"] = {
                        "values": values,
                        "avg": np.mean(values),
                        "min": min(values),
                        "max": max(values),
                    }

        return metrics

    def analyze_pod_utilization(self) -> Dict[str, Any]:
        """Analyze pod utilization and load distribution"""
        print("\n🏗️  Analyzing pod utilization...")

        # Extract pod distribution from request files
        pod_distribution = {"vtc-basic": {}, "random": {}}

        for request_file in self.results_dir.glob("request_*.json"):
            with open(request_file) as f:
                data = json.load(f)

            algorithm = data["routing_algorithm"]
            pod = data["target_pod"]

            if pod not in pod_distribution[algorithm]:
                pod_distribution[algorithm][pod] = 0
            pod_distribution[algorithm][pod] += 1

        # Calculate utilization metrics
        utilization_analysis = {}
        for algorithm, pods in pod_distribution.items():
            if not pods:
                continue

            total_requests = sum(pods.values())
            pod_counts = list(pods.values())

            utilization_analysis[algorithm] = {
                "total_requests": total_requests,
                "pod_distribution": pods,
                "load_balance_score": (
                    1 - (np.std(pod_counts) / np.mean(pod_counts)) if pod_counts else 0
                ),
                "max_min_ratio": (
                    max(pod_counts) / min(pod_counts)
                    if min(pod_counts) > 0
                    else float("inf")
                ),
                "gini_coefficient": self._calculate_gini(pod_counts),
            }

        return utilization_analysis

    def _calculate_gini(self, values: List[float]) -> float:
        """Calculate Gini coefficient for load distribution inequality"""
        if not values or len(values) == 1:
            return 0.0

        sorted_values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)

        return (
            n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(sorted_values, 1))
        ) / (n * sum(sorted_values))

    def generate_fairness_analysis(self) -> Dict[str, Any]:
        """Generate enhanced fairness analysis"""
        print("\n⚖️  Analyzing fairness metrics...")

        if not self.fairness_data:
            return {"status": "no_data"}

        fairness_comparison = self.fairness_data.get("fairness_comparison", {})

        # Calculate overall fairness score
        improvements = []
        for category, metrics in fairness_comparison.items():
            improvement = metrics.get("latency_improvement_pct", 0)
            improvements.append(improvement)

        overall_improvement = np.mean(improvements) if improvements else 0

        # Determine fairness quality
        if overall_improvement > 10:
            fairness_quality = "excellent"
        elif overall_improvement > 0:
            fairness_quality = "good"
        elif overall_improvement > -10:
            fairness_quality = "acceptable"
        else:
            fairness_quality = "poor"

        # Generate recommendations
        recommendations = []
        if overall_improvement < -5:
            recommendations.append(
                "VTC routing is performing worse than random - needs tuning"
            )
            recommendations.append("Consider adjusting bucket size thresholds")
            recommendations.append("Review user categorization logic")

        worst_category = min(
            fairness_comparison.keys(),
            key=lambda k: fairness_comparison[k].get("latency_improvement_pct", 0),
        )

        if fairness_comparison[worst_category]["latency_improvement_pct"] < -15:
            recommendations.append(
                f"{worst_category} users are severely impacted - priority fix needed"
            )

        return {
            "overall_improvement": overall_improvement,
            "fairness_quality": fairness_quality,
            "worst_category": worst_category,
            "recommendations": recommendations,
            "detailed_comparison": fairness_comparison,
        }

    def generate_tuning_recommendations(
        self, bucket_analysis: Dict, fairness_analysis: Dict, utilization_analysis: Dict
    ) -> List[str]:
        """Generate specific VTC tuning recommendations"""
        recommendations = []

        # Bucket adaptation recommendations
        if (
            bucket_analysis.get("adaptation_analysis", {}).get("status")
            == "static_poor"
        ):
            recommendations.extend(
                [
                    "🔧 CRITICAL: Enable VTC bucket adaptation",
                    "   - Check VTC configuration parameters",
                    "   - Increase bucket adaptation sensitivity",
                ]
            )

        # Fairness recommendations
        if fairness_analysis.get("overall_improvement", 0) < -10:
            recommendations.extend(
                [
                    "⚖️  CRITICAL: VTC fairness is degraded",
                    "   - Review user tier classification logic",
                    "   - Adjust bucket size calculation method",
                    "   - Consider request priority weighting",
                ]
            )

        # Load balancing recommendations
        for algorithm, util_data in utilization_analysis.items():
            if util_data.get("gini_coefficient", 0) > 0.3:
                recommendations.append(
                    f"📊 {algorithm}: High load imbalance detected (Gini: {util_data['gini_coefficient']:.3f})"
                )

        # VTC-specific recommendations
        bucket_range = bucket_analysis.get("adaptation_analysis", {}).get(
            "total_range", 0
        )
        if bucket_range > 500:
            recommendations.extend(
                [
                    "🔄 VTC bucket sizes varied significantly",
                    "   - Consider more conservative adaptation rates",
                    "   - Implement bucket size bounds",
                ]
            )

        return recommendations

    def create_visualizations(self, bucket_analysis: Dict, output_dir: Path):
        """Create visualizations for the analysis"""
        print("\n📈 Creating visualizations...")

        # Create output directory
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        # 1. VTC Bucket Evolution Plot
        plt.figure(figsize=(12, 6))
        bucket_data = bucket_analysis.get("bucket_data", {})

        for pod, data in bucket_data.items():
            plt.plot(
                data["timestamps"],
                data["values"],
                label=f"Pod {pod.split('-')[-1]}",
                marker="o",
                linewidth=2,
            )

        plt.title(
            "VTC Bucket Size Evolution During Benchmark", fontsize=14, fontweight="bold"
        )
        plt.xlabel("Time")
        plt.ylabel("Bucket Size")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(vis_dir / "vtc_bucket_evolution.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Fairness Comparison Chart
        if self.fairness_data:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Latency comparison
            categories = []
            vtc_latencies = []
            random_latencies = []

            for category in ["small", "medium", "high"]:
                if category in self.fairness_data.get("fairness_comparison", {}):
                    categories.append(category.title())
                    comp_data = self.fairness_data["fairness_comparison"][category]
                    vtc_latencies.append(comp_data["vtc_avg_latency"])
                    random_latencies.append(comp_data["random_avg_latency"])

            x = np.arange(len(categories))
            width = 0.35

            ax1.bar(
                x - width / 2,
                random_latencies,
                width,
                label="Random",
                alpha=0.8,
                color="skyblue",
            )
            ax1.bar(
                x + width / 2,
                vtc_latencies,
                width,
                label="VTC-Basic",
                alpha=0.8,
                color="lightcoral",
            )

            ax1.set_xlabel("User Category")
            ax1.set_ylabel("Average Latency (s)")
            ax1.set_title("Latency by User Category")
            ax1.set_xticks(x)
            ax1.set_xticklabels(categories)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Improvement percentages
            improvements = []
            for category in ["small", "medium", "high"]:
                if category in self.fairness_data.get("fairness_comparison", {}):
                    improvements.append(
                        self.fairness_data["fairness_comparison"][category][
                            "latency_improvement_pct"
                        ]
                    )

            colors = ["green" if x > 0 else "red" for x in improvements]
            ax2.bar(categories, improvements, color=colors, alpha=0.7)
            ax2.set_xlabel("User Category")
            ax2.set_ylabel("Improvement %")
            ax2.set_title("VTC vs Random Improvement")
            ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                vis_dir / "fairness_comparison.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        print(f"📈 Visualizations saved to {vis_dir}")

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        print("\n📋 Generating comprehensive analysis report...")

        # Run all analyses
        bucket_analysis = self.analyze_vtc_bucket_evolution()
        ttft_analysis = self.analyze_ttft_performance()
        utilization_analysis = self.analyze_pod_utilization()
        fairness_analysis = self.generate_fairness_analysis()

        # Generate recommendations
        recommendations = self.generate_tuning_recommendations(
            bucket_analysis, fairness_analysis, utilization_analysis
        )

        # Create output directory
        output_dir = self.results_dir / "enhanced_analysis"
        output_dir.mkdir(exist_ok=True)

        # Create visualizations
        self.create_visualizations(bucket_analysis, output_dir)

        # Compile report
        report = {
            "metadata": {
                "benchmark_timeframe": {
                    "start": self.start_time.isoformat(),
                    "end": self.end_time.isoformat(),
                    "duration_minutes": (
                        self.end_time - self.start_time
                    ).total_seconds()
                    / 60,
                },
                "analysis_timestamp": datetime.now().isoformat(),
            },
            "vtc_bucket_analysis": bucket_analysis,
            "ttft_performance": ttft_analysis,
            "pod_utilization": utilization_analysis,
            "fairness_analysis": fairness_analysis,
            "tuning_recommendations": recommendations,
            "summary": self._generate_executive_summary(
                bucket_analysis, fairness_analysis, recommendations
            ),
        }

        # Save detailed report
        with open(output_dir / "detailed_analysis_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        return report

    def _generate_executive_summary(
        self, bucket_analysis: Dict, fairness_analysis: Dict, recommendations: List[str]
    ) -> Dict[str, str]:
        """Generate executive summary"""

        # VTC Status
        adaptation_status = bucket_analysis.get("adaptation_analysis", {}).get(
            "status", "unknown"
        )
        vtc_status = {
            "static_poor": "🔴 CRITICAL - VTC not adapting",
            "stable_good": "🟢 GOOD - VTC stable adaptation",
            "oscillating_poor": "🟡 WARNING - VTC oscillating",
            "moderate_adaptation": "🟡 MODERATE - VTC adapting moderately",
        }.get(adaptation_status, "❓ UNKNOWN")

        # Fairness Status
        fairness_quality = fairness_analysis.get("fairness_quality", "unknown")
        fairness_status = {
            "excellent": "🟢 EXCELLENT - VTC outperforming random",
            "good": "🟢 GOOD - VTC performing well",
            "acceptable": "🟡 ACCEPTABLE - VTC performing adequately",
            "poor": "🔴 POOR - VTC underperforming",
        }.get(fairness_quality, "❓ UNKNOWN")

        # Overall recommendation
        critical_issues = len([r for r in recommendations if "CRITICAL" in r])
        if critical_issues > 0:
            overall_status = (
                f"🔴 NEEDS IMMEDIATE ATTENTION - {critical_issues} critical issues"
            )
        elif fairness_quality == "poor":
            overall_status = "🟡 NEEDS TUNING - Fairness issues detected"
        else:
            overall_status = "🟢 PERFORMING ADEQUATELY - Minor optimizations possible"

        return {
            "vtc_status": vtc_status,
            "fairness_status": fairness_status,
            "overall_status": overall_status,
            "key_finding": f"VTC bucket adaptation: {adaptation_status}, Fairness: {fairness_quality}",
        }


def main():
    parser = argparse.ArgumentParser(description="Analyze VTC benchmark results")
    parser.add_argument("results_dir", help="Path to benchmark results directory")
    parser.add_argument(
        "--prometheus-url",
        default="http://localhost:9090",
        help="Prometheus URL (default: http://localhost:9090)",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "text", "both"],
        default="both",
        help="Output format",
    )

    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"❌ Results directory not found: {args.results_dir}")
        sys.exit(1)

    # Run analysis
    analyzer = VTCBenchmarkAnalyzer(args.results_dir, args.prometheus_url)
    report = analyzer.generate_report()

    # Display summary
    print("\n" + "=" * 80)
    print("🎯 VTC BENCHMARK ANALYSIS SUMMARY")
    print("=" * 80)

    summary = report["summary"]
    print(f"VTC Status:     {summary['vtc_status']}")
    print(f"Fairness:       {summary['fairness_status']}")
    print(f"Overall:        {summary['overall_status']}")
    print(f"Key Finding:    {summary['key_finding']}")

    print(
        f"\n📋 TUNING RECOMMENDATIONS ({len(report['tuning_recommendations'])} items):"
    )
    for i, rec in enumerate(report["tuning_recommendations"], 1):
        print(f"{i:2d}. {rec}")

    # Bucket evolution summary
    bucket_data = report["vtc_bucket_analysis"].get("bucket_data", {})
    if bucket_data:
        print(f"\n🔄 VTC BUCKET SIZE EVOLUTION:")
        for pod, data in bucket_data.items():
            pod_short = pod.split("-")[-1]
            print(
                f"   Pod {pod_short}: {data['initial']:3.0f} → {data['final']:3.0f} (range: {data['range']:3.0f})"
            )

    # Fairness summary
    fairness_comp = report["fairness_analysis"].get("detailed_comparison", {})
    if fairness_comp:
        print(f"\n⚖️  FAIRNESS COMPARISON (VTC vs Random):")
        for category, metrics in fairness_comp.items():
            improvement = metrics["latency_improvement_pct"]
            symbol = "🟢" if improvement > 0 else "🔴" if improvement < -10 else "🟡"
            print(
                f"   {symbol} {category.title():6s}: {improvement:+6.1f}% latency change"
            )

    output_dir = Path(args.results_dir) / "enhanced_analysis"
    print(f"\n📁 Detailed analysis saved to: {output_dir}")
    print(f"📈 Visualizations available in: {output_dir}/visualizations")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
