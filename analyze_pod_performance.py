#!/usr/bin/env python3
"""Pod performance analysis script."""

import json
import sys
from pathlib import Path


def analyze_pod_performance(stats_file: str):
    """Analyze pod performance from benchmark stats."""
    try:
        with open(stats_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Stats file not found: {stats_file}")
        sys.exit(1)

    # Group by pod and algorithm
    pod_performance = {}
    for req in data["request_data"]:
        pod = req["target_pod"]
        algo = req["algorithm"]
        latency = req["latency"]

        if pod not in pod_performance:
            pod_performance[pod] = {"vtc-basic": [], "random": []}
        pod_performance[pod][algo].append(latency)

    print("Pod Performance Analysis:")
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

    print("\nKey Insights:")
    print("=" * 60)

    # Find primary pod (most VTC requests)
    primary_pod = max(
        pod_performance.keys(), key=lambda p: len(pod_performance[p]["vtc-basic"])
    )

    pod1_vtc = pod_performance[primary_pod]["vtc-basic"]
    pod1_random = pod_performance[primary_pod]["random"]

    if pod1_vtc and pod1_random:
        pod1_vtc_avg = sum(pod1_vtc) / len(pod1_vtc)
        pod1_random_avg = sum(pod1_random) / len(pod1_random)
        print(f"Primary Pod ({primary_pod}) Performance:")
        print(f"  VTC ({len(pod1_vtc)} reqs): {pod1_vtc_avg:.2f}s average")
        print(f"  Random ({len(pod1_random)} reqs): {pod1_random_avg:.2f}s average")
        print(f"  VTC is {pod1_random_avg - pod1_vtc_avg:.2f}s faster than Random")
        print()

    # Compare with other pods
    other_pods_random = []
    for pod in pod_performance:
        if pod != primary_pod and pod_performance[pod]["random"]:
            other_pods_random.extend(pod_performance[pod]["random"])

    if other_pods_random:
        other_avg = sum(other_pods_random) / len(other_pods_random)
        print(f"Other pods (Random): {other_avg:.2f}s average")
        print(
            f"Primary vs Others: Primary pod is {other_avg - pod1_random_avg:.2f}s faster"
        )


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python analyze_pod_performance.py <comprehensive_benchmark_stats.json>"
        )
        sys.exit(1)

    stats_file = sys.argv[1]
    analyze_pod_performance(stats_file)


if __name__ == "__main__":
    main()
