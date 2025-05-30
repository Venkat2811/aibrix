import json

with open(
    "/tmp/aibrix-benchmark/run_results/run_20250527_222042/comprehensive_benchmark_stats.json",
    "r",
) as f:
    data = json.load(f)

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
            f"  VTC-Basic: {len(vtc_latencies):2d} reqs, avg {avg_vtc:.2f}s, range {min(vtc_latencies):.2f}-{max(vtc_latencies):.2f}s"
        )
    if random_latencies:
        avg_random = sum(random_latencies) / len(random_latencies)
        print(
            f"  Random:    {len(random_latencies):2d} reqs, avg {avg_random:.2f}s, range {min(random_latencies):.2f}-{max(random_latencies):.2f}s"
        )

    # Calculate the difference
    if vtc_latencies and random_latencies:
        diff = avg_random - avg_vtc
        print(f"  Difference: Random is {diff:.2f}s slower than VTC")
    print()

# Overall analysis
print("\nKey Insights:")
print("=" * 60)

# Check if Pod 1 is inherently faster
pod1_vtc = pod_performance["10.244.1.144:8000"]["vtc-basic"]
pod1_random = pod_performance["10.244.1.144:8000"]["random"]

if pod1_vtc and pod1_random:
    pod1_vtc_avg = sum(pod1_vtc) / len(pod1_vtc)
    pod1_random_avg = sum(pod1_random) / len(pod1_random)
    print(f"Pod 1 (10.244.1.144:8000) Performance:")
    print(f"  VTC (47 reqs): {pod1_vtc_avg:.2f}s average")
    print(f"  Random (20 reqs): {pod1_random_avg:.2f}s average")
    print(
        f"  Pod 1 is {pod1_random_avg - pod1_vtc_avg:.2f}s faster with VTC than Random"
    )
    print()

# Check if other pods are slower
other_pods_random = []
for pod in pod_performance:
    if pod != "10.244.1.144:8000" and pod_performance[pod]["random"]:
        other_pods_random.extend(pod_performance[pod]["random"])

if other_pods_random:
    other_avg = sum(other_pods_random) / len(other_pods_random)
    print(f"Other pods (Random): {other_avg:.2f}s average")
    print(f"Pod 1 vs Others: Pod 1 is {other_avg - pod1_random_avg:.2f}s faster")
