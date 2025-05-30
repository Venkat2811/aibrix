#!/usr/bin/env python3

import argparse
import json
import logging
import os
import random
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional

import numpy as np
import redis
import requests
from analyze import analyze_fairness_metrics, log_fairness_summary
from analyze_vtc_config import (
    analyze_pod_concentration,
    analyze_vtc_metrics,
    log_vtc_analysis,
)
from constants import TRAFFIC_PATTERNS, USER_CATEGORIES

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Benchmark configuration
GATEWAY_URL = "http://localhost:8888"
PROMETHEUS_URL = "http://localhost:9090"
DEFAULT_ROUTING_ALGORITHMS = ["random", "vtc-basic"]


def setup_redis_users():
    """Setup Redis with user data for VTC routing."""
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_password = os.getenv("REDIS_PASSWORD", "")

    logger.info(f"Setting up Redis users at {redis_host}:{redis_port}")

    try:
        redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=True,
        )

        # Test connection
        if not redis_client.ping():
            logger.error("Redis connection test failed")
            return False

        # Clear existing users
        existing_keys = redis_client.keys("aibrix-users/*")
        if existing_keys:
            for key in existing_keys:
                redis_client.delete(key)
            logger.info(f"Deleted {len(existing_keys)} existing user keys")

        # Create users based on USER_CATEGORIES
        users_created = []
        for category in USER_CATEGORIES:
            for user_name in category["users"]:
                user_key = f"aibrix-users/{user_name}"
                user_data = json.dumps(
                    {
                        "name": user_name,
                        "rpm": 1000000,  # Very high RPM to avoid rate limiting
                        "tpm": 10000000,  # Very high TPM to avoid rate limiting
                        "category": category["name"],
                    }
                )
                redis_client.set(user_key, user_data)
                users_created.append(user_name)
                logger.info(f"Created user {user_name} (category: {category['name']})")

        logger.info(f"Successfully created {len(users_created)} users in Redis")
        return True

    except redis.ConnectionError as e:
        logger.error(f"Failed to connect to Redis: {e}")
        return False
    except Exception as e:
        logger.error(f"Error creating users in Redis: {e}")
        return False


def make_streaming_req():
    """Make a streaming request to the VTC system."""
    pass


def make_non_streaming_req(
    user: str, prompt: str, routing_algorithm: str, output_tokens: int = 20
) -> Dict:
    """
    Make a non-streaming request to the VTC system.

    Args:
        user: User name for the request
        prompt: The prompt to send
        routing_algorithm: Routing algorithm to use
        output_tokens: Number of output tokens to request

    Returns:
        Dict with request results and metrics
    """
    headers = {
        "Content-Type": "application/json",
        "user": user,
        "routing-strategy": routing_algorithm,
        "model": "tinyllama-1-1b-chat-v1-0",
    }

    payload = {
        "model": "tinyllama-1-1b-chat-v1-0",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_tokens,
        "stream": False,
    }

    start_time = time.time()

    try:
        response = requests.post(
            f"{GATEWAY_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )

        end_time = time.time()
        latency = end_time - start_time

        # Extract pod information from response headers
        target_pod = response.headers.get("target-pod", "unknown")

        if response.status_code == 200:
            response_data = response.json()
            return {
                "success": True,
                "latency": latency,
                "status_code": response.status_code,
                "target_pod": target_pod,
                "response_data": response_data,
                "prompt_tokens": len(prompt.split()),  # Simple approximation
                "completion_tokens": response_data.get("usage", {}).get(
                    "completion_tokens", 0
                ),
                "total_tokens": response_data.get("usage", {}).get("total_tokens", 0),
            }
        else:
            return {
                "success": False,
                "latency": latency,
                "status_code": response.status_code,
                "target_pod": target_pod,
                "error": response.text,
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 0,
                "total_tokens": 0,
            }

    except Exception as e:
        end_time = time.time()
        return {
            "success": False,
            "latency": end_time - start_time,
            "status_code": 0,
            "target_pod": "unknown",
            "error": str(e),
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": 0,
            "total_tokens": 0,
        }


def query_prometheus(query: str, timestamp: Optional[float] = None) -> Dict:
    """
    Query Prometheus for metrics.

    Args:
        query: PromQL query string
        timestamp: Optional timestamp for historical data (default: current time)

    Returns:
        Dict with query results
    """
    try:
        params = {"query": query}
        if timestamp:
            params["time"] = timestamp

        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query", params=params, timeout=10
        )

        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(
                f"Prometheus query failed: {response.status_code} - {response.text}"
            )
            return {"status": "error", "data": {}}

    except Exception as e:
        logger.warning(f"Failed to query Prometheus: {e}")
        return {"status": "error", "data": {}}


def collect_prometheus_metrics(timestamp: Optional[float] = None) -> Dict:
    """
    Collect system metrics from Prometheus.

    Args:
        timestamp: Optional timestamp for historical data

    Returns:
        Dict containing various system metrics
    """
    metrics = {}

    # Pod-level metrics
    pod_queries = {
        "pod_cpu_usage": 'rate(container_cpu_usage_seconds_total{pod=~"tinyllama.*"}[1m])',
        "pod_memory_usage": 'container_memory_working_set_bytes{pod=~"tinyllama.*"}',
        "pod_request_count": "increase(vllm:request_success_total[1m])",
        "pod_request_latency_sum": "vllm:e2e_request_latency_seconds_sum",
        "pod_request_latency_count": "vllm:e2e_request_latency_seconds_count",
        "pod_prompt_tokens_rate": "rate(vllm:prompt_tokens_total[1m])",
        "pod_generation_tokens_rate": "rate(vllm:generation_tokens_total[1m])",
        "pod_requests_running": "vllm:num_requests_running",
        "pod_requests_waiting": "vllm:num_requests_waiting",
        "pod_ttft_latency_sum": "vllm:time_to_first_token_seconds_sum",
        "pod_ttft_latency_count": "vllm:time_to_first_token_seconds_count",
        "pod_time_per_token_sum": "vllm:time_per_output_token_seconds_sum",
        "pod_time_per_token_count": "vllm:time_per_output_token_seconds_count",
    }

    # Gateway metrics
    gateway_queries = {
        "gateway_requests_total": "increase(envoy_http_downstream_rq_total[1m])",
        "gateway_response_time_sum": "envoy_http_downstream_rq_time_sum",
        "gateway_response_time_count": "envoy_http_downstream_rq_time_count",
        "gateway_upstream_requests": "increase(envoy_cluster_upstream_rq_total[1m])",
        "gateway_downstream_connections": "envoy_http_downstream_cx_active",
        "gateway_upstream_response_time": "envoy_cluster_upstream_rq_time_sum",
    }

    # VTC-specific routing metrics
    vtc_queries = {
        "vtc_bucket_size_active": "vtc_bucket_size_active",
        "vtc_bucket_size_changes": "abs(deriv(vtc_bucket_size_active[1m]))",
    }

    # Collect pod metrics
    metrics["pods"] = {}
    for metric_name, query in pod_queries.items():
        result = query_prometheus(query, timestamp)
        if result.get("status") == "success":
            metrics["pods"][metric_name] = result.get("data", {}).get("result", [])

    # Collect gateway metrics
    metrics["gateway"] = {}
    for metric_name, query in gateway_queries.items():
        result = query_prometheus(query, timestamp)
        if result.get("status") == "success":
            metrics["gateway"][metric_name] = result.get("data", {}).get("result", [])

    # Collect VTC routing metrics
    metrics["vtc"] = {}
    for metric_name, query in vtc_queries.items():
        result = query_prometheus(query, timestamp)
        if result.get("status") == "success":
            metrics["vtc"][metric_name] = result.get("data", {}).get("result", [])

    # Add timestamp
    metrics["timestamp"] = timestamp or time.time()

    return metrics


def save_system_metrics(result_dir: str, metrics_type: str, metrics: Dict):
    """Save system metrics to file."""
    filename = f"{result_dir}/system_metrics_{metrics_type}.json"
    with open(filename, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved {metrics_type} system metrics to {filename}")


def prepare_requests(max_req_count, user_distribution):
    """
    Prepare requests based on traffic pattern and user distribution.

    Args:
        max_req_count: Maximum number of requests to prepare
        user_distribution: Traffic pattern name ('balanced', 'high_usage', 'bursty')

    Returns:
        List of prepared requests
    """
    if user_distribution not in TRAFFIC_PATTERNS:
        raise ValueError(f"Unknown user distribution: {user_distribution}")

    pattern = TRAFFIC_PATTERNS[user_distribution]
    requests = []

    # Generate requests based on traffic pattern
    for i in range(max_req_count):
        # Select category based on probabilities
        rand = random.random()
        cumulative_prob = 0
        selected_category = None

        for category_config in pattern:
            cumulative_prob += category_config["prob"]
            if rand <= cumulative_prob:
                selected_category = category_config["category"]
                break

        # Find matching user category from USER_CATEGORIES
        user_category_info = next(
            cat for cat in USER_CATEGORIES if cat["name"] == selected_category
        )

        # Select a random user from this category
        user = random.choice(user_category_info["users"])

        # Create request
        request = {
            "request_id": i,
            "user": user,
            "category": selected_category,
            "burstiness": category_config["burstiness"],
            "token_scale": user_category_info["token_scale"],
            "min_tokens": user_category_info["min_tokens"],
            "max_tokens": user_category_info["max_tokens"],
        }
        requests.append(request)

    return requests


def print_traffic_stats(requests, user_distribution):
    """Print statistics for the generated traffic pattern."""
    print(f"\n=== Traffic Pattern Stats: {user_distribution.upper()} ===")
    print(f"Total requests: {len(requests)}")

    # Category distribution
    category_counts = {}
    user_counts = {}
    burstiness_stats = {}

    for req in requests:
        category = req["category"]
        user = req["user"]
        burstiness = req["burstiness"]

        category_counts[category] = category_counts.get(category, 0) + 1
        user_counts[user] = user_counts.get(user, 0) + 1

        if category not in burstiness_stats:
            burstiness_stats[category] = []
        burstiness_stats[category].append(burstiness)

    # Print category distribution
    print("\nCategory distribution:")
    for category, count in category_counts.items():
        percentage = (count / len(requests)) * 100
        avg_burstiness = sum(burstiness_stats[category]) / len(
            burstiness_stats[category]
        )
        print(
            f"  {category}: {count} requests ({percentage:.1f}%) - Avg burstiness: {avg_burstiness:.1f}"
        )

    # Print user distribution
    print("\nUser distribution:")
    for user, count in sorted(user_counts.items()):
        percentage = (count / len(requests)) * 100
        print(f"  {user}: {count} requests ({percentage:.1f}%)")

    print(f"\nTotal unique users: {len(user_counts)}")
    print("=" * 50)


def create_result_dir() -> str:
    """Create a new results directory in tmp workspace."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"/tmp/aibrix-benchmark/run_results/run_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    logger.info(f"Created result directory: {result_dir}")
    return result_dir


def save_request_result(
    result_dir: str,
    request_id: int,
    user: str,
    routing_algorithm: str,
    traffic_pattern: str,
    result: Dict,
    category: str = None,
    token_range: Dict = None,
):
    """Save individual request result to file."""
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id,
        "user": user,
        "routing_algorithm": routing_algorithm,
        "traffic_pattern": traffic_pattern,
        "category": category,
        "token_range": token_range,
        **result,
    }

    filename = f"{result_dir}/request_{request_id:04d}_{routing_algorithm}_{user}.json"
    with open(filename, "w") as f:
        json.dump(result_data, f, indent=2)


def save_comprehensive_stats(
    result_dir: str, all_requests: List[Dict], system_metrics: Dict
):
    """Save comprehensive benchmark statistics in JSON format similar to advanced benchmark."""
    stats_data = {
        "benchmark_metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_requests": len(all_requests),
            "algorithms_tested": list(
                set(req.get("algorithm") for req in all_requests)
            ),
            "traffic_pattern": (
                all_requests[0].get("traffic_pattern") if all_requests else None
            ),
            "output_tokens_limit": 20,
        },
        "request_data": all_requests,
        "system_metrics": system_metrics,
        "algorithm_summary": {},
        "pod_distribution": {},
        "latency_analysis": {},
        "token_analysis": {},
        "routing_effectiveness": {},
    }

    # Analyze by algorithm
    algorithms = set(req.get("algorithm") for req in all_requests)
    for algorithm in algorithms:
        algo_requests = [
            req for req in all_requests if req.get("algorithm") == algorithm
        ]
        if not algo_requests:
            continue

        # Basic stats
        successful_requests = [
            req for req in algo_requests if req.get("success", False)
        ]
        failed_requests = [
            req for req in algo_requests if not req.get("success", False)
        ]

        # Pod distribution
        pod_counts = {}
        for req in successful_requests:
            pod = req.get("target_pod", "unknown")
            pod_counts[pod] = pod_counts.get(pod, 0) + 1

        # Latency analysis
        latencies = [req.get("latency", 0) for req in successful_requests]

        # Category analysis
        category_stats = {}
        for req in successful_requests:
            category = req.get("category", "unknown")
            if category not in category_stats:
                category_stats[category] = {"count": 0, "latencies": [], "pods": []}
            category_stats[category]["count"] += 1
            category_stats[category]["latencies"].append(req.get("latency", 0))
            category_stats[category]["pods"].append(req.get("target_pod", "unknown"))

        # Calculate category averages
        for category, data in category_stats.items():
            data["avg_latency"] = (
                sum(data["latencies"]) / len(data["latencies"])
                if data["latencies"]
                else 0
            )
            data["min_latency"] = min(data["latencies"]) if data["latencies"] else 0
            data["max_latency"] = max(data["latencies"]) if data["latencies"] else 0
            data["pod_distribution"] = {
                pod: data["pods"].count(pod) for pod in set(data["pods"])
            }

        stats_data["algorithm_summary"][algorithm] = {
            "total_requests": len(algo_requests),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": (
                len(successful_requests) / len(algo_requests) * 100
                if algo_requests
                else 0
            ),
            "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
            "min_latency": min(latencies) if latencies else 0,
            "max_latency": max(latencies) if latencies else 0,
            "latency_std": np.std(latencies) if latencies else 0,
            "category_breakdown": category_stats,
        }

        stats_data["pod_distribution"][algorithm] = pod_counts
        stats_data["latency_analysis"][algorithm] = {
            "all_latencies": latencies,
            "percentiles": {
                "p50": np.percentile(latencies, 50) if latencies else 0,
                "p90": np.percentile(latencies, 90) if latencies else 0,
                "p95": np.percentile(latencies, 95) if latencies else 0,
                "p99": np.percentile(latencies, 99) if latencies else 0,
            },
        }

        # Token analysis
        token_data = []
        for req in successful_requests:
            token_data.append(
                {
                    "prompt_tokens": req.get("prompt_tokens", 0),
                    "completion_tokens": req.get("completion_tokens", 0),
                    "total_tokens": req.get("total_tokens", 0),
                    "latency": req.get("latency", 0),
                    "category": req.get("category", "unknown"),
                }
            )
        stats_data["token_analysis"][algorithm] = token_data

        # Routing effectiveness (clustering analysis)
        pod_user_mapping = {}
        for req in successful_requests:
            user = req.get("user", "unknown")
            pod = req.get("target_pod", "unknown")
            if user not in pod_user_mapping:
                pod_user_mapping[user] = []
            pod_user_mapping[user].append(pod)

        # Calculate clustering effectiveness
        clustering_stats = {}
        for user, pods in pod_user_mapping.items():
            unique_pods = set(pods)
            clustering_stats[user] = {
                "total_requests": len(pods),
                "unique_pods_used": len(unique_pods),
                "clustering_efficiency": (
                    len(pods) / len(unique_pods) if unique_pods else 0
                ),
                "primary_pod": max(set(pods), key=pods.count) if pods else None,
                "primary_pod_percentage": (
                    pods.count(max(set(pods), key=pods.count)) / len(pods) * 100
                    if pods
                    else 0
                ),
            }

        stats_data["routing_effectiveness"][algorithm] = clustering_stats

    # Save to file
    stats_file = f"{result_dir}/comprehensive_benchmark_stats.json"
    with open(stats_file, "w") as f:
        json.dump(
            stats_data, f, indent=2, default=str
        )  # default=str to handle numpy types

    logger.info(f"Comprehensive stats saved to: {stats_file}")
    return stats_file


def validate_success_rates(
    results_by_algorithm: Dict[str, List[Dict]], strict_mode: bool = False
) -> Dict:
    """
    Validate success rates for each algorithm.

    Args:
        results_by_algorithm: Dictionary mapping algorithm names to list of results
        strict_mode: If True, any failure invalidates comparison (default: 2% threshold)

    Returns:
        Dictionary with success rate statistics for each algorithm
    """
    validation_results = {}

    for algorithm, requests in results_by_algorithm.items():
        total_requests = len(requests)
        successful_requests = sum(1 for req in requests if req.get("success", False))
        failed_requests = total_requests - successful_requests

        success_rate = (
            (successful_requests / total_requests * 100) if total_requests > 0 else 0
        )
        failure_rate = (
            (failed_requests / total_requests * 100) if total_requests > 0 else 0
        )

        # Collect failed request details
        failed_details = []
        for req in requests:
            if not req.get("success", False):
                error_msg = req.get("error", "Unknown error")
                status_code = req.get("status_code", "Unknown")

                # Summarize common errors
                if "Connection refused" in error_msg:
                    error_summary = f"Connection refused (status: {status_code})"
                elif "timeout" in error_msg.lower():
                    error_summary = f"Timeout (status: {status_code})"
                elif status_code and status_code != 200:
                    error_summary = f"HTTP {status_code}"
                else:
                    error_summary = (
                        error_msg[:50] + "..." if len(error_msg) > 50 else error_msg
                    )

                failed_details.append(
                    {
                        "request_id": req.get("request_id", "Unknown"),
                        "user": req.get("user", "Unknown"),
                        "status_code": status_code,
                        "error_summary": error_summary,
                    }
                )

        validation_results[algorithm] = {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": success_rate,
            "failure_rate": failure_rate,
            "failed_details": failed_details,
        }

    return validation_results


def run_benchmark(
    traffic_pattern: str = "balanced",
    max_requests: int = 5,
    target_qps: float = 0,
    algorithms: List[str] = None,
    strict_mode: bool = False,
):
    """
    Run benchmark for specified traffic pattern and save results.

    Args:
        traffic_pattern: Traffic pattern to use
        max_requests: Maximum number of requests to send
        target_qps: Target queries per second (0 = no rate limit)
        algorithms: List of algorithms to test (default: both random and vtc-basic)
        strict_mode: If True, any failure invalidates comparison (default: 2% threshold)
    """
    if algorithms is None:
        algorithms = DEFAULT_ROUTING_ALGORITHMS

    logger.info(
        f"Starting benchmark - Pattern: {traffic_pattern}, Max requests: {max_requests}, "
        f"QPS: {target_qps if target_qps > 0 else 'unlimited'}, Algorithms: {algorithms}"
    )

    # Create result directory
    result_dir = create_result_dir()

    # Setup Redis users
    if not setup_redis_users():
        logger.error("Failed to setup Redis users, aborting benchmark")
        return

    # Generate requests
    requests_data = prepare_requests(max_requests, traffic_pattern)
    logger.info(f"Generated {len(requests_data)} requests")

    # Log request distribution
    user_counts = {}
    category_counts = {}
    for req in requests_data:
        user = req["user"]
        category = req["category"]
        user_counts[user] = user_counts.get(user, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1

    logger.info("Request distribution by category:")
    for category, count in sorted(category_counts.items()):
        percentage = (count / len(requests_data)) * 100
        logger.info(f"  {category}: {count} requests ({percentage:.1f}%)")

    logger.info(f"Total unique users: {len(user_counts)}")
    logger.info(
        f"Requests per user: min={min(user_counts.values())}, max={max(user_counts.values())}, avg={sum(user_counts.values())/len(user_counts):.1f}"
    )

    # Collect baseline system metrics
    logger.info("Collecting baseline system metrics...")
    baseline_metrics = collect_prometheus_metrics()
    save_system_metrics(result_dir, "baseline", baseline_metrics)

    # Analyze baseline VTC configuration
    baseline_vtc_analysis = analyze_vtc_metrics(baseline_metrics)
    log_vtc_analysis(baseline_vtc_analysis, "Baseline")

    # Track results for analysis
    results_by_algorithm = {}
    all_requests = []  # For comprehensive stats
    all_system_metrics = {
        "baseline": baseline_metrics,
    }

    # Run benchmark for each routing algorithm
    for routing_algorithm in algorithms:
        logger.info(f"Testing routing algorithm: {routing_algorithm}")

        # Collect pre-algorithm metrics
        pre_algo_metrics = collect_prometheus_metrics()
        save_system_metrics(result_dir, f"pre_{routing_algorithm}", pre_algo_metrics)
        all_system_metrics[f"pre_{routing_algorithm}"] = pre_algo_metrics

        # Analyze VTC metrics before algorithm
        pre_vtc_analysis = analyze_vtc_metrics(pre_algo_metrics)
        log_vtc_analysis(pre_vtc_analysis, f"Pre-{routing_algorithm}")

        # Track requests for this algorithm
        algorithm_requests = []

        # Calculate request timing based on QPS
        if target_qps > 0:
            # Generate requests at specified QPS
            interval = 1.0 / target_qps
            logger.info(
                f"Generating requests at {target_qps} QPS (interval: {interval:.3f}s)"
            )

            # Use threading to maintain QPS rate
            request_queue = Queue()
            results_lock = threading.Lock()

            def worker():
                while True:
                    item = request_queue.get()
                    if item is None:
                        break
                    i, req_data, start_time = item

                    user = req_data["user"]
                    prompt = f"Hello, I am {user}. Please tell me a short story."

                    # Make request
                    result = make_non_streaming_req(
                        user, prompt, routing_algorithm, output_tokens=20
                    )

                    # Find user category and token range
                    user_category = next(
                        (cat for cat in USER_CATEGORIES if user in cat["users"]), None
                    )
                    category_name = (
                        user_category["name"] if user_category else "unknown"
                    )
                    token_range = (
                        {
                            "min_tokens": (
                                user_category["min_tokens"] if user_category else 0
                            ),
                            "max_tokens": (
                                user_category["max_tokens"] if user_category else 0
                            ),
                            "token_scale": (
                                user_category["token_scale"] if user_category else 1
                            ),
                        }
                        if user_category
                        else None
                    )

                    # Save result
                    save_request_result(
                        result_dir,
                        i,
                        user,
                        routing_algorithm,
                        traffic_pattern,
                        result,
                        category_name,
                        token_range,
                    )

                    # Track for analysis
                    result["user"] = user
                    result["algorithm"] = routing_algorithm
                    result["scheduled_time"] = start_time
                    result["category"] = category_name
                    result["token_range"] = token_range
                    result["traffic_pattern"] = traffic_pattern

                    with results_lock:
                        algorithm_requests.append(result)
                        all_requests.append(result.copy())

                    # Log result
                    if result["success"]:
                        logger.info(
                            f"Request {i+1} successful - Latency: {result['latency']:.2f}s, Pod: {result['target_pod']}"
                        )
                    else:
                        status_code = result.get("status_code", "Unknown")
                        error_msg = result.get("error", "Unknown")
                        logger.error(
                            f"Request {i+1} failed - Status: {status_code}, Error: {error_msg[:100]}{'...' if len(error_msg) > 100 else ''}, Pod: {result['target_pod']}"
                        )

                    request_queue.task_done()

            # Start worker threads
            num_workers = min(10, max(1, int(target_qps)))  # Scale workers with QPS
            threads = []
            for _ in range(num_workers):
                t = threading.Thread(target=worker)
                t.start()
                threads.append(t)

            # Schedule requests
            start_time = time.time()
            for i, req_data in enumerate(requests_data):
                scheduled_time = start_time + (i * interval)
                current_time = time.time()

                # Wait until scheduled time
                if scheduled_time > current_time:
                    time.sleep(scheduled_time - current_time)

                logger.info(
                    f"Scheduling request {i+1}/{len(requests_data)} - User: {req_data['user']}, Algorithm: {routing_algorithm}"
                )
                request_queue.put((i, req_data, scheduled_time))

            # Wait for all requests to complete
            request_queue.join()

            # Stop workers
            for _ in threads:
                request_queue.put(None)
            for t in threads:
                t.join()

        else:
            # Sequential execution (no QPS limit)
            for i, req_data in enumerate(requests_data):
                user = req_data["user"]
                prompt = f"Hello, I am {user}. Please tell me a short story."  # Simple prompt

                logger.info(
                    f"Sending request {i+1}/{len(requests_data)} - User: {user}, Algorithm: {routing_algorithm}"
                )

                # Make request
                result = make_non_streaming_req(
                    user, prompt, routing_algorithm, output_tokens=20
                )

                # Find user category and token range
                user_category = next(
                    (cat for cat in USER_CATEGORIES if user in cat["users"]), None
                )
                category_name = user_category["name"] if user_category else "unknown"
                token_range = (
                    {
                        "min_tokens": (
                            user_category["min_tokens"] if user_category else 0
                        ),
                        "max_tokens": (
                            user_category["max_tokens"] if user_category else 0
                        ),
                        "token_scale": (
                            user_category["token_scale"] if user_category else 1
                        ),
                    }
                    if user_category
                    else None
                )

                # Save result
                save_request_result(
                    result_dir,
                    i,
                    user,
                    routing_algorithm,
                    traffic_pattern,
                    result,
                    category_name,
                    token_range,
                )

                # Track for analysis
                result["user"] = user
                result["algorithm"] = routing_algorithm
                result["category"] = category_name
                result["token_range"] = token_range
                result["traffic_pattern"] = traffic_pattern
                algorithm_requests.append(result)
                all_requests.append(result.copy())

                # Log result
                if result["success"]:
                    logger.info(
                        f"Request {i+1} successful - Latency: {result['latency']:.2f}s, Pod: {result['target_pod']}"
                    )
                else:
                    status_code = result.get("status_code", "Unknown")
                    error_msg = result.get("error", "Unknown")
                    logger.error(
                        f"Request {i+1} failed - Status: {status_code}, Error: {error_msg[:100]}{'...' if len(error_msg) > 100 else ''}, Pod: {result['target_pod']}"
                    )

                # Small delay between requests
                time.sleep(0.5)

        # Store requests for this algorithm
        results_by_algorithm[routing_algorithm] = algorithm_requests

        # Collect post-algorithm metrics
        post_algo_metrics = collect_prometheus_metrics()
        save_system_metrics(result_dir, f"post_{routing_algorithm}", post_algo_metrics)
        all_system_metrics[f"post_{routing_algorithm}"] = post_algo_metrics

        # Analyze VTC metrics after algorithm
        post_vtc_analysis = analyze_vtc_metrics(post_algo_metrics)
        log_vtc_analysis(post_vtc_analysis, f"Post-{routing_algorithm}")

        # Wait between algorithms to let metrics settle
        logger.info("Waiting 30 seconds before next algorithm...")
        time.sleep(30)

    # Collect final system metrics
    logger.info("Collecting final system metrics...")
    final_metrics = collect_prometheus_metrics()
    save_system_metrics(result_dir, "final", final_metrics)
    all_system_metrics["final"] = final_metrics

    # Final VTC analysis
    final_vtc_analysis = analyze_vtc_metrics(final_metrics)
    log_vtc_analysis(final_vtc_analysis, "Final")

    # Validate success rates before analysis
    success_rate_validation = validate_success_rates(results_by_algorithm, strict_mode)

    # Log success rate summary
    logger.info("=" * 60)
    logger.info(
        f"SUCCESS RATE VALIDATION {'(STRICT MODE)' if strict_mode else '(RELAXED MODE)'}"
    )
    logger.info("=" * 60)

    valid_for_comparison = True
    threshold = 0.0 if strict_mode else 2.0

    for algorithm, stats in success_rate_validation.items():
        total = stats["total_requests"]
        successful = stats["successful_requests"]
        failed = stats["failed_requests"]
        success_rate = stats["success_rate"]
        failure_rate = stats["failure_rate"]

        logger.info(f"{algorithm.upper()} Algorithm:")
        logger.info(f"  Total requests: {total}")
        logger.info(f"  Successful: {successful} ({success_rate:.1f}%)")
        logger.info(f"  Failed: {failed} ({failure_rate:.1f}%)")

        if failure_rate > threshold:
            if strict_mode:
                logger.error(
                    f"  ❌ STRICT MODE: ANY FAILURE INVALIDATES COMPARISON ({failure_rate:.1f}% > 0.0%)"
                )
            else:
                logger.error(
                    f"  ❌ FAILURE RATE TOO HIGH: {failure_rate:.1f}% > {threshold}%"
                )
            valid_for_comparison = False
        else:
            if strict_mode:
                logger.info(
                    f"  ✅ Strict mode: No failures detected ({failure_rate:.1f}% = 0.0%)"
                )
            else:
                logger.info(
                    f"  ✅ Success rate acceptable: {failure_rate:.1f}% ≤ {threshold}%"
                )

        # Log failed request details
        if failed > 0:
            logger.warning(f"  Failed request details:")
            for req in stats["failed_details"]:
                logger.warning(
                    f"    Request {req['request_id']}: {req['error_summary']}"
                )

    logger.info("=" * 60)

    if not valid_for_comparison:
        logger.error("❌ BENCHMARK INVALID: High failure rates detected!")
        logger.error("   Cannot perform reliable fairness comparison.")
        logger.error("   Please check system connectivity and retry.")

        # Still save the analysis but mark it as invalid
        fairness_analysis = analyze_fairness_metrics(results_by_algorithm)
        fairness_analysis["validation"] = {
            "valid_for_comparison": False,
            "reason": "High failure rates detected",
            "success_rates": success_rate_validation,
        }
    else:
        logger.info(
            "✅ SUCCESS RATE VALIDATION PASSED - Proceeding with fairness analysis"
        )

        # Comprehensive fairness analysis
        fairness_analysis = analyze_fairness_metrics(results_by_algorithm)
        fairness_analysis["validation"] = {
            "valid_for_comparison": True,
            "success_rates": success_rate_validation,
        }

    # Save fairness analysis to file
    fairness_file = f"{result_dir}/fairness_analysis.json"
    with open(fairness_file, "w") as f:
        json.dump(fairness_analysis, f, indent=2)
    logger.info(f"Fairness analysis saved to: {fairness_file}")

    # Check for potential routing issues (pod concentration)
    logger.info("Checking for potential routing concentration issues...")
    for algorithm, requests in results_by_algorithm.items():
        successful_requests = [req for req in requests if req.get("success", False)]
        if not successful_requests:
            continue

        pod_counts = {}
        for req in successful_requests:
            pod = req.get("target_pod", "unknown")
            pod_counts[pod] = pod_counts.get(pod, 0) + 1

        total_requests = len(successful_requests)
        if total_requests > 0:
            # Check if one pod received more than 80% of requests
            max_pod_requests = max(pod_counts.values())
            max_pod_percentage = (max_pod_requests / total_requests) * 100

            if max_pod_percentage > 80:
                logger.warning(f"⚠️  ROUTING CONCENTRATION WARNING for {algorithm}:")
                logger.warning(
                    f"   One pod received {max_pod_requests}/{total_requests} requests ({max_pod_percentage:.1f}%)"
                )
                logger.warning(
                    f"   This may indicate routing is not distributing properly"
                )
                logger.warning(f"   Pod distribution: {pod_counts}")
            else:
                logger.info(
                    f"✅ Pod distribution looks healthy for {algorithm}: {pod_counts}"
                )

    # Save comprehensive benchmark statistics
    logger.info("Generating comprehensive benchmark statistics...")
    comprehensive_stats_file = save_comprehensive_stats(
        result_dir, all_requests, all_system_metrics
    )

    # Log fairness summary only if validation passed
    if len(algorithms) > 1:
        if valid_for_comparison:
            log_fairness_summary(fairness_analysis)
        else:
            logger.error("Skipping fairness comparison due to high failure rates")
    else:
        logger.info(f"Single algorithm test completed: {algorithms[0]}")
        logger.info("Run with multiple algorithms to see fairness comparison")
    logger.info(f"Benchmark completed. Results saved to: {result_dir}")
    logger.info(
        "System metrics collected at: baseline, pre/post each algorithm, and final"
    )
    return result_dir


def main():
    parser = argparse.ArgumentParser(
        description="VTC Benchmarking and Stats Collection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--requests", type=int, default=5, help="Maximum number of requests to send"
    )

    parser.add_argument(
        "--pattern",
        choices=list(TRAFFIC_PATTERNS.keys()),
        default="balanced",
        help="Traffic pattern to use",
    )

    parser.add_argument(
        "--qps",
        type=float,
        default=0,
        help="Target queries per second (0 = sequential execution)",
    )

    parser.add_argument(
        "--stream", action="store_true", help="Use streaming mode (not implemented yet)"
    )

    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=DEFAULT_ROUTING_ALGORITHMS,
        default=None,
        help="Routing algorithms to test (default: all algorithms)",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: any request failure invalidates comparison (default: 2%% threshold)",
    )

    args = parser.parse_args()

    if args.stream:
        logger.error("Streaming mode not implemented yet")
        return

    # Determine which algorithms to run
    algorithms_to_test = (
        args.algorithms if args.algorithms else DEFAULT_ROUTING_ALGORITHMS
    )

    logger.info("=" * 60)
    logger.info("VTC Benchmarking Configuration")
    logger.info("=" * 60)
    logger.info(f"Requests: {args.requests}")
    logger.info(f"Pattern: {args.pattern}")
    logger.info(f"QPS: {args.qps if args.qps > 0 else 'Sequential'}")
    logger.info(f"Algorithms: {', '.join(algorithms_to_test)}")
    logger.info(
        f"Validation: {'Strict (0% failure tolerance)' if args.strict else 'Relaxed (2% failure tolerance)'}"
    )
    logger.info("=" * 60)

    # Run benchmark
    result_dir = run_benchmark(
        traffic_pattern=args.pattern,
        max_requests=args.requests,
        target_qps=args.qps,
        algorithms=algorithms_to_test,
        strict_mode=args.strict,
    )

    if result_dir:
        print(f"\nBenchmark completed successfully!")
        print(f"Results saved to: {result_dir}")
        print("\nTo analyze results, you can:")
        print(f"  python run_analysis.py {result_dir}")
        print(f"  python run_analysis.py {result_dir} --fairness-only")
        print(f"  python run_analysis.py {result_dir} --vtc-only")
    else:
        print("Benchmark failed!")


if __name__ == "__main__":
    main()
