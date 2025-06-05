#!/usr/bin/env python3

import argparse
import json
import logging
import os
import queue
import random
import threading
import time
from datetime import datetime
from pathlib import Path
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

GATEWAY_URL = "http://localhost:8888"
PROMETHEUS_URL = "http://localhost:9090"
DEFAULT_ROUTING_ALGORITHMS = ["random", "vtc-basic"]
VTC_DATASET_PATH = (
    "temp_workspace/dataset/vtc_routing_dataset_varied_cpu_optimized.jsonl"
)


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
                        "rpm": 1000000,
                        "tpm": 10000000,
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


def run_system_warmup(streaming_mode=False):
    """Run system warm-up with short requests using VTC routing."""
    logger.info(f"Running system warm-up (streaming: {streaming_mode})")

    warmup_requests = 10
    warmup_users = [user for category in USER_CATEGORIES for user in category["users"]]
    warmup_prompts = [
        "Hello",
        "Hi there",
        "Test message",
        "Quick check",
        "System ready?",
    ]

    successful_warmup = 0

    for i in range(warmup_requests):
        user = random.choice(warmup_users)
        prompt = random.choice(warmup_prompts)

        logger.info(f"Warm-up {i+1}/{warmup_requests} - User: {user}")

        if streaming_mode:
            result = make_streaming_req(
                user=user, prompt=prompt, routing_algorithm="vtc-basic", output_tokens=5
            )
        else:
            result = make_non_streaming_req(
                user=user, prompt=prompt, routing_algorithm="vtc-basic", output_tokens=5
            )

        if result["success"]:
            successful_warmup += 1
            logger.info(
                f"  Success - Latency: {result['latency']:.2f}s, Pod: {result['target_pod']}"
            )
        else:
            logger.warning(f"  Failed - Error: {result.get('error', 'Unknown')[:50]}")

        time.sleep(0.5)

    warmup_success_rate = (successful_warmup / warmup_requests) * 100

    logger.info(
        f"Warm-up complete: {successful_warmup}/{warmup_requests} successful ({warmup_success_rate:.1f}%)"
    )

    if warmup_success_rate >= 80:
        logger.info("System warm-up completed successfully")
    else:
        logger.warning("Warm-up had high failure rate - system may not be ready")

    logger.info("Waiting 5 seconds for system to stabilize")
    time.sleep(5)
    return warmup_success_rate >= 80


def make_non_streaming_req(
    user: str, prompt: str, routing_algorithm: str, output_tokens: int = 20
) -> Dict:
    """Make a non-streaming request to the VTC system."""
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
        target_pod = response.headers.get("target-pod", "unknown")

        if response.status_code == 200:
            response_data = response.json()
            return {
                "success": True,
                "latency": latency,
                "status_code": response.status_code,
                "target_pod": target_pod,
                "response_data": response_data,
                "prompt_tokens": len(prompt.split()),
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
        latency = end_time - start_time
        return {
            "success": False,
            "latency": latency,
            "status_code": None,
            "target_pod": "unknown",
            "error": str(e),
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": 0,
            "total_tokens": 0,
        }


def make_streaming_req(
    user: str, prompt: str, routing_algorithm: str, output_tokens: int = 20
) -> Dict:
    """Make a streaming request to the VTC system with TTFT and TPOT metrics."""
    headers = {
        "Content-Type": "application/json",
        "user": user,
        "routing-strategy": routing_algorithm,
        "model": "tinyllama-1-1b-chat-v1-0",
        "Accept": "text/event-stream",
    }

    payload = {
        "model": "tinyllama-1-1b-chat-v1-0",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    start_time = time.time()
    ttft = None  # Time to First Token
    tpot_values = []  # Time Per Output Token
    last_token_time = None
    completion_text = ""
    completion_tokens = 0
    target_pod = "unknown"

    try:
        response = requests.post(
            f"{GATEWAY_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=120,
        )

        target_pod = response.headers.get("target-pod", "unknown")

        if response.status_code != 200:
            end_time = time.time()
            latency = end_time - start_time
            return {
                "success": False,
                "latency": latency,
                "status_code": response.status_code,
                "target_pod": target_pod,
                "error": response.text,
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 0,
                "total_tokens": 0,
                "ttft": None,
                "tpot_avg": None,
                "tpot_values": [],
            }

        # Process streaming response
        for line in response.iter_lines():
            if line:
                current_time = time.time()
                line_str = line.decode("utf-8").strip()

                # Skip non-data lines
                if not line_str.startswith("data: "):
                    continue

                data_str = line_str[6:]  # Remove 'data: ' prefix

                # Check for end of stream
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)

                    # Extract token from response
                    if "choices" in data and len(data["choices"]) > 0:
                        choice = data["choices"][0]

                        if "delta" in choice and "content" in choice["delta"]:
                            token_content = choice["delta"]["content"]

                            if token_content:  # Only process non-empty tokens
                                completion_text += token_content
                                completion_tokens += 1

                                # Record TTFT (time to first token)
                                if ttft is None:
                                    ttft = current_time - start_time
                                    last_token_time = current_time
                                else:
                                    # Record TPOT (time per output token)
                                    if last_token_time is not None:
                                        tpot = current_time - last_token_time
                                        tpot_values.append(tpot)
                                    last_token_time = current_time

                        # Check if this is the final message
                        if choice.get("finish_reason") is not None:
                            break

                except json.JSONDecodeError:
                    # Skip malformed JSON lines
                    continue

        end_time = time.time()
        total_latency = end_time - start_time

        # Calculate average TPOT
        tpot_avg = sum(tpot_values) / len(tpot_values) if tpot_values else None

        return {
            "success": True,
            "latency": total_latency,
            "status_code": response.status_code,
            "target_pod": target_pod,
            "response_data": {
                "choices": [{"message": {"content": completion_text}}],
                "usage": {
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": len(prompt.split()),
                    "total_tokens": len(prompt.split()) + completion_tokens,
                },
            },
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": completion_tokens,
            "total_tokens": len(prompt.split()) + completion_tokens,
            "ttft": ttft,
            "tpot_avg": tpot_avg,
            "tpot_values": tpot_values,
            "streaming_metrics": {
                "ttft": ttft,
                "tpot_avg": tpot_avg,
                "tpot_count": len(tpot_values),
                "completion_text": completion_text,
            },
        }

    except Exception as e:
        end_time = time.time()
        latency = end_time - start_time
        return {
            "success": False,
            "latency": latency,
            "status_code": None,
            "target_pod": target_pod,
            "error": str(e),
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": 0,
            "total_tokens": 0,
            "ttft": None,
            "tpot_avg": None,
            "tpot_values": [],
        }


def query_prometheus(query: str, timestamp: Optional[float] = None) -> Dict:
    """Query Prometheus for metrics."""
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
            logger.warning(f"Prometheus query failed: {response.status_code}")
            return {"status": "error", "data": {}}

    except Exception as e:
        logger.warning(f"Failed to query Prometheus: {e}")
        return {"status": "error", "data": {}}


def collect_prometheus_metrics(timestamp: Optional[float] = None) -> Dict:
    """Collect system metrics from Prometheus."""
    metrics = {}

    pod_queries = {
        "pod_cpu_usage": 'rate(container_cpu_usage_seconds_total{pod=~"tinyllama.*"}[1m])',
        "pod_memory_usage": 'container_memory_working_set_bytes{pod=~"tinyllama.*"}',
        "pod_request_count": "increase(vllm:request_success_total[1m])",
        "pod_request_latency_sum": "vllm:e2e_request_latency_seconds_sum",
        "pod_request_latency_count": "vllm:e2e_request_latency_seconds_count",
        "pod_requests_running": "vllm:num_requests_running",
        "pod_requests_waiting": "vllm:num_requests_waiting",
    }

    gateway_queries = {
        "gateway_requests_total": "increase(envoy_http_downstream_rq_total[1m])",
        "gateway_response_time_sum": "envoy_http_downstream_rq_time_sum",
        "gateway_response_time_count": "envoy_http_downstream_rq_time_count",
    }

    vtc_queries = {
        "vtc_bucket_size_active": "vtc_bucket_size_active",
        "vtc_bucket_size_changes": "abs(deriv(vtc_bucket_size_active[1m]))",
    }

    # Collect metrics
    for category, queries in [
        ("pods", pod_queries),
        ("gateway", gateway_queries),
        ("vtc", vtc_queries),
    ]:
        metrics[category] = {}
        for metric_name, query in queries.items():
            result = query_prometheus(query, timestamp)
            if result.get("status") == "success":
                metrics[category][metric_name] = result.get("data", {}).get(
                    "result", []
                )

    metrics["timestamp"] = timestamp or time.time()
    return metrics


def save_system_metrics(result_dir: str, metrics_type: str, metrics: Dict):
    """Save system metrics to file."""
    filename = f"{result_dir}/system_metrics_{metrics_type}.json"
    with open(filename, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved {metrics_type} system metrics to {filename}")


def prepare_requests(max_req_count, user_distribution):
    """Prepare requests based on traffic pattern and user distribution."""
    if user_distribution not in TRAFFIC_PATTERNS:
        raise ValueError(f"Unknown user distribution: {user_distribution}")

    pattern = TRAFFIC_PATTERNS[user_distribution]
    requests = []

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

        user_category_info = next(
            cat for cat in USER_CATEGORIES if cat["name"] == selected_category
        )
        user = random.choice(user_category_info["users"])

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
    """Save comprehensive benchmark statistics."""
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

        successful_requests = [
            req for req in algo_requests if req.get("success", False)
        ]
        latencies = [req.get("latency", 0) for req in successful_requests]

        # Pod distribution
        pod_counts = {}
        for req in successful_requests:
            pod = req.get("target_pod", "unknown")
            pod_counts[pod] = pod_counts.get(pod, 0) + 1

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
            "failed_requests": len(algo_requests) - len(successful_requests),
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

        # Streaming metrics analysis (if available)
        streaming_data = []
        ttft_values = []
        tpot_values = []
        for req in successful_requests:
            if req.get("ttft") is not None:
                streaming_data.append(
                    {
                        "ttft": req.get("ttft"),
                        "tpot_avg": req.get("tpot_avg"),
                        "tpot_count": len(req.get("tpot_values", [])),
                        "completion_tokens": req.get("completion_tokens", 0),
                        "category": req.get("category", "unknown"),
                        "latency": req.get("latency", 0),
                    }
                )
                ttft_values.append(req.get("ttft"))
                if req.get("tpot_avg") is not None:
                    tpot_values.append(req.get("tpot_avg"))

        if streaming_data:
            stats_data["streaming_analysis"] = stats_data.get("streaming_analysis", {})
            stats_data["streaming_analysis"][algorithm] = {
                "streaming_requests": len(streaming_data),
                "ttft_stats": {
                    "avg": sum(ttft_values) / len(ttft_values) if ttft_values else 0,
                    "min": min(ttft_values) if ttft_values else 0,
                    "max": max(ttft_values) if ttft_values else 0,
                    "std": np.std(ttft_values) if ttft_values else 0,
                    "p50": np.percentile(ttft_values, 50) if ttft_values else 0,
                    "p90": np.percentile(ttft_values, 90) if ttft_values else 0,
                    "p95": np.percentile(ttft_values, 95) if ttft_values else 0,
                },
                "tpot_stats": (
                    {
                        "avg": (
                            sum(tpot_values) / len(tpot_values) if tpot_values else 0
                        ),
                        "min": min(tpot_values) if tpot_values else 0,
                        "max": max(tpot_values) if tpot_values else 0,
                        "std": np.std(tpot_values) if tpot_values else 0,
                        "p50": np.percentile(tpot_values, 50) if tpot_values else 0,
                        "p90": np.percentile(tpot_values, 90) if tpot_values else 0,
                        "p95": np.percentile(tpot_values, 95) if tpot_values else 0,
                    }
                    if tpot_values
                    else None
                ),
                "detailed_data": streaming_data,
            }

        # Routing effectiveness
        pod_user_mapping = {}
        for req in successful_requests:
            user = req.get("user", "unknown")
            pod = req.get("target_pod", "unknown")
            if user not in pod_user_mapping:
                pod_user_mapping[user] = []
            pod_user_mapping[user].append(pod)

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
        json.dump(stats_data, f, indent=2, default=str)

    logger.info(f"Comprehensive stats saved to: {stats_file}")
    return stats_file


def validate_success_rates(
    results_by_algorithm: Dict[str, List[Dict]], strict_mode: bool = False
) -> Dict:
    """Validate success rates for each algorithm."""
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
                error_msg = req.get("error", "Unknown")
                status_code = req.get("status_code", "Unknown")

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


def generate_variable_prompt(
    user: str, category: str, min_tokens: int, max_tokens: int
) -> str:
    """Generate a prompt with variable length based on user category."""
    target_tokens = random.randint(max(min_tokens, 10), min(max_tokens, 500))

    base_prompts = {
        "small": [
            f"Hello, I am {user}. Please tell me a short story.",
            f"Hi {user} here. What's the weather like?",
            f"I'm {user}. Can you explain what AI is?",
            f"Hello, I'm {user}. What's your favorite color?",
        ],
        "sm-small": [
            f"Hello, I am {user}. Tell me a short story.",
            f"Hi {user}. What's the weather?",
            f"I'm {user}. Explain AI basics.",
            f"Hello, {user} here. Favorite color?",
        ],
        "medium": [
            f"Hello, I am {user}. I'm working on a project about renewable energy and would like you to explain the differences between solar, wind, and hydroelectric power generation methods.",
            f"Hi, I'm {user}. Can you help me understand machine learning algorithms and provide examples of supervised versus unsupervised learning techniques?",
        ],
        "sm-medium": [
            f"Hello, I am {user}. Can you explain how machine learning works in simple terms?",
            f"Hi {user}. I'm planning a garden and need advice on vegetables that grow well together.",
        ],
        "high": [
            f"Hello, I am {user}. I'm conducting research on the economic implications of climate change policies and their impact on developing nations. Could you provide a comprehensive analysis covering carbon pricing mechanisms, international climate agreements, green technology transfer, and policy recommendations.",
        ],
        "sm-high": [
            f"Hello, I am {user}. I'm developing a mobile app for personal finance and need guidance on features, UX design, security, and banking API integration.",
        ],
    }

    category_prompts = base_prompts.get(category, base_prompts["small"])
    base_prompt = random.choice(category_prompts)

    # Estimate current token count and add padding if needed
    current_tokens = len(base_prompt) // 4
    if current_tokens < target_tokens:
        additional_tokens_needed = target_tokens - current_tokens
        padding_phrases = [
            " Please provide detailed explanations with examples.",
            " I would appreciate comprehensive coverage of this topic.",
            " Include relevant background information and context.",
        ]

        while current_tokens < target_tokens and padding_phrases:
            phrase = random.choice(padding_phrases)
            base_prompt += phrase
            current_tokens = len(base_prompt) // 4

    return base_prompt


def load_vtc_dataset(dataset_path: str = VTC_DATASET_PATH) -> List[Dict]:
    """Load the VTC dataset from JSONL file."""
    dataset = []

    if not os.path.exists(dataset_path):
        logger.warning(f"VTC dataset not found at {dataset_path}")
        return dataset

    with open(dataset_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            dataset.append(entry)

    logger.info(f"Loaded VTC dataset with {len(dataset)} entries from {dataset_path}")
    return dataset


def get_prompt_from_dataset(
    vtc_dataset: List[Dict], user: str, category: str
) -> Optional[str]:
    """Get a prompt from the VTC dataset for a specific user and category."""
    matching_entries = []
    for entry in vtc_dataset:
        for request in entry.get("requests", []):
            if request.get("user_category") == category:
                matching_entries.append(request)

    if not matching_entries:
        logger.warning(f"No prompts found for category {category} in dataset")
        return None

    selected_request = random.choice(matching_entries)
    prompt_content = selected_request.get("prompt", [{}])[0].get("content", "")
    return prompt_content


def send_request_async(
    i,
    req_data,
    send_time,
    routing_algorithm,
    traffic_pattern,
    result_queue,
    vtc_dataset,
    use_real_dataset,
    streaming_mode=False,
):
    """Send a single request asynchronously."""
    user = req_data["user"]

    if use_real_dataset:
        prompt = get_prompt_from_dataset(vtc_dataset, user, req_data["category"])
        if prompt is None:
            prompt = generate_variable_prompt(
                user,
                req_data["category"],
                req_data["min_tokens"],
                req_data["max_tokens"],
            )
    else:
        prompt = generate_variable_prompt(
            user, req_data["category"], req_data["min_tokens"], req_data["max_tokens"]
        )

    logger.info(
        f"Request {i+1} - User: {user}, Algorithm: {routing_algorithm}, Streaming: {streaming_mode}"
    )

    if streaming_mode:
        result = make_streaming_req(user, prompt, routing_algorithm, output_tokens=20)
    else:
        result = make_non_streaming_req(
            user, prompt, routing_algorithm, output_tokens=20
        )

    # Find user category and token range
    user_category = next((cat for cat in USER_CATEGORIES if user in cat["users"]), None)
    category_name = user_category["name"] if user_category else "unknown"
    token_range = (
        {
            "min_tokens": user_category["min_tokens"] if user_category else 0,
            "max_tokens": user_category["max_tokens"] if user_category else 0,
            "token_scale": user_category["token_scale"] if user_category else 1,
        }
        if user_category
        else None
    )

    # Track for analysis
    result.update(
        {
            "user": user,
            "algorithm": routing_algorithm,
            "category": category_name,
            "token_range": token_range,
            "traffic_pattern": traffic_pattern,
            "send_time": send_time,
            "request_id": i,
        }
    )

    result_queue.put((i, result))

    if result["success"]:
        if streaming_mode and result.get("ttft") is not None:
            logger.info(
                f"Request {i+1} successful - Latency: {result['latency']:.2f}s, Pod: {result['target_pod']}, "
                f"TTFT: {result['ttft']:.3f}s, TPOT: {result.get('tpot_avg', 'N/A'):.3f}s"
            )
        else:
            logger.info(
                f"Request {i+1} successful - Latency: {result['latency']:.2f}s, Pod: {result['target_pod']}"
            )
    else:
        logger.error(
            f"Request {i+1} failed - Status: {result.get('status_code', 'Unknown')}, Error: {result.get('error', 'Unknown')[:50]}"
        )


def run_benchmark(
    traffic_pattern: str = "balanced",
    max_requests: int = 5,
    target_qps: float = 0,
    algorithms: List[str] = None,
    strict_mode: bool = False,
    skip_warmup: bool = False,
    streaming_mode: bool = False,
):
    """Run benchmark for specified traffic pattern and save results."""
    if algorithms is None:
        algorithms = DEFAULT_ROUTING_ALGORITHMS

    logger.info(
        f"Starting benchmark - Pattern: {traffic_pattern}, Max requests: {max_requests}, "
        f"QPS: {target_qps if target_qps > 0 else 'unlimited'}, Algorithms: {algorithms}, "
        f"Streaming: {streaming_mode}"
    )

    result_dir = create_result_dir()

    if not setup_redis_users():
        logger.error("Failed to setup Redis users, aborting benchmark")
        return

    if not skip_warmup:
        logger.info("Running system warm-up")
        if not run_system_warmup(streaming_mode):
            logger.warning("System warm-up had issues, continuing")
    else:
        logger.info("Skipping system warm-up")

    # Load VTC dataset
    logger.info("Loading VTC dataset")
    vtc_dataset = load_vtc_dataset()
    use_real_dataset = len(vtc_dataset) > 0

    if use_real_dataset:
        logger.info("Using real ShareGPT prompts from VTC dataset")
    else:
        logger.warning("VTC dataset not available, using synthetic prompts")

    # Generate requests
    requests_data = prepare_requests(max_requests, traffic_pattern)
    logger.info(f"Generated {len(requests_data)} requests")

    # Log distribution
    category_counts = {}
    for req in requests_data:
        category = req["category"]
        category_counts[category] = category_counts.get(category, 0) + 1

    logger.info("Request distribution:")
    for category, count in sorted(category_counts.items()):
        percentage = (count / len(requests_data)) * 100
        logger.info(f"  {category}: {count} requests ({percentage:.1f}%)")

    # Collect baseline metrics
    logger.info("Collecting baseline system metrics")
    baseline_metrics = collect_prometheus_metrics()
    save_system_metrics(result_dir, "baseline", baseline_metrics)

    results_by_algorithm = {}
    all_requests = []
    all_system_metrics = {"baseline": baseline_metrics}

    # Run benchmark for each routing algorithm
    for routing_algorithm in algorithms:
        logger.info(f"Testing routing algorithm: {routing_algorithm}")

        pre_algo_metrics = collect_prometheus_metrics()
        save_system_metrics(result_dir, f"pre_{routing_algorithm}", pre_algo_metrics)
        all_system_metrics[f"pre_{routing_algorithm}"] = pre_algo_metrics

        algorithm_requests = []

        if target_qps > 0:
            # Async execution with QPS control
            interval = 1.0 / target_qps
            logger.info(
                f"Generating requests at {target_qps} QPS (interval: {interval:.3f}s)"
            )

            result_queue = queue.Queue()
            threads = []
            start_time = time.time()

            for i, req_data in enumerate(requests_data):
                scheduled_time = start_time + (i * interval)
                current_time = time.time()

                if current_time < scheduled_time:
                    time.sleep(scheduled_time - current_time)

                actual_send_time = time.time()

                thread = threading.Thread(
                    target=send_request_async,
                    args=(
                        i,
                        req_data,
                        actual_send_time,
                        routing_algorithm,
                        traffic_pattern,
                        result_queue,
                        vtc_dataset,
                        use_real_dataset,
                        streaming_mode,
                    ),
                )
                thread.start()
                threads.append(thread)

            logger.info("Waiting for all requests to complete")
            for thread in threads:
                thread.join()

            # Collect results
            results_by_id = {}
            while not result_queue.empty():
                request_id, result = result_queue.get()
                results_by_id[request_id] = result

            for i in range(len(requests_data)):
                if i in results_by_id:
                    algorithm_requests.append(results_by_id[i])
                    all_requests.append(results_by_id[i].copy())

        else:
            # Sequential execution
            for i, req_data in enumerate(requests_data):
                user = req_data["user"]

                if use_real_dataset:
                    prompt = get_prompt_from_dataset(
                        vtc_dataset, user, req_data["category"]
                    )
                    if prompt is None:
                        prompt = generate_variable_prompt(
                            user,
                            req_data["category"],
                            req_data["min_tokens"],
                            req_data["max_tokens"],
                        )
                else:
                    prompt = generate_variable_prompt(
                        user,
                        req_data["category"],
                        req_data["min_tokens"],
                        req_data["max_tokens"],
                    )

                logger.info(
                    f"Request {i+1}/{len(requests_data)} - User: {user}, Algorithm: {routing_algorithm}, Streaming: {streaming_mode}"
                )

                if streaming_mode:
                    result = make_streaming_req(
                        user, prompt, routing_algorithm, output_tokens=20
                    )
                else:
                    result = make_non_streaming_req(
                        user, prompt, routing_algorithm, output_tokens=20
                    )

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

                result.update(
                    {
                        "user": user,
                        "algorithm": routing_algorithm,
                        "category": category_name,
                        "token_range": token_range,
                        "traffic_pattern": traffic_pattern,
                    }
                )

                algorithm_requests.append(result)
                all_requests.append(result.copy())

                if result["success"]:
                    if streaming_mode and result.get("ttft") is not None:
                        logger.info(
                            f"Request {i+1} successful - Latency: {result['latency']:.2f}s, Pod: {result['target_pod']}, "
                            f"TTFT: {result['ttft']:.3f}s, TPOT: {result.get('tpot_avg', 'N/A'):.3f}s"
                        )
                    else:
                        logger.info(
                            f"Request {i+1} successful - Latency: {result['latency']:.2f}s, Pod: {result['target_pod']}"
                        )
                else:
                    logger.error(
                        f"Request {i+1} failed - Status: {result.get('status_code', 'Unknown')}"
                    )

        results_by_algorithm[routing_algorithm] = algorithm_requests

        post_algo_metrics = collect_prometheus_metrics()
        save_system_metrics(result_dir, f"post_{routing_algorithm}", post_algo_metrics)
        all_system_metrics[f"post_{routing_algorithm}"] = post_algo_metrics

        logger.info("Waiting 30 seconds before next algorithm")
        time.sleep(30)

    # Collect final metrics
    logger.info("Collecting final system metrics")
    final_metrics = collect_prometheus_metrics()
    save_system_metrics(result_dir, "final", final_metrics)
    all_system_metrics["final"] = final_metrics

    # Validate success rates
    success_rate_validation = validate_success_rates(results_by_algorithm, strict_mode)

    logger.info("Success rate validation")
    valid_for_comparison = True
    threshold = 0.0 if strict_mode else 2.0

    for algorithm, stats in success_rate_validation.items():
        success_rate = stats["success_rate"]
        failure_rate = stats["failure_rate"]

        logger.info(
            f"{algorithm}: {stats['successful_requests']}/{stats['total_requests']} successful ({success_rate:.1f}%)"
        )

        if failure_rate > threshold:
            logger.error(f"  Failure rate too high: {failure_rate:.1f}% > {threshold}%")
            valid_for_comparison = False
        else:
            logger.info(
                f"  Success rate acceptable: {failure_rate:.1f}% <= {threshold}%"
            )

    if not valid_for_comparison:
        logger.error("Benchmark invalid: High failure rates detected")
        fairness_analysis = analyze_fairness_metrics(results_by_algorithm)
        fairness_analysis["validation"] = {
            "valid_for_comparison": False,
            "reason": "High failure rates detected",
            "success_rates": success_rate_validation,
        }
    else:
        logger.info("Success rate validation passed")
        fairness_analysis = analyze_fairness_metrics(results_by_algorithm)
        fairness_analysis["validation"] = {
            "valid_for_comparison": True,
            "success_rates": success_rate_validation,
        }

    # Save results
    fairness_file = f"{result_dir}/fairness_analysis.json"
    with open(fairness_file, "w") as f:
        json.dump(fairness_analysis, f, indent=2)
    logger.info(f"Fairness analysis saved to: {fairness_file}")

    # Check routing concentration
    logger.info("Checking routing concentration")
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
            max_pod_requests = max(pod_counts.values())
            max_pod_percentage = (max_pod_requests / total_requests) * 100

            if max_pod_percentage > 80:
                logger.warning(
                    f"Routing concentration warning for {algorithm}: {max_pod_percentage:.1f}%"
                )
            else:
                logger.info(f"Pod distribution healthy for {algorithm}: {pod_counts}")

    # Save comprehensive stats
    logger.info("Generating comprehensive benchmark statistics")
    comprehensive_stats_file = save_comprehensive_stats(
        result_dir, all_requests, all_system_metrics
    )

    if len(algorithms) > 1:
        if valid_for_comparison:
            log_fairness_summary(fairness_analysis)
        else:
            logger.error("Skipping fairness comparison due to high failure rates")
    else:
        logger.info(f"Single algorithm test completed: {algorithms[0]}")

    logger.info(f"Benchmark completed. Results saved to: {result_dir}")
    return result_dir


def main():
    parser = argparse.ArgumentParser(
        description="VTC Benchmarking and Stats Collection"
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
        help="Target queries per second (0 = sequential)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming mode with TTFT/TPOT metrics",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=DEFAULT_ROUTING_ALGORITHMS,
        default=None,
        help="Routing algorithms to test",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: any request failure invalidates comparison",
    )
    parser.add_argument(
        "--no-warmup", action="store_true", help="Skip system warm-up phase"
    )

    args = parser.parse_args()

    # Streaming mode is now implemented
    if args.stream:
        logger.info("Streaming mode enabled - will collect TTFT and TPOT metrics")

    algorithms_to_test = (
        args.algorithms if args.algorithms else DEFAULT_ROUTING_ALGORITHMS
    )

    logger.info("VTC Benchmarking Configuration")
    logger.info(f"Requests: {args.requests}")
    logger.info(f"Pattern: {args.pattern}")
    logger.info(f"QPS: {args.qps if args.qps > 0 else 'Sequential'}")
    logger.info(f"Algorithms: {', '.join(algorithms_to_test)}")
    logger.info(f"Validation: {'Strict' if args.strict else 'Relaxed'}")
    logger.info(f"Warm-up: {'Disabled' if args.no_warmup else 'Enabled'}")
    logger.info(f"Streaming: {'Enabled' if args.stream else 'Disabled'}")

    result_dir = run_benchmark(
        traffic_pattern=args.pattern,
        max_requests=args.requests,
        target_qps=args.qps,
        algorithms=algorithms_to_test,
        strict_mode=args.strict,
        skip_warmup=args.no_warmup,
        streaming_mode=args.stream,
    )

    if result_dir:
        logger.info("Benchmark completed successfully")
        logger.info(f"Results saved to: {result_dir}")
    else:
        logger.error("Benchmark failed")


if __name__ == "__main__":
    main()
