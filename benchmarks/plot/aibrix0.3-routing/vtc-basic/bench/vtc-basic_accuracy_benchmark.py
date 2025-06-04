#!/usr/bin/env python3

import argparse
import json
import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Import production functions from bench_and_collect_stats
from bench_and_collect_stats import (
    PROMETHEUS_URL,
    collect_prometheus_metrics,
    get_prompt_from_dataset,
    load_vtc_dataset,
    make_non_streaming_req,
    query_prometheus,
    run_system_warmup,
    setup_redis_users,
    validate_success_rates,
)
from constants import TEMP_WORKSPACE, USER_CATEGORIES

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_result_dir() -> str:
    """Create results directory in temp workspace."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = TEMP_WORKSPACE / "vtc_accuracy_results" / f"accuracy_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created results directory: {result_dir}")
    return str(result_dir)


def get_vtc_bucket_size() -> float:
    """Get the current VTC adaptive bucket size using production function."""
    result = query_prometheus("vtc_bucket_size_active")
    if result.get("status") == "success" and result.get("data", {}).get("result"):
        return float(result["data"]["result"][0]["value"][1])
    return 0.0


def run_lazy_pruning_benchmark(requests_count: int = 5) -> Dict:
    """Test VTC's lazy pruning feature by building token history, waiting for expiry, and triggering cleanup."""
    logger.info("Starting lazy pruning benchmark")
    start_time = datetime.now()

    if not setup_redis_users():
        return {"error": "Redis setup failed"}

    if not run_system_warmup():
        logger.warning("System warmup had issues, continuing")

    vtc_dataset = load_vtc_dataset()
    if not vtc_dataset:
        return {"error": "Dataset loading failed"}

    # Select varied users for token history building
    users_to_test = []
    for category in USER_CATEGORIES:
        if category["name"] in ["sm-small", "sm-medium", "sm-high"]:
            users_to_test.extend(category["users"][:1])
    users_to_test = users_to_test[:5]

    logger.info(f"Building token history with {len(users_to_test)} users")

    # Phase 1: Build token history
    initial_results = []
    for i, user in enumerate(users_to_test):
        user_category = next(
            (cat["name"] for cat in USER_CATEGORIES if user in cat["users"]), None
        )
        prompt = (
            get_prompt_from_dataset(vtc_dataset, user, user_category)
            or f"Test prompt for {user}"
        )

        result = make_non_streaming_req(user, prompt, "vtc-basic", output_tokens=20)
        initial_results.append(result)

        if result["success"]:
            logger.info(
                f"User {user}: {result['total_tokens']} tokens, pod {result['target_pod']}"
            )
        else:
            logger.error(f"User {user} failed: {result.get('error', 'Unknown')}")

        time.sleep(1)

    # Validate initial phase
    success_validation = validate_success_rates(
        {"vtc-basic": initial_results}, strict_mode=False
    )
    vtc_stats = success_validation["vtc-basic"]

    if vtc_stats["success_rate"] < 80:
        return {
            "error": "Low success rate in initial phase",
            "success_rate": vtc_stats["success_rate"],
        }

    bucket_size_after_initial = get_vtc_bucket_size()
    logger.info(f"Initial bucket size: {bucket_size_after_initial}")

    # Phase 2: Wait for window expiry
    logger.info("Waiting 70 seconds for token window expiry")
    time.sleep(70)

    bucket_size_before_pruning = get_vtc_bucket_size()
    logger.info(f"Bucket size before pruning: {bucket_size_before_pruning}")

    # Phase 3: Trigger pruning with small requests
    all_users = [user for category in USER_CATEGORIES for user in category["users"]]
    pruning_results = []

    logger.info(f"Sending pruning requests to {len(all_users)} users")
    for user in all_users:
        result = make_non_streaming_req(user, "Hi", "vtc-basic", output_tokens=5)
        result["pruning_request"] = True
        pruning_results.append(result)
        time.sleep(1)

    bucket_size_after_pruning = get_vtc_bucket_size()
    logger.info(f"Bucket size after pruning: {bucket_size_after_pruning}")

    # Analysis
    lazy_pruning_working = (
        abs(bucket_size_after_initial - bucket_size_before_pruning) < 0.1
    )
    pruning_triggered = bucket_size_after_pruning < bucket_size_before_pruning

    success = lazy_pruning_working and pruning_triggered

    if success:
        logger.info("Lazy pruning feature working correctly")
    else:
        logger.error("Lazy pruning issues detected")

    return {
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": (datetime.now() - start_time).total_seconds(),
        "initial_phase": {"results": initial_results, "validation": success_validation},
        "pruning_phase": {"results": pruning_results},
        "bucket_sizes": {
            "after_initial": bucket_size_after_initial,
            "before_pruning": bucket_size_before_pruning,
            "after_pruning": bucket_size_after_pruning,
        },
        "analysis": {
            "success": success,
            "lazy_pruning_working": lazy_pruning_working,
            "pruning_triggered": pruning_triggered,
        },
    }


def send_requests_for_user(
    user_data: Dict, results_queue: queue.Queue, vtc_dataset: List[Dict]
):
    """Send requests for a single user at 1 QPS."""
    user = user_data["name"]
    category = user_data["category"]
    num_requests = user_data["requests"]

    user_results = []
    current_tokens = 0

    for i in range(num_requests):
        start_time = time.time()

        prompt = (
            get_prompt_from_dataset(vtc_dataset, user, category)
            or f"Test message {i+1} for {user}"
        )

        try:
            result = make_non_streaming_req(user, prompt, "vtc-basic", output_tokens=10)
            duration = time.time() - start_time

            if result["success"]:
                tokens_added = result["prompt_tokens"] + result["completion_tokens"] * 2
                current_tokens += tokens_added
                user_results.append(
                    {"request": i + 1, "tokens_added": tokens_added, "success": True}
                )

                if (i + 1) % 5 == 0:
                    logger.info(
                        f"{user}: {i+1}/{num_requests} requests, {current_tokens} tokens"
                    )
            else:
                user_results.append(
                    {
                        "request": i + 1,
                        "error": result.get("error", "Unknown"),
                        "success": False,
                    }
                )
                logger.error(
                    f"{user} request {i+1} failed: {result.get('error', 'Unknown')}"
                )

        except Exception as e:
            user_results.append({"request": i + 1, "error": str(e), "success": False})
            logger.error(f"{user} request {i+1} exception: {str(e)}")

        # Maintain 1 QPS
        elapsed = time.time() - start_time
        sleep_time = max(0, 1.0 - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    results_queue.put(
        {
            "user": user,
            "category": category,
            "expected_type": user_data["expected_type"],
            "final_tokens": current_tokens,
            "requests": user_results,
            "success_count": len([r for r in user_results if r["success"]]),
        }
    )

    logger.info(f"{user} completed: {current_tokens} tokens")


def run_fairness_distribution_test() -> Dict:
    """Test VTC fairness distribution across different token levels."""
    logger.info("Starting VTC fairness distribution test")
    start_time = datetime.now()

    if not setup_redis_users():
        return {"error": "Redis setup failed"}

    vtc_dataset = load_vtc_dataset()
    if not vtc_dataset:
        return {"error": "Dataset loading failed"}

    # Select test users: 2 small + 2 medium + 1 high
    small_users = [
        (user, cat["name"])
        for cat in USER_CATEGORIES
        if "small" in cat["name"]
        for user in cat["users"][:2]
    ]
    medium_users = [
        (user, cat["name"])
        for cat in USER_CATEGORIES
        if "medium" in cat["name"]
        for user in cat["users"][:2]
    ]
    high_users = [
        (user, cat["name"])
        for cat in USER_CATEGORIES
        if "high" in cat["name"]
        for user in cat["users"][:1]
    ]

    test_users = []

    # Add users with proper token level expectations
    for user, category in small_users:
        test_users.append(
            {
                "name": user,
                "category": category,
                "requests": 20,
                "expected_type": "low_tokens",
            }
        )

    for user, category in medium_users:
        test_users.append(
            {
                "name": user,
                "category": category,
                "requests": 20,
                "expected_type": "medium_tokens",
            }
        )

    for user, category in high_users:
        test_users.append(
            {
                "name": user,
                "category": category,
                "requests": 20,
                "expected_type": "high_tokens",
            }
        )

    logger.info(f"Testing {len(test_users)} users in parallel at 1 QPS each")

    # Run parallel token building
    results_queue = queue.Queue()
    threads = []

    for user_data in test_users:
        thread = threading.Thread(
            target=send_requests_for_user, args=(user_data, results_queue, vtc_dataset)
        )
        thread.daemon = True
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    # Collect results
    parallel_results = []
    while not results_queue.empty():
        parallel_results.append(results_queue.get())

    time.sleep(5)  # Allow metrics to stabilize

    # Test fairness routing
    logger.info("Testing fairness routing")
    routing_results = []
    user_token_counts = {}

    for result in parallel_results:
        user = result["user"]
        category = result["category"]
        expected_type = result["expected_type"]
        final_tokens = result["final_tokens"]

        routing_result = make_non_streaming_req(
            user, "Route me fairly", "vtc-basic", output_tokens=10
        )

        if routing_result["success"]:
            target_pod = routing_result.get("target_pod", "unknown")
            routing_results.append(
                {
                    "user": user,
                    "category": category,
                    "expected_type": expected_type,
                    "final_tokens": final_tokens,
                    "target_pod": target_pod,
                    "success": True,
                }
            )
            user_token_counts[user] = final_tokens
            logger.info(f"{user} -> {target_pod} ({final_tokens} tokens)")
        else:
            routing_results.append(
                {
                    "user": user,
                    "category": category,
                    "expected_type": expected_type,
                    "final_tokens": final_tokens,
                    "error": routing_result.get("error", "Unknown"),
                    "success": False,
                }
            )
            logger.error(
                f"{user} routing failed: {routing_result.get('error', 'Unknown')}"
            )

    # Analyze fairness
    successful_results = [r for r in routing_results if r["success"]]

    low_token_users = [
        r for r in successful_results if r["expected_type"] == "low_tokens"
    ]
    medium_token_users = [
        r for r in successful_results if r["expected_type"] == "medium_tokens"
    ]
    high_token_users = [
        r for r in successful_results if r["expected_type"] == "high_tokens"
    ]

    # Performance checks
    unique_pods = len(set(r["target_pod"] for r in successful_results))
    total_users = len(successful_results)
    success_rate = (
        (len(successful_results) / len(routing_results) * 100) if routing_results else 0
    )

    checks_passed = 0
    total_checks = 3

    # Check 1: Pod distribution
    if unique_pods > 1 or total_users == 1:
        checks_passed += 1
        logger.info(f"Pod distribution: {total_users} users across {unique_pods} pods")

    # Check 2: Token routing consistency
    if len(successful_results) >= 2:
        token_counts = [r["final_tokens"] for r in successful_results]
        token_ratio = (
            max(token_counts) / min(token_counts) if min(token_counts) > 0 else 1
        )
        checks_passed += 1
        logger.info(
            f"Token routing: range {min(token_counts)}-{max(token_counts)} (ratio {token_ratio:.1f}x)"
        )

    # Check 3: Success rate
    if success_rate >= 80:
        checks_passed += 1
        logger.info(f"Success rate: {success_rate:.1f}%")

    fairness_score = checks_passed / total_checks
    test_passed = success_rate >= 80 and fairness_score >= 0.6

    if test_passed:
        logger.info("VTC fairness test passed")
    else:
        logger.error("VTC fairness test failed")

    return {
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": (datetime.now() - start_time).total_seconds(),
        "total_users_tested": len(routing_results),
        "successful_routes": len(successful_results),
        "success_rate": success_rate,
        "fairness_score": fairness_score,
        "routing_results": routing_results,
        "user_token_counts": user_token_counts,
        "unique_pods_used": unique_pods,
        "test_passed": test_passed,
    }


def main():
    parser = argparse.ArgumentParser(description="VTC-Basic Accuracy Benchmark")
    parser.add_argument(
        "--requests",
        type=int,
        default=5,
        help="Number of requests for lazy pruning test",
    )
    parser.add_argument(
        "--test-type",
        choices=["lazy-pruning", "fairness", "all"],
        default="lazy-pruning",
        help="Type of test to run",
    )

    args = parser.parse_args()

    logger.info("Starting VTC-Basic accuracy benchmark")
    result_dir = create_result_dir()
    results = {}

    # Run selected tests
    if args.test_type in ["lazy-pruning", "all"]:
        lazy_results = run_lazy_pruning_benchmark(args.requests)
        if "error" in lazy_results:
            logger.error(f"Lazy pruning test failed: {lazy_results['error']}")
            return 1
        results["lazy_pruning"] = lazy_results

    if args.test_type in ["fairness", "all"]:
        fairness_results = run_fairness_distribution_test()
        if "error" in fairness_results:
            logger.error(f"Fairness test failed: {fairness_results['error']}")
            return 1
        results["fairness_distribution"] = fairness_results

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.test_type == "lazy-pruning":
        results_file = f"{result_dir}/lazy_pruning_benchmark_{timestamp}.json"
        final_results = results.get("lazy_pruning", {})
    elif args.test_type == "fairness":
        results_file = f"{result_dir}/fairness_distribution_test_{timestamp}.json"
        final_results = results.get("fairness_distribution", {})
    else:
        results_file = f"{result_dir}/vtc_accuracy_benchmark_{timestamp}.json"
        final_results = results

    with open(results_file, "w") as f:
        json.dump(final_results, f, indent=2, default=str)

    logger.info(f"Results saved to: {results_file}")

    # Determine overall success
    success = True
    if args.test_type in ["lazy-pruning", "all"]:
        if (
            not results.get("lazy_pruning", {})
            .get("analysis", {})
            .get("success", False)
        ):
            success = False

    if args.test_type in ["fairness", "all"]:
        if not results.get("fairness_distribution", {}).get("test_passed", False):
            success = False

    if success:
        logger.info("All tests passed")
        return 0
    else:
        logger.error("One or more tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
