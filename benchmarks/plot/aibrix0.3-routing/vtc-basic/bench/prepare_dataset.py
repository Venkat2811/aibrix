#!/usr/bin/env python3

import argparse
import json
import logging
import os
import random
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tiktoken
from constants import (
    DATASET_DIR,
    DEFAULT_INTERVAL_MS,
    DEFAULT_MIN_REQUESTS,
    DEFAULT_SESSION_PROBABILITY,
    SHAREGPT_DATASET_FILENAME,
    SHAREGPT_DATASET_URL,
    TEMP_WORKSPACE,
    USER_CATEGORIES,
    VTC_DATASET_PATH,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TOKENIZER = tiktoken.get_encoding("cl100k_base")

CPU_ONLY_USER_CATEGORIES = [
    {
        "name": "sm-small",
        "users": ["sm-user-small-1", "sm-user-small-2", "sm-user-small-3"],
        "token_scale": 0.5,
        "min_tokens": 5,
        "max_tokens": 35,
        "target_percentage": 0.4,
    },
    {
        "name": "sm-medium",
        "users": ["sm-user-med-1", "sm-user-med-2", "sm-user-med-3"],
        "token_scale": 1.0,
        "min_tokens": 45,
        "max_tokens": 75,
        "target_percentage": 0.35,
    },
    {
        "name": "sm-high",
        "users": ["sm-user-high-1", "sm-user-high-2", "sm-user-high-3"],
        "token_scale": 1.5,
        "min_tokens": 85,
        "max_tokens": 130,
        "target_percentage": 0.25,
    },
]


def count_tokens(text):
    return len(TOKENIZER.encode(text, disallowed_special=()))


def download_if_not_exists_dataset(dataset_file: Optional[str] = None) -> str:
    if not dataset_file:
        dataset_dir = Path(DATASET_DIR)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = dataset_dir / SHAREGPT_DATASET_FILENAME

        if dataset_path.exists():
            logger.info(f"Dataset already exists at {dataset_path}")
        else:
            try:
                logger.info(f"Downloading ShareGPT dataset from {SHAREGPT_DATASET_URL}")
                urllib.request.urlretrieve(SHAREGPT_DATASET_URL, dataset_path)
                logger.info(f"Dataset downloaded to {dataset_path}")
            except Exception as e:
                logger.error(f"Failed to download dataset: {e}")
                sys.exit(1)
    else:
        dataset_path = Path(dataset_file)
        if not dataset_path.exists():
            logger.error(f"Dataset file not found at {dataset_file}")
            sys.exit(1)

    return str(dataset_path)


def load_sharegpt_data(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} conversations from ShareGPT")
    return data


def extract_prompts_with_tokens(data: List[Dict]) -> List[Dict]:
    prompts = []
    for conversation in data:
        if "conversations" not in conversation:
            continue

        messages = conversation["conversations"]
        for i, message in enumerate(messages):
            if (
                message["from"] == "human"
                and i + 1 < len(messages)
                and messages[i + 1]["from"] == "gpt"
            ):

                prompt_tokens = count_tokens(message["value"])
                response_tokens = count_tokens(messages[i + 1]["value"])

                if prompt_tokens > 0 and response_tokens > 0:
                    prompts.append(
                        {
                            "prompt": message["value"],
                            "response": messages[i + 1]["value"],
                            "prompt_tokens": prompt_tokens,
                            "response_tokens": response_tokens,
                        }
                    )

    logger.info(f"Extracted {len(prompts)} valid prompt-response pairs")
    return prompts


def assign_user_categories(
    prompts: List[Dict],
    min_requests: int = DEFAULT_MIN_REQUESTS,
    use_cpu_categories: bool = False,
) -> List[Dict]:
    categories_to_use = (
        CPU_ONLY_USER_CATEGORIES if use_cpu_categories else USER_CATEGORIES
    )
    prompts_sorted = sorted(prompts, key=lambda x: x["prompt_tokens"])
    categorized_prompts = []

    required_per_category = {
        cat["name"]: int(cat["target_percentage"] * min_requests)
        for cat in categories_to_use
    }

    for category in categories_to_use:
        matching_prompts = [
            p
            for p in prompts_sorted
            if category["min_tokens"] <= p["prompt_tokens"] <= category["max_tokens"]
        ]

        num_to_select = min(
            len(matching_prompts), required_per_category[category["name"]]
        )

        if len(matching_prompts) < required_per_category[category["name"]]:
            logger.warning(
                f"Category {category['name']}: found {len(matching_prompts)}, need {required_per_category[category['name']]}"
            )

        if num_to_select > 0:
            selected = random.sample(matching_prompts, num_to_select)
            for prompt in selected:
                prompt_with_user = prompt.copy()
                prompt_with_user.update(
                    {
                        "user": random.choice(category["users"]),
                        "user_category": category["name"],
                        "token_scale": category["token_scale"],
                    }
                )
                categorized_prompts.append(prompt_with_user)

    # Fill remaining slots if needed
    if len(categorized_prompts) < min_requests:
        additional_needed = min_requests - len(categorized_prompts)
        logger.warning(
            f"Duplicating {additional_needed} prompts to reach {min_requests}"
        )

        for _ in range(additional_needed):
            if categorized_prompts:
                original = random.choice(categorized_prompts)
                duplicate = original.copy()
                category_info = next(
                    cat
                    for cat in categories_to_use
                    if cat["name"] == duplicate["user_category"]
                )
                users = [u for u in category_info["users"] if u != duplicate["user"]]
                if users:
                    duplicate["user"] = random.choice(users)
                categorized_prompts.append(duplicate)

    random.shuffle(categorized_prompts)

    # Log distribution stats
    category_counts = {}
    user_counts = {}
    category_tokens = {}

    for p in categorized_prompts:
        category = p["user_category"]
        user = p["user"]
        category_counts[category] = category_counts.get(category, 0) + 1
        if category not in user_counts:
            user_counts[category] = {}
            category_tokens[category] = []
        user_counts[category][user] = user_counts[category].get(user, 0) + 1
        category_tokens[category].append(p["prompt_tokens"])

    logger.info(f"Dataset distribution: {category_counts}")

    for category, users in user_counts.items():
        tokens = category_tokens[category]
        avg_tokens = sum(tokens) / len(tokens) if tokens else 0
        min_tokens = min(tokens) if tokens else 0
        max_tokens = max(tokens) if tokens else 0
        logger.info(
            f"{category}: tokens({min_tokens}-{max_tokens}, avg:{avg_tokens:.1f})"
        )
        for user, count in users.items():
            logger.info(f"  {user}: {count} requests")

    return categorized_prompts


def create_workload_format(
    prompts: List[Dict], interval_ms: int = DEFAULT_INTERVAL_MS
) -> List[Dict]:
    workload = []
    current_timestamp = 0

    for i, prompt_data in enumerate(prompts):
        request = {
            "prompt": [{"role": "user", "content": prompt_data["prompt"]}],
            "output_length": prompt_data["response_tokens"],
            "user": prompt_data["user"],
            "user_category": prompt_data["user_category"],
        }

        if random.random() < DEFAULT_SESSION_PROBABILITY:
            request["session_id"] = f"session-{prompt_data['user']}-{i // 2}"

        workload.append({"timestamp": current_timestamp, "requests": [request]})
        current_timestamp += interval_ms

    return workload


def save_vtc_dataset(dataset: List[Dict], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
    logger.info(f"Saved dataset to {output_path}")


def create_vtc_bench_dataset(
    sharegpt_path: Optional[str] = None,
    output_path: Optional[str] = None,
    min_requests: int = DEFAULT_MIN_REQUESTS,
    interval_ms: int = DEFAULT_INTERVAL_MS,
    use_cpu_categories: bool = False,
) -> str:
    if not sharegpt_path:
        sharegpt_path = download_if_not_exists_dataset()

    if not output_path:
        if use_cpu_categories:
            output_path = str(VTC_DATASET_PATH).replace(
                ".jsonl", "_cpu_optimized.jsonl"
            )
        else:
            output_path = str(VTC_DATASET_PATH)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sharegpt_data = load_sharegpt_data(sharegpt_path)
    prompts_with_tokens = extract_prompts_with_tokens(sharegpt_data)

    token_counts = [p["prompt_tokens"] for p in prompts_with_tokens]
    logger.info(
        f"Token stats - Min: {min(token_counts)}, Max: {max(token_counts)}, "
        f"Avg: {np.mean(token_counts):.1f}, Median: {np.median(token_counts):.1f}"
    )

    if use_cpu_categories:
        logger.info("Using CPU-optimized categories")
        for cat in CPU_ONLY_USER_CATEGORIES:
            logger.info(
                f"  {cat['name']}: {cat['min_tokens']}-{cat['max_tokens']} tokens ({cat['target_percentage']*100:.0f}%)"
            )

    categorized_prompts = assign_user_categories(
        prompts_with_tokens, min_requests, use_cpu_categories
    )
    workload = create_workload_format(categorized_prompts, interval_ms)
    save_vtc_dataset(workload, output_path)

    logger.info(
        f"Generated VTC dataset: {len(workload)} entries, {len(categorized_prompts)} requests"
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare VTC benchmark dataset from ShareGPT data"
    )
    parser.add_argument(
        "--sharegpt-path", type=str, help="Path to ShareGPT dataset file"
    )
    parser.add_argument("--output-path", type=str, help="Output path for VTC dataset")
    parser.add_argument(
        "--min-requests",
        type=int,
        default=DEFAULT_MIN_REQUESTS,
        help="Minimum number of requests to generate",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=DEFAULT_INTERVAL_MS,
        help="Interval between requests in milliseconds",
    )
    parser.add_argument(
        "--cpu-only-users",
        action="store_true",
        help="Use CPU-optimized user categories (5-35, 45-75, 85-130 tokens)",
    )

    args = parser.parse_args()

    if args.cpu_only_users:
        logger.info("CPU-optimized dataset generation")
        logger.info("Token ranges: SMALL(5-35), MEDIUM(45-75), HIGH(85-130)")
    else:
        logger.info("Standard dataset generation")

    dataset_path = args.sharegpt_path or download_if_not_exists_dataset()
    logger.info(f"Using dataset: {dataset_path}")

    vtc_dataset_path = create_vtc_bench_dataset(
        sharegpt_path=dataset_path,
        output_path=args.output_path,
        min_requests=args.min_requests,
        interval_ms=args.interval_ms,
        use_cpu_categories=args.cpu_only_users,
    )

    logger.info(f"VTC benchmark dataset created: {vtc_dataset_path}")
    if args.cpu_only_users:
        logger.info("Dataset optimized for CPU-only processing")


if __name__ == "__main__":
    main()
