#!/usr/bin/env python3

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

TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    return len(TOKENIZER.encode(text, disallowed_special=()))


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


def download_if_not_exists_dataset(dataset_file: Optional[str] = None) -> str:
    if not dataset_file:
        dataset_dir = Path(DATASET_DIR)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created temporary workspace at {TEMP_WORKSPACE}")

        dataset_path = dataset_dir / SHAREGPT_DATASET_FILENAME
        if dataset_path.exists():
            print(f"Dataset already exists at {dataset_path}, skipping download")
        else:
            try:
                print(f"Downloading ShareGPT dataset from {SHAREGPT_DATASET_URL}")
                urllib.request.urlretrieve(SHAREGPT_DATASET_URL, dataset_path)
                print(f"Dataset downloaded to {dataset_path}")
            except Exception as e:
                print(f"Failed to download dataset: {e}")
                sys.exit(1)
    else:
        dataset_path = Path(dataset_file)
        if not dataset_path.exists():
            print(f"Error: Dataset file not found at {dataset_file}")
            sys.exit(1)
        print(f"Using specified dataset file: {dataset_file}")

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
    prompts: List[Dict], min_requests: int = DEFAULT_MIN_REQUESTS
) -> List[Dict]:
    prompts_sorted = sorted(prompts, key=lambda x: x["prompt_tokens"])
    categorized_prompts = []
    required_per_category = {
        cat["name"]: int(cat["target_percentage"] * min_requests)
        for cat in USER_CATEGORIES
    }

    for category in USER_CATEGORIES:
        matching_prompts = [
            p
            for p in prompts_sorted
            if category["min_tokens"] <= p["prompt_tokens"] <= category["max_tokens"]
        ]

        if len(matching_prompts) < required_per_category[category["name"]]:
            logger.warning(
                f"Not enough prompts for category {category['name']}. "
                f"Found {len(matching_prompts)}, need {required_per_category[category['name']]}"
            )

        num_to_select = min(
            len(matching_prompts), required_per_category[category["name"]]
        )
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

    if len(categorized_prompts) < min_requests:
        additional_needed = min_requests - len(categorized_prompts)
        logger.warning(
            f"Duplicating {additional_needed} prompts to reach {min_requests}"
        )

        for _ in range(additional_needed):
            original = random.choice(categorized_prompts)
            duplicate = original.copy()
            category_info = next(
                cat
                for cat in USER_CATEGORIES
                if cat["name"] == duplicate["user_category"]
            )
            users = [u for u in category_info["users"] if u != duplicate["user"]]
            if users:
                duplicate["user"] = random.choice(users)
            categorized_prompts.append(duplicate)

    random.shuffle(categorized_prompts)

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

    # Print user distribution stats for each category
    for category, users in user_counts.items():
        tokens = category_tokens[category]
        avg_tokens = sum(tokens) / len(tokens) if tokens else 0
        min_tokens = min(tokens) if tokens else 0
        max_tokens = max(tokens) if tokens else 0
        print(f"\n{category.upper()} category user distribution:")
        print(
            f"  Token length stats - Min: {min_tokens}, Max: {max_tokens}, Avg: {avg_tokens:.1f}"
        )
        for user, count in users.items():
            print(f"  {user}: {count} requests")

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
) -> str:
    if not sharegpt_path:
        sharegpt_path = download_if_not_exists_dataset()
    if not output_path:
        output_path = str(VTC_DATASET_PATH)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sharegpt_data = load_sharegpt_data(sharegpt_path)
    prompts_with_tokens = extract_prompts_with_tokens(sharegpt_data)

    token_counts = [p["prompt_tokens"] for p in prompts_with_tokens]
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    avg_tokens = np.mean(token_counts)
    logger.info(
        f"Token length stats - Min: {min_tokens}, Max: {max_tokens}, Avg: {avg_tokens:.1f}, "
        f"Median: {np.median(token_counts):.1f}"
    )

    categorized_prompts = assign_user_categories(prompts_with_tokens, min_requests)
    workload = create_workload_format(categorized_prompts, interval_ms)
    save_vtc_dataset(workload, output_path)

    logger.info(
        f"Generated VTC dataset: {len(workload)} entries, {len(categorized_prompts)} requests"
    )
    return output_path


if __name__ == "__main__":
    print("=== Downloading ShareGPT Dataset ===")
    dataset_path = download_if_not_exists_dataset(
        "temp_workspace/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
    )
    print(f"Dataset ready at: {dataset_path}")

    print("\n=== Creating VTC Benchmark Dataset ===")
    vtc_dataset_path = create_vtc_bench_dataset(sharegpt_path=dataset_path)
    print(f"VTC benchmark dataset created at: {vtc_dataset_path}")
