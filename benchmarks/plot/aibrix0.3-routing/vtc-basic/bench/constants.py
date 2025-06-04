#!/usr/bin/env python3

from pathlib import Path

# Existing constants for ShareGPT dataset
TEMP_WORKSPACE = Path("temp_workspace")
DATASET_DIR = TEMP_WORKSPACE / "dataset"
SHAREGPT_DATASET_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"
SHAREGPT_DATASET_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

# VTC dataset constants
VTC_DATASET_FILENAME = "vtc_routing_dataset_varied_cpu_optimized.jsonl"
VTC_DATASET_PATH = DATASET_DIR / VTC_DATASET_FILENAME

# User categories for VTC dataset - Updated for CPU-optimized processing
USER_CATEGORIES = [
    {
        "name": "sm-small",
        "users": ["sm-user-small-1", "sm-user-small-2", "sm-user-small-3"],
        "token_scale": 0.5,
        "min_tokens": 5,
        "max_tokens": 25,
        "target_percentage": 0.4,  # 40% of requests
    },
    {
        "name": "sm-medium",
        "users": ["sm-user-med-1", "sm-user-med-2", "sm-user-med-3"],
        "token_scale": 1.0,
        "min_tokens": 26,
        "max_tokens": 50,
        "target_percentage": 0.35,  # 35% of requests
    },
    {
        "name": "sm-high",
        "users": ["sm-user-high-1", "sm-user-high-2", "sm-user-high-3"],
        "token_scale": 1.5,
        "min_tokens": 51,
        "max_tokens": 80,
        "target_percentage": 0.25,  # 25% of requests
    },
]

# VTC dataset generation defaults
DEFAULT_MIN_REQUESTS = 1000
DEFAULT_INTERVAL_MS = 10
DEFAULT_SESSION_PROBABILITY = 0.5

# Traffic pattern configurations for benchmarking - CPU-optimized
TRAFFIC_PATTERNS = {
    "balanced": [
        {"category": "sm-small", "prob": 0.4, "burstiness": 0.1},
        {"category": "sm-medium", "prob": 0.35, "burstiness": 0.15},
        {"category": "sm-high", "prob": 0.25, "burstiness": 0.2},
    ],
    "high_usage": [
        {"category": "sm-small", "prob": 0.3, "burstiness": 0.1},
        {"category": "sm-medium", "prob": 0.4, "burstiness": 0.15},
        {"category": "sm-high", "prob": 0.3, "burstiness": 0.2},
    ],
    "bursty": [
        {"category": "sm-small", "prob": 0.5, "burstiness": 0.4},
        {"category": "sm-medium", "prob": 0.3, "burstiness": 0.5},
        {"category": "sm-high", "prob": 0.2, "burstiness": 0.6},
    ],
    "high_med_pressure": [
        {
            "category": "sm-small",
            "prob": 0.15,
            "burstiness": 0.1,
        },  # Only 15% small users (vulnerable)
        {
            "category": "sm-medium",
            "prob": 0.50,
            "burstiness": 0.7,
        },  # 50% medium users with high burst
        {
            "category": "sm-high",
            "prob": 0.35,
            "burstiness": 0.8,
        },  # 35% high users with very high burst
    ],
}
