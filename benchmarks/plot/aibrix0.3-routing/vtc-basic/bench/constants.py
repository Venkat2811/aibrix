#!/usr/bin/env python3

from pathlib import Path

# Existing constants for ShareGPT dataset
TEMP_WORKSPACE = Path("temp_workspace")
DATASET_DIR = TEMP_WORKSPACE / "dataset"
SHAREGPT_DATASET_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"
SHAREGPT_DATASET_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

# VTC dataset constants
VTC_DATASET_FILENAME = "vtc_routing_dataset_varied.jsonl"
VTC_DATASET_PATH = DATASET_DIR / VTC_DATASET_FILENAME

# User categories for VTC dataset
USER_CATEGORIES = [
    {
        "name": "small",
        "users": ["user-small-1", "user-small-2"],
        "token_scale": 1.0,
        "min_tokens": 10,
        "max_tokens": 100,
        "target_percentage": 0.33,
    },
    {
        "name": "medium",
        "users": ["user-med-1", "user-med-2"],
        "token_scale": 3.0,
        "min_tokens": 100,
        "max_tokens": 300,
        "target_percentage": 0.33,
    },
    {
        "name": "high",
        "users": ["user-high-1", "user-high-2"],
        "token_scale": 6.0,
        "min_tokens": 300,
        "max_tokens": 800,
        "target_percentage": 0.34,
    },
]

# VTC dataset generation defaults
DEFAULT_MIN_REQUESTS = 1000
DEFAULT_INTERVAL_MS = 10
DEFAULT_SESSION_PROBABILITY = 0.5

# Traffic pattern configurations for benchmarking
TRAFFIC_PATTERNS = {
    "balanced": [
        {"category": "small", "prob": 0.33, "burstiness": 0.1},
        {"category": "medium", "prob": 0.33, "burstiness": 0.2},
        {"category": "high", "prob": 0.34, "burstiness": 0.3},
    ],
    "high_usage": [
        {"category": "small", "prob": 0.2, "burstiness": 0.1},
        {"category": "medium", "prob": 0.3, "burstiness": 0.2},
        {"category": "high", "prob": 0.5, "burstiness": 0.3},
    ],
    "bursty": [
        {"category": "small", "prob": 0.4, "burstiness": 0.6},
        {"category": "medium", "prob": 0.4, "burstiness": 0.7},
        {"category": "high", "prob": 0.2, "burstiness": 0.8},
    ],
}
