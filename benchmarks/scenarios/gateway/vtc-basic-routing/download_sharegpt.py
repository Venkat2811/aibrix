#!/usr/bin/env python3
import logging
import os
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_sharegpt_dataset():
    """
    Download the ShareGPT dataset for benchmark usage.
    """
    # Set paths
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = base_dir / "dataset"
    target_dataset = dataset_dir / "ShareGPT_V3_unfiltered_cleaned_split.json"

    # Create dataset directory if it doesn't exist
    os.makedirs(dataset_dir, exist_ok=True)

    # Check if dataset already exists
    if target_dataset.exists():
        logger.info(f"ShareGPT dataset already exists at {target_dataset}")
        return str(target_dataset)

    # Download dataset
    logger.info(f"Downloading ShareGPT dataset to {target_dataset}")
    download_url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

    try:
        subprocess.run(["wget", download_url, "-O", str(target_dataset)], check=True)
        logger.info("ShareGPT dataset downloaded successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download ShareGPT dataset: {e}")
        raise

    return str(target_dataset)


if __name__ == "__main__":
    download_sharegpt_dataset()
