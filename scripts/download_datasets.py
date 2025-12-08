#!/usr/bin/env python3
"""
Download all benchmark datasets for offline use.

This script pre-downloads datasets from HuggingFace to your local cache,
so future benchmark runs don't require network access.
"""

from datasets import load_dataset
import structlog

logger = structlog.get_logger()

DATASETS = {
    # === EVALUATION DATASETS ===
    "satml": {
        "path": "ethz-spylab/ctf-satml24",
        "name": "interaction_chats",
        "split": "attack",
        "description": "SaTML CTF 2024 - Adaptive attacks"
    },
    "deepset": {
        "path": "deepset/prompt-injections",
        "split": "train",
        "description": "Deepset prompt injections (mixed)"
    },
    "llmail": {
        "path": "microsoft/llmail-inject-challenge",
        "split": "Phase1",
        "description": "LLMail indirect injections"
    },
    "notinject_hf": {
        "path": "leolee99/NotInject",
        "split": "NotInject_one",  # Also has NotInject_two, NotInject_three
        "description": "NotInject - Benign samples with trigger words"
    },
    # Note: BrowseSafe may require different path or be private
    # "browsesafe": {
    #     "path": "perplexity-ai/browsesafe",
    #     "split": "test",
    #     "description": "BrowseSafe - HTML-embedded attacks (Dec 2025)"
    # },
    "tensortrust": {
        "path": "tensortrust/tensortrust-data",
        "split": "test",
        "description": "TensorTrust - Human-generated adversarial examples"
    },
    # === TRAINING DATA SOURCES ===
    "sharegpt": {
        "path": "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "split": "train",
        "description": "ShareGPT - Real user conversations (for benign training)"
    },
    "lmsys": {
        "path": "lmsys/lmsys-chat-1m",
        "split": "train",
        "description": "LMSYS Chat 1M - Diverse ChatGPT conversations"
    }
}

def download_all():
    """Download all datasets to local cache."""
    print("="*60)
    print("DOWNLOADING BENCHMARK DATASETS")
    print("="*60)
    
    for name, config in DATASETS.items():
        print(f"\n[{name}] Downloading...")
        try:
            if "name" in config:
                ds = load_dataset(
                    config["path"],
                    config["name"],
                    split=config["split"],
                    streaming=False  # Force full download
                )
            else:
                ds = load_dataset(
                    config["path"],
                    split=config["split"],
                    streaming=False
                )
            print(f"    ✓ Downloaded {len(ds)} samples")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("Datasets cached in: ~/.cache/huggingface/datasets/")
    print("="*60)

if __name__ == "__main__":
    download_all()
