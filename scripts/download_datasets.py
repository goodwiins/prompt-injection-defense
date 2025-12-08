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
    # === EVALUATION DATASETS (Paper Table) ===
    "satml": {
        "path": "ethz-spylab/ctf-satml24",
        "name": "interaction_chats",
        "split": "attack",
        "description": "SaTML CTF 2024 - 300 adaptive attacks"
    },
    "deepset": {
        "path": "deepset/prompt-injections",
        "split": "train",
        "description": "Deepset - 518 injection samples for recall"
    },
    "llmail": {
        "path": "microsoft/llmail-inject-challenge",
        "split": "Phase1",
        "description": "LLMail - 200 indirect injection scenarios"
    },
    "notinject_hf": {
        "path": "leolee99/NotInject",
        "split": "NotInject_one",  # Also has NotInject_two, NotInject_three
        "description": "NotInject - 339 benign samples with trigger words"
    },
    "browsesafe": {
        "path": "perplexity-ai/browsesafe-bench",
        "split": "test",
        "description": "BrowseSafe-Bench - 14,719 web-based attacks (Perplexity Dec 2025)"
    },
    # Note: AgentDojo requires pip install agentdojo, not direct HF download
    # "agentdojo": {
    #     "path": "ethz-spylab/agentdojo",
    #     "split": "test",
    #     "description": "AgentDojo - 97 multi-agent workflows"
    # },
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
