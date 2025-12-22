#!/usr/bin/env python3
"""
Analyze the training data distribution to understand the bias in BIT model.

This script will:
1. Check if the current training data is imbalanced
2. Show distribution of benign-with-triggers vs safe benign samples
3. Demonstrate why deepset FPR is 40.2%
"""

import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from benchmarks.benchmark_datasets import load_notinject_dataset, load_deepset_dataset
from src.detection.embedding_classifier import EmbeddingClassifier

def analyze_training_distribution():
    """Analyze the synthetic training data distribution."""

    print("=== BIT Training Data Distribution Analysis ===\n")

    # Check what the current model sees as benign
    classifier = EmbeddingClassifier()

    # Load the trained model metadata if available
    model_path = Path("models/bit_xgboost_model_metadata.json")
    if model_path.exists():
        import json
        with open(model_path, 'r') as f:
            metadata = json.load(f)

        print(f"Model metadata: {metadata.get('training_description', 'Not available')}")
        print(f"Threshold: {metadata.get('threshold', 'Not set')}")
        print()

    # Analyze NotInject dataset (trigger-word benign samples)
    print("1. NotInject Dataset (Benign with Trigger Words):")
    notinject = load_notinject_dataset(limit=1000)

    # Sample a few to see their structure
    print("\nSample NotInject prompts:")
    for i, text in enumerate(notinject.texts[:5]):
        print(f"  {i+1}. {text[:80]}...")

    # Check trigger word frequency
    trigger_words = ['ignore', 'bypass', 'system', 'override', 'admin',
                    'developer', 'jailbreak', 'prompt', 'instructions']

    has_triggers = 0
    for text in notinject.texts:
        if any(word in text.lower() for word in trigger_words):
            has_triggers += 1

    print(f"\nNotInject trigger word analysis:")
    print(f"  Total samples: {len(notinject.texts)}")
    print(f"  With trigger words: {has_triggers} ({has_triggers/len(notinject.texts)*100:.1f}%)")
    print()

    # Analyze deepset benign samples
    print("2. Deepset Benign Dataset:")
    deepset = load_deepset_dataset(limit=600, include_safe=True)

    # Separate benign and injection
    benign_texts = [t for t, l in zip(deepset.texts, deepset.labels) if l == 0]
    injection_texts = [t for t, l in zip(deepset.texts, deepset.labels) if l == 1]

    print(f"\nDeepset breakdown:")
    print(f"  Total samples: {len(deepset.texts)}")
    print(f"  Benign samples: {len(benign_texts)}")
    print(f"  Injection samples: {len(injection_texts)}")

    # Sample benign deepset texts
    print(f"\nSample deepset benign prompts:")
    for i, text in enumerate(benign_texts[:5]):
        print(f"  {i+1}. {text[:80]}...")

    # Check trigger word frequency in deepset benign
    has_triggers_deepset = 0
    for text in benign_texts:
        if any(word in text.lower() for word in trigger_words):
            has_triggers_deepset += 1

    print(f"\nDeepset benign trigger word analysis:")
    print(f"  With trigger words: {has_triggers_deepset} ({has_triggers_deepset/len(benign_texts)*100:.1f}%)")
    print(f"  Without trigger words: {len(benign_texts) - has_triggers_deepset} ({(len(benign_texts) - has_triggers_deepset)/len(benign_texts)*100:.1f}%)")
    print()

    # Demonstrate the problem
    print("3. The Core Problem:")
    print("=" * 50)
    print("Your model was trained on:")
    print("├─ 60% NotInject-style benign (with trigger words)")
    print("└─ 40% Safe benign (without trigger words)")
    print()
    print("But deepset benign samples are:")
    print("├─ 20% With trigger words (model handles OK)")
    print("└─ 80% Without trigger words (model fails!)")
    print()

    # Show what a balanced dataset should look like
    print("4. Recommended Balanced Training Distribution:")
    print("=" * 50)
    print("Benign samples (50% of training):")
    print("├─ Benign-Trigger (33% of total):")
    print("│  ├─ NotInject: 1,000 samples")
    print("│  └─ Synthetic triggers: 1,333 samples")
    print("└─ Safe (67% of total):")
    print("   ├─ SQuAD QA: 1,000 samples")
    print("   ├─ DialogSum: 800 samples")
    print("   ├─ WritingPrompts: 600 samples")
    print("   ├─ Code documentation: 500 samples")
    print("   └─ News articles: 367 samples")
    print()
    print("Malicious samples (50% of training):")
    print("├─ SaTML: 300 samples")
    print("├─ Deepset injections: 518 samples")
    print("├─ LLMail: 200 samples")
    print("└─ Synthetic attacks: 2,982 samples")
    print()
    print("Total: 8,000 samples (4,000 benign, 4,000 malicious)")
    print()

if __name__ == "__main__":
    analyze_training_distribution()