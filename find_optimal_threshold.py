#!/usr/bin/env python3
"""Find the optimal threshold for the balanced model."""

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))

from benchmarks.benchmark_datasets import load_deepset_dataset
from src.detection.embedding_classifier import EmbeddingClassifier

# Load the model
classifier = EmbeddingClassifier()
classifier.load_model('models/bit_xgboost_balanced_v2_classifier.json')

# Load test data with limited samples
print("Loading test data...")
benign_dataset = load_deepset_dataset(include_injections=False, include_safe=True, limit=200)
injection_dataset = load_deepset_dataset(include_injections=True, include_safe=False, limit=200)

# Get embeddings and probabilities (BenchmarkDataset returns tuples of (text, label))
benign_texts = [item[0] for item in benign_dataset]
injection_texts = [item[0] for item in injection_dataset]

print(f"Getting probabilities for {len(benign_texts)} benign samples...")
benign_probs = classifier.predict_proba(benign_texts)[:, 1]

print(f"Getting probabilities for {len(injection_texts)} injection samples...")
injection_probs = classifier.predict_proba(injection_texts)[:, 1]

# Find stats
print(f"\nBenign probabilities - Min: {benign_probs.min():.4f}, Max: {benign_probs.max():.4f}, Mean: {benign_probs.mean():.4f}")
print(f"Injection probabilities - Min: {injection_probs.min():.4f}, Max: {injection_probs.max():.4f}, Mean: {injection_probs.mean():.4f}")

# Test different thresholds
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

print("\nThreshold Analysis:")
print("Threshold | FPR     | Recall  | F1")
print("-" * 40)

for threshold in thresholds:
    benign_pred = (benign_probs >= threshold).sum()
    injection_pred = (injection_probs >= threshold).sum()

    fpr = benign_pred / len(benign_probs)
    recall = injection_pred / len(injection_probs)

    # Calculate F1
    tp = injection_pred
    fp = benign_pred
    fn = len(injection_texts) - injection_pred

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"{threshold:9.1f} | {fpr:7.1%} | {recall:7.1%} | {f1:6.1%}")

# Find threshold with FPR < 5% and max recall
print("\nFinding optimal threshold with FPR < 5%...")
best_threshold = 0.5
best_recall = 0

for threshold in np.linspace(0.1, 0.9, 81):
    fpr = (benign_probs >= threshold).sum() / len(benign_probs)
    recall = (injection_probs >= threshold).sum() / len(injection_probs)

    if fpr < 0.05 and recall > best_recall:
        best_recall = recall
        best_threshold = threshold

print(f"\n✅ Best threshold: {best_threshold:.3f}")
print(f"   Expected FPR: {(benign_probs >= best_threshold).sum() / len(benign_probs):.1%}")
print(f"   Expected Recall: {(injection_probs >= best_threshold).sum() / len(injection_probs):.1%}")