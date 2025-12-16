#!/usr/bin/env python3
"""
Optimize threshold for all-mpnet-base-v2 model.

Finds the optimal classification threshold that maximizes performance
while keeping FPR within acceptable bounds.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

import structlog
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve

from src.detection.embedding_classifier import EmbeddingClassifier
from benchmarks.benchmark_datasets import (
    load_satml_dataset,
    load_deepset_dataset,
    load_notinject_hf_dataset,
    load_llmail_dataset,
    load_browsesafe_dataset
)

logger = structlog.get_logger()


def evaluate_threshold(y_true, y_scores, threshold):
    """Evaluate metrics at a specific threshold."""
    y_pred = (y_scores >= threshold).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'fnr': fnr,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def main():
    print("=" * 70)
    print("THRESHOLD OPTIMIZATION FOR all-mpnet-base-v2")
    print("=" * 70)
    print()

    # Load model
    print("1️⃣ Loading all-mpnet-base-v2 model...")
    classifier = EmbeddingClassifier()
    classifier.load_model('models/all-mpnet-base-v2_classifier.json')
    print(f"   ✓ Current threshold: {classifier.threshold}")
    print()

    # Load datasets for threshold optimization
    print("2️⃣ Loading evaluation datasets...")

    print("   Loading SaTML (300 samples)...")
    satml = load_satml_dataset(limit=300)

    print("   Loading deepset (400 samples)...")
    deepset = load_deepset_dataset(limit=400)

    print("   Loading NotInject (339 samples)...")
    notinject = load_notinject_hf_dataset(limit=339)

    print("   Loading LLMail (200 samples)...")
    llmail = load_llmail_dataset(limit=200)

    print("   Loading BrowseSafe (500 samples)...")
    browsesafe = load_browsesafe_dataset(limit=500)
    print()

    # Combine all datasets
    all_texts = []
    all_labels = []

    for dataset in [satml, deepset, notinject, llmail, browsesafe]:
        all_texts.extend(dataset.texts)
        all_labels.extend(dataset.labels)

    all_labels = np.array(all_labels)

    print(f"3️⃣ Getting predictions on {len(all_texts)} samples...")
    print(f"   - Injections: {np.sum(all_labels == 1)}")
    print(f"   - Safe: {np.sum(all_labels == 0)}")
    print()

    # Get probability scores
    all_scores = classifier.predict_proba(all_texts)[:, 1]

    # Calculate AUC
    auc = roc_auc_score(all_labels, all_scores)
    print(f"   Overall AUC: {auc:.4f}")
    print()

    # Sweep thresholds
    print("4️⃣ Sweeping thresholds from 0.3 to 0.9...")
    print()

    thresholds = np.arange(0.3, 0.91, 0.05)
    results = []

    print(f"{'Threshold':>10} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>8} | {'F1':>8} | {'FPR':>8}")
    print("-" * 75)

    for thresh in thresholds:
        result = evaluate_threshold(all_labels, all_scores, thresh)
        results.append(result)

        print(f"{result['threshold']:>10.2f} | "
              f"{result['accuracy']:>8.1%} | "
              f"{result['precision']:>9.1%} | "
              f"{result['recall']:>8.1%} | "
              f"{result['f1']:>8.1%} | "
              f"{result['fpr']:>8.1%}")

    print()

    # Find optimal thresholds for different objectives
    print("5️⃣ Finding optimal thresholds...")
    print()

    # Objective 1: Max F1 with FPR ≤ 5%
    valid_results = [r for r in results if r['fpr'] <= 0.05]
    if valid_results:
        best_f1 = max(valid_results, key=lambda r: r['f1'])
        print("📊 Optimal for MAX F1 (FPR ≤ 5%):")
        print(f"   Threshold: {best_f1['threshold']:.3f}")
        print(f"   F1: {best_f1['f1']:.1%}")
        print(f"   Accuracy: {best_f1['accuracy']:.1%}")
        print(f"   Precision: {best_f1['precision']:.1%}")
        print(f"   Recall: {best_f1['recall']:.1%}")
        print(f"   FPR: {best_f1['fpr']:.1%}")
        print()

    # Objective 2: Max Recall with FPR ≤ 5%
    if valid_results:
        best_recall = max(valid_results, key=lambda r: r['recall'])
        print("📊 Optimal for MAX RECALL (FPR ≤ 5%):")
        print(f"   Threshold: {best_recall['threshold']:.3f}")
        print(f"   Recall: {best_recall['recall']:.1%}")
        print(f"   F1: {best_recall['f1']:.1%}")
        print(f"   Accuracy: {best_recall['accuracy']:.1%}")
        print(f"   FPR: {best_recall['fpr']:.1%}")
        print()

    # Objective 3: Balanced (closest to FPR = 2.5% and high F1)
    target_fpr_results = sorted(results, key=lambda r: abs(r['fpr'] - 0.025))[:5]
    best_balanced = max(target_fpr_results, key=lambda r: r['f1'])
    print("📊 Optimal for BALANCED (FPR ≈ 2.5%, Max F1):")
    print(f"   Threshold: {best_balanced['threshold']:.3f}")
    print(f"   F1: {best_balanced['f1']:.1%}")
    print(f"   Accuracy: {best_balanced['accuracy']:.1%}")
    print(f"   Recall: {best_balanced['recall']:.1%}")
    print(f"   FPR: {best_balanced['fpr']:.1%}")
    print()

    # Objective 4: Max Accuracy
    best_acc = max(results, key=lambda r: r['accuracy'])
    print("📊 Optimal for MAX ACCURACY:")
    print(f"   Threshold: {best_acc['threshold']:.3f}")
    print(f"   Accuracy: {best_acc['accuracy']:.1%}")
    print(f"   F1: {best_acc['f1']:.1%}")
    print(f"   Recall: {best_acc['recall']:.1%}")
    print(f"   FPR: {best_acc['fpr']:.1%}")
    print()

    # Recommendation
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    recommended = best_f1 if valid_results else best_balanced

    print(f"✅ Recommended threshold: {recommended['threshold']:.3f}")
    print()
    print("   Expected Performance:")
    print(f"   - Accuracy: {recommended['accuracy']:.1%}")
    print(f"   - Precision: {recommended['precision']:.1%}")
    print(f"   - Recall: {recommended['recall']:.1%}")
    print(f"   - F1: {recommended['f1']:.1%}")
    print(f"   - FPR: {recommended['fpr']:.1%}")
    print()
    print(f"   Current threshold: {classifier.threshold}")
    if abs(recommended['threshold'] - classifier.threshold) > 0.01:
        print(f"   ⚠️  Recommended to update threshold from {classifier.threshold} to {recommended['threshold']:.3f}")
    else:
        print(f"   ✓ Current threshold is already optimal")
    print()

    # Save results
    output = {
        'model': 'all-mpnet-base-v2',
        'current_threshold': classifier.threshold,
        'recommended_threshold': recommended['threshold'],
        'auc': auc,
        'all_results': results,
        'optimal': {
            'max_f1_fpr5': best_f1 if valid_results else None,
            'max_recall_fpr5': best_recall if valid_results else None,
            'balanced': best_balanced,
            'max_accuracy': best_acc,
            'recommended': recommended
        }
    }

    output_path = 'results/mpnet_threshold_optimization.json'
    Path('results').mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"📁 Results saved to: {output_path}")
    print()


if __name__ == '__main__':
    main()
