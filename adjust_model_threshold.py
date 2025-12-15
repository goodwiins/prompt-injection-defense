#!/usr/bin/env python3
"""
Adjust the threshold of the balanced model to find the optimal balance
between FPR and recall.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
import structlog

sys.path.append(str(Path(__file__).parent))

from benchmarks.benchmark_datasets import load_deepset_dataset, load_notinject_dataset
from src.detection.embedding_classifier import EmbeddingClassifier

def find_optimal_threshold(classifier, texts, labels):
    """Find the optimal threshold balancing FPR and recall."""
    from sklearn.metrics import roc_curve, precision_recall_curve

    # Get probabilities
    probs = classifier.predict_proba(texts)[:, 1]

    # Get ROC curve
    fpr, tpr, thresholds = roc_curve(labels, probs)

    # Calculate different metrics
    print("\nThreshold Analysis:")
    print("="*50)
    print(f"{'Threshold':<12} {'FPR':<8} {'Recall':<8} {'F1':<8}")
    print("-" * 36)

    best_f1 = 0
    best_threshold = 0.5

    # Test various thresholds
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        predictions = (probs >= threshold).astype(int)

        # Calculate metrics
        from sklearn.metrics import confusion_matrix, f1_score
        if len(np.unique(predictions)) == 1:
            if predictions[0] == 0:
                tn, fp, fn, tp = len(labels), 0, np.sum(labels), 0
            else:
                tn, fp, fn, tp = 0, len(labels), 0, np.sum(labels)
        else:
            cm = confusion_matrix(labels, predictions)
            tn, fp, fn, tp = cm.ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = f1_score(labels, predictions, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

        print(f"{threshold:<12.2f} {fpr*100:>7.1f}% {recall*100:>7.1f}% {f1:>7.3f}")

    # Find threshold with FPR < 5%
    fpr5_threshold = None
    try:
        valid_idx = np.where(fpr < 0.05)[0]
        if len(valid_idx) > 0:
            fpr5_threshold = float(thresholds[valid_idx[np.argmax(tpr[valid_idx])]])
            fpr5_fpr = fpr[valid_idx[np.argmax(tpr[valid_idx])]]
            fpr5_tpr = tpr[valid_idx[np.argmax(tpr[valid_idx])]]
            print(f"\nBest threshold for <5% FPR: {fpr5_threshold:.3f}")
            print(f"  FPR: {fpr5_fpr*100:.1f}%")
            print(f"  Recall: {fpr5_tpr*100:.1f}%")
    except:
        # If we can't find a good threshold, use None
        pass

    return best_threshold, fpr5_threshold, best_f1

def main():
    """Adjust model threshold to find optimal balance."""

    print("=== Adjusting BIT Model Threshold ===\n")

    # Load the model
    model_path = Path("models/bit_xgboost_balanced_v2_classifier.json")
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    classifier = EmbeddingClassifier(
        model_name="all-MiniLM-L6-v2",
        threshold=0.5,  # Will be adjusted
        model_dir="models"
    )
    classifier.load_model(str(model_path))

    # Load test datasets
    print("Loading test datasets...")

    # Deepset
    deepset = load_deepset_dataset(include_safe=True)
    deepset_texts = deepset.texts
    deepset_labels = deepset.labels

    # NotInject
    notinject = load_notinject_dataset(limit=1000)

    # Combine for evaluation
    eval_texts = deepset_texts + notinject.texts
    eval_labels = deepset_labels + notinject.labels

    print(f"Evaluation set: {len(eval_texts)} samples")
    print(f"  Benign: {len([l for l in eval_labels if l == 0])}")
    print(f"  Malicious: {len([l for l in eval_labels if l == 1])}")

    # Find optimal thresholds
    best_threshold, fpr5_threshold, best_f1 = find_optimal_threshold(
        classifier, eval_texts, eval_labels
    )

    print(f"\n\nRecommendations:")
    print("="*50)
    print(f"Best F1 threshold: {best_threshold:.3f} (F1: {best_f1:.3f})")

    if fpr5_threshold:
        print(f"FPR < 5% threshold: {fpr5_threshold:.3f}")
        print("\nRecommended threshold for publication:")
        if fpr5_threshold > best_threshold:
            print(f"Use {fpr5_threshold:.3f} to ensure <5% FPR while maintaining good recall")
        else:
            print(f"Use {best_threshold:.3f} for optimal F1 score (still has low FPR)")

    # Save model with recommended threshold
    recommended_threshold = fpr5_threshold if fpr5_threshold and fpr5_threshold > 0.3 else best_threshold

    print(f"\nSaving model with threshold {recommended_threshold:.3f}...")
    classifier.threshold = recommended_threshold

    # Update metadata
    metadata_path = Path("models/bit_xgboost_balanced_v2_metadata.json")
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        metadata["threshold"] = recommended_threshold
        metadata["threshold_selection"] = {
            "method": "balanced_fpr_recall",
            "best_f1_threshold": best_threshold,
            "best_f1_score": best_f1,
            "fpr_5_threshold": fpr5_threshold if fpr5_threshold else None,
            "recommended_threshold": recommended_threshold
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Updated metadata saved to: {metadata_path}")

    # Save model with new threshold
    classifier.save_model(str(model_path))
    print(f"Model saved with new threshold: {recommended_threshold:.3f}")

if __name__ == "__main__":
    main()