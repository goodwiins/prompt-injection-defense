#!/usr/bin/env python3
"""
Evaluate the DistilBERT-based BIT model.

This script evaluates the DistilBERT model and compares it with the MiniLM baseline.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import structlog

sys.path.append(str(Path(__file__).parent))

from benchmarks.benchmark_datasets import (
    load_satml_dataset,
    load_deepset_dataset,
    load_llmail_dataset,
    load_notinject_dataset
)
from src.detection.embedding_classifier import EmbeddingClassifier
from src.detection.html_preprocessor import preprocess_for_detection

logger = structlog.get_logger()

def evaluate_classifier(
    classifier: EmbeddingClassifier,
    dataset_name: str,
    texts: List[str],
    labels: List[int],
    preprocess: bool = False,
    sample_size: int = None
) -> Dict:
    """Evaluate classifier on a dataset (with optional sampling)."""

    # Sample if requested
    if sample_size and len(texts) > sample_size:
        indices = np.random.choice(len(texts), sample_size, replace=False)
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]

    print(f"\nEvaluating on {dataset_name}...")
    print(f"  Samples: {len(texts)}")

    # Preprocess if needed
    if preprocess:
        processed_texts = []
        for text in texts:
            processed = preprocess_for_detection(text, source_type="auto")
            processed_texts.append(processed)
        texts = processed_texts

    # Get predictions
    start_time = time.time()
    probs = classifier.predict_proba(texts)
    predictions = classifier.predict(texts)
    duration = time.time() - start_time

    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    # Basic metrics - handle single class case
    if len(np.unique(labels)) == 1:
        # All samples are the same class
        unique_label = np.unique(labels)[0]
        if unique_label == 0:
            # All benign
            tn = len(labels) - np.sum(predictions)
            fp = np.sum(predictions)
            fn = 0
            tp = 0
        else:
            # All malicious
            tn = 0
            fp = 0
            fn = len(predictions) - np.sum(predictions)
            tp = np.sum(predictions)
    else:
        # Multi-class case
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel()

    # Detailed metrics
    accuracy = (tp + tn) / len(labels)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    # AUC (only if we have both classes)
    auc = roc_auc_score(labels, probs[:, 1]) if len(np.unique(labels)) > 1 else 0.0

    results = {
        "dataset": dataset_name,
        "samples": len(texts),
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "auc": auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "duration_ms": duration * 1000,
        "threshold": float(classifier.threshold)
    }

    # Print summary
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  FPR: {fpr:.4f} ({fpr*100:.1f}%)")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")

    return results

def main():
    """Run evaluation on all benchmarks."""

    print("=== DistilBERT BIT Model Evaluation ===")
    print("Model: bit_distilbert_balanced_classifier.json")
    print("Base Model: distilbert-base-uncased")
    print("Training: 50% benign, 50% malicious (properly balanced)")
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # Load the DistilBERT model
    model_path = Path("models/bit_distilbert_balanced_classifier.json")
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please run: python train_distilbert_model.py")
        return

    # Load model metadata to get threshold
    metadata_path = Path("models/bit_distilbert_balanced_metadata.json")
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        threshold = metadata.get('threshold', 0.5)
    else:
        threshold = 0.5

    classifier = EmbeddingClassifier(
        model_name="distilbert-base-uncased",
        threshold=threshold,
        model_dir="models"
    )

    # Load the model
    classifier.load_model(str(model_path))
    print(f"Model loaded with threshold: {threshold:.3f}\n")

    # Load datasets
    print("Loading evaluation datasets...")

    # 1. deepset (full dataset - it's manageable)
    print("\nLoading deepset dataset...")
    deepset = load_deepset_dataset(include_safe=True)

    # Separate benign and injection
    deepset_benign_texts = [t for t, l in zip(deepset.texts, deepset.labels) if l == 0]
    deepset_benign_labels = [0] * len(deepset_benign_texts)
    deepset_injection_texts = [t for t, l in zip(deepset.texts, deepset.labels) if l == 1]
    deepset_injection_labels = [1] * len(deepset_injection_texts)

    # 2. NotInject (sample 1000)
    notinject = load_notinject_dataset(limit=1000)

    # 3. SaTML (sample 1000 from the large dataset)
    try:
        satml_full = load_satml_dataset()
        satml_indices = np.random.choice(len(satml_full.texts), 1000, replace=False)
        satml_texts = [satml_full.texts[i] for i in satml_indices]
        satml_labels = [satml_full.labels[i] for i in satml_indices]
    except:
        print("Warning: Could not load SaTML dataset")
        satml_texts = []
        satml_labels = []

    # Evaluate on each dataset
    all_results = []

    # deepset benign (CRITICAL - this was 40.2% FPR with original)
    print("\n" + "="*50)
    print("CRITICAL TEST: deepset benign FPR")
    print("="*50)
    deepset_benign_results = evaluate_classifier(
        classifier, "deepset_benign", deepset_benign_texts, deepset_benign_labels
    )
    all_results.append(deepset_benign_results)

    # Check if we fixed the issue
    if deepset_benign_results["fpr"] < 0.05:
        print(f"\n✅ SUCCESS: deepset benign FPR is now {deepset_benign_results['fpr']*100:.1f}% (<5%)")
    else:
        print(f"\n❌ FAILED: deepset benign FPR is still {deepset_benign_results['fpr']*100:.1f}% (≥5%)")

    # Other evaluations
    print("\n" + "="*50)
    print("Other Dataset Evaluations")
    print("="*50)

    # deepset injections
    if len(deepset_injection_texts) > 0:
        deepset_injection_results = evaluate_classifier(
            classifier, "deepset_injections", deepset_injection_texts, deepset_injection_labels
        )
        all_results.append(deepset_injection_results)

    # NotInject
    notinject_results = evaluate_classifier(
        classifier, "NotInject", notinject.texts, notinject.labels
    )
    all_results.append(notinject_results)

    # SaTML
    if satml_texts:
        satml_results = evaluate_classifier(
            classifier, "SaTML", satml_texts, satml_labels
        )
        all_results.append(satml_results)

    # Overall metrics
    print("\n" + "="*50)
    print("Overall Performance")
    print("="*50)

    # Combine all datasets for overall
    all_texts = deepset_benign_texts + deepset_injection_texts + notinject.texts
    all_labels = deepset_benign_labels + deepset_injection_labels + notinject.labels

    if satml_texts:
        all_texts.extend(satml_texts)
        all_labels.extend(satml_labels)

    overall_results = evaluate_classifier(
        classifier, "Overall", all_texts, all_labels
    )
    all_results.append(overall_results)

    # Load MiniLM results for comparison
    minilm_results_file = Path("results/balanced_v2_evaluation.json")
    minilm_results = None
    if minilm_results_file.exists():
        with open(minilm_results_file, 'r') as f:
            minilm_results = json.load(f)

    # Print comparison table
    print("\n" + "="*50)
    print("Performance Comparison: DistilBERT vs MiniLM")
    print("="*50)

    if minilm_results:
        print(f"{'Dataset':<20} {'DistilBERT FPR':<15} {'MiniLM FPR':<12} {'Winner':<10}")
        print("-" * 58)

        for result in all_results:
            dataset = result["dataset"]
            if dataset in ["deepset_injections"]:
                continue  # Skip injection-only for FPR comparison

            distilbert_fpr = result["fpr"] * 100

            # Get MiniLM result
            minilm_fpr = minilm_results['datasets'].get(dataset, {}).get('fpr', 0) * 100

            # Determine winner
            if distilbert_fpr < minilm_fpr:
                winner = "DistilBERT"
            elif minilm_fpr < distilbert_fpr:
                winner = "MiniLM"
            else:
                winner = "Tie"

            print(f"{dataset:<20} {distilbert_fpr:>13.1f}% {minilm_fpr:>12.1f}% {winner:<10}")
    else:
        # Just show DistilBERT results
        print(f"{'Dataset':<20} {'FPR':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 56)

        for result in all_results:
            dataset = result["dataset"]
            fpr = result["fpr"] * 100
            recall = result["recall"] * 100 if result["recall"] > 0 else 0
            f1 = result["f1"]

            print(f"{dataset:<20} {fpr:>10.1f}% {recall:>10.1f}% {f1:>11.3f}")

    # Save results
    results_file = Path("results/distilbert_evaluation.json")
    results_file.parent.mkdir(exist_ok=True)

    save_data = {
        "model": "bit_distilbert_balanced",
        "base_model": "distilbert-base-uncased",
        "threshold": threshold,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": {r["dataset"]: r for r in all_results},
        "training_info": {
            "safe_benign_samples": 1333,
            "benign_with_triggers": 667,
            "malicious_samples": 2000,
            "total_samples": 4000,
            "benign_ratio": 0.5,
            "malicious_ratio": 0.5
        },
        "comparison_with_minilm": {
            "available": minilm_results is not None,
            "minilm_file": str(minilm_results_file) if minilm_results else None
        }
    }

    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Summary
    print("\n" + "="*50)
    print("Summary")
    print("="*50)

    print("\nDistilBERT Model Results:")
    print(f"✅ deepset benign FPR: {deepset_benign_results['fpr']*100:.1f}%")
    print(f"✅ Overall FPR: {overall_results['fpr']*100:.1f}%")
    print(f"✅ Overall Recall: {overall_results['recall']*100:.1f}%")
    print(f"✅ Overall F1: {overall_results['f1']:.3f}")

    if minilm_results:
        minilm_overall = minilm_results['datasets']['Overall']
        print(f"\nComparison with MiniLM:")
        print(f"FPR: DistilBERT {overall_results['fpr']*100:.1f}% vs MiniLM {minilm_overall['fpr']*100:.1f}%")
        print(f"Recall: DistilBERT {overall_results['recall']*100:.1f}% vs MiniLM {minilm_overall['recall']*100:.1f}%")
        print(f"F1: DistilBERT {overall_results['f1']:.3f} vs MiniLM {minilm_overall['f1']:.3f}")

    print("\nRecommendation:")
    if overall_results["fpr"] < 0.05:
        print("1. DistilBERT model is ready for use!")
        if minilm_results and overall_results["f1"] > minilm_overall["f1"]:
            print("2. DistilBERT shows better F1 score - consider using it for production")
        else:
            print("2. Compare with MiniLM and choose based on your priorities (FPR vs F1)")
    else:
        print("1. Model may need threshold adjustment")
        print("2. Consider training with more diverse data")

if __name__ == "__main__":
    main()