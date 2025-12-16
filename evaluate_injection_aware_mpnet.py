#!/usr/bin/env python3
"""
Quick evaluation of the injection_aware_mpnet model with sampled datasets.
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
    sample_size: int = None,
    print_false_negatives: bool = False
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

    if print_false_negatives and fn > 0:
        print(f"\n--- False Negatives in {dataset_name} ({fn} cases) ---")
        fn_indices = np.where((np.array(labels) == 1) & (np.array(predictions) == 0))[0]
        for i, index in enumerate(fn_indices):
            print(f"FN {i+1}:")
            print(f"  Text: {texts[index]}")
            print("-" * 20)
        print(f"--- End of False Negatives ---")

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
    }

    # Print summary
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  FPR: {fpr:.4f} ({fpr*100:.1f}%)")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")

    return results

def main():
    """Run quick evaluation on sampled datasets."""

    print("=== Quick Injection Aware MPNET Model Evaluation ===")
    print("Model: injection_aware_mpnet_classifier.json")
    print("Using sampled datasets for fast evaluation\n")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Load the balanced model
    model_path = Path("models/injection_aware_mpnet_classifier.json")
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please run: python train_injection_aware_classifier.py")
        return

    classifier = EmbeddingClassifier(
        model_name="models/injection_aware_mpnet",
        threshold=0.764,
        model_dir="models"
    )

    # Load the balanced model
    classifier.load_model(str(model_path))

    # Load datasets with sampling
    print("Loading datasets (with sampling for large datasets)...\n")

    # 1. deepset (full dataset - it's manageable)
    print("Loading deepset dataset...")
    deepset = load_deepset_dataset(include_safe=True)

    # Separate benign and injection
    deepset_benign_texts = [t for t, l in zip(deepset.texts, deepset.labels) if l == 0]
    deepset_benign_labels = [0] * len(deepset_benign_texts)
    deepset_injection_texts = [t for t, l in zip(deepset.texts, deepset.labels) if l == 1]
    deepset_injection_labels = [1] * len(deepset_injection_texts)

    # 2. NotInject (sample 1000)
    notinject = load_notinject_dataset(limit=1000)

    # 3. SaTML (sample 2000 from the large dataset)
    try:
        satml_full = load_satml_dataset()
        satml_indices = np.random.choice(len(satml_full.texts), 2000, replace=False)
        satml_texts = [satml_full.texts[i] for i in satml_indices]
        satml_labels = [satml_full.labels[i] for i in satml_indices]
    except:
        print("Warning: Could not load SaTML dataset")
        satml_texts = []
        satml_labels = []

    # 4. LLMail (sample 2000 from the large dataset)
    try:
        llmail_full = load_llmail_dataset()
        llmail_indices = np.random.choice(len(llmail_full.texts), 2000, replace=False)
        llmail_texts = [llmail_full.texts[i] for i in llmail_indices]
        llmail_labels = [llmail_full.labels[i] for i in llmail_indices]
    except:
        print("Warning: Could not load LLMail dataset")
        llmail_texts = []
        llmail_labels = []

    # Test a few samples to understand what's happening
    print("\n" + "="*50)
    print("Debugging: Testing a few samples")
    print("="*50)

    test_samples = deepset_benign_texts[:5]
    test_probs = classifier.predict_proba(test_samples)

    print("Sample deepset benign texts and their predictions:")
    for i, (text, prob) in enumerate(zip(test_samples, test_probs)):
        print(f"\nSample {i+1}:")
        print(f"  Text: {text[:100]}...")
        print(f"  Benign prob: {prob[0]:.3f}")
        print(f"  Malicious prob: {prob[1]:.3f}")
        print(f"  Prediction: {'Malicious' if prob[1] > 0.764 else 'Benign'}")

    # Evaluate on each dataset
    all_results = []

    # deepset benign (CRITICAL - this was 40.2% FPR)
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

    if llmail_texts:
        all_texts.extend(llmail_texts)
        all_labels.extend(llmail_labels)

    overall_results = evaluate_classifier(
        classifier, "Overall", all_texts, all_labels, print_false_negatives=True
    )
    all_results.append(overall_results)

    # Print comparison table
    print("\n" + "="*50)
    print("Performance Comparison")
    print("="*50)
    print(f"{'Dataset':<20} {'FPR':<12} {'Target':<12} {'Status':<15}")
    print("-" * 60)

    # Target values from paper
    targets = {
        "deepset_benign": 0.023,  # 2.3% from paper
        "NotInject": 0.018,       # 1.8% from paper
        "Overall": 0.05          # <5% target
    }

    for result in all_results:
        dataset = result["dataset"]
        if dataset in ["deepset_injections"]:
            continue  # Skip injection-only for FPR comparison

        fpr = result["fpr"]

        if dataset in targets:
            target = targets[dataset]
            status = "✅ PASS" if fpr <= target else f"❌ FAIL ({fpr*100:.1f}% > {target*100:.1f}%)"
            target_str = f"{target*100:.1f}%"
        else:
            target_str = "N/A"
            status = "✓"

        print(f"{dataset:<20} {fpr*100:>10.1f}% {target_str:>12} {status:<15}")

    # Save results
    results_file = Path("results/injection_aware_mpnet_evaluation.json")
    results_file.parent.mkdir(exist_ok=True)

    save_data = {
        "model": "injection_aware_mpnet",
        "threshold": 0.764,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sampling_info": {
            "satml": "2000/136000 samples" if satml_texts else "Not available",
            "llmail": "2000/370000 samples" if llmail_texts else "Not available",
            "deepset": "Full dataset",
            "notinject": f"{len(notinject.texts)} samples"
        },
        "datasets": {r["dataset"]: r for r in all_results}
    }

    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Summary
    print("\n" + "="*50)
    print("Summary")
    print("="*50)

    print("\nKey Results:")
    print(f"deepset benign FPR: {deepset_benign_results['fpr']*100:.1f}%")
    print(f"Overall FPR: {overall_results['fpr']*100:.1f}%")
    print(f"Model threshold: 0.764")

    if deepset_benign_results["fpr"] < 0.05:
        print(f"\n✅ SUCCESS: deepset benign FPR improved!")
    else:
        print(f"\n❌ Model still needs improvement")

    print("\nNext Steps:")
    print("1. If FPRs are <5%, run full evaluation with run_balanced_eval.py")
    print("2. Test HTML preprocessing on BrowseSafe dataset")
    print("3. Update paper results table with new numbers")

if __name__ == "__main__":
    main()