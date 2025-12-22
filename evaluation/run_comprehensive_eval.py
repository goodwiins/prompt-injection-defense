#!/usr/bin/env python3
"""
Comprehensive evaluation of the θ=0.764 optimized model.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, auc
)

sys.path.append(str(Path(__file__).parent))

from src.detection.embedding_classifier import EmbeddingClassifier
from benchmarks.benchmark_datasets import (
    load_satml_dataset,
    load_deepset_dataset,
    load_llmail_dataset,
    load_notinject_dataset
)

def main():
    """Run comprehensive evaluation."""

    print("=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)

    # Check for model
    model_path = "models/bit_xgboost_theta_764_classifier.json"

    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        print("Please run: python train_with_theta_764.py")
        return

    # Load model
    print(f"\nLoading model: {model_path}")
    classifier = EmbeddingClassifier(
        model_name="all-MiniLM-L6-v2",
        threshold=0.764,
        model_dir="models"
    )
    classifier.load_model(model_path)

    # Evaluate on benchmarks
    print("\n" + "=" * 60)
    print("BENCHMARK EVALUATION")
    print("=" * 60)

    results = {}

    # 1. Deepset datasets
    print("\n1. Deepset Datasets")
    print("-" * 40)

    deepset = load_deepset_dataset(include_safe=True)

    # Split into benign and injections
    benign_texts = [t for t, l in zip(deepset.texts, deepset.labels) if l == 0]
    injection_texts = [t for t, l in zip(deepset.texts, deepset.labels) if l == 1]

    # Evaluate benign
    print(f"  Evaluating {len(benign_texts)} benign samples...")
    benign_probs = classifier.predict_proba(benign_texts)
    benign_preds = classifier.predict(benign_texts)
    benign_fp = np.sum(benign_preds)
    benign_tn = len(benign_preds) - benign_fp
    benign_fpr = benign_fp / (benign_fp + benign_tn) if (benign_fp + benign_tn) > 0 else 0

    results['deepset_benign'] = {
        'fpr': float(benign_fpr),
        'fp': int(benign_fp),
        'tn': int(benign_tn),
        'samples': len(benign_texts)
    }

    print(f"    FPR: {benign_fpr*100:.1f}%")

    # Evaluate injections
    print(f"  Evaluating {len(injection_texts)} injection samples...")
    injection_probs = classifier.predict_proba(injection_texts)
    injection_preds = classifier.predict(injection_texts)
    injection_tp = np.sum(injection_preds)
    injection_fn = len(injection_preds) - injection_tp
    injection_recall = injection_tp / (injection_tp + injection_fn) if (injection_tp + injection_fn) > 0 else 0

    results['deepset_injections'] = {
        'recall': float(injection_recall),
        'tp': int(injection_tp),
        'fn': int(injection_fn),
        'samples': len(injection_texts)
    }

    print(f"    Recall: {injection_recall*100:.1f}%")

    # 2. NotInject dataset
    print("\n2. NotInject Dataset")
    print("-" * 40)

    try:
        notinject = load_notinject_dataset()
        notinject_probs = classifier.predict_proba(notinject.texts)
        notinject_preds = classifier.predict(notinject.texts)
        notinject_fp = np.sum(notinject_preds)
        notinject_tn = len(notinject_preds) - notinject_fp
        notinject_fpr = notinject_fp / (notinject_fp + notinject_tn) if (notinject_fp + notinject_tn) > 0 else 0

        results['NotInject'] = {
            'fpr': float(notinject_fpr),
            'fp': int(notinject_fp),
            'tn': int(notinject_tn),
            'samples': len(notinject.texts)
        }

        print(f"  Samples: {len(notinject.texts)}")
        print(f"  FPR: {notinject_fpr*100:.1f}%")
    except Exception as e:
        print(f"  Error: {e}")

    # 3. SaTML dataset (sample)
    print("\n3. SaTML Dataset (sample)")
    print("-" * 40)

    try:
        satml = load_satml_dataset(limit=1000)
        satml_probs = classifier.predict_proba(satml.texts)
        satml_preds = classifier.predict(satml.texts)
        satml_tp = np.sum(satml_preds)
        satml_fn = len(satml_preds) - satml_tp
        satml_recall = satml_tp / (satml_tp + satml_fn) if (satml_tp + satml_fn) > 0 else 0

        results['SaTML'] = {
            'recall': float(satml_recall),
            'tp': int(satml_tp),
            'fn': int(satml_fn),
            'samples': len(satml.texts)
        }

        print(f"  Samples: {len(satml.texts)}")
        print(f"  Recall: {satml_recall*100:.1f}%")
    except Exception as e:
        print(f"  Error: {e}")

    # 4. LLMail dataset (sample)
    print("\n4. LLMail Dataset (sample)")
    print("-" * 40)

    try:
        llmail = load_llmail_dataset(limit=1000)
        llmail_probs = classifier.predict_proba(llmail.texts)
        llmail_preds = classifier.predict(llmail.texts)
        llmail_tp = np.sum(llmail_preds)
        llmail_fn = len(llmail_preds) - llmail_tp
        llmail_recall = llmail_tp / (llmail_tp + llmail_fn) if (llmail_tp + llmail_fn) > 0 else 0

        results['LLMail'] = {
            'recall': float(llmail_recall),
            'tp': int(llmail_tp),
            'fn': int(llmail_fn),
            'samples': len(llmail.texts)
        }

        print(f"  Samples: {len(llmail.texts)}")
        print(f"  Recall: {llmail_recall*100:.1f}%")
    except Exception as e:
        print(f"  Error: {e}")

    # Threshold sweep analysis
    print("\n" + "=" * 60)
    print("THRESHOLD SWEEP ANALYSIS")
    print("=" * 60)

    # Create mixed dataset
    benign_sample = benign_texts[:50]
    injection_sample = injection_texts[:50]
    mixed_texts = benign_sample + injection_sample
    mixed_labels = [0] * len(benign_sample) + [1] * len(injection_sample)

    # Get probabilities
    mixed_probs = classifier.predict_proba(mixed_texts)
    mixed_scores = mixed_probs[:, 1]

    print(f"{'Threshold':<10} {'Recall':<10} {'FPR':<10} {'Precision':<10} {'F1':<10}")
    print("-" * 50)

    thresholds = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.764, 0.8]

    for thresh in thresholds:
        preds = (mixed_scores >= thresh).astype(int)

        tp = np.sum((preds == 1) & (np.array(mixed_labels) == 1))
        fp = np.sum((preds == 1) & (np.array(mixed_labels) == 0))
        fn = np.sum((preds == 0) & (np.array(mixed_labels) == 1))
        tn = np.sum((preds == 0) & (np.array(mixed_labels) == 0))

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        marker = " ★" if thresh == 0.764 else "  "
        print(f"{thresh:.2f}{marker}      {recall*100:8.1f}% {fpr*100:8.1f}% {precision*100:8.1f}% {f1:8.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print("\n📊 Dataset Performance:")
    for dataset_name, metrics in results.items():
        if dataset_name in ['deepset_benign', 'NotInject']:
            print(f"  {dataset_name}: FPR={metrics['fpr']*100:.1f}%")
        else:
            print(f"  {dataset_name}: Recall={metrics['recall']*100:.1f}%")

    # Target assessment
    print("\n🎯 Target Assessment:")

    targets_met = True
    if results.get('deepset_benign', {}).get('fpr', 1) > 0.05:
        print("  ❌ deepset_benign: FPR > 5%")
        targets_met = False
    else:
        print("  ✅ deepset_benign: FPR ≤ 5%")

    if results.get('deepset_injections', {}).get('recall', 0) < 0.85:
        print("  ❌ deepset_injections: Recall < 85%")
        targets_met = False
    else:
        print("  ✅ deepset_injections: Recall ≥ 85%")

    # Save results
    final_results = {
        'model': 'bit_xgboost_theta_764',
        'threshold': 0.764,
        'date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'datasets': results,
        'targets_met': targets_met,
        'summary': {
            'model_ready': '✅ Yes' if targets_met else '❌ Needs improvement',
            'strengths': [],
            'weaknesses': []
        }
    }

    # Analyze strengths and weaknesses
    if results.get('deepset_benign', {}).get('fpr', 1) <= 0.05:
        final_results['summary']['strengths'].append('Low false positive rate on benign prompts')

    if results.get('deepset_injections', {}).get('recall', 0) >= 0.8:
        final_results['summary']['strengths'].append('Good detection of injection attempts')

    if results.get('SaTML', {}).get('recall', 0) < 0.8:
        final_results['summary']['weaknesses'].append('Poor performance on SaTML dataset')

    if results.get('LLMail', {}).get('recall', 0) < 0.8:
        final_results['summary']['weaknesses'].append('Poor performance on LLMail dataset')

    # Save
    Path("results").mkdir(exist_ok=True)
    with open("results/comprehensive_evaluation.json", 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n✅ Results saved to: results/comprehensive_evaluation.json")
    print(f"\nModel Status: {'Ready for production' if targets_met else 'Needs improvement'}")

    return final_results


if __name__ == "__main__":
    main()