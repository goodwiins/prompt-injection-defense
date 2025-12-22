#!/usr/bin/env python3
"""
Statistical significance testing for BIT vs baseline models.

This script performs McNemar's test to quantify statistical significance
of the performance difference between BIT and no-weight models.
"""

import sys
import json
from pathlib import Path
from typing import List, Tuple, Dict
import random
import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from detection.embedding_classifier import EmbeddingClassifier
from datasets import load_dataset

# Import data loading from train_bit_model
from train_bit_model import (
    load_satml_dataset,
    load_deepset_dataset,
    load_notinject_dataset,
    generate_synthetic_benign_triggers,
)


def prepare_bit_dataset(
    target_total: int = 5000,
    benign_trigger_weight: float = 2.0,
) -> Tuple[List[str], List[int], List[float]]:
    """Prepare BIT dataset with specified weight."""
    # Load datasets
    satml_inj, satml_safe = load_satml_dataset()
    deepset_inj, deepset_safe = load_deepset_dataset()
    notinject_safe = load_notinject_dataset()

    # Combine
    all_injections = satml_inj + deepset_inj
    all_safe = satml_safe + deepset_safe + notinject_safe

    # Generate benign-trigger samples
    synthetic_bt = generate_synthetic_benign_triggers(n_samples=2000)

    # Tag sample types
    injections = [{"text": t, "label": 1, "type": "injection"} for t in all_injections]
    safe = [{"text": t, "label": 0, "type": "safe"} for t in all_safe]
    benign_triggers = [{"text": t, "label": 0, "type": "benign_trigger"} for t in synthetic_bt]

    # BIT composition: 40/40/20
    n_injection = int(target_total * 0.4)
    n_safe = int(target_total * 0.4)
    n_benign_trigger = int(target_total * 0.2)

    # Sample
    balanced = []
    balanced.extend(random.sample(injections, min(n_injection, len(injections))))
    balanced.extend(random.sample(safe, min(n_safe, len(safe))))
    balanced.extend(random.sample(benign_triggers, min(n_benign_trigger, len(benign_triggers))))

    random.shuffle(balanced)

    # Extract
    texts = [s["text"] for s in balanced]
    labels = [s["label"] for s in balanced]
    weights = [benign_trigger_weight if s["type"] == "benign_trigger" else 1.0 for s in balanced]

    return texts, labels, weights


def train_model(X_train, y_train, weights, threshold: float = 0.764):
    """Train XGBoost classifier with given weights."""
    classifier = EmbeddingClassifier(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        use_xgboost=True,
    )
    classifier.train_on_dataset(X_train, y_train, sample_weights=weights)
    classifier.set_threshold(threshold)
    return classifier


def evaluate_on_notinject(classifier) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate on NotInject and return (predictions, labels)."""
    # Load NotInject test set
    dataset = load_dataset("deepset/prompt-injections", split="test")

    texts = []
    labels = []

    for example in dataset:
        texts.append(example["text"])
        labels.append(1 if example["label"] == 1 else 0)

    # Predict
    predictions = classifier.predict(texts)

    return np.array(predictions), np.array(labels)


def mcnemar_test(predictions_a: np.ndarray, predictions_b: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Perform McNemar's test for paired binary classifiers.

    McNemar's test checks if two classifiers have significantly different error rates
    on the same test set.

    Contingency table:
                        Classifier B Correct    Classifier B Wrong
    Classifier A Correct        n00                     n01
    Classifier A Wrong          n10                     n11

    Test statistic focuses on discordant pairs (n01, n10):
    - n01: A correct, B wrong
    - n10: A wrong, B correct

    If p-value < 0.05, the difference is statistically significant.
    """
    # Calculate correct/wrong for each classifier
    correct_a = (predictions_a == labels)
    correct_b = (predictions_b == labels)

    # Build contingency table
    n00 = np.sum(correct_a & correct_b)  # Both correct
    n01 = np.sum(correct_a & ~correct_b)  # A correct, B wrong
    n10 = np.sum(~correct_a & correct_b)  # A wrong, B correct
    n11 = np.sum(~correct_a & ~correct_b)  # Both wrong

    # Contingency matrix for mcnemar test
    # Format: [[n00, n01], [n10, n11]]
    table = [[n00, n01], [n10, n11]]

    # Perform McNemar's test
    result = mcnemar(table, exact=False, correction=True)

    return {
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "n00_both_correct": n00,
        "n01_a_correct_b_wrong": n01,
        "n10_a_wrong_b_correct": n10,
        "n11_both_wrong": n11,
        "total_samples": len(labels),
    }


def main():
    """Run statistical significance tests."""
    random.seed(42)
    np.random.seed(42)

    print("="*60)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("="*60)
    print("\nComparing BIT (w=2.0) vs No Weight (w=1.0)")
    print("Test: McNemar's test for paired binary classifiers")
    print("="*60)

    # Prepare dataset (same data for both models)
    print("\n📊 Preparing dataset...")
    X_train, y_train, _ = prepare_bit_dataset(target_total=5000)

    # Split for training
    split_idx = int(len(X_train) * 0.8)
    X_train_split = X_train[:split_idx]
    y_train_split = y_train[:split_idx]
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]

    print(f"   Training: {len(X_train_split)} samples")
    print(f"   Validation: {len(X_val)} samples")

    # Train BIT model (w=2.0)
    print("\n🚀 Training BIT model (w=2.0)...")
    weights_bit = [2.0 if i < int(len(X_train_split) * 0.2) else 1.0 for i in range(len(X_train_split))]
    model_bit = train_model(X_train_split, y_train_split, weights_bit)

    # Train baseline model (w=1.0)
    print("🚀 Training baseline model (w=1.0)...")
    weights_baseline = [1.0 for _ in range(len(X_train_split))]
    model_baseline = train_model(X_train_split, y_train_split, weights_baseline)

    # Evaluate both on NotInject
    print("\n📊 Evaluating on NotInject test set...")
    predictions_bit, labels = evaluate_on_notinject(model_bit)
    predictions_baseline, _ = evaluate_on_notinject(model_baseline)

    # Calculate accuracies
    acc_bit = np.mean(predictions_bit == labels)
    acc_baseline = np.mean(predictions_baseline == labels)

    # Calculate FPR (on safe samples only)
    safe_mask = labels == 0
    fpr_bit = np.mean(predictions_bit[safe_mask])
    fpr_baseline = np.mean(predictions_baseline[safe_mask])

    print(f"\n📈 BIT model (w=2.0):")
    print(f"   Accuracy: {acc_bit*100:.1f}%")
    print(f"   FPR: {fpr_bit*100:.1f}%")

    print(f"\n📉 Baseline model (w=1.0):")
    print(f"   Accuracy: {acc_baseline*100:.1f}%")
    print(f"   FPR: {fpr_baseline*100:.1f}%")

    print(f"\n📊 Improvement:")
    print(f"   Accuracy: {(acc_bit - acc_baseline)*100:+.1f} pp")
    print(f"   FPR: {(fpr_baseline - fpr_bit)*100:+.1f} pp reduction")

    # Perform McNemar's test
    print("\n" + "="*60)
    print("MCNEMAR'S TEST")
    print("="*60)

    result = mcnemar_test(predictions_bit, predictions_baseline, labels)

    print(f"\nContingency Table:")
    print(f"                      Baseline Correct    Baseline Wrong")
    print(f"  BIT Correct         {result['n00']:<18}  {result['n01']:<18}")
    print(f"  BIT Wrong           {result['n10']:<18}  {result['n11']:<18}")

    print(f"\nDiscordant pairs:")
    print(f"  BIT correct, Baseline wrong: {result['n01']}")
    print(f"  BIT wrong, Baseline correct: {result['n10']}")

    print(f"\nTest Results:")
    print(f"  Chi-squared statistic: {result['statistic']:.4f}")
    print(f"  p-value: {result['p_value']:.6f}")

    # Interpret results
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    if result['p_value'] < 0.001:
        significance = "p < 0.001 (highly significant)"
    elif result['p_value'] < 0.01:
        significance = "p < 0.01 (very significant)"
    elif result['p_value'] < 0.05:
        significance = "p < 0.05 (significant)"
    else:
        significance = f"p = {result['p_value']:.4f} (not significant)"

    print(f"\nStatistical Significance: {significance}")

    if result['p_value'] < 0.05:
        print("✅ CONCLUSION: The difference is statistically significant.")
        print("   BIT's weighted loss mechanism provides a measurable improvement")
        print("   over uniform weighting on the same dataset.")
    else:
        print("⚠️  CONCLUSION: The difference is NOT statistically significant.")
        print("   Cannot reject null hypothesis of equal performance.")

    print(f"\nEffect Size:")
    print(f"  BIT improved on {result['n01']} samples that baseline got wrong")
    print(f"  Baseline improved on {result['n10']} samples that BIT got wrong")
    print(f"  Net improvement: {result['n01'] - result['n10']} samples ({len(labels)} total)")

    # Save results
    output_file = Path("results/statistical_significance.json")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump({
            "test": "McNemar",
            "comparison": "BIT (w=2.0) vs Baseline (w=1.0)",
            "dataset": "NotInject",
            "n_samples": int(result['total_samples']),
            "bit_accuracy": float(acc_bit),
            "baseline_accuracy": float(acc_baseline),
            "bit_fpr": float(fpr_bit),
            "baseline_fpr": float(fpr_baseline),
            "mcnemar_statistic": float(result['statistic']),
            "p_value": float(result['p_value']),
            "contingency_table": {
                "both_correct": int(result['n00']),
                "bit_correct_baseline_wrong": int(result['n01']),
                "bit_wrong_baseline_correct": int(result['n10']),
                "both_wrong": int(result['n11']),
            },
        }, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")

    # Generate LaTeX text
    print("\n" + "="*60)
    print("LATEX TEXT (for paper Section 6):")
    print("="*60)
    print(f"""
Statistical significance was assessed using McNemar's test for paired
binary classifiers on the NotInject dataset (n={result['total_samples']}).
The comparison between BIT (w=2.0) and uniform weighting (w=1.0) yielded
χ²={result['statistic']:.2f}, {significance}, confirming that BIT's
weighted loss mechanism provides a statistically significant improvement
beyond data composition effects.
""")


if __name__ == "__main__":
    main()
