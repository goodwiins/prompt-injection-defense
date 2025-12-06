#!/usr/bin/env python
"""
Statistical Analysis for Benchmark Results

Includes:
- Confidence intervals for metrics (bootstrap)
- McNemar's test for comparing classifiers
- Statistical significance testing
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = structlog.get_logger()


@dataclass
class ConfidenceInterval:
    """Confidence interval result."""
    mean: float
    lower: float
    upper: float
    std: float
    confidence_level: float
    
    def to_dict(self) -> Dict:
        return {
            "mean": round(self.mean, 4),
            "lower": round(self.lower, 4),
            "upper": round(self.upper, 4),
            "std": round(self.std, 4),
            "confidence_level": self.confidence_level
        }
    
    def __repr__(self):
        return f"{self.mean:.4f} [{self.lower:.4f}, {self.upper:.4f}] (95% CI)"


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_func=np.mean,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> ConfidenceInterval:
    """
    Calculate bootstrap confidence interval for a statistic.
    
    Args:
        data: Input data array
        statistic_func: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default: 0.95 for 95% CI)
        
    Returns:
        ConfidenceInterval object
    """
    n = len(data)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return ConfidenceInterval(
        mean=float(statistic_func(data)),
        lower=float(lower),
        upper=float(upper),
        std=float(np.std(bootstrap_stats)),
        confidence_level=confidence_level
    )


def calculate_metric_with_ci(
    true_labels: List[int],
    predictions: List[int],
    metric: str = "accuracy",
    n_bootstrap: int = 1000
) -> ConfidenceInterval:
    """
    Calculate a metric with bootstrap confidence interval.
    
    Args:
        true_labels: Ground truth labels
        predictions: Model predictions
        metric: Metric name ('accuracy', 'precision', 'recall', 'f1')
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        ConfidenceInterval for the metric
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    
    metric_funcs = {
        "accuracy": accuracy_score,
        "precision": lambda y, p: precision_score(y, p, zero_division=0),
        "recall": lambda y, p: recall_score(y, p, zero_division=0),
        "f1": lambda y, p: f1_score(y, p, zero_division=0)
    }
    
    if metric not in metric_funcs:
        raise ValueError(f"Unknown metric: {metric}")
    
    func = metric_funcs[metric]
    n = len(true_labels)
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        y_sample = true_labels[indices]
        p_sample = predictions[indices]
        bootstrap_scores.append(func(y_sample, p_sample))
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    return ConfidenceInterval(
        mean=float(func(true_labels, predictions)),
        lower=float(np.percentile(bootstrap_scores, 2.5)),
        upper=float(np.percentile(bootstrap_scores, 97.5)),
        std=float(np.std(bootstrap_scores)),
        confidence_level=0.95
    )


def mcnemars_test(
    y_true: List[int],
    pred_a: List[int],
    pred_b: List[int]
) -> Dict:
    """
    Perform McNemar's test to compare two classifiers.
    
    McNemar's test compares the disagreement between two classifiers.
    It's particularly useful when you can't assume independence.
    
    Args:
        y_true: Ground truth labels
        pred_a: Predictions from classifier A
        pred_b: Predictions from classifier B
        
    Returns:
        Dictionary with test results
    """
    y_true = np.array(y_true)
    pred_a = np.array(pred_a)
    pred_b = np.array(pred_b)
    
    # Correct/incorrect for each classifier
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    
    # Contingency table
    # b01: A correct, B wrong
    # b10: A wrong, B correct
    b01 = np.sum(correct_a & ~correct_b)  # A correct, B wrong
    b10 = np.sum(~correct_a & correct_b)  # A wrong, B correct
    
    # McNemar's test statistic
    if b01 + b10 == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "b01": int(b01),
            "b10": int(b10),
            "interpretation": "No disagreement between classifiers"
        }
    
    # Chi-squared test with continuity correction
    statistic = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    # Exact binomial test for small samples
    if b01 + b10 < 25:
        p_value = stats.binom_test(b01, b01 + b10, 0.5)
    
    significant = p_value < 0.05
    
    # Interpretation
    if significant:
        if b01 > b10:
            interpretation = "Classifier A is significantly BETTER than B"
        else:
            interpretation = "Classifier B is significantly BETTER than A"
    else:
        interpretation = "No significant difference between classifiers"
    
    return {
        "statistic": round(float(statistic), 4),
        "p_value": round(float(p_value), 4),
        "significant": significant,
        "b01": int(b01),
        "b10": int(b10),
        "interpretation": interpretation
    }


def paired_bootstrap_test(
    y_true: List[int],
    pred_a: List[int],
    pred_b: List[int],
    n_bootstrap: int = 10000
) -> Dict:
    """
    Paired bootstrap test for comparing classifier accuracy.
    
    Args:
        y_true: Ground truth labels
        pred_a: Predictions from classifier A
        pred_b: Predictions from classifier B
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Dictionary with test results
    """
    y_true = np.array(y_true)
    pred_a = np.array(pred_a)
    pred_b = np.array(pred_b)
    
    n = len(y_true)
    
    # Original difference in accuracy
    acc_a = np.mean(pred_a == y_true)
    acc_b = np.mean(pred_b == y_true)
    original_diff = acc_a - acc_b
    
    # Bootstrap
    diffs = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        acc_a_boot = np.mean(pred_a[indices] == y_true[indices])
        acc_b_boot = np.mean(pred_b[indices] == y_true[indices])
        diffs.append(acc_a_boot - acc_b_boot)
    
    diffs = np.array(diffs)
    
    # Two-tailed p-value
    p_value = np.mean(np.abs(diffs) >= np.abs(original_diff))
    
    return {
        "accuracy_a": round(float(acc_a), 4),
        "accuracy_b": round(float(acc_b), 4),
        "difference": round(float(original_diff), 4),
        "ci_lower": round(float(np.percentile(diffs, 2.5)), 4),
        "ci_upper": round(float(np.percentile(diffs, 97.5)), 4),
        "p_value": round(float(p_value), 4),
        "significant": p_value < 0.05
    }


def main():
    """Run statistical analysis on benchmark results."""
    print("=" * 60)
    print("Statistical Analysis for Benchmarks")
    print("=" * 60)
    
    # Load test data
    from src.detection.embedding_classifier import EmbeddingClassifier
    from datasets import load_dataset
    
    print("\nLoading data and models...")
    
    detector = EmbeddingClassifier()
    detector.load_model("models/mof_classifier.json")
    
    # Load test data
    ds = load_dataset("deepset/prompt-injections", split="train", streaming=True)
    texts, labels = [], []
    for i, sample in enumerate(ds):
        if i >= 500:
            break
        texts.append(sample["text"])
        labels.append(sample["label"])
    
    print(f"Loaded {len(texts)} samples")
    
    # Get predictions from our model
    print("\nGenerating predictions...")
    our_preds = detector.predict(texts)
    
    # Calculate metrics with CIs
    print("\n" + "=" * 60)
    print("METRICS WITH 95% CONFIDENCE INTERVALS")
    print("=" * 60)
    
    metrics = ["accuracy", "precision", "recall", "f1"]
    results = {}
    
    for metric in metrics:
        ci = calculate_metric_with_ci(labels, our_preds, metric)
        results[metric] = ci.to_dict()
        print(f"{metric.capitalize():12} {ci}")
    
    # McNemar's test against a simple baseline
    print("\n" + "=" * 60)
    print("McNEMAR'S TEST: MOF vs Baseline")
    print("=" * 60)
    
    # Simple baseline: predict 1 if text contains trigger words
    trigger_words = ["ignore", "forget", "bypass", "override", "system"]
    baseline_preds = [
        1 if any(w in t.lower() for w in trigger_words) else 0
        for t in texts
    ]
    
    mcnemar = mcnemars_test(labels, our_preds, baseline_preds)
    print(f"Chi-squared statistic: {mcnemar['statistic']}")
    print(f"P-value: {mcnemar['p_value']}")
    print(f"b01 (MOF correct, baseline wrong): {mcnemar['b01']}")
    print(f"b10 (MOF wrong, baseline correct): {mcnemar['b10']}")
    print(f"Result: {mcnemar['interpretation']}")
    
    results["mcnemars_vs_baseline"] = mcnemar
    
    # Paired bootstrap test
    print("\n" + "=" * 60)
    print("PAIRED BOOTSTRAP TEST")
    print("=" * 60)
    
    bootstrap = paired_bootstrap_test(labels, our_preds, baseline_preds)
    print(f"MOF Accuracy: {bootstrap['accuracy_a']:.1%}")
    print(f"Baseline Accuracy: {bootstrap['accuracy_b']:.1%}")
    print(f"Difference: {bootstrap['difference']:.1%}")
    print(f"95% CI: [{bootstrap['ci_lower']:.1%}, {bootstrap['ci_upper']:.1%}]")
    print(f"P-value: {bootstrap['p_value']}")
    print(f"Significant: {bootstrap['significant']}")
    
    results["bootstrap_test"] = bootstrap
    
    # Save results
    output_path = Path("results/statistical_analysis.json")
    with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else x)
    
    print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
