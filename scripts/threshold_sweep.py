#!/usr/bin/env python
"""
Threshold Sweep for Optimal Detection

Sweep thresholds and generate ROC curves to find optimal
operating points for different datasets.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from sklearn.metrics import roc_curve, auc, precision_recall_curve

logger = structlog.get_logger()


@dataclass
class ThresholdResult:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    fpr: float
    fnr: float


def sweep_thresholds(
    y_true: List[int],
    y_scores: List[float],
    thresholds: List[float] = None
) -> List[ThresholdResult]:
    """Evaluate metrics at different thresholds."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05).tolist()
    
    results = []
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        
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
        
        results.append(ThresholdResult(
            threshold=thresh,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            fpr=fpr,
            fnr=fnr
        ))
    
    return results


def find_optimal_threshold(
    results: List[ThresholdResult],
    metric: str = "f1",
    max_fpr: float = 0.05
) -> ThresholdResult:
    """Find optimal threshold for given metric with FPR constraint."""
    valid = [r for r in results if r.fpr <= max_fpr]
    
    if not valid:
        # Fallback: find lowest FPR
        return min(results, key=lambda r: r.fpr)
    
    if metric == "f1":
        return max(valid, key=lambda r: r.f1)
    elif metric == "accuracy":
        return max(valid, key=lambda r: r.accuracy)
    elif metric == "recall":
        return max(valid, key=lambda r: r.recall)
    else:
        return max(valid, key=lambda r: r.f1)


def generate_roc_data(y_true: List[int], y_scores: List[float]) -> Dict:
    """Generate ROC curve data."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": roc_auc
    }


def run_sweep_on_datasets(detector, datasets: Dict, output_path: str = None):
    """Run threshold sweep on multiple datasets."""
    from benchmarks import BenchmarkRunner
    
    results = {}
    
    for name, dataset in datasets.items():
        logger.info(f"Running sweep on {name}")
        
        # Get scores
        runner = BenchmarkRunner(detector, threshold=0.5)
        texts = dataset.texts
        labels = dataset.labels
        
        # Get probability scores
        scores = detector.predict_proba(texts)[:, 1].tolist()
        
        # Sweep thresholds
        thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        sweep_results = sweep_thresholds(labels, scores, thresholds)
        
        # Find optimal
        optimal = find_optimal_threshold(sweep_results, metric="f1", max_fpr=0.05)
        
        # ROC data
        roc_data = generate_roc_data(labels, scores)
        
        results[name] = {
            "sweep": [
                {
                    "threshold": r.threshold,
                    "accuracy": round(r.accuracy, 4),
                    "precision": round(r.precision, 4),
                    "recall": round(r.recall, 4),
                    "f1": round(r.f1, 4),
                    "fpr": round(r.fpr, 4)
                }
                for r in sweep_results
            ],
            "optimal": {
                "threshold": optimal.threshold,
                "f1": round(optimal.f1, 4),
                "fpr": round(optimal.fpr, 4)
            },
            "roc_auc": round(roc_data["auc"], 4)
        }
        
        logger.info(f"  Optimal threshold: {optimal.threshold} (F1: {optimal.f1:.3f}, FPR: {optimal.fpr:.3f})")
    
    # Save results
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    return results


def print_summary(results: Dict):
    """Print sweep summary."""
    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<20} {'Optimal Thresh':>15} {'F1':>10} {'FPR':>10} {'AUC':>10}")
    print("-" * 70)
    
    for name, data in results.items():
        opt = data["optimal"]
        print(f"{name:<20} {opt['threshold']:>15.2f} {opt['f1']:>10.3f} {opt['fpr']:>10.3f} {data['roc_auc']:>10.3f}")
    
    print("=" * 70)


def main():
    """Main entry point."""
    from src.detection.embedding_classifier import EmbeddingClassifier
    from benchmarks import load_all_datasets
    
    print("=" * 60)
    print("Threshold Sweep Analysis")
    print("=" * 60)
    
    # Load detector
    detector = EmbeddingClassifier()
    detector.load_model("models/bit_classifier.json")
    
    # Load datasets
    print("\nLoading datasets...")
    datasets = load_all_datasets(limit_per_dataset=300)
    
    # Run sweep
    print("\nRunning threshold sweep...")
    output_path = "results/threshold_sweep.json"
    results = run_sweep_on_datasets(detector, datasets, output_path)
    
    # Print summary
    print_summary(results)
    
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
