"""
Benchmark Metrics

Extended metrics calculation for prompt injection detection evaluation.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve
)
import structlog

logger = structlog.get_logger()


@dataclass
class BenchmarkMetrics:
    """
    Comprehensive metrics for benchmark evaluation.
    """
    # Basic classification metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Error rates
    false_positive_rate: float = 0.0  # FPR
    false_negative_rate: float = 0.0  # FNR
    
    # Ranking metrics
    roc_auc: float = 0.0
    
    # Custom metrics
    tivs: float = 0.0  # Total Injection Vulnerability Score
    over_defense_rate: float = 0.0  # FPR on trigger-word benign samples
    
    # Latency metrics (in milliseconds)
    latency_avg: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    latency_min: float = 0.0
    latency_max: float = 0.0
    
    # Counts
    total_samples: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    # Metadata
    dataset_name: str = ""
    threshold: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "false_negative_rate": round(self.false_negative_rate, 4),
            "roc_auc": round(self.roc_auc, 4),
            "tivs": round(self.tivs, 4),
            "over_defense_rate": round(self.over_defense_rate, 4),
            "latency": {
                "avg_ms": round(self.latency_avg, 2),
                "p50_ms": round(self.latency_p50, 2),
                "p95_ms": round(self.latency_p95, 2),
                "p99_ms": round(self.latency_p99, 2),
                "min_ms": round(self.latency_min, 2),
                "max_ms": round(self.latency_max, 2),
            },
            "counts": {
                "total": self.total_samples,
                "tp": self.true_positives,
                "tn": self.true_negatives,
                "fp": self.false_positives,
                "fn": self.false_negatives,
            },
            "dataset_name": self.dataset_name,
            "threshold": self.threshold,
        }
    
    def meets_targets(self, 
                     min_accuracy: float = 0.95,
                     max_fpr: float = 0.05,
                     max_latency_p95: float = 100.0) -> Dict[str, bool]:
        """Check if metrics meet target thresholds."""
        return {
            "accuracy": self.accuracy >= min_accuracy,
            "fpr": self.false_positive_rate <= max_fpr,
            "latency_p95": self.latency_p95 <= max_latency_p95,
            "all_met": (
                self.accuracy >= min_accuracy and
                self.false_positive_rate <= max_fpr and
                self.latency_p95 <= max_latency_p95
            )
        }


def calculate_metrics(
    true_labels: List[int],
    predictions: List[int],
    scores: Optional[List[float]] = None,
    latencies: Optional[List[float]] = None,
    dataset_name: str = "",
    threshold: float = 0.5
) -> BenchmarkMetrics:
    """
    Calculate comprehensive benchmark metrics.
    
    Args:
        true_labels: Ground truth labels (0=safe, 1=injection)
        predictions: Model predictions (0=safe, 1=injection)
        scores: Confidence scores (optional, for ROC-AUC)
        latencies: Inference latencies in ms (optional)
        dataset_name: Name of the dataset
        threshold: Classification threshold used
        
    Returns:
        BenchmarkMetrics object with all calculated metrics
    """
    metrics = BenchmarkMetrics(
        dataset_name=dataset_name,
        threshold=threshold,
        total_samples=len(true_labels)
    )
    
    if len(true_labels) == 0:
        return metrics
    
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    
    # Basic classification metrics
    metrics.accuracy = accuracy_score(true_labels, predictions)
    
    # Handle edge cases where one class might be missing
    try:
        metrics.precision = precision_score(true_labels, predictions, zero_division=0)
        metrics.recall = recall_score(true_labels, predictions, zero_division=0)
        metrics.f1_score = f1_score(true_labels, predictions, zero_division=0)
    except Exception as e:
        logger.warning("Error calculating precision/recall/f1", error=str(e))
    
    # Confusion matrix
    try:
        cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        metrics.true_positives = int(tp)
        metrics.true_negatives = int(tn)
        metrics.false_positives = int(fp)
        metrics.false_negatives = int(fn)
        
        # Error rates
        metrics.false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics.false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    except Exception as e:
        logger.warning("Error calculating confusion matrix", error=str(e))
    
    # ROC-AUC (requires scores)
    if scores is not None and len(set(true_labels)) > 1:
        try:
            metrics.roc_auc = roc_auc_score(true_labels, scores)
        except Exception as e:
            logger.warning("Error calculating ROC-AUC", error=str(e))
    
    # TIVS (Total Injection Vulnerability Score)
    # Lower is better, negative is excellent
    metrics.tivs = calculate_tivs(
        isr=metrics.false_negative_rate,  # Injection Success Rate
        fpr=metrics.false_positive_rate
    )
    
    # Latency metrics
    if latencies:
        latencies = np.array(latencies)
        metrics.latency_avg = float(np.mean(latencies))
        metrics.latency_p50 = float(np.percentile(latencies, 50))
        metrics.latency_p95 = float(np.percentile(latencies, 95))
        metrics.latency_p99 = float(np.percentile(latencies, 99))
        metrics.latency_min = float(np.min(latencies))
        metrics.latency_max = float(np.max(latencies))
    
    return metrics


def calculate_tivs(
    isr: float,
    fpr: float,
    pof: float = 0.0,
    psr: float = 0.0,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate Total Injection Vulnerability Score (TIVS).
    
    Formula:
    TIVS = (ISR × w1) + (POF × w2) + (FPR × w3) - (PSR × w4)
    
    Lower/more negative is better.
    
    Args:
        isr: Injection Success Rate (false negative rate)
        fpr: False Positive Rate
        pof: Policy Override Frequency (optional)
        psr: Prompt Sanitization Rate (optional)
        weights: Custom weights for each component
        
    Returns:
        TIVS score
    """
    if weights is None:
        weights = {
            "isr": 0.4,   # Injection Success Rate (most important)
            "fpr": 0.3,   # False Positive Rate
            "pof": 0.2,   # Policy Override Frequency
            "psr": 0.1,   # Prompt Sanitization Rate
        }
    
    tivs = (
        (isr * weights["isr"]) +
        (fpr * weights["fpr"]) +
        (pof * weights["pof"]) -
        (psr * weights["psr"])
    )
    
    return tivs


def calculate_over_defense_rate(
    predictions: List[int],
    all_safe: bool = True
) -> float:
    """
    Calculate over-defense rate on a dataset that is all benign samples.
    
    Used with NotInject dataset to measure false positives on trigger-word
    enriched benign samples.
    
    Args:
        predictions: Model predictions (0=safe, 1=injection)
        all_safe: Whether all samples are actually safe (True for NotInject)
        
    Returns:
        Over-defense rate (ratio of false positives)
    """
    if not predictions:
        return 0.0
    
    if all_safe:
        # All samples are safe, so any prediction of 1 is a false positive
        false_positives = sum(p == 1 for p in predictions)
        return false_positives / len(predictions)
    
    return 0.0


def find_optimal_threshold(
    true_labels: List[int],
    scores: List[float],
    metric: str = "f1"
) -> Tuple[float, float]:
    """
    Find optimal classification threshold for given metric.
    
    Args:
        true_labels: Ground truth labels
        scores: Confidence scores
        metric: Metric to optimize ('f1', 'accuracy', 'balanced')
        
    Returns:
        Tuple of (optimal_threshold, metric_value)
    """
    thresholds = np.arange(0.1, 0.95, 0.05)
    best_threshold = 0.5
    best_score = 0.0
    
    for thresh in thresholds:
        preds = [1 if s >= thresh else 0 for s in scores]
        
        if metric == "f1":
            score = f1_score(true_labels, preds, zero_division=0)
        elif metric == "accuracy":
            score = accuracy_score(true_labels, preds)
        elif metric == "balanced":
            # Balance between precision and recall
            p = precision_score(true_labels, preds, zero_division=0)
            r = recall_score(true_labels, preds, zero_division=0)
            score = (p + r) / 2
        else:
            score = f1_score(true_labels, preds, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score


# Baseline metrics from industry solutions
BASELINE_METRICS = {
    "Lakera Guard": {
        "accuracy": 0.8791,
        "fpr": 0.057,
        "latency_p50": 66.0,
    },
    "ProtectAI LLM Guard": {
        "accuracy": 0.90,
        "latency_p50": 500.0,
    },
    "ActiveFence": {
        "f1_score": 0.857,
        "fpr": 0.054,
    },
    "Glean AI": {
        "accuracy": 0.978,
        "fpr": 0.030,
    },
    "PromptArmor": {
        "fpr": 0.0056,
        "fnr": 0.0013,
    }
}


def compare_to_baselines(metrics: BenchmarkMetrics) -> Dict[str, Dict[str, Any]]:
    """
    Compare metrics against industry baselines.
    
    Args:
        metrics: Calculated metrics
        
    Returns:
        Comparison results with improvement percentages
    """
    comparisons = {}
    
    for baseline_name, baseline_values in BASELINE_METRICS.items():
        comparison = {"baseline": baseline_values, "improvements": {}}
        
        for metric_name, baseline_value in baseline_values.items():
            current_value = None
            
            # Map metric names
            if metric_name == "accuracy":
                current_value = metrics.accuracy
            elif metric_name == "fpr":
                current_value = metrics.false_positive_rate
            elif metric_name == "fnr":
                current_value = metrics.false_negative_rate
            elif metric_name == "f1_score":
                current_value = metrics.f1_score
            elif metric_name.startswith("latency"):
                current_value = metrics.latency_p50
            
            if current_value is not None and baseline_value > 0:
                # For error rates and latency, lower is better
                if metric_name in ["fpr", "fnr", "latency_p50"]:
                    improvement = (baseline_value - current_value) / baseline_value * 100
                else:
                    improvement = (current_value - baseline_value) / baseline_value * 100
                
                comparison["improvements"][metric_name] = {
                    "current": round(current_value, 4),
                    "baseline": baseline_value,
                    "improvement_pct": round(improvement, 2),
                    "is_better": improvement > 0
                }
        
        comparisons[baseline_name] = comparison
    
    return comparisons
