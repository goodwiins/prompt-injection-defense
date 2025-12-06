#!/usr/bin/env python
"""
Run Baseline Comparisons

Compare our MOF model against baseline detectors:
- TF-IDF + SVM
- HuggingFace prompt injection classifier
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import structlog

logger = structlog.get_logger()


@dataclass
class BaselineResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    latency_ms: float


def load_test_data(limit: int = 500) -> tuple:
    """Load test data from datasets."""
    texts, labels = [], []
    
    try:
        from datasets import load_dataset
        
        # Load deepset for balanced test set
        ds = load_dataset("deepset/prompt-injections", split="train", streaming=True)
        injection_count = safe_count = 0
        
        for sample in ds:
            if injection_count >= limit // 2 and safe_count >= limit // 2:
                break
            label = sample.get("label", 0)
            text = sample.get("text", "")
            if text.strip():
                if label == 1 and injection_count < limit // 2:
                    texts.append(text)
                    labels.append(1)
                    injection_count += 1
                elif label == 0 and safe_count < limit // 2:
                    texts.append(text)
                    labels.append(0)
                    safe_count += 1
        
        logger.info(f"Loaded {len(texts)} test samples")
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
    
    return texts, labels


def evaluate_mof_model(texts: List[str], labels: List[int]) -> BaselineResult:
    """Evaluate our MOF model."""
    from src.detection.embedding_classifier import EmbeddingClassifier
    
    logger.info("Evaluating MOF model...")
    detector = EmbeddingClassifier()
    detector.load_model("models/mof_classifier.json")
    
    start = time.time()
    predictions = detector.predict(texts)
    elapsed = (time.time() - start) * 1000 / len(texts)
    
    return BaselineResult(
        name="MOF Model (Ours)",
        accuracy=accuracy_score(labels, predictions),
        precision=precision_score(labels, predictions, zero_division=0),
        recall=recall_score(labels, predictions, zero_division=0),
        f1=f1_score(labels, predictions, zero_division=0),
        latency_ms=elapsed
    )


def evaluate_tfidf_svm(texts: List[str], labels: List[int]) -> BaselineResult:
    """Evaluate TF-IDF + SVM baseline."""
    from benchmarks.baselines import TfidfSvmBaseline
    
    logger.info("Training and evaluating TF-IDF + SVM...")
    
    # Split for training
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    baseline = TfidfSvmBaseline()
    baseline.train(X_train, y_train)
    
    start = time.time()
    predictions = baseline.predict(X_test)
    elapsed = (time.time() - start) * 1000 / len(X_test)
    
    return BaselineResult(
        name="TF-IDF + SVM",
        accuracy=accuracy_score(y_test, predictions),
        precision=precision_score(y_test, predictions, zero_division=0),
        recall=recall_score(y_test, predictions, zero_division=0),
        f1=f1_score(y_test, predictions, zero_division=0),
        latency_ms=elapsed
    )


def evaluate_huggingface(texts: List[str], labels: List[int]) -> BaselineResult:
    """Evaluate HuggingFace classifier."""
    try:
        from benchmarks.baselines import HuggingFaceBaseline
        
        logger.info("Evaluating HuggingFace classifier...")
        baseline = HuggingFaceBaseline()
        
        # Use subset for speed
        subset_size = min(100, len(texts))
        subset_texts = texts[:subset_size]
        subset_labels = labels[:subset_size]
        
        start = time.time()
        predictions = baseline.predict(subset_texts)
        elapsed = (time.time() - start) * 1000 / len(subset_texts)
        
        return BaselineResult(
            name="HuggingFace DeBERTa",
            accuracy=accuracy_score(subset_labels, predictions),
            precision=precision_score(subset_labels, predictions, zero_division=0),
            recall=recall_score(subset_labels, predictions, zero_division=0),
            f1=f1_score(subset_labels, predictions, zero_division=0),
            latency_ms=elapsed
        )
    except Exception as e:
        logger.warning(f"HuggingFace evaluation failed: {e}")
        return None


def print_results(results: List[BaselineResult]):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Latency':>12}")
    print("-" * 80)
    
    for r in results:
        if r:
            print(f"{r.name:<25} {r.accuracy:>9.1%} {r.precision:>9.1%} {r.recall:>9.1%} {r.f1:>9.1%} {r.latency_ms:>10.1f}ms")
    
    print("=" * 80)


def main():
    print("=" * 60)
    print("Baseline Comparison")
    print("=" * 60)
    
    # Load test data
    texts, labels = load_test_data(limit=500)
    
    if not texts:
        print("No test data available")
        return
    
    results = []
    
    # Our model
    results.append(evaluate_mof_model(texts, labels))
    
    # TF-IDF + SVM
    results.append(evaluate_tfidf_svm(texts, labels))
    
    # HuggingFace (optional, may fail if not installed)
    hf_result = evaluate_huggingface(texts, labels)
    if hf_result:
        results.append(hf_result)
    
    # Print results
    print_results(results)
    
    # Save results
    output_path = "results/baseline_comparison.json"
    Path("results").mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([{
            "name": r.name,
            "accuracy": round(r.accuracy, 4),
            "precision": round(r.precision, 4),
            "recall": round(r.recall, 4),
            "f1": round(r.f1, 4),
            "latency_ms": round(r.latency_ms, 2)
        } for r in results if r], f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
