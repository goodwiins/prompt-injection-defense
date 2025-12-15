#!/usr/bin/env python3
"""
Train BIT Model (Balanced Intent Training) - Paper-Accurate Configuration

Implements the exact BIT strategy from the paper:
- Total: 10,240 samples
- 40% injection samples (4,096)
- 40% safe samples (4,096)
- 20% benign-trigger samples (2,048)
- Training: 8,192 samples (80%)
- Test: 2,048 samples (20%)
- Random seed: 42
- Weighted loss: w_benign_trigger = 2.0

Dataset sources:
- Injections: SaTML CTF 2024, deepset/prompt-injections, synthetic
- Safe: Alpaca/conversational, synthetic
- Benign-triggers: NotInject (HuggingFace), synthetic
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

import structlog
import numpy as np
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
from datasets import Dataset

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ]
)
logger = structlog.get_logger()


def optimize_threshold(classifier, X_val: List[str], y_val: List[int], 
                      target_recall: float = 0.98) -> float:
    """
    Optimize classification threshold for target recall.
    Paper claims 97.1% recall, so we target 97%.
    """
    print(f"\n🎯 Optimizing threshold for {target_recall*100:.0f}% recall...")
    
    # Get probability predictions
    probs = classifier.predict_proba(X_val)
    if probs.ndim == 2:
        probs = probs[:, 1]  # Probability of positive class
    
    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
    
    # Find threshold that achieves target recall
    valid_idx = recalls[:-1] >= target_recall
    if not any(valid_idx):
        logger.warning(f"Cannot achieve {target_recall*100}% recall!")
        # Find best available recall
        best_idx = np.argmax(recalls[:-1])
        optimal_threshold = float(thresholds[best_idx])
        print(f"   Best achievable recall: {recalls[best_idx]*100:.1f}%")
    else:
        # Among thresholds achieving target recall, choose highest threshold (lowest FPR)
        valid_thresholds = thresholds[valid_idx]
        optimal_threshold = float(max(valid_thresholds))
    
    print(f"   Optimal threshold: {optimal_threshold:.4f}")
    return optimal_threshold


def evaluate_on_notinject(classifier, threshold: float, count: int = 500) -> dict:
    """
    Evaluate over-defense on NotInject-style samples.
    Paper target: <1.8% FPR with 95% CI [0.8-3.4%]
    """
    from src.utils.bit_dataset_loader import BITDatasetLoader
    
    print("\n" + "=" * 60)
    print("OVER-DEFENSE EVALUATION (NotInject)")
    print("=" * 60)
    
    loader = BITDatasetLoader()
    notinject_samples = loader._generate_benign_trigger_samples(count)
    notinject_texts = [s["text"] for s in notinject_samples]
    
    # Get predictions
    probs = classifier.predict_proba(notinject_texts)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    fpr = sum(preds) / len(preds)
    
    # Calculate 95% confidence interval (Wilson score)
    n = len(preds)
    p_hat = fpr
    z = 1.96  # 95% CI
    denominator = 1 + z**2/n
    center = (p_hat + z**2/(2*n)) / denominator
    margin = z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)) / denominator
    ci_low, ci_high = max(0, center - margin), min(1, center + margin)
    
    print(f"\n✨ Over-Defense (FPR) on NotInject:")
    print(f"   Samples tested: {n}")
    print(f"   False positives: {sum(preds)}")
    print(f"   FPR: {fpr:.4f} ({fpr*100:.2f}%)")
    print(f"   95% CI: [{ci_low:.4f}, {ci_high:.4f}] ({ci_low*100:.2f}%-{ci_high*100:.2f}%)")
    print(f"   Paper target: 1.8% [0.8-3.4% CI]")
    
    if fpr <= 0.018:
        print("   ✅ PASS - Meets paper claims!")
    elif fpr <= 0.05:
        print("   ⚠️  ACCEPTABLE - Within 5%")
    else:
        print("   ❌ FAIL - Exceeds acceptable FPR")
    
    return {
        "fpr": float(fpr),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "samples": n,
        "false_positives": int(sum(preds))
    }


def main():
    """Main BIT training function with paper-accurate configuration."""
    print("=" * 70)
    print("BIT Training: Balanced Intent Training - Paper Configuration")
    print("=" * 70)
    print("Paper spec: 10,240 samples = 8,192 train + 2,048 test")
    print("Composition: 40% injections, 40% safe, 20% benign-triggers")
    print("Weighted loss: benign-triggers = 2.0x")
    print("=" * 70)
    
    # Load data using BIT dataset loader
    from src.utils.bit_dataset_loader import BITDatasetLoader
    
    loader = BITDatasetLoader()
    train_dataset, test_dataset, train_weights = loader.load_bit_training_data()
    
    # Extract train/test data
    X_train = train_dataset["text"]
    y_train = train_dataset["label"]
    X_test = test_dataset["text"]
    y_test = test_dataset["label"]
    
    print(f"\n📦 Final Data Splits:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Create validation split from training for threshold optimization
    from sklearn.model_selection import train_test_split
    
    X_train_final, X_val, y_train_final, y_val, w_train_final, w_val = train_test_split(
        X_train, y_train, train_weights,
        test_size=0.15, random_state=42, stratify=y_train
    )
    
    print(f"   (After val split: {len(X_train_final)} train, {len(X_val)} val)")
    
    # Train classifier with weighted loss
    from src.detection.embedding_classifier import EmbeddingClassifier
    
    classifier = EmbeddingClassifier(
        model_name="all-MiniLM-L6-v2",
        model_dir="models",
        threshold=0.5  # Initial threshold, will be optimized
    )
    
    print("\n🚀 Training BIT classifier with weighted loss...")
    print("   (This may take 3-5 minutes)\n")
    
    # Create Dataset object for training
    train_data = Dataset.from_dict({
        "text": X_train_final,
        "label": y_train_final,
    })
    
    # Use train_on_dataset method with weights
    stats = classifier.train_on_dataset(
        train_data,
        sample_weights=w_train_final,
        batch_size=500,
        validation_split=True
    )
    
    print("   ✓ Training complete!")
    print(f"   Training AUC: {stats.get('train_auc', 0):.4f}")
    if 'val_auc' in stats:
        print(f"   Validation AUC: {stats['val_auc']:.4f}")
    
    # Optimize threshold on validation set
    optimal_threshold = optimize_threshold(
        classifier, X_val, y_val, 
        target_recall=0.97  # Paper claims 97.1%
    )
    classifier.threshold = optimal_threshold
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    
    test_probs = classifier.predict_proba(X_test)[:, 1]
    y_pred = (test_probs >= optimal_threshold).astype(int)
    
    print(classification_report(y_test, y_pred, target_names=["Safe", "Injection"]))
    
    # Calculate test metrics
    tp = sum((p == 1 and l == 1) for p, l in zip(y_pred, y_test))
    tn = sum((p == 0 and l == 0) for p, l in zip(y_pred, y_test))
    fp = sum((p == 1 and l == 0) for p, l in zip(y_pred, y_test))
    fn = sum((p == 0 and l == 1) for p, l in zip(y_pred, y_test))
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fpr_test = fp / (fp + tn) if (fp + tn) > 0 else 0
    accuracy = (tp + tn) / len(y_test)
    
    print(f"\n📊 Test Metrics:")
    print(f"   Accuracy: {accuracy*100:.1f}%")
    print(f"   Recall (Detection): {recall*100:.1f}%")
    print(f"   Precision: {precision*100:.1f}%")
    print(f"   FPR: {fpr_test*100:.2f}%")
    
    # Evaluate on NotInject for over-defense
    notinject_results = evaluate_on_notinject(classifier, optimal_threshold, count=500)
    
    # Save model
    model_path = "models/bit_xgboost_paper_config.json"
    classifier.save_model(model_path)
    
    # Save comprehensive metadata
    metadata = {
        "model_name": "all-MiniLM-L6-v2",
        "threshold": float(optimal_threshold),
        "paper_configuration": {
            "total_samples": 10240,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "injection_pct": 0.40,
            "safe_pct": 0.40,
            "benign_trigger_pct": 0.20,
            "random_seed": 42,
            "train_test_split": "80/20"
        },
        "bit_composition": {
            "injections": 0.4,
            "safe": 0.4,
            "benign_triggers": 0.2
        },
        "sample_weights": {
            "benign_trigger": 2.0,
            "other": 1.0
        },
        "training_stats": stats,
        "test_results": {
            "accuracy": float(accuracy),
            "recall": float(recall),
            "precision": float(precision),
            "fpr": float(fpr_test),
            "samples": len(y_test)
        },
        "notinject_results": notinject_results,
        "is_trained": True,
        "paper_aligned": True
    }
    
    metadata_path = model_path.replace(".json", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print(f"\n✅ Model saved to {model_path}")
    print(f"✅ Metadata saved to {metadata_path}")
    print(f"   Threshold: {optimal_threshold:.4f}")
    
    print("\n" + "=" * 70)
    print("Training complete! Paper-aligned model ready.")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run full benchmark: python -m benchmarks.run_benchmark --all")
    print("  2. Verify claims: python test_paper_claims.py")
    print("  3. Run evaluation: python run_comprehensive_eval.py")
    print("=" * 70)
    
    return stats


if __name__ == "__main__":
    main()
