#!/usr/bin/env python3
"""
Analyze False Positive Rate on deepset dataset.

This script diagnoses why the model has high FPR (26.5%) on deepset
when the paper claims <2.3%.

Outputs:
- Sample of false positives with their scores
- Analysis of trigger words in false positives
- Recommendations for improvement
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.embedding_classifier import EmbeddingClassifier
from benchmarks.datasets import load_deepset_dataset

# Trigger words that commonly cause false positives
TRIGGER_WORDS = [
    "ignore", "system", "override", "bypass", "admin", 
    "jailbreak", "prompt", "instruction", "forget", "disable"
]


def analyze_deepset_fpr(threshold: float = 0.5, limit: int = 500):
    """
    Analyze false positives on deepset dataset.
    
    Args:
        threshold: Detection threshold to use
        limit: Max samples to analyze
    """
    print("="*60)
    print("DEEPSET FALSE POSITIVE RATE ANALYSIS")
    print("="*60)
    print(f"\nThreshold: {threshold}")
    
    # Load model
    print("\n[1] Loading model...")
    classifier = EmbeddingClassifier(
        model_name="all-MiniLM-L6-v2",
        model_dir="models",
        threshold=threshold
    )
    
    model_path = "models/bit_xgboost_model.json"
    classifier.load_model(model_path)
    classifier.threshold = threshold  # Force threshold
    print(f"    Model loaded, threshold set to: {classifier.threshold}")
    
    # Load deepset dataset
    print("\n[2] Loading deepset dataset...")
    dataset = load_deepset_dataset(limit=limit)
    print(f"    Loaded {len(dataset)} samples ({dataset.injection_count} injections, {dataset.safe_count} safe)")
    
    # Get predictions
    print("\n[3] Running predictions...")
    predictions = classifier.batch_predict(dataset.texts)
    
    # Analyze false positives
    false_positives = []
    true_positives = []
    true_negatives = []
    false_negatives = []
    
    for i, (text, label) in enumerate(dataset):
        pred = predictions[i]
        score = pred["score"]
        is_pred_injection = pred["is_injection"]
        
        if label == 0 and is_pred_injection:  # False Positive
            false_positives.append({"text": text, "score": score, "idx": i})
        elif label == 1 and is_pred_injection:  # True Positive
            true_positives.append({"text": text, "score": score, "idx": i})
        elif label == 0 and not is_pred_injection:  # True Negative
            true_negatives.append({"text": text, "score": score, "idx": i})
        else:  # False Negative
            false_negatives.append({"text": text, "score": score, "idx": i})
    
    # Calculate metrics
    total_safe = len(false_positives) + len(true_negatives)
    total_attacks = len(true_positives) + len(false_negatives)
    fpr = len(false_positives) / total_safe if total_safe > 0 else 0
    recall = len(true_positives) / total_attacks if total_attacks > 0 else 0
    
    print(f"\n[4] RESULTS:")
    print(f"    Total Safe Samples: {total_safe}")
    print(f"    Total Attack Samples: {total_attacks}")
    print(f"    False Positives: {len(false_positives)}")
    print(f"    True Negatives: {len(true_negatives)}")
    print(f"    False Positive Rate: {fpr*100:.1f}%")
    print(f"    Recall: {recall*100:.1f}%")
    
    # Analyze false positives
    print(f"\n[5] FALSE POSITIVE ANALYSIS")
    print("-"*60)
    
    if false_positives:
        # Check for trigger words
        trigger_counts = {w: 0 for w in TRIGGER_WORDS}
        
        print("\nSample False Positives (up to 10):")
        for fp in false_positives[:10]:
            text = fp["text"][:100] + "..." if len(fp["text"]) > 100 else fp["text"]
            print(f"  [Score: {fp['score']:.3f}] {text}")
            
            # Count trigger words
            text_lower = fp["text"].lower()
            for word in TRIGGER_WORDS:
                if word in text_lower:
                    trigger_counts[word] += 1
        
        print(f"\nTrigger Word Frequency in False Positives:")
        for word, count in sorted(trigger_counts.items(), key=lambda x: -x[1]):
            pct = count / len(false_positives) * 100
            print(f"  '{word}': {count} ({pct:.1f}%)")
    
    # Score distribution analysis
    print(f"\n[6] SCORE DISTRIBUTION")
    print("-"*60)
    
    safe_scores = [predictions[i]["score"] for i, (_, label) in enumerate(dataset) if label == 0]
    attack_scores = [predictions[i]["score"] for i, (_, label) in enumerate(dataset) if label == 1]
    
    print(f"Safe samples score distribution:")
    print(f"  Min: {min(safe_scores):.3f}, Max: {max(safe_scores):.3f}")
    print(f"  Mean: {np.mean(safe_scores):.3f}, Std: {np.std(safe_scores):.3f}")
    print(f"  Percentiles: p25={np.percentile(safe_scores, 25):.3f}, p50={np.percentile(safe_scores, 50):.3f}, p75={np.percentile(safe_scores, 75):.3f}")
    
    print(f"\nAttack samples score distribution:")
    print(f"  Min: {min(attack_scores):.3f}, Max: {max(attack_scores):.3f}")
    print(f"  Mean: {np.mean(attack_scores):.3f}, Std: {np.std(attack_scores):.3f}")
    print(f"  Percentiles: p25={np.percentile(attack_scores, 25):.3f}, p50={np.percentile(attack_scores, 50):.3f}, p75={np.percentile(attack_scores, 75):.3f}")
    
    # Recommendations
    print(f"\n[7] RECOMMENDATIONS")
    print("-"*60)
    
    if fpr > 0.10:
        print("! HIGH FPR DETECTED. Possible causes:")
        print("  1. Training data lacks deepset-style negative examples")
        print("  2. Threshold may need calibration based on score distribution")
        
        # Suggest optimal threshold
        optimal_threshold = None
        for t in np.arange(0.3, 0.9, 0.05):
            test_fpr = sum(1 for s in safe_scores if s >= t) / len(safe_scores)
            test_recall = sum(1 for s in attack_scores if s >= t) / len(attack_scores)
            if test_fpr <= 0.025 and test_recall >= 0.90:
                optimal_threshold = t
                break
        
        if optimal_threshold:
            print(f"\n  Suggested threshold for FPR<2.5% and Recall>90%: {optimal_threshold:.2f}")
        else:
            print("\n  No threshold achieves both FPR<2.5% AND Recall>90%")
            print("  Trade-off table:")
            for t in [0.4, 0.5, 0.6, 0.7, 0.8]:
                test_fpr = sum(1 for s in safe_scores if s >= t) / len(safe_scores)
                test_recall = sum(1 for s in attack_scores if s >= t) / len(attack_scores)
                print(f"    Threshold {t:.1f}: FPR={test_fpr*100:.1f}%, Recall={test_recall*100:.1f}%")
    
    return {
        "fpr": fpr,
        "recall": recall,
        "false_positives": len(false_positives),
        "safe_scores": safe_scores,
        "attack_scores": attack_scores
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze deepset FPR")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    
    analyze_deepset_fpr(threshold=args.threshold, limit=args.limit)
