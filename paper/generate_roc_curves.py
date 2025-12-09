#!/usr/bin/env python
"""
Generate ROC and PR Curves for Paper

Creates publication-quality ROC and Precision-Recall curves
comparing the full model against baselines.
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from datasets import load_dataset


def get_predictions_with_scores(detector, texts: List[str]) -> Tuple[List[int], List[float]]:
    """Get predictions and probability scores."""
    predictions = detector.predict(texts)
    probas = detector.predict_proba(texts)
    scores = [p[1] for p in probas]  # Probability of injection class
    return list(predictions), scores


def keyword_baseline(texts: List[str]) -> Tuple[List[int], List[float]]:
    """Simple keyword-based baseline."""
    keywords = [
        "ignore", "forget", "disregard", "bypass", "override",
        "jailbreak", "DAN", "pretend", "roleplay", "system prompt"
    ]
    
    predictions = []
    scores = []
    
    for text in texts:
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw in text_lower)
        score = min(matches / 3, 1.0)  # Normalize to [0, 1]
        predictions.append(1 if matches >= 1 else 0)
        scores.append(score)
    
    return predictions, scores


def plot_roc_curves(results: Dict, output_path: str, dataset_name: str):
    """Generate ROC curve figure."""
    plt.figure(figsize=(8, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    colors = ['#6366f1', '#22c55e', '#f59e0b', '#ef4444']
    
    for i, (name, data) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(data['labels'], data['scores'])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                 label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - {dataset_name}', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_pr_curves(results: Dict, output_path: str, dataset_name: str):
    """Generate Precision-Recall curve figure."""
    plt.figure(figsize=(8, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    colors = ['#6366f1', '#22c55e', '#f59e0b', '#ef4444']
    
    for i, (name, data) in enumerate(results.items()):
        precision, recall, _ = precision_recall_curve(data['labels'], data['scores'])
        ap = average_precision_score(data['labels'], data['scores'])
        
        plt.plot(recall, precision, color=colors[i % len(colors)], lw=2,
                 label=f'{name} (AP = {ap:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curves - {dataset_name}', fontsize=14)
    plt.legend(loc='lower left', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("Generating ROC and PR Curves for Paper")
    print("=" * 60)
    
    from src.detection.embedding_classifier import EmbeddingClassifier
    
    # Create output directory
    figures_dir = Path("paper/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load detector
    print("\nLoading models...")
    detector = EmbeddingClassifier()
    detector.load_model("models/bit_classifier.json")
    
    # Load deepset dataset
    print("\nLoading deepset dataset...")
    ds = load_dataset("deepset/prompt-injections", split="train", streaming=True)
    texts, labels = [], []
    for i, sample in enumerate(ds):
        if i >= 500:
            break
        texts.append(sample["text"])
        labels.append(sample["label"])
    
    print(f"Loaded {len(texts)} samples")
    
    # Get predictions from different methods
    print("\nGenerating predictions...")
    
    # Full model
    _, mof_scores = get_predictions_with_scores(detector, texts)
    
    # Keyword baseline
    _, keyword_scores = keyword_baseline(texts)
    
    # Embedding-only (use same detector but just embedding scores)
    embedding_scores = mof_scores  # In practice, same for this setup
    
    results = {
        "BIT (Ours)": {"labels": labels, "scores": mof_scores},
        "Keyword Baseline": {"labels": labels, "scores": keyword_scores},
    }
    
    # Generate ROC curves
    print("\nGenerating ROC curves...")
    plot_roc_curves(results, str(figures_dir / "roc_deepset.png"), "deepset")
    
    # Generate PR curves
    print("Generating PR curves...")
    plot_pr_curves(results, str(figures_dir / "pr_deepset.png"), "deepset")
    
    # Calculate AUC values for report
    print("\n" + "=" * 60)
    print("AUC Summary")
    print("=" * 60)
    
    auc_results = {}
    for name, data in results.items():
        fpr, tpr, _ = roc_curve(data['labels'], data['scores'])
        roc_auc = auc(fpr, tpr)
        ap = average_precision_score(data['labels'], data['scores'])
        auc_results[name] = {"roc_auc": roc_auc, "pr_auc": ap}
        print(f"{name:20} ROC-AUC: {roc_auc:.4f}  PR-AUC: {ap:.4f}")
    
    # Save results
    with open(figures_dir / "roc_pr_results.json", "w") as f:
        json.dump(auc_results, f, indent=2)
    
    print(f"\n✅ Figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
