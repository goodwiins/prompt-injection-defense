#!/usr/bin/env python
"""
Error Analysis for Prompt Injection Detection

Analyzes failure cases to identify patterns and improvement opportunities.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

logger = structlog.get_logger()


@dataclass
class ErrorCase:
    """Represents a single error case."""
    text: str
    true_label: int
    predicted_label: int
    confidence: float
    error_type: str  # 'false_positive' or 'false_negative'
    category: str = ""
    length: int = 0


def categorize_error(text: str, error_type: str) -> str:
    """Categorize an error case based on text content."""
    text_lower = text.lower()
    
    if error_type == "false_positive":
        # Benign text incorrectly flagged as injection
        if any(w in text_lower for w in ["code", "programming", "function", "python"]):
            return "coding_question"
        elif any(w in text_lower for w in ["system", "admin", "server", "database"]):
            return "sysadmin_question"
        elif any(w in text_lower for w in ["security", "vulnerability", "attack"]):
            return "security_education"
        elif any(w in text_lower for w in ["explain", "what", "how", "why"]):
            return "general_question"
        else:
            return "other_benign"
    else:
        # Injection missed by detector
        if any(w in text_lower for w in ["ignore", "forget", "disregard"]):
            return "instruction_override"
        elif any(w in text_lower for w in ["dan", "jailbreak", "bypass"]):
            return "jailbreak_attempt"
        elif any(w in text_lower for w in ["system prompt", "initial prompt", "reveal"]):
            return "prompt_extraction"
        elif any(w in text_lower for w in ["pretend", "roleplay", "imagine"]):
            return "roleplay_attack"
        elif any(w in text_lower for w in ["base64", "decode", "encode"]):
            return "encoded_attack"
        else:
            return "other_injection"


def analyze_errors(
    texts: List[str],
    true_labels: List[int],
    predictions: List[int],
    confidences: List[float]
) -> Dict:
    """
    Analyze error cases from predictions.
    
    Args:
        texts: Input texts
        true_labels: Ground truth labels
        predictions: Model predictions
        confidences: Prediction confidence scores
        
    Returns:
        Dictionary with error analysis results
    """
    errors = []
    
    for text, true_label, pred, conf in zip(texts, true_labels, predictions, confidences):
        if true_label != pred:
            error_type = "false_positive" if pred == 1 else "false_negative"
            category = categorize_error(text, error_type)
            
            errors.append(ErrorCase(
                text=text,
                true_label=true_label,
                predicted_label=pred,
                confidence=conf,
                error_type=error_type,
                category=category,
                length=len(text)
            ))
    
    # Aggregate statistics
    fp_count = sum(1 for e in errors if e.error_type == "false_positive")
    fn_count = sum(1 for e in errors if e.error_type == "false_negative")
    
    # Category breakdown
    fp_categories = Counter(e.category for e in errors if e.error_type == "false_positive")
    fn_categories = Counter(e.category for e in errors if e.error_type == "false_negative")
    
    # Confidence distribution for errors
    fp_confidences = [e.confidence for e in errors if e.error_type == "false_positive"]
    fn_confidences = [e.confidence for e in errors if e.error_type == "false_negative"]
    
    # Length analysis
    fp_lengths = [e.length for e in errors if e.error_type == "false_positive"]
    fn_lengths = [e.length for e in errors if e.error_type == "false_negative"]
    
    # Find hardest cases (high confidence errors)
    hardest_fp = sorted(
        [e for e in errors if e.error_type == "false_positive"],
        key=lambda x: x.confidence,
        reverse=True
    )[:5]
    
    hardest_fn = sorted(
        [e for e in errors if e.error_type == "false_negative"],
        key=lambda x: x.confidence,
        reverse=True
    )[:5]
    
    return {
        "summary": {
            "total_errors": len(errors),
            "false_positives": fp_count,
            "false_negatives": fn_count,
            "error_rate": len(errors) / len(texts) if texts else 0
        },
        "false_positive_categories": dict(fp_categories),
        "false_negative_categories": dict(fn_categories),
        "confidence_stats": {
            "fp_mean": float(np.mean(fp_confidences)) if fp_confidences else 0,
            "fp_std": float(np.std(fp_confidences)) if fp_confidences else 0,
            "fn_mean": float(np.mean(fn_confidences)) if fn_confidences else 0,
            "fn_std": float(np.std(fn_confidences)) if fn_confidences else 0
        },
        "length_stats": {
            "fp_mean": float(np.mean(fp_lengths)) if fp_lengths else 0,
            "fn_mean": float(np.mean(fn_lengths)) if fn_lengths else 0
        },
        "hardest_false_positives": [
            {"text": e.text[:100], "confidence": e.confidence, "category": e.category}
            for e in hardest_fp
        ],
        "hardest_false_negatives": [
            {"text": e.text[:100], "confidence": e.confidence, "category": e.category}
            for e in hardest_fn
        ]
    }


def print_error_report(analysis: Dict):
    """Print formatted error analysis report."""
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS REPORT")
    print("=" * 70)
    
    s = analysis["summary"]
    print(f"\nTotal Errors: {s['total_errors']}")
    print(f"  False Positives: {s['false_positives']}")
    print(f"  False Negatives: {s['false_negatives']}")
    print(f"  Error Rate: {s['error_rate']:.2%}")
    
    print("\n" + "-" * 70)
    print("FALSE POSITIVE BREAKDOWN (Benign flagged as injection)")
    print("-" * 70)
    for cat, count in sorted(analysis["false_positive_categories"].items(), 
                             key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    
    print("\n" + "-" * 70)
    print("FALSE NEGATIVE BREAKDOWN (Injections missed)")
    print("-" * 70)
    for cat, count in sorted(analysis["false_negative_categories"].items(), 
                             key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    
    print("\n" + "-" * 70)
    print("HARDEST FALSE POSITIVES (High confidence errors)")
    print("-" * 70)
    for i, case in enumerate(analysis["hardest_false_positives"], 1):
        print(f"\n{i}. [{case['category']}] Confidence: {case['confidence']:.3f}")
        print(f"   Text: {case['text']}...")
    
    print("\n" + "-" * 70)
    print("HARDEST FALSE NEGATIVES (Missed with high confidence)")
    print("-" * 70)
    for i, case in enumerate(analysis["hardest_false_negatives"], 1):
        print(f"\n{i}. [{case['category']}] Confidence: {case['confidence']:.3f}")
        print(f"   Text: {case['text']}...")
    
    print("\n" + "=" * 70)


def suggest_improvements(analysis: Dict) -> List[str]:
    """Generate improvement suggestions based on error analysis."""
    suggestions = []
    
    fp_cats = analysis["false_positive_categories"]
    fn_cats = analysis["false_negative_categories"]
    
    # High FP in specific categories
    if fp_cats.get("coding_question", 0) > 2:
        suggestions.append(
            "Add more coding-related benign samples to NotInject training data"
        )
    
    if fp_cats.get("security_education", 0) > 2:
        suggestions.append(
            "Add security education samples to distinguish from real attacks"
        )
    
    # High FN in specific categories
    if fn_cats.get("roleplay_attack", 0) > 2:
        suggestions.append(
            "Add more roleplay/persona attack patterns to adversarial training"
        )
    
    if fn_cats.get("encoded_attack", 0) > 2:
        suggestions.append(
            "Implement preprocessing to decode base64/hex before classification"
        )
    
    if fn_cats.get("instruction_override", 0) > 2:
        suggestions.append(
            "Strengthen detection of instruction override patterns"
        )
    
    # Confidence analysis
    if analysis["confidence_stats"]["fn_mean"] > 0.7:
        suggestions.append(
            "Consider lowering detection threshold - missed attacks have high base confidence"
        )
    
    if analysis["confidence_stats"]["fp_mean"] < 0.6:
        suggestions.append(
            "Consider raising detection threshold - false positives have low confidence"
        )
    
    if not suggestions:
        suggestions.append("Model is performing well - continue monitoring for edge cases")
    
    return suggestions


def main():
    """Run error analysis on benchmark results."""
    print("=" * 60)
    print("Error Analysis for Prompt Injection Detection")
    print("=" * 60)
    
    from src.detection.embedding_classifier import EmbeddingClassifier
    from datasets import load_dataset
    
    print("\nLoading model and data...")
    
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
    
    # Get predictions with confidence
    print("Running predictions...")
    predictions = detector.predict(texts)
    probas = detector.predict_proba(texts)
    confidences = [p[1] if pred == 1 else p[0] for p, pred in zip(probas, predictions)]
    
    # Analyze errors
    print("Analyzing errors...")
    analysis = analyze_errors(texts, labels, predictions, confidences)
    
    # Print report
    print_error_report(analysis)
    
    # Generate suggestions
    print("\n" + "=" * 70)
    print("IMPROVEMENT SUGGESTIONS")
    print("=" * 70)
    suggestions = suggest_improvements(analysis)
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. {suggestion}")
    
    # Save results
    output = {
        "analysis": analysis,
        "suggestions": suggestions
    }
    
    output_path = Path("results/error_analysis.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
