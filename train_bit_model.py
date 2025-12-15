#!/usr/bin/env python3
"""
Train BIT Model (Balanced Intent Training)

Implements the BIT strategy from the paper:
- 40% injection samples
- 40% safe samples  
- 20% benign-trigger samples (safe but with trigger words)
- Weighted loss: w_benign_trigger = 2.0

This corrects the previous training issues:
- Proper 40/40/20 composition (was 61/6/33)
- Weighted loss for benign triggers (was missing)
- Threshold optimization on validation set
- No deepset safe samples (caused 55.7% FPR)
"""

import sys
import os
import json
import random
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent))

import structlog
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ]
)
logger = structlog.get_logger()

# Trigger words for identifying benign-trigger samples
TRIGGER_WORDS = {
    "ignore", "forget", "override", "system", "admin", "bypass",
    "disable", "unlock", "secret", "hidden", "mode", "instructions",
    "rules", "restrictions", "prompt", "jailbreak", "disregard",
    "previous", "new", "execute", "command", "password", "access",
    "root", "sudo", "hack", "exploit", "inject", "payload"
}


def extract_text_from_html(html: str) -> str:
    """
    Extract readable text from HTML content.
    This helps MiniLM understand HTML-embedded attacks better.
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean up whitespace
        text = soup.get_text(separator=' ')
        text = ' '.join(text.split())
        
        # If extraction results in very short text, keep some HTML context
        if len(text) < 50 and len(html) > 100:
            # Keep a snippet of the original HTML for context
            text = html[:500] + " [TEXT]: " + text
        
        return text
    except Exception:
        # Fallback: simple regex-based extraction
        import re
        text = re.sub(r'<[^>]+>', ' ', html)
        text = ' '.join(text.split())
        return text


def has_trigger_words(text: str) -> bool:
    """Check if text contains any trigger words."""
    text_lower = text.lower()
    return any(word in text_lower for word in TRIGGER_WORDS)


def get_sample_type(text: str, label: int) -> str:
    """
    Categorize sample for BIT training:
    - injection: attack samples
    - safe: benign without triggers
    - benign_trigger: benign WITH triggers (NotInject-style)
    """
    if label == 1:
        return "injection"
    elif has_trigger_words(text):
        return "benign_trigger"
    else:
        return "safe"


def generate_notinject_samples(count: int = 2000) -> List[Dict]:
    """Load NotInject-style benign samples with trigger words.
    
    First tries to load from HuggingFace, then generates synthetic if needed.
    """
    samples = []
    
    # First, try to load from HuggingFace NotInject dataset
    try:
        from datasets import load_dataset
        print(f"\n📥 Loading NotInject from HuggingFace (target: {count})...")
        
        loaded = 0
        for split_name in ["NotInject_one", "NotInject_two", "NotInject_three"]:
            if loaded >= count:
                break
            ds = load_dataset("leolee99/NotInject", split=split_name)
            for sample in ds:
                if loaded >= count:
                    break
                text = sample.get("text") or sample.get("prompt") or str(sample)
                if text and text.strip():
                    samples.append({
                        "text": text,
                        "label": 0,
                        "type": "benign_trigger"
                    })
                    loaded += 1
        
        print(f"   ✓ Loaded {len(samples)} NotInject HF samples")
        
    except Exception as e:
        logger.warning(f"Could not load NotInject HF: {e}, generating synthetic")
    
    # Generate synthetic to fill the gap
    if len(samples) < count:
        remaining = count - len(samples)
        print(f"\n📝 Generating {remaining} synthetic NotInject samples...")
        
        trigger_words = list(TRIGGER_WORDS)
        templates = [
            "How do I {trigger} whitespace in Python regex?",
            "Can you explain how to {trigger} a method in Java?",
            "What's the best way to {trigger} null pointer exceptions?",
            "The {trigger} key in the hotel room wasn't working.",
            "I need to {trigger} my flight reservation for tomorrow.",
            "My grandmother taught me how to {trigger} vegetables.",
            "The {trigger} feature in this software is very useful.",
            "Can you help me {trigger} this configuration file?",
            "I want to {trigger} my old emails from the archive.",
            "The teacher asked us to {trigger} the previous chapter.",
            "Please {trigger} my subscription to the newsletter.",
            "How can I {trigger} my phone's screen timeout?",
            "The {trigger} button on my keyboard is broken.",
            "I'd like to {trigger} my appointment for next week.",
            "The {trigger} permissions are needed for this task.",
            "Can you show me how to {trigger} user settings?",
            "We need to {trigger} this outdated policy.",
            "The IT team will {trigger} the old server.",
            "My manager asked me to {trigger} these files.",
            "The {trigger} process takes about 30 minutes.",
            "Please don't {trigger} my changes to the document.",
            "I accidentally clicked {trigger} on my browser.",
            "The software will {trigger} unused data automatically.",
            "How do I {trigger} pending notifications?",
            "The {trigger} mechanism in this machine needs repair.",
        ]
        
        for _ in range(remaining):
            template = random.choice(templates)
            trigger = random.choice(trigger_words)
            text = template.format(trigger=trigger)
            samples.append({
                "text": text,
                "label": 0,
                "type": "benign_trigger"
            })
    
    return samples


def load_attack_samples(target_count: int = 4000, include_browsesafe: bool = False) -> List[Dict]:
    """Load attack samples from SaTML, deepset, and optionally BrowseSafe as supplement."""
    samples = []
    
    try:
        from datasets import load_dataset
        
        # Load full text samples (SaTML + deepset), then add BrowseSafe as supplement
        per_text_source = target_count // 2  # 50/50 split for text
        
        # SaTML attacks (text-based)
        print(f"\n📥 Loading SaTML attacks (target: {per_text_source})...")
        ds = load_dataset("ethz-spylab/ctf-satml24", "interaction_chats",
                         split="attack", streaming=True)
        count = 0
        for sample in tqdm(ds, total=per_text_source, desc="SaTML"):
            if count >= per_text_source:
                break
            history = sample.get("history", [])
            if history and len(history) > 0:
                # Get user message
                for msg in history:
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if content.strip() and len(content) > 10:
                            samples.append({
                                "text": content,
                                "label": 1,
                                "type": "injection"
                            })
                            count += 1
                            break
        
        print(f"   ✓ Loaded {count} SaTML attacks")
        
        # Deepset injections ONLY (not safe samples!)
        print(f"\n📥 Loading deepset attacks (target: {per_text_source})...")
        ds = load_dataset("deepset/prompt-injections", split="train")
        count = 0
        for sample in ds:
            if count >= per_text_source:
                break
            if sample.get("label") == 1:  # Only injections!
                text = sample.get("text", "")
                if text.strip():
                    samples.append({
                        "text": text,
                        "label": 1,
                        "type": "injection"
                    })
                    count += 1
        
        print(f"   ✓ Loaded {count} deepset attacks")
        
        # BrowseSafe HTML-embedded attacks (with text extraction)
        if include_browsesafe:
            browsesafe_target = 2000  # Supplement to text samples (reduced from 5K)
            print(f"\n📥 Loading BrowseSafe attacks (target: {browsesafe_target})...")
            ds = load_dataset("perplexity-ai/browsesafe-bench", split="train")
            count = 0
            for sample in ds:
                if count >= browsesafe_target:
                    break
                # Get label - BrowseSafe uses "yes"/"no" for malicious/benign
                label = sample.get("label", "")
                if label == "yes":  # Only malicious samples
                    html = sample.get("content", sample.get("html", sample.get("text", "")))
                    if html and html.strip():
                        # Extract text from HTML for better embeddings
                        text = extract_text_from_html(html)
                        samples.append({
                            "text": text,
                            "label": 1,
                            "type": "injection"
                        })
                        count += 1
            
            print(f"   ✓ Loaded {count} BrowseSafe attacks (text extracted from HTML)")
        
        print(f"   Total attack samples: {len(samples)}")
        
    except Exception as e:
        logger.error(f"Failed to load attack samples: {e}")
        raise
    
    return samples


def load_diverse_benign_samples(target_count: int = 4000, include_browsesafe: bool = False) -> List[Dict]:
    """
    Load diverse benign samples from local data, synthetic, and optionally BrowseSafe.
    Avoids deepset safe samples (which cause high FPR).
    """
    samples = []
    
    # Try to load from local synthetic_safe_prompts.json
    try:
        safe_path = Path("data/synthetic_safe_prompts.json")
        if safe_path.exists():
            print(f"\n📥 Loading local safe prompts...")
            with open(safe_path, "r") as f:
                safe_data = json.load(f)
                
            if isinstance(safe_data, dict) and "safe_prompts" in safe_data:
                prompts = safe_data["safe_prompts"]
            elif isinstance(safe_data, list):
                prompts = safe_data
            else:
                prompts = []
                
            for text in prompts[:target_count // 2]:
                if isinstance(text, str) and text.strip() and not has_trigger_words(text):
                    samples.append({
                        "text": text,
                        "label": 0,
                        "type": "safe"
                    })
            print(f"   ✓ Loaded {len(samples)} local safe prompts")
    except Exception as e:
        logger.warning(f"Could not load local safe prompts: {e}")
    
    # Load BrowseSafe benign HTML samples (with text extraction)
    if include_browsesafe:
        try:
            from datasets import load_dataset
            browsesafe_benign_target = 2000  # Match attack samples (reduced from 5K)
            print(f"\n📥 Loading BrowseSafe benign HTML samples (target: {browsesafe_benign_target})...")
            ds = load_dataset("perplexity-ai/browsesafe-bench", split="train")
            count = 0
            for sample in ds:
                if count >= browsesafe_benign_target:
                    break
                label = sample.get("label", "")
                if label == "no":  # Benign HTML
                    html = sample.get("content", sample.get("html", sample.get("text", "")))
                    if html and html.strip():
                        # Extract text from HTML for better embeddings
                        text = extract_text_from_html(html)
                        samples.append({
                            "text": text,
                            "label": 0,
                            "type": "safe"
                        })
                        count += 1
            print(f"   ✓ Loaded {count} BrowseSafe benign HTML samples (text extracted)")
        except Exception as e:
            logger.warning(f"Could not load BrowseSafe benign: {e}")
    
    # Generate more if needed
    if len(samples) < target_count:
        remaining = target_count - len(samples)
        print(f"\n📝 Generating {remaining} additional safe prompts...")
        
        safe_templates = [
            "What's the weather like in {}?",
            "Can you help me write a {} for my {}?",
            "Explain {} in simple terms.",
            "What are some healthy {} ideas?",
            "How do I learn {} programming?",
            "Tell me about the history of {}.",
            "What's a good {} recommendation for beginners?",
            "How can I improve my {}?",
            "What are the best practices for {}?",
            "Can you help me plan a trip to {}?",
            "What's the difference between {} and {}?",
            "How do I cook {}?",
            "What books should I read about {}?",
            "Can you explain how {} works?",
            "What's the best way to {}?",
        ]
        
        fill_words = [
            "New York", "Tokyo", "Paris", "London", "Berlin",
            "Python", "JavaScript", "cooking", "fitness", "meditation",
            "birthday message", "cover letter", "resume", "essay",
            "science", "art", "music", "dance", "photography",
            "chicken", "pasta", "salad", "soup", "dessert",
            "machine learning", "web development", "data analysis",
        ]
        
        for _ in range(remaining):
            template = random.choice(safe_templates)
            words = random.sample(fill_words, min(2, template.count("{}")))
            try:
                text = template.format(*words)
            except:
                text = template.replace("{}", random.choice(fill_words))
            
            if not has_trigger_words(text):
                samples.append({
                    "text": text,
                    "label": 0,
                    "type": "safe"
                })
        
        print(f"   ✓ Generated additional safe prompts")
    
    print(f"   Total safe samples: {len(samples)}")
    return samples


def balance_to_bit_composition(samples: List[Dict], target_total: int = 10000) -> Tuple[List[str], List[int], List[float]]:
    """
    Balance dataset to BIT composition (40/40/20).
    
    Returns: (texts, labels, sample_weights)
    """
    # Separate by type
    injections = [s for s in samples if s["type"] == "injection"]
    safe = [s for s in samples if s["type"] == "safe"]
    benign_triggers = [s for s in samples if s["type"] == "benign_trigger"]
    
    print(f"\n📊 Available samples before balancing:")
    print(f"   Injections: {len(injections)}")
    print(f"   Safe: {len(safe)}")
    print(f"   Benign-triggers: {len(benign_triggers)}")
    
    # Calculate target sizes (40/40/20)
    n_injection = int(target_total * 0.4)
    n_safe = int(target_total * 0.4)
    n_benign_trigger = int(target_total * 0.2)
    
    # Sample to targets (with replacement if needed)
    balanced = []
    
    if len(injections) >= n_injection:
        balanced.extend(random.sample(injections, n_injection))
    else:
        # Oversample if not enough
        balanced.extend(random.choices(injections, k=n_injection))
        logger.warning(f"Oversampled injections: {len(injections)} -> {n_injection}")
    
    if len(safe) >= n_safe:
        balanced.extend(random.sample(safe, n_safe))
    else:
        balanced.extend(random.choices(safe, k=n_safe))
        logger.warning(f"Oversampled safe: {len(safe)} -> {n_safe}")
    
    if len(benign_triggers) >= n_benign_trigger:
        balanced.extend(random.sample(benign_triggers, n_benign_trigger))
    else:
        balanced.extend(random.choices(benign_triggers, k=n_benign_trigger))
        logger.warning(f"Oversampled benign-triggers: {len(benign_triggers)} -> {n_benign_trigger}")
    
    random.shuffle(balanced)
    
    # Extract texts, labels, and weights
    texts = [s["text"] for s in balanced]
    labels = [s["label"] for s in balanced]
    weights = [2.0 if s["type"] == "benign_trigger" else 1.0 for s in balanced]
    
    # Report final composition
    final_types = [s["type"] for s in balanced]
    composition = Counter(final_types)
    
    print(f"\n✅ Final BIT composition ({len(balanced)} samples):")
    for sample_type, count in composition.items():
        pct = 100 * count / len(balanced)
        print(f"   {sample_type}: {count} ({pct:.1f}%)")
    
    return texts, labels, weights


def optimize_threshold(classifier, X_val: List[str], y_val: List[int], target_recall: float = 0.98) -> float:
    """
    Optimize classification threshold for target recall.
    Paper claims 98.8% recall, so we target 98%.
    """
    print(f"\n🎯 Optimizing threshold for {target_recall*100:.0f}% recall...")
    
    # Get probability predictions
    probs = classifier.predict_proba(X_val)
    if probs.ndim == 2:
        probs = probs[:, 1]  # Probability of positive class
    
    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
    
    # Find threshold that achieves target recall
    valid_idx = recalls[:-1] >= target_recall  # Last element is recall=1.0 at threshold=0
    if not any(valid_idx):
        logger.warning(f"Cannot achieve {target_recall*100}% recall!")
        optimal_threshold = 0.5
    else:
        # Among thresholds achieving target recall, choose highest threshold (lowest FPR)
        valid_thresholds = thresholds[valid_idx]
        optimal_threshold = float(max(valid_thresholds))
    
    print(f"   Optimal threshold: {optimal_threshold:.3f}")
    return optimal_threshold


def main():
    """Main BIT training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train BIT model for prompt injection detection")
    parser.add_argument(
        "--include-browsesafe", 
        action="store_true",
        help="Include BrowseSafe HTML dataset in training (adds ~2K attacks + ~2K benign)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/bit_xgboost_model.json",
        help="Output model path (default: models/bit_xgboost_model.json)"
    )
    args = parser.parse_args()
    
    include_browsesafe = args.include_browsesafe
    model_output_path = args.output
    
    print("=" * 60)
    print("BIT Training: Balanced Intent Training")
    print("Paper: 40% injections, 40% safe, 20% benign-triggers")
    print("Weighted loss: benign-triggers = 2.0x")
    if include_browsesafe:
        print("🌐 Including BrowseSafe HTML dataset")
    print("=" * 60)
    
    # Collect all samples
    all_samples = []
    
    # 1. Load attack samples (40%)
    all_samples.extend(load_attack_samples(target_count=4000, include_browsesafe=include_browsesafe))
    
    # 2. Load diverse benign samples (40%)
    all_samples.extend(load_diverse_benign_samples(target_count=4000, include_browsesafe=include_browsesafe))
    
    # 3. Generate NotInject benign-trigger samples (20%)
    print("\n📝 Generating NotInject benign-trigger samples...")
    notinject = generate_notinject_samples(count=2000)
    all_samples.extend(notinject)
    print(f"   ✓ Generated {len(notinject)} benign-trigger samples")
    
    # Balance to BIT composition
    texts, labels, weights = balance_to_bit_composition(all_samples, target_total=10000)
    
    # Split with stratification (70% train, 15% val, 15% test)
    X_train, X_temp, y_train, y_temp, w_train, w_temp = train_test_split(
        texts, labels, weights,
        test_size=0.3, random_state=42, stratify=labels
    )
    
    X_val, X_test, y_val, y_test, w_val, w_test = train_test_split(
        X_temp, y_temp, w_temp,
        test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\n📦 Data splits:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Train classifier with weighted loss
    from src.detection.embedding_classifier import EmbeddingClassifier
    
    classifier = EmbeddingClassifier(
        model_name="all-MiniLM-L6-v2",
        model_dir="models",
        threshold=0.5  # Initial threshold, will be optimized
    )
    
    print("\n🚀 Training BIT classifier with weighted loss...")
    print("   (This may take 2-3 minutes)\n")
    
    # Create Dataset object for training
    from datasets import Dataset
    train_data = Dataset.from_dict({
        "text": X_train,
        "label": y_train,
    })
    
    # Use train_on_dataset method with weights
    stats = classifier.train_on_dataset(
        train_data,
        sample_weights=w_train,
        batch_size=500,
        validation_split=True
    )
    print("   ✓ Training complete!")
    print(f"   Training AUC: {stats.get('train_auc', 0):.4f}")
    if 'val_auc' in stats:
        print(f"   Validation AUC: {stats['val_auc']:.4f}")
    
    # Optimize threshold on validation set
    optimal_threshold = optimize_threshold(classifier, X_val, y_val, target_recall=0.98)
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
    fpr_test = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"\nTest Recall: {recall*100:.1f}%")
    print(f"Test FPR: {fpr_test*100:.1f}%")
    
    # Critical: Test over-defense on NotInject
    print("\n" + "=" * 60)
    print("OVER-DEFENSE EVALUATION (NotInject-style)")
    print("=" * 60)
    
    notinject_test = generate_notinject_samples(count=500)
    notinject_texts = [s["text"] for s in notinject_test]
    notinject_probs = classifier.predict_proba(notinject_texts)[:, 1]
    notinject_preds = (notinject_probs >= optimal_threshold).astype(int)
    
    fpr = sum(notinject_preds) / len(notinject_preds)
    
    # Calculate 95% confidence interval (Wilson score)
    n = len(notinject_preds)
    p_hat = fpr
    z = 1.96  # 95% CI
    denominator = 1 + z**2/n
    center = (p_hat + z**2/(2*n)) / denominator
    margin = z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)) / denominator
    ci_low, ci_high = max(0, center - margin), min(1, center + margin)
    
    print(f"\n✨ Over-Defense (FPR) on NotInject:")
    print(f"   Point estimate: {fpr:.4f} ({fpr*100:.2f}%)")
    print(f"   95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   Paper target: <1.5% (0.015)")
    
    if fpr <= 0.015:
        print("   ✅ PASS - Meets paper claims!")
    elif fpr <= 0.05:
        print("   ⚠️  ACCEPTABLE - Within 5%")
    else:
        print("   ❌ FAIL - Exceeds acceptable FPR")
    
    # Save model
    model_path = model_output_path  # Use CLI-specified path
    classifier.save_model(model_path)
    
    # Update metadata with correct values
    metadata = {
        "model_name": "all-MiniLM-L6-v2",
        "threshold": float(optimal_threshold),
        "training_stats": stats,
        "bit_composition": {
            "injections": 0.4,
            "safe": 0.4,
            "benign_triggers": 0.2
        },
        "sample_weights": {
            "benign_trigger": 2.0,
            "other": 1.0
        },
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_results": {
            "recall": float(recall),
            "fpr": float(fpr_test),
            "notinject_fpr": float(fpr),
            "notinject_fpr_ci": [float(ci_low), float(ci_high)]
        },
        "is_trained": True
    }
    
    metadata_path = model_path.replace(".json", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Model saved to {model_path}")
    print(f"✅ Metadata saved to {metadata_path}")
    print(f"   Threshold: {optimal_threshold:.3f}")
    
    print("\n" + "=" * 60)
    print("Training complete! Next steps:")
    print("  1. Run full benchmark: python -m benchmarks.run_benchmark --all")
    print("  2. Verify claims: python test_paper_claims.py")
    print("=" * 60)
    
    return stats


if __name__ == "__main__":
    main()
