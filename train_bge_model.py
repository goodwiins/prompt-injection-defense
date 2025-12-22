#!/usr/bin/env python3
"""
Train BGE-Large Classifier for Prompt Injection Detection

Uses BAAI/bge-large-en-v1.5 embeddings (1024 dimensions) with XGBoost classifier.
Similar to BIT training but with BGE embeddings instead of MiniLM.
"""

import sys
import os
import json
import random
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from typing import List, Dict, Tuple

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

# Use BGE-large-en-v1.5
BGE_MODEL = "BAAI/bge-large-en-v1.5"

# Trigger words for identifying benign-trigger samples
TRIGGER_WORDS = {
    "ignore", "forget", "override", "system", "admin", "bypass",
    "disable", "unlock", "secret", "hidden", "mode", "instructions",
    "rules", "restrictions", "prompt", "jailbreak", "disregard",
    "previous", "new", "execute", "command", "password", "access",
    "root", "sudo", "hack", "exploit", "inject", "payload"
}


def has_trigger_words(text: str) -> bool:
    """Check if text contains any trigger words."""
    text_lower = text.lower()
    return any(word in text_lower for word in TRIGGER_WORDS)


def load_attack_samples(target_count: int = 4000) -> List[Dict]:
    """Load attack samples from SaTML and deepset."""
    samples = []
    
    try:
        from datasets import load_dataset
        
        per_source = target_count // 2
        
        # SaTML attacks
        print(f"\n📥 Loading SaTML attacks (target: {per_source})...")
        ds = load_dataset("ethz-spylab/ctf-satml24", "interaction_chats",
                         split="attack", streaming=True)
        count = 0
        for sample in tqdm(ds, total=per_source, desc="SaTML"):
            if count >= per_source:
                break
            history = sample.get("history", [])
            if history and len(history) > 0:
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
        
        # Deepset injections
        print(f"\n📥 Loading deepset attacks (target: {per_source})...")
        ds = load_dataset("deepset/prompt-injections", split="train")
        count = 0
        for sample in ds:
            if count >= per_source:
                break
            if sample.get("label") == 1:
                text = sample.get("text", "")
                if text.strip():
                    samples.append({
                        "text": text,
                        "label": 1,
                        "type": "injection"
                    })
                    count += 1
        
        print(f"   ✓ Loaded {count} deepset attacks")
        print(f"   Total attack samples: {len(samples)}")
        
    except Exception as e:
        logger.error(f"Failed to load attack samples: {e}")
        raise
    
    return samples


def load_benign_samples(target_count: int = 4000) -> List[Dict]:
    """Load diverse benign samples."""
    samples = []
    
    # Try local safe prompts
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
        ]
        
        fill_words = [
            "New York", "Tokyo", "Paris", "London", "Berlin",
            "Python", "JavaScript", "cooking", "fitness", "meditation",
            "birthday message", "cover letter", "resume", "essay",
            "science", "art", "music", "dance", "photography",
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
    
    print(f"   Total safe samples: {len(samples)}")
    return samples


def generate_notinject_samples(count: int = 2000) -> List[Dict]:
    """Load NotInject-style benign samples with trigger words."""
    samples = []
    
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
    
    # Generate synthetic to fill gap
    if len(samples) < count:
        remaining = count - len(samples)
        print(f"\n📝 Generating {remaining} synthetic NotInject samples...")
        
        trigger_words = list(TRIGGER_WORDS)
        templates = [
            "How do I {} whitespace in Python regex?",
            "Can you explain how to {} a method in Java?",
            "The {} key in the hotel room wasn't working.",
            "I need to {} my flight reservation for tomorrow.",
            "The {} feature in this software is very useful.",
            "Can you help me {} this configuration file?",
            "Please {} my subscription to the newsletter.",
            "How can I {} my phone's screen timeout?",
        ]
        
        for _ in range(remaining):
            template = random.choice(templates)
            trigger = random.choice(trigger_words)
            text = template.format(trigger)
            samples.append({
                "text": text,
                "label": 0,
                "type": "benign_trigger"
            })
    
    return samples


def main():
    """Main BGE training function."""
    print("=" * 60)
    print("BGE-Large Classifier Training")
    print(f"Embedding Model: {BGE_MODEL}")
    print("=" * 60)
    
    # Collect samples
    all_samples = []
    all_samples.extend(load_attack_samples(target_count=4000))
    all_samples.extend(load_benign_samples(target_count=4000))
    all_samples.extend(generate_notinject_samples(count=2000))
    
    # Balance dataset (40/40/20)
    injections = [s for s in all_samples if s["type"] == "injection"]
    safe = [s for s in all_samples if s["type"] == "safe"]
    benign_triggers = [s for s in all_samples if s["type"] == "benign_trigger"]
    
    target_total = 10000
    n_injection = int(target_total * 0.4)
    n_safe = int(target_total * 0.4)
    n_benign_trigger = int(target_total * 0.2)
    
    balanced = []
    balanced.extend(random.choices(injections, k=n_injection) if len(injections) < n_injection else random.sample(injections, n_injection))
    balanced.extend(random.choices(safe, k=n_safe) if len(safe) < n_safe else random.sample(safe, n_safe))
    balanced.extend(random.choices(benign_triggers, k=n_benign_trigger) if len(benign_triggers) < n_benign_trigger else random.sample(benign_triggers, n_benign_trigger))
    
    random.shuffle(balanced)
    
    texts = [s["text"] for s in balanced]
    labels = [s["label"] for s in balanced]
    weights = [2.0 if s["type"] == "benign_trigger" else 1.0 for s in balanced]
    
    print(f"\n📊 Final composition ({len(balanced)} samples):")
    composition = Counter([s["type"] for s in balanced])
    for sample_type, count in composition.items():
        pct = 100 * count / len(balanced)
        print(f"   {sample_type}: {count} ({pct:.1f}%)")
    
    # Split data
    X_train, X_temp, y_train, y_temp, w_train, w_temp = train_test_split(
        texts, labels, weights, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test, w_val, w_test = train_test_split(
        X_temp, y_temp, w_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\n📦 Data splits:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Initialize BGE embedding model
    print(f"\n🔄 Loading BGE embedding model: {BGE_MODEL}")
    from sentence_transformers import SentenceTransformer
    import xgboost as xgb
    
    embedding_model = SentenceTransformer(BGE_MODEL)
    print(f"   ✓ Loaded! Embedding dimension: {embedding_model.get_sentence_embedding_dimension()}")
    
    # Generate embeddings
    print("\n🔄 Generating embeddings for training data...")
    X_train_emb = embedding_model.encode(X_train, show_progress_bar=True, convert_to_numpy=True)
    print("🔄 Generating embeddings for validation data...")
    X_val_emb = embedding_model.encode(X_val, show_progress_bar=True, convert_to_numpy=True)
    print("🔄 Generating embeddings for test data...")
    X_test_emb = embedding_model.encode(X_test, show_progress_bar=True, convert_to_numpy=True)
    
    # Train XGBoost classifier
    print("\n🚀 Training XGBoost classifier...")
    classifier = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.01,
        reg_lambda=1,
        use_label_encoder=False,
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    )
    
    classifier.fit(
        X_train_emb, y_train,
        sample_weight=w_train,
        eval_set=[(X_val_emb, y_val)],
        verbose=True
    )
    
    print("   ✓ Training complete!")
    
    # Optimize threshold
    print("\n🎯 Optimizing threshold for 98% recall...")
    val_probs = classifier.predict_proba(X_val_emb)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    
    target_recall = 0.98
    valid_idx = recalls[:-1] >= target_recall
    if any(valid_idx):
        optimal_threshold = float(max(thresholds[valid_idx]))
    else:
        optimal_threshold = 0.5
    
    print(f"   Optimal threshold: {optimal_threshold:.3f}")
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    
    test_probs = classifier.predict_proba(X_test_emb)[:, 1]
    y_pred = (test_probs >= optimal_threshold).astype(int)
    
    print(classification_report(y_test, y_pred, target_names=["Safe", "Injection"]))
    
    # Calculate metrics
    tp = sum((p == 1 and l == 1) for p, l in zip(y_pred, y_test))
    tn = sum((p == 0 and l == 0) for p, l in zip(y_pred, y_test))
    fp = sum((p == 1 and l == 0) for p, l in zip(y_pred, y_test))
    fn = sum((p == 0 and l == 1) for p, l in zip(y_pred, y_test))
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr_test = fp / (fp + tn) if (fp + tn) > 0 else 0
    auc_score = roc_auc_score(y_test, test_probs)
    
    print(f"\nTest Recall: {recall*100:.1f}%")
    print(f"Test FPR: {fpr_test*100:.1f}%")
    print(f"Test AUC: {auc_score:.4f}")
    
    # Save model
    model_path = "models/bge-large-en-v1.5_classifier.json"
    classifier.save_model(model_path)
    
    # Save metadata
    metadata = {
        "model_name": BGE_MODEL,
        "embedding_dimension": 1024,
        "threshold": float(optimal_threshold),
        "training_samples": len(X_train),
        "test_results": {
            "recall": float(recall),
            "fpr": float(fpr_test),
            "auc": float(auc_score)
        },
        "is_trained": True,
        "classes_": [0, 1]
    }
    
    metadata_path = model_path.replace(".json", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Model saved to {model_path}")
    print(f"✅ Metadata saved to {metadata_path}")
    print(f"   Threshold: {optimal_threshold:.3f}")
    
    print("\n" + "=" * 60)
    print("Training complete! To run benchmark:")
    print(f"  python -m benchmarks.run_benchmark --datasets satml deepset --model {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
