#!/usr/bin/env python3
"""
Fine-tune all-mpnet-base-v2 embeddings for prompt injection detection.

Uses contrastive learning to make embeddings more specialized for distinguishing
between injections, benign prompts, and benign-with-triggers.

Expected improvement: 85.4% → 92-95% recall at same FPR.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

# Disable wandb logging
os.environ['WANDB_DISABLED'] = 'true'

# Enable memory-efficient CUDA allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

sys.path.insert(0, str(Path(__file__).parent))

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

from benchmarks.benchmark_datasets import (
    load_satml_dataset,
    load_deepset_dataset,
    load_deepset_injections_only,
    load_notinject_hf_dataset,
    load_llmail_dataset,
    load_browsesafe_dataset,
    load_notinject_dataset,
    load_tensortrust_dataset
)

print("=" * 80)
print("FINE-TUNING all-mpnet-base-v2 FOR PROMPT INJECTION DETECTION")
print("=" * 80)
print()

# Configuration
BASE_MODEL = "sentence-transformers/all-mpnet-base-v2"
OUTPUT_PATH = "models/injection_aware_mpnet"
BATCH_SIZE = 4  # Reduced from 16 to avoid OOM
EPOCHS = 4
WARMUP_STEPS = 100

print(f"Configuration:")
print(f"  Base model: {BASE_MODEL}")
print(f"  Output path: {OUTPUT_PATH}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Warmup steps: {WARMUP_STEPS}")
print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
print()

# ============================================================================
# 1. Load Training Data
# ============================================================================

print("1️⃣ Loading training data...")
print()

# ============================================================================
# 0. Build the EVAL HOLDOUT set (texts that must NOT enter training)
# ============================================================================
# These limits MUST match every benchmark invocation that produces a paper
# number. --paper uses: satml=300, deepset_injections=203, notinject_hf=339,
# llmail=200. The full per-dataset table also reports deepset(full), browsesafe,
# tensortrust. We hold out a superset of all of them so no evaluated text can
# enter the encoder fine-tuning pool or the XGBoost training set.

def _normalize(t: str) -> str:
    """Normalize text so trivial whitespace/case differences still collide."""
    return " ".join(t.split()).strip().lower()

print("0️⃣ Computing eval holdout (zero-leak guard)...")
_eval_holdout_datasets = {
    "satml":              load_satml_dataset(limit=300),
    "deepset_full":       load_deepset_dataset(limit=400),
    "deepset_injections": load_deepset_injections_only(limit=203),
    "notinject_hf":       load_notinject_hf_dataset(limit=339),
    "llmail":             load_llmail_dataset(limit=200),
    "browsesafe":         load_browsesafe_dataset(limit=500),
    "tensortrust":        load_tensortrust_dataset(limit=1000),
}
EVAL_HOLDOUT = set()
for _name, _ds in _eval_holdout_datasets.items():
    for _t in _ds.texts:
        EVAL_HOLDOUT.add(_normalize(_t))
print(f"   ✓ Eval holdout contains {len(EVAL_HOLDOUT)} unique normalized texts")
print()

def _drop_holdout(samples: List[str], label_name: str) -> List[str]:
    before = len(samples)
    kept = [s for s in samples if _normalize(s) not in EVAL_HOLDOUT]
    print(f"   [holdout] {label_name}: {before} -> {len(kept)} "
          f"(removed {before - len(kept)})")
    return kept

# Load injection samples
print("   Loading injection samples...")
satml = load_satml_dataset(limit=1500)
deepset_attacks = load_deepset_dataset(limit=500, include_safe=False, include_injections=True)
llmail = load_llmail_dataset(limit=500)
browsesafe = load_browsesafe_dataset(limit=2000)

injection_samples = []
for ds in [satml, deepset_attacks, llmail]:
    injection_samples.extend([text for text, label in ds if label == 1])

# Add BrowseSafe injections
for text, label in browsesafe:
    if label == 1:
        injection_samples.append(text)

print(f"   ✓ Loaded {len(injection_samples)} injection samples")
injection_samples = _drop_holdout(injection_samples, "injection_samples")

# Load safe samples
print("   Loading safe samples...")
deepset_safe = load_deepset_dataset(limit=1000, include_safe=True, include_injections=False)
safe_samples = [text for text, label in deepset_safe if label == 0]

# Add BrowseSafe safe samples
for text, label in browsesafe:
    if label == 0:
        safe_samples.append(text)

print(f"   ✓ Loaded {len(safe_samples)} safe samples")
safe_samples = _drop_holdout(safe_samples, "safe_samples")

# Load benign-trigger samples (NotInject)
print("   Loading benign-trigger samples...")
notinject_hf = load_notinject_hf_dataset(limit=500)
notinject_synthetic = load_notinject_dataset(limit=1000)

benign_trigger_samples = []
for ds in [notinject_hf, notinject_synthetic]:
    benign_trigger_samples.extend([text for text, label in ds if label == 0])

print(f"   ✓ Loaded {len(benign_trigger_samples)} benign-trigger samples")
benign_trigger_samples = _drop_holdout(benign_trigger_samples, "benign_trigger_samples")
print()

# Guard against a pool collapsing to nothing after holdout subtraction.
assert len(injection_samples) > 100, "injection pool collapsed after holdout"
assert len(safe_samples) > 100, "safe pool collapsed after holdout"
assert len(benign_trigger_samples) > 100, "benign-trigger pool collapsed after holdout"
print(f"   Post-holdout pools: {len(injection_samples)} inj / "
      f"{len(safe_samples)} safe / {len(benign_trigger_samples)} benign-trigger")
print()

# ============================================================================
# 2. Create Contrastive Training Pairs
# ============================================================================

print("2️⃣ Creating contrastive training pairs...")
print()

def create_contrastive_pairs(
    injections: List[str],
    safe: List[str],
    benign_triggers: List[str],
    samples_per_type: int = 2000
) -> List[InputExample]:
    """
    Create contrastive pairs for fine-tuning.

    Pair types:
    1. (injection, injection) - similar (label=1.0)
    2. (safe, safe) - similar (label=1.0)
    3. (benign_trigger, benign_trigger) - similar (label=1.0)
    4. (injection, safe) - dissimilar (label=0.0)
    5. (injection, benign_trigger) - dissimilar (label=0.0)
    6. (safe, benign_trigger) - similar (label=0.8)  # Both benign but different style
    """

    np.random.seed(42)
    examples = []

    # Type 1: Injection-Injection pairs (similar)
    print(f"   Creating {samples_per_type} injection-injection pairs (similar)...")
    for _ in range(samples_per_type):
        idx1, idx2 = np.random.choice(len(injections), 2, replace=False)
        examples.append(InputExample(
            texts=[injections[idx1], injections[idx2]],
            label=1.0
        ))

    # Type 2: Safe-Safe pairs (similar)
    print(f"   Creating {samples_per_type} safe-safe pairs (similar)...")
    for _ in range(samples_per_type):
        idx1, idx2 = np.random.choice(len(safe), 2, replace=False)
        examples.append(InputExample(
            texts=[safe[idx1], safe[idx2]],
            label=1.0
        ))

    # Type 3: Benign-trigger - Benign-trigger pairs (similar)
    print(f"   Creating {samples_per_type} benign-trigger pairs (similar)...")
    for _ in range(samples_per_type):
        idx1, idx2 = np.random.choice(len(benign_triggers), 2, replace=False)
        examples.append(InputExample(
            texts=[benign_triggers[idx1], benign_triggers[idx2]],
            label=1.0
        ))

    # Type 4: Injection-Safe pairs (dissimilar)
    print(f"   Creating {samples_per_type} injection-safe pairs (dissimilar)...")
    for _ in range(samples_per_type):
        inj_idx = np.random.choice(len(injections))
        safe_idx = np.random.choice(len(safe))
        examples.append(InputExample(
            texts=[injections[inj_idx], safe[safe_idx]],
            label=0.0
        ))

    # Type 5: Injection-Benign-trigger pairs (dissimilar - CRITICAL!)
    print(f"   Creating {samples_per_type * 2} injection-benign-trigger pairs (dissimilar)...")
    # Double the samples for this critical distinction
    for _ in range(samples_per_type * 2):
        inj_idx = np.random.choice(len(injections))
        bt_idx = np.random.choice(len(benign_triggers))
        examples.append(InputExample(
            texts=[injections[inj_idx], benign_triggers[bt_idx]],
            label=0.0
        ))

    # Type 6: Safe-Benign-trigger pairs (somewhat similar - both benign)
    print(f"   Creating {samples_per_type} safe-benign-trigger pairs (similar)...")
    for _ in range(samples_per_type):
        safe_idx = np.random.choice(len(safe))
        bt_idx = np.random.choice(len(benign_triggers))
        examples.append(InputExample(
            texts=[safe[safe_idx], benign_triggers[bt_idx]],
            label=0.8  # Similar but not identical
        ))

    return examples

# Carve disjoint train / internal-val splits per pool (no overlap, seed 42).
# Previously the validation pairs were drawn from pool[-200:] while those same
# samples remained in the training pairs (an internal leak); we now split first.
VAL_TAIL = 200
np.random.seed(42)

def _split_pool(pool: List[str]) -> Tuple[List[str], List[str]]:
    pool = list(pool)
    np.random.shuffle(pool)            # deterministic under seed 42
    if len(pool) <= VAL_TAIL * 2:
        cut = max(1, len(pool) // 10)  # too small to spare 200; use 10%
    else:
        cut = VAL_TAIL
    return pool[cut:], pool[:cut]      # (train_part, val_part) — disjoint

inj_train,  inj_val  = _split_pool(injection_samples)
safe_train, safe_val = _split_pool(safe_samples)
bt_train,   bt_val   = _split_pool(benign_trigger_samples)

train_examples = create_contrastive_pairs(
    inj_train,
    safe_train,
    bt_train,
    samples_per_type=2000
)

print(f"   ✓ Created {len(train_examples)} training pairs")
print()

# Create validation set from the held-out tails (disjoint from train pairs)
print("   Creating validation pairs (disjoint from train)...")
val_examples = create_contrastive_pairs(
    inj_val,
    safe_val,
    bt_val,
    samples_per_type=100
)
print(f"   ✓ Created {len(val_examples)} validation pairs")
print()

# ============================================================================
# 3. Load Base Model and Setup Training
# ============================================================================

print("3️⃣ Loading base model...")

# Clear GPU cache before loading model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("   GPU cache cleared")

model = SentenceTransformer(BASE_MODEL)
print(f"   ✓ Loaded {BASE_MODEL}")
print(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}")
print()

# ============================================================================
# 4. Fine-tune with Contrastive Loss
# ============================================================================

print("4️⃣ Fine-tuning with contrastive learning...")
print()

# Create data loader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.CosineSimilarityLoss(model)

# Setup evaluator
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    val_examples,
    name='injection-detection-val'
)

# Train
print(f"   Training for {EPOCHS} epochs...")
print(f"   Total training steps: {len(train_dataloader) * EPOCHS}")
print()

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=WARMUP_STEPS,
    evaluator=evaluator,
    evaluation_steps=500,
    output_path=OUTPUT_PATH,
    save_best_model=True,
    show_progress_bar=True
)

print()
print(f"✅ Fine-tuning complete!")
print(f"   Model saved to: {OUTPUT_PATH}")
print()

# ============================================================================
# 5. Train XGBoost on Fine-tuned Embeddings
# ============================================================================

print("5️⃣ Training XGBoost classifier on fine-tuned embeddings...")
print()

from src.detection.embedding_classifier import EmbeddingClassifier
from datasets import Dataset

# Load the fine-tuned model
print("   Loading fine-tuned model...")
finetuned_model = SentenceTransformer(OUTPUT_PATH)

# Create classifier with fine-tuned embeddings
classifier = EmbeddingClassifier(threshold=0.700)
classifier.model_name = "injection_aware_mpnet"
classifier.embedding_model = finetuned_model

# Prepare balanced training dataset (same as before)
print("   Preparing balanced training dataset...")

# Sample balanced data
np.random.seed(42)
n_injection = 2800
n_safe = 2800
n_benign_trigger = 1400

# Sample from the train halves only (already holdout-subtracted AND disjoint
# from the encoder's internal validation tail) so the classifier is doubly clean.
selected_injections = np.random.choice(inj_train, min(n_injection, len(inj_train)), replace=False).tolist()
selected_safe = np.random.choice(safe_train, min(n_safe, len(safe_train)), replace=False).tolist()
selected_benign = np.random.choice(bt_train, min(n_benign_trigger, len(bt_train)), replace=False).tolist()

all_texts = selected_safe + selected_benign + selected_injections
all_labels = [0] * len(selected_safe) + [0] * len(selected_benign) + [1] * len(selected_injections)

training_dataset = Dataset.from_dict({'text': all_texts, 'label': all_labels})
training_dataset = training_dataset.shuffle(seed=42)

print(f"   Training samples: {len(training_dataset)}")
print(f"     - Safe: {len(selected_safe)}")
print(f"     - Benign-trigger: {len(selected_benign)}")
print(f"     - Injection: {len(selected_injections)}")
print()

# Train XGBoost
print("   Training XGBoost classifier...")
stats = classifier.train_on_dataset(
    training_dataset,
    batch_size=1000,
    validation_split=True
)

print()
print("   Training statistics:")
print(f"     Train AUC: {stats.get('train_auc', 'N/A'):.4f}")
print(f"     Val AUC: {stats.get('val_auc', 'N/A'):.4f}")
print()

# Save the classifier
model_path = f"models/injection_aware_mpnet_classifier.json"
classifier.save_model(model_path)
print(f"✅ Classifier saved to: {model_path}")
print()

# ============================================================================
# 6. Evaluate and Compare
# ============================================================================

print("6️⃣ Evaluating fine-tuned model...")
print()

# Load test datasets
print("   Loading test datasets...")
test_satml = load_satml_dataset(limit=300)
test_deepset = load_deepset_dataset(limit=400)
test_notinject = load_notinject_hf_dataset(limit=339)
test_llmail = load_llmail_dataset(limit=200)
test_browsesafe = load_browsesafe_dataset(limit=500)

test_texts = []
test_labels = []
for ds in [test_satml, test_deepset, test_notinject, test_llmail, test_browsesafe]:
    test_texts.extend([text for text, label in ds])
    test_labels.extend([label for text, label in ds])

test_labels = np.array(test_labels)

print(f"   Test set: {len(test_texts)} samples")
print(f"     - Injections: {np.sum(test_labels == 1)}")
print(f"     - Safe: {np.sum(test_labels == 0)}")
print()

# Evaluate at different thresholds
print("   Evaluating at multiple thresholds...")
print()
print(f"{'Threshold':>10} | {'Recall':>8} | {'FPR':>8} | {'Precision':>9} | {'F1':>8} | {'Accuracy':>8}")
print("-" * 75)

test_probs = classifier.predict_proba(test_texts)[:, 1]

for threshold in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85]:
    y_pred = (test_probs >= threshold).astype(int)

    tp = np.sum((y_pred == 1) & (test_labels == 1))
    tn = np.sum((y_pred == 0) & (test_labels == 0))
    fp = np.sum((y_pred == 1) & (test_labels == 0))
    fn = np.sum((y_pred == 0) & (test_labels == 1))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(test_labels)

    print(f"{threshold:>10.2f} | {recall:>8.1%} | {fpr:>8.1%} | {precision:>9.1%} | {f1:>8.1%} | {accuracy:>8.1%}")

print()

# ============================================================================
# Summary
# ============================================================================

print("=" * 80)
print("FINE-TUNING COMPLETE")
print("=" * 80)
print()
print("📁 Outputs:")
print(f"   - Fine-tuned embeddings: {OUTPUT_PATH}/")
print(f"   - Trained classifier: {model_path}")
print()
print("Next steps:")
print("   1. Run benchmarks: python -m benchmarks.run_benchmark --model {model_path}")
print("   2. Compare with baseline all-mpnet-base-v2")
print("   3. Deploy if performance meets targets (>90% recall, <5% FPR)")
print()
