#!/usr/bin/env python3
"""
Custom training script for all-mpnet-base-v2 model with BIT (Balanced Injection Training)
"""

import torch
from datasets import Dataset
from src.detection.embedding_classifier import EmbeddingClassifier
import numpy as np
from tqdm import tqdm
import json

print("🚀 Starting Custom Training with all-mpnet-base-v2...")
print("   Target Total Samples: 7000")
print("   - Injections: 2800")
print("   - Safe: 2800")
print("   - Benign Triggers: 1400")
print()

# 1️⃣ Loading Data
print("1️⃣ Loading Data...")
print()

# Import dataset loading utilities
from benchmarks.benchmark_datasets import (
    load_satml_dataset,
    load_deepset_dataset,
    load_browsesafe_dataset,
    load_notinject_hf_dataset,
    load_notinject_dataset
)

# Load attack samples
print("📥 Loading SaTML attacks (target: 1500)...")
satml_ds = load_satml_dataset(limit=1500)
print(f"   ✓ Loaded {len(satml_ds)} SaTML attacks")
print()

print("📥 Loading deepset attacks (target: 1500)...")
deepset_ds = load_deepset_dataset(limit=1500, include_safe=False, include_injections=True)
print(f"   ✓ Loaded {len(deepset_ds)} deepset attacks")
print()

print("📥 Loading BrowseSafe dataset (target: 2000 attacks + safe)...")
browsesafe_ds = load_browsesafe_dataset(limit=4000)
print(f"   ✓ Loaded {len(browsesafe_ds)} BrowseSafe samples (text extracted from HTML)")
print(f"     - Attacks: {browsesafe_ds.injection_count}")
print(f"     - Safe: {browsesafe_ds.safe_count}")
print()

# Combine attack samples
attack_samples = []
for text, label in satml_ds:
    attack_samples.append({'text': text, 'label': label})
for text, label in deepset_ds:
    if label == 1:
        attack_samples.append({'text': text, 'label': label})
for text, label in browsesafe_ds:
    if label == 1:
        attack_samples.append({'text': text, 'label': label})

print(f"   Total attack samples: {len(attack_samples)}")
print()

# Load safe samples from BrowseSafe
safe_samples = []
for text, label in browsesafe_ds:
    if label == 0:
        safe_samples.append({'text': text, 'label': label})

print(f"   BrowseSafe safe samples: {len(safe_samples)}")
print()

# Load benign-trigger samples (NotInject)
print("📥 Loading NotInject from HuggingFace (target: 1500)...")
notinject_hf_ds = load_notinject_hf_dataset(limit=1500)
print(f"   ✓ Loaded {len(notinject_hf_ds)} NotInject HF samples")
print()

# Add synthetic NotInject if needed
notinject_samples = []
for text, label in notinject_hf_ds:
    notinject_samples.append({'text': text, 'label': label})

remaining_notinject = 1500 - len(notinject_samples)
if remaining_notinject > 0:
    print(f"📝 Generating {remaining_notinject} synthetic NotInject samples...")
    synthetic_ds = load_notinject_dataset(limit=remaining_notinject)
    for text, label in synthetic_ds:
        notinject_samples.append({'text': text, 'label': label})
    print(f"   ✓ Generated {remaining_notinject} synthetic NotInject samples")
print()

print(f"   Total safe samples before balancing: {len(safe_samples)}")
print(f"   Total benign-trigger samples: {len(notinject_samples)}")
print()

# 2️⃣ Balance Dataset
print("2️⃣ Balancing Dataset...")
print()

print("📊 Available samples before balancing:")
print(f"   Injections: {len(attack_samples)}")
print(f"   Safe: {len(safe_samples)}")
print(f"   Benign-triggers: {len(notinject_samples)}")
print()

# Target distribution
target_total = 7000
target_injection = 2800
target_safe = 2800
target_benign = 1400

# Sample to reach targets
np.random.seed(42)

injection_texts = [s['text'] for s in attack_samples]
safe_texts = [s['text'] for s in safe_samples]
benign_texts = [s['text'] for s in notinject_samples]

# Sample with replacement if needed
if len(injection_texts) < target_injection:
    injection_indices = np.random.choice(len(injection_texts), target_injection, replace=True)
    injection_texts = [injection_texts[i] for i in injection_indices]
else:
    injection_texts = np.random.choice(injection_texts, target_injection, replace=False).tolist()

if len(safe_texts) < target_safe:
    safe_indices = np.random.choice(len(safe_texts), target_safe, replace=True)
    safe_texts = [safe_texts[i] for i in safe_indices]
else:
    safe_texts = np.random.choice(safe_texts, target_safe, replace=False).tolist()

if len(benign_texts) < target_benign:
    benign_indices = np.random.choice(len(benign_texts), target_benign, replace=True)
    benign_texts = [benign_texts[i] for i in benign_indices]
else:
    benign_texts = np.random.choice(benign_texts, target_benign, replace=False).tolist()

# Combine all samples
all_texts = safe_texts + benign_texts + injection_texts
all_labels = [0] * len(safe_texts) + [0] * len(benign_texts) + [1] * len(injection_texts)

# Create dataset
dataset_dict = {
    'text': all_texts,
    'label': all_labels
}
full_dataset = Dataset.from_dict(dataset_dict)

# Shuffle
full_dataset = full_dataset.shuffle(seed=42)

print(f"✅ Final BIT composition ({len(full_dataset)} samples):")
n_safe = len(safe_texts)
n_benign = len(benign_texts)
n_injection = len(injection_texts)
print(f"   safe: {n_safe} ({n_safe/len(full_dataset)*100:.1f}%)")
print(f"   benign_trigger: {n_benign} ({n_benign/len(full_dataset)*100:.1f}%)")
print(f"   injection: {n_injection} ({n_injection/len(full_dataset)*100:.1f}%)")
print()

# Split into train/val/test
train_size = 4900
val_size = 1050
test_size = 1050

train_dataset = Dataset.from_dict({
    'text': full_dataset['text'][:train_size],
    'label': full_dataset['label'][:train_size]
})
val_dataset = Dataset.from_dict({
    'text': full_dataset['text'][train_size:train_size+val_size],
    'label': full_dataset['label'][train_size:train_size+val_size]
})
test_dataset = Dataset.from_dict({
    'text': full_dataset['text'][train_size+val_size:],
    'label': full_dataset['label'][train_size+val_size:]
})

print(f"📦 Splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
print()

# 3️⃣ Initialize Model
print("3️⃣ Initializing all-mpnet-base-v2...")
classifier = EmbeddingClassifier(
    model_name="all-mpnet-base-v2",
    threshold=0.764
)

# FIX: Enable GPU for XGBoost if available
if torch.cuda.is_available():
    print("   ✅ Enabling GPU for XGBoost")
    classifier.classifier.set_params(device='cuda', tree_method='hist')
else:
    print("   ℹ️  GPU not available, using CPU")
print()

# 4️⃣ Train
print("4️⃣ Training...")
print()

# Combine train + val for final training
combined_texts = list(train_dataset['text']) + list(val_dataset['text'])
combined_labels = list(train_dataset['label']) + list(val_dataset['label'])
combined_dataset = Dataset.from_dict({
    'text': combined_texts,
    'label': combined_labels
})

stats = classifier.train_on_dataset(
    combined_dataset,
    batch_size=1000,
    validation_split=True
)

print()
print("📊 Training Statistics:")
print(json.dumps(stats, indent=2, default=str))
print()

# 5️⃣ Evaluate on test set
print("5️⃣ Evaluating on test set...")
test_stats = classifier.evaluate(
    test_dataset['text'],
    test_dataset['label']
)

print()
print("📊 Test Results:")
print(json.dumps(test_stats, indent=2, default=str))
print()

print("✅ Training complete!")
print(f"   Model saved to: models/all-mpnet-base-v2_classifier.json")
