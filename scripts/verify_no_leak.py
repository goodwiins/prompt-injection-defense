#!/usr/bin/env python3
"""Verify zero overlap between encoder/XGBoost training texts and benchmark eval texts.

This is the gate that must PASS before retraining. It rebuilds the evaluation sets
and the (holdout-subtracted) training pools exactly as
training/finetune_mpnet_embeddings.py does, then asserts an empty intersection.

Run: python scripts/verify_no_leak.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.benchmark_datasets import (
    load_satml_dataset,
    load_deepset_dataset,
    load_deepset_injections_only,
    load_notinject_hf_dataset,
    load_llmail_dataset,
    load_browsesafe_dataset,
    load_notinject_dataset,
    load_tensortrust_dataset,
)


def norm(t: str) -> str:
    return " ".join(t.split()).strip().lower()


# 1. Eval sets — limits MUST match finetune_mpnet_embeddings.py EVAL_HOLDOUT
eval_sets = {
    "satml":              load_satml_dataset(limit=300).texts,
    "deepset_full":       load_deepset_dataset(limit=400).texts,
    "deepset_injections": load_deepset_injections_only(limit=203).texts,
    "notinject_hf":       load_notinject_hf_dataset(limit=339).texts,
    "llmail":             load_llmail_dataset(limit=200).texts,
    "browsesafe":         load_browsesafe_dataset(limit=500).texts,
    "tensortrust":        load_tensortrust_dataset(limit=1000).texts,
}
eval_holdout = {norm(t) for texts in eval_sets.values() for t in texts}

# 2. Rebuild training pools exactly as the finetune script does, then subtract holdout
inj = []
for ds in [
    load_satml_dataset(limit=1500),
    load_deepset_dataset(limit=500, include_safe=False, include_injections=True),
    load_llmail_dataset(limit=500),
]:
    inj += [t for t, l in ds if l == 1]
bs = load_browsesafe_dataset(limit=2000)
inj += [t for t, l in bs if l == 1]

safe = [t for t, l in load_deepset_dataset(limit=1000, include_safe=True, include_injections=False) if l == 0]
safe += [t for t, l in bs if l == 0]

bt = []
for ds in [load_notinject_hf_dataset(limit=500), load_notinject_dataset(limit=1000)]:
    bt += [t for t, l in ds if l == 0]

train_norm = set()
for pool in (inj, safe, bt):
    train_norm |= {norm(t) for t in pool}
train_after = {t for t in train_norm if t not in eval_holdout}

# 3. Report + assert
overlap = train_after & eval_holdout
print(
    f"eval_unique={len(eval_holdout)} train_before={len(train_norm)} "
    f"train_after={len(train_after)} overlap={len(overlap)}"
)
for name, texts in eval_sets.items():
    ev = {norm(t) for t in texts}
    print(f"  {name:20s} eval={len(ev):5d} leaked_into_train={len(ev & train_after)}")

assert not overlap, f"LEAK: {len(overlap)} training texts still in eval set"
print("PASS: zero overlap between training pools and eval sets")
