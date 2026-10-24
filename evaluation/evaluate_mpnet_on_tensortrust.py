#!/usr/bin/env python3
"""Evaluate the Injection-Aware MPNet detector on TensorTrust.

The TensorTrust dataset is cached locally in ``data/tensortrust_cache.json``
(populated from the qxcv/tensor-trust HuggingFace dataset).  All cached
samples are adversarial attacks (label = 1).  This script measures recall
(equivalently accuracy, since there are no benign samples) and writes the
result to ``results/tensortrust_mpnet_evaluation.json``.

Run from the repo root:

    PYTHONPATH=. .venv/bin/python evaluation/evaluate_mpnet_on_tensortrust.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_datasets import load_tensortrust_dataset  # noqa: E402
from src.detection.embedding_classifier import EmbeddingClassifier  # noqa: E402


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + z * z / total
    centre = p + z * z / (2 * total)
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return (centre - half) / denom, (centre + half) / denom


def main(limit: int = 500) -> None:
    print("Loading TensorTrust …")
    ds = load_tensortrust_dataset(limit=limit, use_cache=True)
    texts = list(ds.texts)
    labels = list(ds.labels)
    n = len(texts)
    print(f"  {n} samples loaded")

    print("Loading Injection-Aware MPNet detector at θ=0.764 …")
    clf = EmbeddingClassifier(
        model_name=str(ROOT / "models" / "injection_aware_mpnet"),
        threshold=0.764,
    )
    clf.load_model(str(ROOT / "models" / "injection_aware_mpnet_classifier.json"))

    t0 = time.time()
    preds = clf.predict(texts)
    duration_ms = (time.time() - t0) * 1000.0

    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    rec_lo, rec_hi = wilson_ci(tp, tp + fn)

    result = {
        "dataset": "TensorTrust",
        "model": "injection_aware_mpnet",
        "threshold": 0.764,
        "samples": n,
        "tp": tp,
        "fn": fn,
        "recall": recall,
        "accuracy": recall,
        "recall_ci_95": [rec_lo, rec_hi],
        "duration_ms": duration_ms,
    }

    out = ROOT / "results" / "tensortrust_mpnet_evaluation.json"
    out.write_text(json.dumps(result, indent=2))

    print()
    print(json.dumps(result, indent=2))
    print(f"\nResult written to {out}")


if __name__ == "__main__":
    main()
