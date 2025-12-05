"""
Baseline Implementations for Benchmark Comparison

Simple baseline models to compare against our system:
- TF-IDF + SVM
- DistilBERT fine-tuned
- HuggingFace prompt injection classifier
- GPT-4 as zero-shot judge
"""

from .tfidf_svm import TfidfSvmBaseline
from .hf_classifier import HuggingFaceBaseline

__all__ = [
    "TfidfSvmBaseline",
    "HuggingFaceBaseline",
]
