"""
Prompt Injection Detection Benchmarks

Standardized evaluation scripts for benchmarking detection performance
against public datasets including SaTML, deepset, NotInject, LLMail,
and BrowseSafe.

Key Components:
- BenchmarkDataset: Standardized dataset wrapper
- Loaders: load_satml_dataset(), load_deepset_dataset(), etc.
- BenchmarkRunner: Execute evaluation pipelines
- BenchmarkMetrics: Calculate accuracy, FPR, latency, TIVS
- BenchmarkReporter: Format and display results
"""

__version__ = "1.0.0"
__author__ = "BIT Authors"

from .benchmark_datasets import (
    BenchmarkDataset,
    load_satml_dataset,
    load_deepset_dataset,
    load_notinject_dataset,
    load_llmail_dataset,
    load_browsesafe_dataset,
    load_all_datasets,
    AVAILABLE_DATASETS
)

from .runner import BenchmarkRunner, BenchmarkResults
from .metrics import BenchmarkMetrics
from .reporter import BenchmarkReporter

__all__ = [
    # Datasets
    "BenchmarkDataset",
    "load_satml_dataset",
    "load_deepset_dataset", 
    "load_notinject_dataset",
    "load_llmail_dataset",
    "load_browsesafe_dataset",
    "load_all_datasets",
    "AVAILABLE_DATASETS",
    # Runner
    "BenchmarkRunner",
    "BenchmarkResults",
    # Metrics
    "BenchmarkMetrics",
    # Reporter
    "BenchmarkReporter",
    # Helpers
    "quick_benchmark",
]

def quick_benchmark(detector, verbose: bool = True) -> dict:
    """
    Quick benchmark on all major datasets.
    
    Args:
        detector: The detector instance (must have predict_proba or predict method)
        verbose: Whether to print progress and results
        
    Returns:
        Dictionary containing benchmark results
    """
    from .runner import BenchmarkRunner
    runner = BenchmarkRunner(detector)
    return runner.run_all(limit_per_dataset=100, verbose=verbose)
