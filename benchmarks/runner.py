"""
Benchmark Runner

Orchestrates benchmark execution across datasets and detectors.
"""

import time
import os
import sys
from typing import List, Dict, Any, Optional, Union, Protocol
from dataclasses import dataclass, field
from pathlib import Path
import json
import structlog

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .benchmark_datasets import BenchmarkDataset, load_all_datasets, AVAILABLE_DATASETS
from .metrics import BenchmarkMetrics, calculate_metrics, calculate_over_defense_rate

logger = structlog.get_logger()


class DetectorProtocol(Protocol):
    """Protocol for detector compatibility."""
    
    def predict(self, texts: List[str]) -> List[Any]: ...
    def predict_proba(self, texts: List[str]) -> Any: ...


@dataclass
class BenchmarkResults:
    """
    Container for benchmark results across multiple datasets.
    """
    results: Dict[str, BenchmarkMetrics] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __getitem__(self, dataset_name: str) -> BenchmarkMetrics:
        return self.results[dataset_name]
    
    def __iter__(self):
        return iter(self.results.items())
    
    @property
    def overall_accuracy(self) -> float:
        """Calculate weighted average accuracy across datasets."""
        if not self.results:
            return 0.0
        
        total_samples = sum(m.total_samples for m in self.results.values())
        if total_samples == 0:
            return 0.0
        
        weighted_sum = sum(
            m.accuracy * m.total_samples 
            for m in self.results.values()
        )
        return weighted_sum / total_samples
    
    @property
    def overall_fpr(self) -> float:
        """Calculate weighted average FPR across datasets."""
        if not self.results:
            return 0.0
        
        total_negatives = sum(
            m.true_negatives + m.false_positives 
            for m in self.results.values()
        )
        if total_negatives == 0:
            return 0.0
        
        total_fp = sum(m.false_positives for m in self.results.values())
        return total_fp / total_negatives
    
    @property
    def overall_fnr(self) -> float:
        """Calculate weighted average FNR across datasets."""
        if not self.results:
            return 0.0
        
        total_positives = sum(
            m.true_positives + m.false_negatives 
            for m in self.results.values()
        )
        if total_positives == 0:
            return 0.0
        
        total_fn = sum(m.false_negatives for m in self.results.values())
        return total_fn / total_positives

    @property
    def overall_recall(self) -> float:
        """Calculate weighted average Recall across datasets."""
        if not self.results:
            return 0.0
        
        total_positives = sum(
            m.true_positives + m.false_negatives 
            for m in self.results.values()
        )
        if total_positives == 0:
            return 0.0
        
        total_tp = sum(m.true_positives for m in self.results.values())
        return total_tp / total_positives

    @property
    def overall_precision(self) -> float:
        """Calculate weighted average Precision across datasets."""
        if not self.results:
            return 0.0
        
        total_predicted_positives = sum(
            m.true_positives + m.false_positives 
            for m in self.results.values()
        )
        if total_predicted_positives == 0:
            return 0.0
        
        total_tp = sum(m.true_positives for m in self.results.values())
        return total_tp / total_predicted_positives

    @property
    def overall_f1(self) -> float:
        """Calculate weighted average F1 across datasets."""
        p = self.overall_precision
        r = self.overall_recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
    
    @property
    def over_defense_rate(self) -> Optional[float]:
        """Get over-defense rate from NotInject dataset if available."""
        if "notinject" in self.results:
            return self.results["notinject"].over_defense_rate
        return None
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary of all results."""
        return {
            "datasets": {
                name: metrics.to_dict() 
                for name, metrics in self.results.items()
            },
            "overall": {
                "accuracy": round(self.overall_accuracy, 4),
                "fpr": round(self.overall_fpr, 4),
                "over_defense_rate": (
                    round(self.over_defense_rate, 4) 
                    if self.over_defense_rate is not None else None
                ),
            },
            "metadata": self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert results to JSON string."""
        return json.dumps(self.summary(), indent=indent)
    
    def save(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())
        logger.info("Results saved", path=path)


class BenchmarkRunner:
    """
    Orchestrates benchmark execution across datasets.
    """
    
    def __init__(
        self,
        detector: Any,
        batch_size: int = 32,
        threshold: float = 0.5,
        cache_embeddings: bool = True
    ):
        """
        Initialize benchmark runner.
        
        Args:
            detector: Detection model (EmbeddingClassifier or InjectionDetector)
            batch_size: Batch size for processing
            threshold: Classification threshold
            cache_embeddings: Whether to cache embeddings for repeated runs
        """
        self.detector = detector
        self.batch_size = batch_size
        self.threshold = threshold
        self.cache_embeddings = cache_embeddings
        self._embedding_cache: Dict[str, Any] = {}
        
        # Detect detector type
        self.detector_type = self._detect_detector_type()
        logger.info("BenchmarkRunner initialized", 
                   detector_type=self.detector_type, 
                   threshold=threshold)
    
    def _detect_detector_type(self) -> str:
        """Detect the type of detector being used."""
        class_name = type(self.detector).__name__
        
        if class_name == "EmbeddingClassifier":
            return "embedding_classifier"
        elif class_name == "InjectionDetector":
            return "injection_detector"
        elif hasattr(self.detector, "scan"):
            return "ensemble"
        elif hasattr(self.detector, "predict_proba"):
            return "sklearn_like"
        else:
            return "unknown"
    
    def run(
        self,
        dataset: BenchmarkDataset,
        verbose: bool = True
    ) -> BenchmarkMetrics:
        """
        Run benchmark on a single dataset.
        
        Args:
            dataset: BenchmarkDataset to evaluate
            verbose: Whether to log progress
            
        Returns:
            BenchmarkMetrics for this dataset
        """
        if len(dataset) == 0:
            logger.warning("Empty dataset", name=dataset.name)
            return BenchmarkMetrics(dataset_name=dataset.name)
        
        if verbose:
            logger.info(f"Running benchmark on {dataset.name}", samples=len(dataset))
        
        texts = dataset.texts
        true_labels = dataset.labels
        
        # Run predictions with latency tracking
        predictions, scores, latencies = self._predict_batch(texts, verbose)
        
        # Calculate metrics
        metrics = calculate_metrics(
            true_labels=true_labels,
            predictions=predictions,
            scores=scores,
            latencies=latencies,
            dataset_name=dataset.name,
            threshold=self.threshold
        )
        
        # Calculate over-defense rate for safe-only datasets
        if dataset.metadata.get("dataset_type") == "safe_only":
            metrics.over_defense_rate = calculate_over_defense_rate(
                predictions=predictions,
                all_safe=True
            )
        
        if verbose:
            logger.info(
                f"Benchmark complete: {dataset.name}",
                accuracy=f"{metrics.accuracy:.2%}",
                f1=f"{metrics.f1_score:.2%}",
                fpr=f"{metrics.false_positive_rate:.2%}",
                latency_p95=f"{metrics.latency_p95:.1f}ms"
            )
        
        return metrics
    
    def _predict_batch(
        self, 
        texts: List[str],
        verbose: bool = True
    ) -> tuple:
        """
        Run predictions on a batch of texts with latency tracking.
        
        Returns:
            Tuple of (predictions, scores, latencies)
        """
        predictions = []
        scores = []
        latencies = []
        
        total = len(texts)
        
        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Time each batch
            start = time.time()
            
            if self.detector_type == "injection_detector" or hasattr(self.detector, "scan"):
                # InjectionDetector / Ensemble style
                batch_results = []
                for text in batch:
                    result = self.detector.scan(text)
                    batch_results.append(result)
                
                batch_preds = [1 if r.get("is_injection", False) else 0 for r in batch_results]
                batch_scores = [r.get("score", 0.5) for r in batch_results]
                
            elif hasattr(self.detector, "predict_proba"):
                # EmbeddingClassifier / sklearn style
                proba = self.detector.predict_proba(batch)
                batch_scores = proba[:, 1].tolist() if len(proba.shape) > 1 else proba.tolist()
                batch_preds = [1 if s >= self.threshold else 0 for s in batch_scores]
                
            elif hasattr(self.detector, "predict"):
                # Basic predict style
                batch_preds = self.detector.predict(batch)
                if hasattr(batch_preds, "tolist"):
                    batch_preds = batch_preds.tolist()
                batch_scores = [float(p) for p in batch_preds]
                
            else:
                raise ValueError(f"Unsupported detector type: {self.detector_type}")
            
            batch_time = (time.time() - start) * 1000  # Convert to ms
            per_sample_latency = batch_time / len(batch)
            
            predictions.extend(batch_preds)
            scores.extend(batch_scores)
            latencies.extend([per_sample_latency] * len(batch))
            
            if verbose and (i + self.batch_size) % (self.batch_size * 10) == 0:
                progress = min(i + self.batch_size, total) / total * 100
                logger.info(f"Progress: {progress:.1f}%")
        
        return predictions, scores, latencies
    
    def run_all(
        self,
        datasets: Optional[Dict[str, BenchmarkDataset]] = None,
        limit_per_dataset: Optional[int] = None,
        include_datasets: Optional[List[str]] = None,
        verbose: bool = True
    ) -> BenchmarkResults:
        """
        Run benchmarks on all available datasets.
        
        Args:
            datasets: Pre-loaded datasets (optional, will load if not provided)
            limit_per_dataset: Maximum samples per dataset
            include_datasets: List of dataset names to include
            verbose: Whether to log progress
            
        Returns:
            BenchmarkResults with all dataset metrics
        """
        if datasets is None:
            if verbose:
                logger.info("Loading benchmark datasets...")
            datasets = load_all_datasets(
                limit_per_dataset=limit_per_dataset,
                include_datasets=include_datasets
            )
        
        results = BenchmarkResults(
            metadata={
                "detector_type": self.detector_type,
                "threshold": self.threshold,
                "batch_size": self.batch_size,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        )
        
        for name, dataset in datasets.items():
            if len(dataset) == 0:
                logger.warning(f"Skipping empty dataset: {name}")
                continue
            
            metrics = self.run(dataset, verbose=verbose)
            results.results[name] = metrics
        
        if verbose:
            logger.info("All benchmarks complete", 
                       datasets=len(results.results),
                       overall_accuracy=f"{results.overall_accuracy:.2%}")
        
        return results
    
    def run_quick(
        self,
        samples_per_dataset: int = 100,
        verbose: bool = True
    ) -> BenchmarkResults:
        """
        Run a quick benchmark with limited samples for testing.
        
        Args:
            samples_per_dataset: Number of samples per dataset
            verbose: Whether to log progress
            
        Returns:
            BenchmarkResults with limited sample metrics
        """
        if verbose:
            logger.info("Running quick benchmark", samples=samples_per_dataset)
        
        return self.run_all(
            limit_per_dataset=samples_per_dataset,
            verbose=verbose
        )


def create_runner_from_model(
    model_path: str,
    model_type: str = "auto",
    **kwargs
) -> BenchmarkRunner:
    """
    Create a BenchmarkRunner from a saved model file.
    
    Args:
        model_path: Path to saved model
        model_type: Type of model ('embedding_classifier', 'ensemble', 'auto')
        **kwargs: Additional arguments for BenchmarkRunner
        
    Returns:
        Configured BenchmarkRunner
    """
    from src.detection.embedding_classifier import EmbeddingClassifier
    
    if model_type == "auto":
        if "classifier" in model_path.lower():
            model_type = "embedding_classifier"
        elif "ensemble" in model_path.lower():
            model_type = "ensemble"
        else:
            model_type = "embedding_classifier"
    
    if model_type == "embedding_classifier":
        detector = EmbeddingClassifier()
        detector.load_model(model_path)
    elif model_type == "ensemble":
        from src.detection.ensemble import InjectionDetector
        detector = InjectionDetector()
        # Load model if path provided
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return BenchmarkRunner(detector, **kwargs)
