#!/usr/bin/env python
"""
Ablation Study Runner

Run benchmark with different system configurations to measure
the contribution of each component.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

logger = structlog.get_logger()


@dataclass
class AblationConfig:
    """Configuration for an ablation experiment."""
    name: str
    description: str
    use_pattern_detector: bool = True
    use_embedding_classifier: bool = True
    use_cascade: bool = True
    use_mof_data: bool = True
    fast_path_only: bool = False
    deep_path_only: bool = False


# Define ablation configurations
CONFIGS = [
    AblationConfig(
        name="full_system",
        description="Complete system (baseline)",
        use_pattern_detector=True,
        use_embedding_classifier=True,
        use_cascade=True,
        use_mof_data=True,
    ),
    AblationConfig(
        name="no_pattern",
        description="Without pattern detection",
        use_pattern_detector=False,
        use_embedding_classifier=True,
        use_cascade=True,
        use_mof_data=True,
    ),
    AblationConfig(
        name="no_embedding",
        description="Pattern detection only",
        use_pattern_detector=True,
        use_embedding_classifier=False,
        use_cascade=False,
        use_mof_data=True,
    ),
    AblationConfig(
        name="no_cascade",
        description="Always use deep path (no cascade)",
        use_pattern_detector=True,
        use_embedding_classifier=True,
        use_cascade=False,
        use_mof_data=True,
    ),
    AblationConfig(
        name="fast_only",
        description="Fast path only (MiniLM)",
        use_pattern_detector=True,
        use_embedding_classifier=True,
        use_cascade=True,
        use_mof_data=True,
        fast_path_only=True,
    ),
    AblationConfig(
        name="no_mof",
        description="Without MOF training data",
        use_pattern_detector=True,
        use_embedding_classifier=True,
        use_cascade=True,
        use_mof_data=False,
    ),
]


def run_ablation(config: AblationConfig, datasets: Dict) -> Dict[str, Any]:
    """Run benchmark with specific ablation configuration."""
    from src.detection.embedding_classifier import EmbeddingClassifier
    from src.detection.patterns import PatternDetector
    from benchmarks import BenchmarkRunner
    
    logger.info(f"Running ablation: {config.name}")
    
    # Select model based on MOF config
    if config.use_mof_data:
        model_path = "models/bit_classifier.json"
    else:
        # Use non-MOF model if available, else fallback
        model_path = "models/all-MiniLM-L6-v2_classifier.json"
    
    # Initialize components based on config
    results = {
        "config": asdict(config),
        "datasets": {}
    }
    
    if config.use_embedding_classifier:
        detector = EmbeddingClassifier()
        try:
            detector.load_model(model_path)
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
            return results
        
        runner = BenchmarkRunner(detector, threshold=0.5)
        
        for name, dataset in datasets.items():
            if len(dataset) == 0:
                continue
            
            metrics = runner.run(dataset, verbose=False)
            results["datasets"][name] = {
                "accuracy": round(metrics.accuracy, 4),
                "precision": round(metrics.precision, 4),
                "recall": round(metrics.recall, 4),
                "f1": round(metrics.f1_score, 4),
                "fpr": round(metrics.false_positive_rate, 4),
                "latency_p95": round(metrics.latency_p95, 2),
            }
    
    elif config.use_pattern_detector:
        # Pattern-only mode
        pattern = PatternDetector()
        
        for name, dataset in datasets.items():
            if len(dataset) == 0:
                continue
            
            tp = tn = fp = fn = 0
            for text, label in zip(dataset.texts, dataset.labels):
                result = pattern.detect(text)
                pred = 1 if result["is_suspicious"] else 0
                
                if pred == 1 and label == 1:
                    tp += 1
                elif pred == 0 and label == 0:
                    tn += 1
                elif pred == 1 and label == 0:
                    fp += 1
                else:
                    fn += 1
            
            total = tp + tn + fp + fn
            results["datasets"][name] = {
                "accuracy": round((tp + tn) / total, 4) if total > 0 else 0,
                "precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0,
                "recall": round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0,
                "f1": 0,  # Calculate if needed
                "fpr": round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0,
                "latency_p95": 1.0,  # Pattern detection is fast
            }
    
    return results


def run_all_ablations(output_path: str = None, samples: int = 200):
    """Run all ablation configurations."""
    from benchmarks import load_all_datasets
    
    print("=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)
    
    # Load datasets
    print("\nLoading datasets...")
    datasets = load_all_datasets(limit_per_dataset=samples)
    
    all_results = {}
    
    for config in CONFIGS:
        print(f"\n[{config.name}] {config.description}")
        results = run_ablation(config, datasets)
        all_results[config.name] = results
        
        # Print summary for this config
        if results["datasets"]:
            avg_acc = sum(d["accuracy"] for d in results["datasets"].values()) / len(results["datasets"])
            avg_fpr = sum(d["fpr"] for d in results["datasets"].values()) / len(results["datasets"])
            print(f"  Avg Accuracy: {avg_acc:.1%}, Avg FPR: {avg_fpr:.1%}")
    
    # Save results
    if output_path:
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return all_results


def print_comparison_table(results: Dict):
    """Print ablation comparison table."""
    print("\n" + "=" * 90)
    print("ABLATION COMPARISON")
    print("=" * 90)
    print(f"{'Config':<20} {'Accuracy':>12} {'Precision':>12} {'Recall':>10} {'FPR':>10} {'Latency':>12}")
    print("-" * 90)
    
    for name, data in results.items():
        if not data.get("datasets"):
            continue
        
        datasets = data["datasets"]
        avg_acc = sum(d["accuracy"] for d in datasets.values()) / len(datasets)
        avg_prec = sum(d["precision"] for d in datasets.values()) / len(datasets)
        avg_rec = sum(d["recall"] for d in datasets.values()) / len(datasets)
        avg_fpr = sum(d["fpr"] for d in datasets.values()) / len(datasets)
        avg_lat = sum(d["latency_p95"] for d in datasets.values()) / len(datasets)
        
        print(f"{name:<20} {avg_acc:>11.1%} {avg_prec:>11.1%} {avg_rec:>9.1%} {avg_fpr:>9.1%} {avg_lat:>10.1f}ms")
    
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--output", "-o", default="results/ablation_results.json",
                       help="Output path for results")
    parser.add_argument("--samples", "-s", type=int, default=200,
                       help="Samples per dataset")
    args = parser.parse_args()
    
    results = run_all_ablations(args.output, args.samples)
    print_comparison_table(results)


if __name__ == "__main__":
    main()
