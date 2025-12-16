#!/usr/bin/env python
"""
Benchmark CLI

Command-line interface for running prompt injection detection benchmarks.

Usage:
    python -m benchmarks.run_benchmark --all
    python -m benchmarks.run_benchmark --datasets satml deepset
    python -m benchmarks.run_benchmark --model models/comprehensive_classifier.json
    python -m benchmarks.run_benchmark --output-format markdown --output report.md
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ]
)
logger = structlog.get_logger()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run prompt injection detection benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all benchmarks
    python -m benchmarks.run_benchmark --all
    
    # Run specific datasets
    python -m benchmarks.run_benchmark --datasets satml deepset
    
    # Use specific model
    python -m benchmarks.run_benchmark --model models/comprehensive_classifier.json
    
    # Quick test with limited samples
    python -m benchmarks.run_benchmark --quick --samples 100
    
    # Output to file
    python -m benchmarks.run_benchmark --output-format markdown --output report.md
"""
    )
    
    # Dataset selection
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--all", 
        action="store_true",
        help="Run benchmarks on all available datasets"
    )
    dataset_group.add_argument(
        "--paper",
        action="store_true",
        help="Paper-aligned benchmark: SaTML(300), deepset_attacks(203), NotInject(339), LLMail(200)"
    )
    dataset_group.add_argument(
        "--datasets",
        nargs="+",
        choices=["satml", "deepset", "deepset_injections", "notinject", "notinject_hf", "llmail", "browsesafe", "agentdojo", "tensortrust"],
        help="Specific datasets to benchmark"
    )
    dataset_group.add_argument(
        "--quick",
        action="store_true",
        help="Quick benchmark with limited samples"
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model file (default: load from models/)"
    )
    parser.add_argument(
        "--model-type",
        choices=["embedding_classifier", "ensemble", "auto"],
        default="auto",
        help="Type of model (default: auto-detect)"
    )
    
    # Benchmark parameters
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.764,
        help="Classification threshold (default: 0.5)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Maximum samples per dataset (default: all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )
    
    # Output options
    parser.add_argument(
        "--output-format",
        choices=["console", "json", "markdown"],
        default="console",
        help="Output format (default: console)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (optional, default: print to console)"
    )
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Don't show baseline comparisons"
    )
    parser.add_argument(
        "--exclude-notinject",
        action="store_true",
        help="Exclude NotInject dataset (over-defense testing) for cleaner attack-only metrics"
    )
    
    # Other options
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit"
    )
    
    # GPU option
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration (if available)"
    )
    
    return parser.parse_args()


def list_datasets():
    """Print available datasets and exit."""
    from .benchmark_datasets import AVAILABLE_DATASETS
    
    print("\nAvailable Benchmark Datasets:")
    print("=" * 60)
    
    for key, info in AVAILABLE_DATASETS.items():
        print(f"\n  {key}:")
        print(f"    Name: {info['name']}")
        print(f"    Source: {info['source']}")
        print(f"    Type: {info['type']}")
        print(f"    Description: {info['description']}")
    
    print()
    sys.exit(0)


def load_detector(model_path: str = None, model_type: str = "auto", threshold: float = 0.5, gpu: bool = False):
    """Load detector model with threshold override."""
    from src.detection.embedding_classifier import EmbeddingClassifier
    
    logger.info("Loading detector model", path=model_path, type=model_type, threshold=threshold, gpu=gpu)
    
    # Initialize with threshold and GPU option
    detector = EmbeddingClassifier(threshold=threshold, gpu=gpu)
    
    if model_path and Path(model_path).exists():
        detector.load_model(model_path)
        # Force override threshold (metadata might have different value)
        detector.threshold = threshold
        logger.info("Model loaded from file", path=model_path, threshold=threshold)
    else:
        # Try to find a trained model in models directory
        models_dir = Path("models")
        if models_dir.exists():
            # Prioritize BIT model for paper validation
            priority_models = [
                "bit_xgboost_model.json",
                "comprehensive_classifier.json",
                "all-MiniLM-L6-v2_classifier.json",
            ]
            
            for model_name in priority_models:
                model_file = models_dir / model_name
                if model_file.exists():
                    detector.load_model(str(model_file))
                    # Force override threshold
                    detector.threshold = threshold
                    logger.info("Auto-loaded model", path=str(model_file), threshold=threshold)
                    break
    
    if not detector.is_trained:
        logger.warning("No trained model found. Results may be inaccurate.")
    
    return detector


def main():
    """Main entry point."""
    args = parse_args()
    
    # Handle list datasets
    if args.list_datasets:
        list_datasets()
    
    # Import benchmark modules
    from .runner import BenchmarkRunner
    from .reporter import BenchmarkReporter
    from .benchmark_datasets import load_all_datasets
    
    # Load detector with threshold and GPU option
    detector = load_detector(args.model, args.model_type, args.threshold, args.gpu)
    
    # Create runner
    runner = BenchmarkRunner(
        detector=detector,
        batch_size=args.batch_size,
        threshold=args.threshold
    )
    
    # Determine which datasets to include
    include_datasets = None
    if args.exclude_notinject:
        include_datasets = ["satml", "deepset", "llmail"]
        logger.info("Excluding NotInject dataset for attack-only metrics")
    
    # Determine which datasets to run
    if args.quick:
        samples = args.samples or 100
        logger.info(f"Running quick benchmark with {samples} samples per dataset")
        results = runner.run_quick(
            samples_per_dataset=samples,
            verbose=not args.quiet
        )
        # Filter out notinject if excluded
        if args.exclude_notinject and "notinject" in results.results:
            del results.results["notinject"]
    elif getattr(args, 'paper', False):
        # Paper-aligned benchmark with statistically valid sample counts
        logger.info("Running PAPER-ALIGNED benchmark (1,042 total samples)")
        logger.info("  SaTML: 300, deepset_attacks: 203, NotInject: 339, LLMail: 200")
        
        # Paper-specific sample limits
        paper_config = {
            "satml": 300,
            "deepset_injections": 203,  # Attacks only for recall
            "notinject_hf": 339,        # Full HF dataset
            "llmail": 200               # Phase 1
        }
        
        from .benchmark_datasets import load_all_datasets
        all_results = {}
        
        for dataset_name, limit in paper_config.items():
            datasets = load_all_datasets(
                include_datasets=[dataset_name],
                limit_per_dataset=limit
            )
            if datasets:
                for ds_name, ds in datasets.items():
                    if len(ds.texts) > 0:  # Only run if we got data
                        metrics = runner.run(ds, verbose=not args.quiet)
                        all_results[ds_name] = metrics
        
        # Create combined result
        from .runner import BenchmarkResults
        results = BenchmarkResults(
            results=all_results,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "detector_name": type(detector).__name__,
                "threshold": args.threshold,
                "paper_aligned": True
            }
        )
    elif args.datasets:
        datasets_to_run = args.datasets
        if args.exclude_notinject:
            datasets_to_run = [d for d in args.datasets if d != "notinject"]
        logger.info(f"Running benchmark on datasets: {datasets_to_run}")
        results = runner.run_all(
            limit_per_dataset=args.samples,
            include_datasets=datasets_to_run,
            verbose=not args.quiet
        )
    else:
        # Default to all (or all except notinject)
        logger.info("Running benchmark on all datasets")
        results = runner.run_all(
            limit_per_dataset=args.samples,
            include_datasets=include_datasets,
            verbose=not args.quiet
        )
    
    # Create reporter
    reporter = BenchmarkReporter(results)
    
    # Output results
    if args.output_format == "console":
        reporter.print_console(show_baselines=not args.no_baselines)
    elif args.output_format == "json":
        if args.output:
            reporter.save(args.output, format="json")
        else:
            print(reporter.to_json())
    elif args.output_format == "markdown":
        if args.output:
            reporter.save(args.output, format="markdown")
        else:
            print(reporter.to_markdown())
    
    # Save to file if specified for non-file outputs
    if args.output and args.output_format == "console":
        reporter.save(args.output, format="json")
    
    # Return success/failure based on targets
    targets_met = results.overall_accuracy >= 0.95 and results.overall_fpr <= 0.05
    sys.exit(0 if targets_met else 1)


if __name__ == "__main__":
    main()
