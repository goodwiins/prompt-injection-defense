#!/usr/bin/env python3
"""
Production training script for prompt injection defense system.
Uses large-scale datasets and optimized training procedures.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.dataset_loader import DatasetLoader
from detection.embedding_classifier import EmbeddingClassifier
from detection.ensemble import InjectionDetector
import structlog

logger = structlog.get_logger()

def main():
    parser = argparse.ArgumentParser(description="Train production-grade prompt injection detector")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2",
                       help="Sentence transformer model name")
    parser.add_argument("--ensemble", action="store_true",
                       help="Train ensemble classifier instead of single model")
    parser.add_argument("--batch-size", type=int, default=1000,
                       help="Batch size for embedding generation")
    parser.add_argument("--no-validation", action="store_true",
                       help="Skip validation split")
    parser.add_argument("--cross-validation", type=int, default=0,
                       help="Number of CV folds (0 to disable)")
    parser.add_argument("--data-dir", default="data",
                       help="Data directory for local datasets")
    parser.add_argument("--model-dir", default="models",
                       help="Model directory for saving trained models")

    args = parser.parse_args()

    logger.info("Starting production training", args=vars(args))

    # Initialize dataset loader
    dataset_loader = DatasetLoader(data_dir=args.data_dir)

    try:
        # Load comprehensive dataset
        logger.info("Loading comprehensive dataset...")
        train_dataset, val_dataset, test_dataset = dataset_loader.load_and_split(
            test_size=0.1,
            val_size=0.1,
            include_local=True,
            include_hf=True
        )

        logger.info(f"Dataset loaded successfully")
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")

        if args.ensemble:
            # Train ensemble classifier
            logger.info("Training ensemble classifier...")
            classifier = InjectionDetector(
                fast_model_name="all-MiniLM-L6-v2",
                deep_model_name="all-mpnet-base-v2",
                model_dir=args.model_dir
            )

            # Train on training data
            try:
                training_stats = classifier.train_on_dataset(
                    train_dataset,
                    batch_size=args.batch_size,
                    validation_split=not args.no_validation
                )
                logger.info(f"Training completed successfully. is_trained={classifier.is_trained}")
            except Exception as e:
                logger.error("Training failed", error=str(e))
                raise e

            # Evaluate on test set (skip if training failed)
            try:
                test_stats = classifier.evaluate(test_dataset["text"], test_dataset["label"])
                logger.info("Ensemble evaluation complete", test_stats=test_stats)
            except Exception as e:
                logger.warning("Evaluation failed, but training completed", error=str(e))
                test_stats = {"auc": 0.0, "tivs": 0.0}

        else:
            # Train single embedding classifier
            logger.info("Training embedding classifier...")
            classifier = EmbeddingClassifier(
                model_name=args.model_name,
                model_dir=args.model_dir
            )

            # Perform cross-validation if requested
            if args.cross_validation > 0:
                cv_texts = train_dataset["text"][:5000]  # Limit for CV
                cv_labels = train_dataset["label"][:5000]
                cv_stats = classifier.cross_validate(cv_texts, cv_labels, args.cross_validation)
                logger.info("Cross-validation complete", cv_stats=cv_stats)

            # Train on full training data
            training_stats = classifier.train_on_dataset(
                train_dataset,
                batch_size=args.batch_size,
                validation_split=not args.no_validation
            )

            # Evaluate on test set
            test_stats = classifier.evaluate(test_dataset["text"], test_dataset["label"])
            logger.info("Test evaluation complete", test_stats=test_stats)

        # Print summary
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Model: {'Ensemble' if args.ensemble else args.model_name}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test AUC: {test_stats.get('auc', 'N/A'):.4f}")
        print(f"Test TIVS: {test_stats.get('tivs', 'N/A'):.4f}")
        print(f"Model saved to: {args.model_dir}")
        print("="*80)

    except Exception as e:
        logger.error("Training failed", error=str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()