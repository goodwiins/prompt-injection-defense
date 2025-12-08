#!/usr/bin/env python3
"""
Train BIT (Balanced Intent Training) Model for Prompt Injection Defense.
Implements the specific strategy required for the paper:
1. 40/40/20 Data Split (Injection / Safe / Benign-Trigger)
2. Weighted Loss (2.0x for Benign Triggers)
3. Full Dataset utilization
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import Dataset, concatenate_datasets
import structlog

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from detection.embedding_classifier import EmbeddingClassifier
from utils.dataset_loader import DatasetLoader

logger = structlog.get_logger()

def load_bit_datasets(dataset_loader):
    """
    Load the three specific components for BIT:
    1. Injections (Positive class)
    2. Safe Prompts (Negative class)
    3. Benign Triggers (Negative class, hard negatives)
    """
    logger.info("Loading BIT dataset components...")
    
    # --- 1. Injections ---
    # Load from local and HF
    raw_injections = []
    
    # Local
    try:
        with open("data/prompt_injections.json", "r") as f:
            local_inj = pd.read_json(f)
            if "injections" in local_inj:
                local_inj = pd.DataFrame({"text": local_inj["injections"], "label": 1})
            elif "text" in local_inj:
                local_inj = local_inj[local_inj["label"] == 1][["text", "label"]]
            raw_injections.append(local_inj)
            logger.info(f"Loaded {len(local_inj)} local injections")
    except Exception as e:
        logger.warning(f"Could not load local prompt_injections.json: {e}")

    # HF Datasets (via loader or manual)
    # Using loader's load_all_datasets brings mixed data, so we filter
    all_hf = dataset_loader.load_all_datasets(include_local=False, include_hf=True)
    if all_hf:
        hf_df = all_hf.to_pandas()
        hf_inj = hf_df[hf_df["label"] == 1][["text", "label"]]
        raw_injections.append(hf_inj)
        logger.info(f"Loaded {len(hf_inj)} HF injections")
    
    injections_df = pd.concat(raw_injections).drop_duplicates(subset=["text"])
    
    # --- 2. Benign Triggers (Hard Negatives) ---
    # Load from data/notinject_expanded.json if exists, else generate
    try:
        ni_path = Path("data/notinject_expanded.json")
        if ni_path.exists():
            with open(ni_path, "r") as f:
                ni_data = pd.read_json(f)
                # Assume structure list of dicts or list of strings
                if "text" in ni_data.columns:
                     ni_df = ni_data[["text", "label"]]
                else:
                     # It might be list of strings
                     pass # Todo handle formats
                
                # Filter for safe logic if needed, usually these are all safe triggers
                ni_df["label"] = 0
                benign_triggers_df = ni_df
                logger.info(f"Loaded {len(benign_triggers_df)} from notinject_expanded")
        else:
            raise FileNotFoundError("notinject_expanded.json not found")
    except Exception as e:
        logger.warning(f"Falling back to generator for NotInject: {e}")
        ni_data = dataset_loader.generate_not_inject_prompts(count=2000)
        benign_triggers_df = pd.DataFrame(ni_data)
        
    # --- 3. Safe Prompts (Normal Negatives) ---
    try:
        safe_path = Path("data/synthetic_safe_prompts.json")
        if safe_path.exists():
            with open(safe_path, "r") as f:
                safe_df = pd.read_json(f)
                if "safe_prompts" in safe_df:    
                    safe_df = pd.DataFrame({"text": safe_df["safe_prompts"], "label": 0})
        else:
            raise FileNotFoundError
    except:
        safe_data = dataset_loader.generate_synthetic_safe_prompts(count=4000)
        safe_df = pd.DataFrame(safe_data)
        
    logger.info("BIT Components Loaded", 
                injections=len(injections_df), 
                benign_triggers=len(benign_triggers_df), 
                safe=len(safe_df))
                
    return injections_df, benign_triggers_df, safe_df

def create_bit_dataset(injections_df, benign_triggers_df, safe_df, target_size=10000):
    """
    Combine into 40/40/20 mix.
    """
    n_inj = int(target_size * 0.4)
    n_safe = int(target_size * 0.4)
    n_benign = int(target_size * 0.2)
    
    # Resample if needed
    inv_inj = injections_df.sample(n=n_inj, replace=True, random_state=42)
    inv_safe = safe_df.sample(n=n_safe, replace=True, random_state=42)
    inv_benign = benign_triggers_df.sample(n=n_benign, replace=True, random_state=42)
    
    # Assign weights
    inv_inj["weight"] = 1.0
    inv_safe["weight"] = 1.0
    inv_benign["weight"] = 2.0 # BIT Strategy: Upweight hard negatives
    
    # Combine
    combined_df = pd.concat([inv_inj, inv_safe, inv_benign])
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return Dataset.from_pandas(combined_df)

def main():
    logger.info("Starting BIT Model Training")
    
    loader = DatasetLoader()
    inj, benign, safe = load_bit_datasets(loader)
    
    # Create final training set
    train_dataset = create_bit_dataset(inj, benign, safe, target_size=12000)
    
    logger.info("Training Dataset composed", 
                total=len(train_dataset),
                columns=train_dataset.column_names)
    
    # Initialize Classifier with threshold matching paper claims
    classifier = EmbeddingClassifier(
        model_name="all-MiniLM-L6-v2",
        model_dir="models",
        threshold=0.5  # MUST match paper claims - saved in metadata
    )
    
    # Extract weights
    weights = train_dataset["weight"]
    
    # Train
    stats = classifier.train_on_dataset(
        train_dataset, 
        batch_size=1000,
        sample_weights=weights
    )
    
    # Save specifically as BIT model
    classifier.save_model("models/bit_xgboost_model.json")
    print(f"BIT Model saved to models/bit_xgboost_model.json")
    print(f"Training Stats: {stats}")

if __name__ == "__main__":
    main()
