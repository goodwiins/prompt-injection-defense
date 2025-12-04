import sys
import os
import time
import json
from typing import Dict, List, Any

# Add the project to path
sys.path.insert(0, os.path.abspath('.'))

from src.utils.dataset_loader import DatasetLoader

def main():
    # Initialize the enhanced dataset loader
    print("🔄 Initializing enhanced dataset loader...")
    dataset_loader = DatasetLoader(data_dir="data")

    # Load comprehensive dataset (this may take a while for first run)
    print("\n🚀 Loading comprehensive dataset...")
    try:
        train_dataset, val_dataset, test_dataset = dataset_loader.load_and_split(
            test_size=0.1,
            val_size=0.1,
            include_local=True,
            include_hf=True
        )
        
        print(f"\n📈 Dataset Statistics:")
        print(f"  • Training samples: {len(train_dataset):,}")
        print(f"  • Validation samples: {len(val_dataset):,}")
        print(f"  • Test samples: {len(test_dataset):,}")
        
        # Show label distribution
        train_labels = train_dataset["label"]
        safe_count = train_labels.count(0)
        injection_count = train_labels.count(1)
        
        print(f"\n📊 Training Label Distribution:")
        print(f"  • Safe prompts: {safe_count:,} ({safe_count/len(train_labels)*100:.1f}%)")
        print(f"  • Injection attempts: {injection_count:,} ({injection_count/len(train_labels)*100:.1f}%)")
        
    except Exception as e:
        print(f"⚠️ Could not load datasets: {e}")
        print("Using minimal synthetic dataset for demonstration")

if __name__ == "__main__":
    main()
