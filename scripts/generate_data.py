
import sys
import os
import json
from pathlib import Path

# Add the project to path
sys.path.insert(0, os.path.abspath('.'))

from src.utils.dataset_loader import DatasetLoader

def main():
    print("🚀 Generating synthetic safe prompts...")
    
    # Initialize loader
    loader = DatasetLoader()
    
    # Generate prompts
    prompts = loader.generate_synthetic_safe_prompts(count=1000)
    
    # Save to file
    output_path = Path("data/synthetic_safe_prompts.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, indent=2)
        
    print(f"✅ Generated {len(prompts)} prompts and saved to {output_path}")

if __name__ == "__main__":
    main()
