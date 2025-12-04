import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.dataset_loader import DatasetLoader
import structlog

logger = structlog.get_logger()

def verify_mof():
    loader = DatasetLoader()
    
    print("Generating NotInject samples...")
    samples = loader.generate_not_inject_prompts(count=10)
    
    print(f"\nGenerated {len(samples)} samples.")
    print("Sample 1:", samples[0])
    
    # Check labels
    all_safe = all(s['label'] == 0 for s in samples)
    print(f"\nAll labels are 0 (Safe): {all_safe}")
    
    # Check trigger words
    triggers = ["ignore", "system", "override", "bypass", "delete", "update"]
    has_triggers = any(any(t in s['text'].lower() for t in triggers) for s in samples)
    print(f"Contains trigger words: {has_triggers}")
    
    if all_safe and has_triggers:
        print("\nSUCCESS: MOF strategy verification passed.")
    else:
        print("\nFAILURE: MOF strategy verification failed.")

    # Verify integration into load_all_datasets
    print("\nVerifying integration into full dataset...")
    # We mock the other loaders to avoid network calls/large downloads for this quick check
    loader.datasets = {} # Disable HF downloads
    loader.local_datasets = {} # Disable local files
    
    ds = loader.load_all_datasets(include_local=False, include_hf=False)
    # Should contain synthetic safe + synthetic injection + NotInject
    # 2 synthetic + 50 NotInject = 52
    print(f"Total dataset size (synthetic only): {len(ds)}")
    
    # Check if we have enough samples
    if len(ds) >= 52:
        print("Integration verification passed.")
    else:
        print(f"Integration verification failed. Expected >= 52, got {len(ds)}")

if __name__ == "__main__":
    verify_mof()
