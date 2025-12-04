from datasets import load_dataset
import structlog

logger = structlog.get_logger()

def inspect_dataset():
    print("Loading SaTML dataset...")
    try:
        # Load a small portion to inspect
        dataset = load_dataset("ethz-spylab/ctf-satml24", "interaction_chats", split="attack", streaming=True)
        
        print("\nDataset Structure:")
        sample = next(iter(dataset))
        for key, value in sample.items():
            print(f"  {key}: {type(value)} - {str(value)[:100]}...")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")

if __name__ == "__main__":
    inspect_dataset()
