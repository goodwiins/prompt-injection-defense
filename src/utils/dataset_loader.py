from typing import Tuple, Dict
import datasets
from datasets import load_dataset, concatenate_datasets, Dataset
import structlog
import pandas as pd

logger = structlog.get_logger()

class DatasetLoader:
    """
    Loader for prompt injection datasets from HuggingFace.
    """

    def __init__(self):
        self.deepset_name = "deepset/prompt-injections"
        self.imoxto_name = "imoxto/prompt_injection_cleaned"

    def load_and_split(self, test_size: float = 0.1, val_size: float = 0.1) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load datasets, combine, deduplicate, and split into train/val/test.
        
        Args:
            test_size: Fraction of data for testing.
            val_size: Fraction of data for validation.
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).
        """
        logger.info("Loading datasets...")
        
        # Load deepset dataset
        try:
            ds_deepset = load_dataset(self.deepset_name, split="train")
            # Standardize columns: text, label (0 for safe, 1 for injection)
            # deepset has 'text' and 'label' (1=injection, 0=safe)
            logger.info(f"Loaded {self.deepset_name}", count=len(ds_deepset))
        except Exception as e:
            logger.error(f"Failed to load {self.deepset_name}", error=str(e))
            ds_deepset = None

        # Load imoxto dataset
        try:
            ds_imoxto = load_dataset(self.imoxto_name, split="train")
            # imoxto might need column mapping. Assuming 'text' and 'label' exist or similar.
            # Checking common column names if needed, but for now assuming standard
            logger.info(f"Loaded {self.imoxto_name}", count=len(ds_imoxto))
        except Exception as e:
            logger.error(f"Failed to load {self.imoxto_name}", error=str(e))
            ds_imoxto = None

        if not ds_deepset and not ds_imoxto:
            raise RuntimeError("Could not load any datasets")

        # Combine
        datasets_list = []
        if ds_deepset:
            datasets_list.append(ds_deepset)
        if ds_imoxto:
            # Ensure columns match before concatenating
            # This is a simplification; in production we'd map columns explicitly
            # For this task, we assume they are compatible or we'd select specific columns
            datasets_list.append(ds_imoxto)
            
        combined = concatenate_datasets(datasets_list)
        logger.info("Datasets combined", total_count=len(combined))

        # Deduplicate using pandas for efficiency
        df = combined.to_pandas()
        initial_count = len(df)
        df.drop_duplicates(subset=["text"], inplace=True)
        deduped_count = len(df)
        logger.info("Deduplication complete", removed=initial_count - deduped_count)

        # Convert back to HF Dataset
        dataset = Dataset.from_pandas(df)
        
        # Split
        # First split off test set
        test_split = dataset.train_test_split(test_size=test_size, seed=42)
        test_dataset = test_split["test"]
        remaining = test_split["train"]
        
        # Then split remaining into train and val
        # Adjust val_size to be relative to the original total
        # If we want 10% val of total, and we have 90% left, we need 1/9 of remaining
        adjusted_val_size = val_size / (1 - test_size)
        val_split = remaining.train_test_split(test_size=adjusted_val_size, seed=42)
        val_dataset = val_split["test"]
        train_dataset = val_split["train"]

        logger.info("Splitting complete", 
                    train=len(train_dataset), 
                    val=len(val_dataset), 
                    test=len(test_dataset))

        return train_dataset, val_dataset, test_dataset

    def get_dataloader(self, dataset: Dataset, batch_size: int = 32):
        """
        Create a PyTorch DataLoader (if needed later).
        """
        # Placeholder for PyTorch integration
        pass
