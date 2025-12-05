"""
HuggingFace Classifier Baseline

Use pre-trained HuggingFace prompt injection classifier.
"""

from typing import List, Optional
import numpy as np
import structlog

logger = structlog.get_logger()


class HuggingFaceBaseline:
    """HuggingFace prompt injection classifier."""
    
    def __init__(self, model_name: str = "protectai/deberta-v3-base-prompt-injection"):
        """
        Initialize HuggingFace baseline.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.classifier = None
        self.is_trained = True  # Pre-trained
    
    def load(self):
        """Load the HuggingFace model."""
        try:
            from transformers import pipeline
            logger.info("Loading HuggingFace model", model=self.model_name)
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                device=-1  # CPU
            )
            logger.info("Model loaded")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict labels for texts."""
        if self.classifier is None:
            self.load()
        
        results = self.classifier(texts)
        # Convert to binary: INJECTION=1, SAFE=0
        labels = []
        for r in results:
            if isinstance(r, list):
                r = r[0]
            label = 1 if r.get("label", "").upper() in ["INJECTION", "1", "POSITIVE"] else 0
            labels.append(label)
        
        return np.array(labels)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict probabilities."""
        if self.classifier is None:
            self.load()
        
        results = self.classifier(texts)
        proba = []
        for r in results:
            if isinstance(r, list):
                r = r[0]
            score = r.get("score", 0.5)
            is_injection = r.get("label", "").upper() in ["INJECTION", "1", "POSITIVE"]
            if is_injection:
                proba.append([1 - score, score])
            else:
                proba.append([score, 1 - score])
        
        return np.array(proba)
