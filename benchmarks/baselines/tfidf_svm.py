"""
TF-IDF + SVM Baseline

Simple classical ML baseline for prompt injection detection.
"""

from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import structlog

logger = structlog.get_logger()


class TfidfSvmBaseline:
    """TF-IDF + SVM baseline classifier."""
    
    def __init__(self, max_features: int = 5000, C: float = 1.0):
        """
        Initialize baseline.
        
        Args:
            max_features: Maximum vocabulary size
            C: SVM regularization parameter
        """
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                stop_words='english'
            )),
            ('svm', SVC(
                C=C,
                kernel='rbf',
                probability=True,
                random_state=42
            ))
        ])
        self.is_trained = False
    
    def train(self, texts: List[str], labels: List[int]):
        """Train the baseline model."""
        logger.info("Training TF-IDF + SVM baseline", samples=len(texts))
        self.pipeline.fit(texts, labels)
        self.is_trained = True
        logger.info("Training complete")
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict labels for texts."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.pipeline.predict(texts)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.pipeline.predict_proba(texts)
