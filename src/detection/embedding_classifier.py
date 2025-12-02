import time
from typing import List, Dict, Union, Optional
import numpy as np
import xgboost as xgb
from sentence_transformers import SentenceTransformer
import structlog

logger = structlog.get_logger()

class EmbeddingClassifier:
    """
    Classifier that uses sentence embeddings and XGBoost to detect prompt injections.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.85):
        """
        Initialize the embedding classifier.
        
        Args:
            model_name: Name of the sentence-transformer model to use.
            threshold: Confidence threshold for classifying as injection.
        """
        self.model_name = model_name
        self.threshold = threshold
        self.embedding_model = None
        self.classifier = None
        self._load_models()

    def _load_models(self):
        """Load the embedding model and initialize the classifier."""
        logger.info("Loading embedding model", model=self.model_name)
        try:
            self.embedding_model = SentenceTransformer(self.model_name)
            # Initialize XGBoost classifier - in a real scenario, we would load a trained model
            # For now, we'll initialize a blank one or load if a path was provided (future improvement)
            self.classifier = xgb.XGBClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=5, 
                use_label_encoder=False, 
                eval_metric='logloss'
            )
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error("Failed to load models", error=str(e))
            raise

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of strings to embed.
            
        Returns:
            Numpy array of embeddings.
        """
        start_time = time.time()
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        duration = (time.time() - start_time) * 1000
        logger.debug("Embeddings generated", count=len(texts), duration_ms=duration)
        return embeddings

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict probabilities of prompt injection for a list of texts.
        
        Args:
            texts: List of strings to classify.
            
        Returns:
            Numpy array of probabilities (N, 2) where column 1 is injection probability.
        """
        embeddings = self.embed(texts)
        
        # Check if classifier is trained (has classes_ attribute)
        # If not trained, return zeros (or handle appropriately for untraised state)
        if not hasattr(self.classifier, "classes_"):
            logger.warning("Classifier not trained. Returning default probabilities.")
            # Return 0.0 probability for injection for now to avoid errors
            return np.zeros((len(texts), 2))

        return self.classifier.predict_proba(embeddings)

    def predict(self, texts: List[str]) -> List[int]:
        """
        Predict labels (0 for safe, 1 for injection) for a list of texts.
        
        Args:
            texts: List of strings to classify.
            
        Returns:
            List of integer labels.
        """
        probs = self.predict_proba(texts)
        # Use column 1 (injection probability) and compare with threshold
        return (probs[:, 1] >= self.threshold).astype(int).tolist()

    def batch_predict(self, texts: List[str]) -> List[Dict[str, Union[bool, float]]]:
        """
        Predict with detailed metadata for a batch of texts.
        
        Args:
            texts: List of strings to classify.
            
        Returns:
            List of dictionaries containing 'is_injection' and 'score'.
        """
        start_time = time.time()
        probs = self.predict_proba(texts)
        
        results = []
        for i, prob in enumerate(probs):
            score = float(prob[1])
            is_injection = score >= self.threshold
            results.append({
                "is_injection": is_injection,
                "score": score
            })
            
        duration = (time.time() - start_time) * 1000
        logger.info("Batch prediction complete", count=len(texts), duration_ms=duration)
        return results

    def train(self, texts: List[str], labels: List[int]):
        """
        Train the classifier on provided texts and labels.
        
        Args:
            texts: List of training texts.
            labels: List of corresponding labels (0 or 1).
        """
        logger.info("Starting training", samples=len(texts))
        embeddings = self.embed(texts)
        self.classifier.fit(embeddings, labels)
        logger.info("Training complete")

    def save_model(self, path: str):
        """Save the trained classifier to a file."""
        self.classifier.save_model(path)
        logger.info("Model saved", path=path)

    def load_model(self, path: str):
        """Load a trained classifier from a file."""
        self.classifier.load_model(path)
        logger.info("Model loaded", path=path)
