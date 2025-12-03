import time
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np
import structlog
from sentence_transformers import SentenceTransformer
import xgboost as xgb

logger = structlog.get_logger()

class DetectionPath(Enum):
    """Detection path types in ensemble."""
    FAST = "fast"  # Lightweight, quick screening
    DEEP = "deep"  # Detailed semantic analysis
    SPECIALIZED = "specialized"  # Domain-specific models

class EnsembleClassifier:
    """
    Multi-embedding ensemble approach for prompt injection detection.

    Based on research showing that combining multiple embedding models
    (all-MiniLM-L6-v2, gte-large, text-embedding-3-small) with Random Forest
    and XGBoost outperforms single-model approaches.

    Implements three detection paths:
    - Fast: Lightweight embeddings for real-time screening (all-MiniLM-L6-v2)
    - Deep: Larger embeddings for flagged inputs (gte-large or similar)
    - Specialized: Domain-specific fine-tuned models (deberta-v3-base-prompt-injection)
    """

    def __init__(self,
                 fast_model: str = "all-MiniLM-L6-v2",
                 deep_model: Optional[str] = "sentence-transformers/all-mpnet-base-v2",
                 specialized_model: Optional[str] = None,
                 fast_threshold: float = 0.5,
                 deep_threshold: float = 0.85,
                 use_cascade: bool = True):
        """
        Initialize ensemble classifier.

        Args:
            fast_model: Model for fast path
            deep_model: Model for deep path (optional)
            specialized_model: Domain-specific model (optional)
            fast_threshold: Threshold for fast path
            deep_threshold: Threshold for deep path
            use_cascade: If True, only use deep path for flagged samples
        """
        self.fast_model_name = fast_model
        self.deep_model_name = deep_model
        self.specialized_model_name = specialized_model
        self.fast_threshold = fast_threshold
        self.deep_threshold = deep_threshold
        self.use_cascade = use_cascade

        # Load models
        self.fast_model: Optional[SentenceTransformer] = None
        self.deep_model: Optional[SentenceTransformer] = None
        self.specialized_model: Optional[SentenceTransformer] = None

        # Classifiers
        self.fast_classifier: Optional[xgb.XGBClassifier] = None
        self.deep_classifier: Optional[xgb.XGBClassifier] = None
        self.specialized_classifier: Optional[xgb.XGBClassifier] = None

        # Performance tracking
        self.stats = {
            "fast_path_count": 0,
            "deep_path_count": 0,
            "specialized_path_count": 0,
            "total_latency_ms": 0,
            "fast_latency_ms": 0,
            "deep_latency_ms": 0
        }

        self._load_models()

    def _load_models(self) -> None:
        """Load embedding models and initialize classifiers."""
        logger.info("Loading ensemble models")

        # Fast path (always loaded)
        try:
            self.fast_model = SentenceTransformer(self.fast_model_name)
            self.fast_classifier = xgb.XGBClassifier(
                n_estimators=50,
                learning_rate=0.15,
                max_depth=4,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            logger.info("Fast model loaded", model=self.fast_model_name)
        except Exception as e:
            logger.error("Failed to load fast model", error=str(e))
            raise

        # Deep path (optional)
        if self.deep_model_name:
            try:
                self.deep_model = SentenceTransformer(self.deep_model_name)
                self.deep_classifier = xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
                logger.info("Deep model loaded", model=self.deep_model_name)
            except Exception as e:
                logger.warning("Failed to load deep model", error=str(e))

        # Specialized path (optional)
        if self.specialized_model_name:
            try:
                self.specialized_model = SentenceTransformer(self.specialized_model_name)
                self.specialized_classifier = xgb.XGBClassifier(
                    n_estimators=75,
                    learning_rate=0.12,
                    max_depth=5,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
                logger.info("Specialized model loaded", model=self.specialized_model_name)
            except Exception as e:
                logger.warning("Failed to load specialized model", error=str(e))

        logger.info("Ensemble models loaded successfully")

    def predict_proba(self, texts: List[str],
                     force_deep: bool = False) -> Tuple[np.ndarray, List[DetectionPath]]:
        """
        Predict injection probabilities using ensemble.

        Args:
            texts: List of input texts
            force_deep: Force deep path for all samples

        Returns:
            Tuple of (probabilities array, detection paths used)
        """
        start_time = time.time()
        n_samples = len(texts)

        # Initialize results
        probabilities = np.zeros((n_samples, 2))
        paths_used = []

        # Step 1: Fast path for all samples
        fast_start = time.time()
        fast_embeddings = self._embed(texts, self.fast_model)
        fast_probs = self._predict_with_classifier(
            fast_embeddings, self.fast_classifier, "fast"
        )
        self.stats["fast_latency_ms"] += (time.time() - fast_start) * 1000
        self.stats["fast_path_count"] += n_samples

        if self.use_cascade and not force_deep:
            # Cascade approach: only use deep path for uncertain cases
            for i, prob in enumerate(fast_probs):
                injection_score = prob[1]

                # High confidence from fast path
                if injection_score < self.fast_threshold * 0.5:
                    # Very safe
                    probabilities[i] = prob
                    paths_used.append(DetectionPath.FAST)

                elif injection_score > 0.9:
                    # Very suspicious
                    probabilities[i] = prob
                    paths_used.append(DetectionPath.FAST)

                else:
                    # Uncertain - use deep path
                    if self.deep_model and self.deep_classifier:
                        deep_start = time.time()
                        deep_embedding = self._embed([texts[i]], self.deep_model)
                        deep_prob = self._predict_with_classifier(
                            deep_embedding, self.deep_classifier, "deep"
                        )[0]
                        self.stats["deep_latency_ms"] += (time.time() - deep_start) * 1000
                        self.stats["deep_path_count"] += 1

                        # Ensemble: average fast and deep predictions
                        probabilities[i] = (prob + deep_prob) / 2
                        paths_used.append(DetectionPath.DEEP)
                    else:
                        probabilities[i] = prob
                        paths_used.append(DetectionPath.FAST)

        else:
            # Full ensemble: use all available models
            for i in range(n_samples):
                scores = [fast_probs[i]]
                paths = [DetectionPath.FAST]

                # Add deep path
                if self.deep_model and self.deep_classifier:
                    deep_embedding = self._embed([texts[i]], self.deep_model)
                    deep_prob = self._predict_with_classifier(
                        deep_embedding, self.deep_classifier, "deep"
                    )[0]
                    scores.append(deep_prob)
                    paths.append(DetectionPath.DEEP)
                    self.stats["deep_path_count"] += 1

                # Add specialized path
                if self.specialized_model and self.specialized_classifier:
                    spec_embedding = self._embed([texts[i]], self.specialized_model)
                    spec_prob = self._predict_with_classifier(
                        spec_embedding, self.specialized_classifier, "specialized"
                    )[0]
                    scores.append(spec_prob)
                    paths.append(DetectionPath.SPECIALIZED)
                    self.stats["specialized_path_count"] += 1

                # Ensemble: weighted average (give more weight to specialized if available)
                if len(scores) == 3:
                    # Fast: 0.2, Deep: 0.3, Specialized: 0.5
                    probabilities[i] = 0.2 * scores[0] + 0.3 * scores[1] + 0.5 * scores[2]
                elif len(scores) == 2:
                    # Fast: 0.4, Deep: 0.6
                    probabilities[i] = 0.4 * scores[0] + 0.6 * scores[1]
                else:
                    probabilities[i] = scores[0]

                paths_used.append(paths[-1])  # Most advanced path used

        total_latency = (time.time() - start_time) * 1000
        self.stats["total_latency_ms"] += total_latency

        logger.info("Ensemble prediction complete",
                   samples=n_samples,
                   latency_ms=total_latency,
                   avg_latency_ms=total_latency / n_samples)

        return probabilities, paths_used

    def predict(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Predict with detailed results.

        Returns:
            List of prediction dictionaries with scores and paths
        """
        probabilities, paths = self.predict_proba(texts)

        results = []
        for i, (prob, path) in enumerate(zip(probabilities, paths)):
            injection_score = float(prob[1])
            is_injection = injection_score >= self.deep_threshold

            results.append({
                "is_injection": is_injection,
                "score": injection_score,
                "safe_score": float(prob[0]),
                "detection_path": path.value,
                "confidence": abs(injection_score - 0.5) * 2  # 0 to 1
            })

        return results

    def _embed(self, texts: List[str],
               model: SentenceTransformer) -> np.ndarray:
        """Generate embeddings using specified model."""
        if model is None:
            raise ValueError("Model not loaded")
        return model.encode(texts, convert_to_numpy=True)

    def _predict_with_classifier(self, embeddings: np.ndarray,
                                 classifier: xgb.XGBClassifier,
                                 path_name: str) -> np.ndarray:
        """Predict with a classifier, handling untrained state."""
        if not hasattr(classifier, "classes_"):
            logger.warning(f"{path_name} classifier not trained, returning defaults")
            return np.zeros((len(embeddings), 2))

        return classifier.predict_proba(embeddings)

    def train(self, texts: List[str], labels: List[int],
             train_all_paths: bool = True) -> None:
        """
        Train ensemble classifiers.

        Args:
            texts: Training texts
            labels: Training labels (0=safe, 1=injection)
            train_all_paths: Train all available models or just fast path
        """
        logger.info("Training ensemble", samples=len(texts))

        # Train fast path
        fast_embeddings = self._embed(texts, self.fast_model)
        self.fast_classifier.fit(fast_embeddings, labels)
        logger.info("Fast classifier trained")

        if train_all_paths:
            # Train deep path
            if self.deep_model and self.deep_classifier:
                deep_embeddings = self._embed(texts, self.deep_model)
                self.deep_classifier.fit(deep_embeddings, labels)
                logger.info("Deep classifier trained")

            # Train specialized path
            if self.specialized_model and self.specialized_classifier:
                spec_embeddings = self._embed(texts, self.specialized_model)
                self.specialized_classifier.fit(spec_embeddings, labels)
                logger.info("Specialized classifier trained")

        logger.info("Ensemble training complete")

    def get_stats(self) -> Dict[str, any]:
        """Get performance statistics."""
        total_predictions = (
            self.stats["fast_path_count"] +
            self.stats["deep_path_count"] +
            self.stats["specialized_path_count"]
        )

        if total_predictions == 0:
            return self.stats

        return {
            **self.stats,
            "avg_latency_ms": self.stats["total_latency_ms"] / max(1, total_predictions),
            "deep_path_usage": self.stats["deep_path_count"] / max(1, self.stats["fast_path_count"]),
            "total_predictions": total_predictions
        }

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.stats = {
            "fast_path_count": 0,
            "deep_path_count": 0,
            "specialized_path_count": 0,
            "total_latency_ms": 0,
            "fast_latency_ms": 0,
            "deep_latency_ms": 0
        }
