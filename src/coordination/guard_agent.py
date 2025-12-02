from typing import Dict, Any, Optional
import structlog
from src.detection.embedding_classifier import EmbeddingClassifier
from src.detection.patterns import PatternDetector

logger = structlog.get_logger()

class GuardAgent:
    """
    Coordination agent that orchestrates the detection pipeline.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.pattern_detector = PatternDetector()
        # Initialize embedding classifier with config if available
        model_name = self.config.get("detection", {}).get("fast_model", "all-MiniLM-L6-v2")
        threshold = self.config.get("detection", {}).get("threshold", 0.85)
        self.embedding_classifier = EmbeddingClassifier(model_name=model_name, threshold=threshold)

    def analyze(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze a prompt for injection attacks using all available detectors.
        
        Args:
            prompt: The input text to analyze.
            
        Returns:
            Dictionary with analysis results.
        """
        logger.info("Analyzing prompt", prompt_length=len(prompt))

        # 1. Pattern Detection (Fastest)
        pattern_result = self.pattern_detector.detect(prompt)
        
        # 2. Embedding Classification (Slower but more robust)
        # We get probability of injection (index 1)
        embedding_probs = self.embedding_classifier.predict_proba([prompt])[0]
        embedding_score = float(embedding_probs[1])
        embedding_is_injection = embedding_score >= self.embedding_classifier.threshold

        # Combine results
        # Logic: If either detector is very confident, flag it.
        # Pattern detection is high precision for known attacks.
        # Embedding is better for semantic variations.
        
        is_safe = not (pattern_result["is_suspicious"] or embedding_is_injection)
        
        # Calculate overall confidence
        # If pattern matched, confidence is high (severity).
        # If embedding matched, confidence is the score.
        confidence = max(pattern_result["severity"], embedding_score)

        recommendation = "allow"
        if not is_safe:
            recommendation = "block"
            if confidence < 0.9:
                recommendation = "flag_for_review"

        result = {
            "is_safe": is_safe,
            "confidence": confidence,
            "matched_patterns": pattern_result["matched_categories"],
            "embedding_score": embedding_score,
            "recommendation": recommendation,
            "details": {
                "pattern_analysis": pattern_result,
                "embedding_analysis": {"score": embedding_score, "threshold": self.embedding_classifier.threshold}
            }
        }
        
        logger.info("Analysis complete", result=result)
        return result
