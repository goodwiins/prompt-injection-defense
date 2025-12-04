import numpy as np
from typing import List, Dict, Any, Optional, Union
import structlog

logger = structlog.get_logger()

class AttentionMonitor:
    """
    Monitors attention weights to detect 'distraction effects' where
    attention shifts significantly to injected commands.
    
    Based on research: 'Stopping AI Hallucinations: New Research on Prompt Injection Attacks'
    and 'Attention-Guided Jailbreak Detection'.
    """
    
    def __init__(self, distraction_threshold: float = 0.6, top_k_heads: int = 5):
        """
        Initialize the Attention Monitor.
        
        Args:
            distraction_threshold: Threshold for attention score concentration on injected parts.
            top_k_heads: Number of attention heads to monitor (simulated for now).
        """
        self.distraction_threshold = distraction_threshold
        self.top_k_heads = top_k_heads
        
    def analyze_attention(self, 
                         full_text: str, 
                         suspected_injection_ranges: List[tuple[int, int]], 
                         attention_weights: Optional[Any] = None) -> Dict[str, Any]:
        """
        Analyze attention weights to check if the model is 'distracted' by the suspected injection.
        
        Args:
            full_text: The complete prompt text.
            suspected_injection_ranges: List of (start, end) indices of suspected injected text.
            attention_weights: Optional tensor/array of attention weights. 
                               If None, runs in simulation mode based on heuristics.
                               
        Returns:
            Dictionary containing analysis results.
        """
        if not suspected_injection_ranges:
            return {
                "is_distracted": False,
                "distraction_score": 0.0,
                "details": "No suspected injection ranges provided"
            }

        # Simulation mode if no weights provided (common for API-based usage or dev/test)
        if attention_weights is None:
            return self._simulate_attention_analysis(full_text, suspected_injection_ranges)
            
        # Real analysis would go here (requires model internals)
        # For this implementation, we'll focus on the interface and simulation
        # as we don't have a running local LLM with exposed weights in this environment.
        return self._simulate_attention_analysis(full_text, suspected_injection_ranges)

    def _simulate_attention_analysis(self, 
                                   text: str, 
                                   ranges: List[tuple[int, int]]) -> Dict[str, Any]:
        """
        Simulate attention analysis using heuristics.
        
        Heuristic: If the suspected injection is at the very end or beginning, 
        or constitutes a large portion of the text, we assume higher 'distraction' potential.
        """
        text_len = len(text)
        if text_len == 0:
            return {"is_distracted": False, "distraction_score": 0.0}
            
        max_score = 0.0
        
        for start, end in ranges:
            injection_len = end - start
            ratio = injection_len / text_len
            
            # Heuristic 1: Length ratio
            score = ratio
            
            # Heuristic 2: Position (Recency bias - end of prompt is often more attended to)
            if end > text_len * 0.8:
                score *= 1.2
                
            max_score = max(max_score, score)
            
        # Normalize score to 0-1 range roughly
        max_score = min(max_score, 1.0)
        
        is_distracted = max_score > self.distraction_threshold
        
        if is_distracted:
            logger.warning("Attention distraction detected (simulated)", 
                          score=max_score, 
                          threshold=self.distraction_threshold)
            
        return {
            "is_distracted": is_distracted,
            "distraction_score": max_score,
            "details": "Simulated analysis based on position and length heuristics"
        }

    def get_vulnerable_heads(self) -> List[int]:
        """
        Return indices of attention heads identified as vulnerable.
        (Placeholder for future implementation where we track specific heads)
        """
        return list(range(self.top_k_heads))
