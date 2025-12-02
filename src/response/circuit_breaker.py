import time
from typing import List
import structlog

logger = structlog.get_logger()

class CircuitBreaker:
    """
    Circuit breaker pattern to stop processing when attack threshold is exceeded.
    """

    def __init__(self, threshold: int = 10, time_window: int = 60):
        """
        Args:
            threshold: Number of events allowed within the time window.
            time_window: Time window in seconds.
        """
        self.threshold = threshold
        self.time_window = time_window
        self.events: List[float] = []
        self._is_open = False

    def record_event(self):
        """Record a suspicious event."""
        current_time = time.time()
        self.events.append(current_time)
        self._cleanup(current_time)
        
        if len(self.events) >= self.threshold:
            if not self._is_open:
                logger.critical("Circuit breaker tripped!", threshold=self.threshold, window=self.time_window)
                self._is_open = True

    def is_open(self) -> bool:
        """
        Check if the circuit breaker is open (tripped).
        
        Returns:
            True if open (should stop processing), False otherwise.
        """
        self._cleanup(time.time())
        return self._is_open

    def reset(self):
        """Reset the circuit breaker manually."""
        self.events = []
        self._is_open = False
        logger.info("Circuit breaker reset")

    def _cleanup(self, current_time: float):
        """Remove events outside the time window."""
        # Keep only events within the window
        cutoff = current_time - self.time_window
        self.events = [t for t in self.events if t > cutoff]
        
        # If count drops below threshold, we could auto-close, 
        # but usually circuit breakers require manual reset or cool-down.
        # For this implementation, we'll keep it open until reset or if we want auto-recovery logic.
        # Let's implement auto-recovery if no events happen for a while?
        # The requirement says "Auto-opens when events exceed", doesn't specify auto-close.
        # I'll stick to manual reset or if the list clears up significantly, 
        # but standard pattern is usually strict. 
        # However, if the list is empty, it should probably be closed.
        
        if self._is_open and len(self.events) == 0:
             # Auto-recover if window passes completely? 
             # Let's keep it simple: if it's open, it stays open until reset 
             # UNLESS we want a "half-open" state. 
             # Given the prompt "Auto-shutdown on alert threshold", implies a hard stop.
             pass
