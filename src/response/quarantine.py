import time
from typing import Dict, List, Optional
import structlog

logger = structlog.get_logger()

class QuarantineManager:
    """
    Manages isolation of compromised agents.
    Prevents agents from communicating until cleared.
    """
    def __init__(self, default_timeout: int = 300, redis_client: Optional[Any] = None):
        """
        Initialize quarantine manager.
        
        Args:
            default_timeout: Default quarantine duration in seconds
            redis_client: Optional RedisClient for persistence
        """
        self.default_timeout = default_timeout
        self.redis_client = redis_client
        
        # In-memory fallback
        self.quarantined_agents: Dict[str, float] = {}  # agent_id -> release_timestamp

    def isolate(self, agent_id: str, duration: Optional[int] = None) -> None:
        """
        Place an agent in quarantine.
        
        Args:
            agent_id: The agent to isolate
            duration: Duration in seconds (defaults to configured default)
        """
        timeout = duration or self.default_timeout
        release_time = time.time() + timeout
        
        # 1. Update in-memory
        self.quarantined_agents[agent_id] = release_time
        
        # 2. Persist to Redis if available
        if self.redis_client and self.redis_client.enabled:
            # Add to set of quarantined agents
            self.redis_client.sadd("quarantine:agents", agent_id)
            # Set a key with expiration for auto-cleanup
            self.redis_client.set(f"quarantine:agent:{agent_id}", str(release_time), ex=timeout)
            
        logger.warning("Agent quarantined", agent_id=agent_id, duration=timeout)

    def peer_guard_check(self, agent_id: str, history: List[Dict]) -> bool:
        """
        Triggers a 'Peer Review' where the KPI Evaluator reviews the Guard Agent's recent logs.
        Returns True if the agent is cleared, False otherwise.
        """
        # Placeholder for peer review logic
        # In a real system, this would analyze the history for anomalies
        logger.info("Performing peer guard check", agent_id=agent_id)
        return False # Default to not clearing without explicit review

    def lift_quarantine(self, agent_id: str) -> None:
        """Manually lift quarantine for an agent."""
        if agent_id in self.quarantined_agents:
            del self.quarantined_agents[agent_id]
            
        if self.redis_client and self.redis_client.enabled:
            self.redis_client.delete(f"quarantine:agent:{agent_id}")
            self.redis_client.srem("quarantine:agents", agent_id)
            
        logger.info("Quarantine lifted", agent_id=agent_id)

    def is_quarantined(self, agent_id: str) -> bool:
        """
        Check if an agent is currently quarantined.
        
        Args:
            agent_id: ID of the agent to check.
            
        Returns:
            True if quarantined and not expired, False otherwise.
        """
        expiry = self.quarantined_agents.get(agent_id)
        if expiry is None:
            return False
        
        if time.time() > expiry:
            # Expired, remove from list
            self.release(agent_id)
            return False
            
        return True

    def get_all_quarantined(self) -> List[str]:
        """
        Get a list of all currently quarantined agents.
        
        Returns:
            List of agent IDs.
        """
        # Clean up expired ones first
        current_time = time.time()
        expired = [aid for aid, exp in self.quarantined_agents.items() if current_time > exp]
        for aid in expired:
            self.release(aid)
            
        return list(self.quarantined_agents.keys())
