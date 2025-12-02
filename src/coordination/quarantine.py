import time
from typing import Dict, List, Optional
import structlog

logger = structlog.get_logger()

class QuarantineManager:
    """
    Manages the isolation of compromised agents.
    """

    def __init__(self, default_timeout: int = 300):
        """
        Args:
            default_timeout: Default quarantine duration in seconds.
        """
        self.quarantined_agents: Dict[str, float] = {}
        self.default_timeout = default_timeout

    def isolate(self, agent_id: str, duration: Optional[int] = None):
        """
        Place an agent in quarantine.
        
        Args:
            agent_id: ID of the agent to isolate.
            duration: Duration in seconds. Defaults to default_timeout.
        """
        timeout = duration if duration is not None else self.default_timeout
        expiry = time.time() + timeout
        self.quarantined_agents[agent_id] = expiry
        logger.warning("Agent quarantined", agent_id=agent_id, duration=timeout)

    def release(self, agent_id: str):
        """
        Release an agent from quarantine.
        
        Args:
            agent_id: ID of the agent to release.
        """
        if agent_id in self.quarantined_agents:
            del self.quarantined_agents[agent_id]
            logger.info("Agent released from quarantine", agent_id=agent_id)

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
