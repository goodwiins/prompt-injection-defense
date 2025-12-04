import time
from typing import Dict, List, Optional
import structlog

logger = structlog.get_logger()

class QuarantineProtocol:
    """
    Manages isolation of compromised agents.
    Prevents agents from communicating until cleared.
    """
    def __init__(self, default_duration: int = 300):
        self.quarantined_agents: Dict[str, float] = {}
        self.default_duration = default_duration
        self.history: Dict[str, List[Dict]] = {}

    def isolate_agent(self, agent_id: str, duration: Optional[int] = None) -> None:
        """
        Isolate an agent for a specified duration.
        """
        duration = duration or self.default_duration
        release_time = time.time() + duration
        self.quarantined_agents[agent_id] = release_time
        logger.warning("Agent quarantined", agent_id=agent_id, duration=duration)

    def peer_guard_check(self, agent_id: str, history: List[Dict]) -> bool:
        """
        Triggers a 'Peer Review' where the KPI Evaluator reviews the Guard Agent's recent logs.
        Returns True if the agent is cleared, False otherwise.
        """
        # Placeholder for peer review logic
        # In a real system, this would analyze the history for anomalies
        logger.info("Performing peer guard check", agent_id=agent_id)
        return False # Default to not clearing without explicit review

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
