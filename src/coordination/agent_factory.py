from typing import Dict, Any, Optional
import structlog
from .messaging import SecureMessage, OVONContent
from ..detection.ensemble import InjectionDetector
from ..response.quarantine import QuarantineProtocol

logger = structlog.get_logger()

class SecureAgent:
    """
    Base class for secure agents in the defense framework.
    Wraps standard LLM calls with security checks.
    """
    def __init__(self, agent_id: str, role: str):
        self.agent_id = agent_id
        self.role = role
        self.quarantine_status = False
        self.logger = logger.bind(agent_id=agent_id, role=role)

    def process(self, message: SecureMessage) -> Optional[SecureMessage]:
        """Process an incoming message and return a response."""
        raise NotImplementedError

    def _create_response(self, target_id: str, content: str, 
                        security_verified: bool = False, score: float = 0.0) -> SecureMessage:
        """Helper to create a secure response message."""
        msg = SecureMessage(
            source_agent=self.agent_id,
            destination_agent=target_id,
            content=OVONContent(utterance=content)
        )
        # Add whisper metadata
        msg.add_whisper_metadata(
            verified=security_verified,
            score=score,
            signature=f"sig_{self.agent_id}" # Simplified signature
        )
        return msg

class PreprocessorAgent(SecureAgent):
    """
    Agent responsible for cleaning and normalizing input.
    Performs regex cleaning and encoding checks.
    """
    def process(self, message: SecureMessage) -> Optional[SecureMessage]:
        self.logger.info("Preprocessing message")
        # Simple normalization
        cleaned_text = message.content.utterance.strip()
        # In a real impl, we'd do more complex cleaning here
        
        return self._create_response(
            target_id=message.sender_id, # Echo back or forward? Usually forward in a pipeline
            content=cleaned_text,
            security_verified=True
        )

class GuardAgent(SecureAgent):
    """
    Primary defense agent using InjectionDetector.
    """
    def __init__(self, agent_id: str, detector: InjectionDetector, quarantine: QuarantineProtocol):
        super().__init__(agent_id, "guard")
        self.detector = detector
        self.quarantine = quarantine

    def process(self, message: SecureMessage) -> Optional[SecureMessage]:
        self.logger.info("Analyzing message for injection")
        
        # Check quarantine
        if self.quarantine.is_quarantined(message.sender_id):
            self.logger.warning("Sender is quarantined", sender=message.sender_id)
            return self._create_response(
                target_id=message.sender_id,
                content="Access Denied: You are in quarantine.",
                security_verified=False,
                score=1.0
            )

        # Detect injection
        result = self.detector.scan(message.content.utterance)
        
        if result["is_injection"]:
            self.logger.warning("Injection detected", score=result["score"])
            # Isolate the sender
            self.quarantine.isolate_agent(message.sender_id)
            
            return self._create_response(
                target_id=message.sender_id,
                content="Security Alert: Prompt Injection Detected.",
                security_verified=False,
                score=result["score"]
            )
            
        self.logger.info("Message safe", score=result["score"])
        return self._create_response(
            target_id=message.sender_id,
            content=message.content.utterance,
            security_verified=True,
            score=result["score"]
        )

class PolicyAgent(SecureAgent):
    """
    Agent responsible for verifying output against corporate policy.
    """
    def process(self, message: SecureMessage) -> Optional[SecureMessage]:
        self.logger.info("Checking policy compliance")
        # Placeholder for policy check
        return self._create_response(
            target_id=message.sender_id,
            content=message.content.utterance,
            security_verified=True
        )

class AgentFactory:
    """Factory to spawn secure agents."""
    
    @staticmethod
    def create_guard(agent_id: str, detector: InjectionDetector, quarantine: QuarantineProtocol) -> GuardAgent:
        return GuardAgent(agent_id, detector, quarantine)
        
    @staticmethod
    def create_preprocessor(agent_id: str) -> PreprocessorAgent:
        return PreprocessorAgent(agent_id, "preprocessor")
        
    @staticmethod
    def create_policy(agent_id: str) -> PolicyAgent:
        return PolicyAgent(agent_id, "policy")
