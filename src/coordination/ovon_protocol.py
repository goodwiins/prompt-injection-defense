from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
import uuid
import hashlib
import structlog

logger = structlog.get_logger()

class LLMTag(BaseModel):
    """
    LLM Tagging mechanism to track message provenance and prevent injection spread.
    Based on Lee & Tiwari (2024) research on mitigating prompt infection.
    """
    agent_id: str
    agent_type: str  # e.g., "guard", "preprocessor", "policy_enforcer"
    trust_level: float = Field(ge=0.0, le=1.0, default=1.0)
    security_clearance: str = Field(default="standard")  # standard, elevated, restricted
    injection_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    def compute_signature(self, message_content: str) -> str:
        """Compute cryptographic signature of message content."""
        data = f"{self.agent_id}:{message_content}:{self.trust_level}"
        return hashlib.sha256(data.encode()).hexdigest()

class OVONContent(BaseModel):
    """Content payload for OVON messages."""
    utterance: str
    whisper: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Metadata not visible to the user (detection results, security tags, etc.)"
    )

class OVONMessage(BaseModel):
    """
    Open Voice Interoperability Initiative (OVON) compatible message schema.
    Enhanced with LLM Tagging for multi-agent security.
    """
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_agent: str
    destination_agent: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    content: OVONContent
    tags: List[str] = Field(default_factory=list)
    llm_tag: Optional[LLMTag] = None
    signature: Optional[str] = None
    hop_count: int = Field(default=0, description="Number of agents this message has passed through")

    @validator('timestamp')
    def validate_timestamp(cls, v):
        if v > datetime.utcnow():
            # Allow small clock skew but generally shouldn't be in future
            pass
        return v

    def add_llm_tag(self, agent_id: str, agent_type: str,
                    trust_level: float = 1.0,
                    security_clearance: str = "standard",
                    injection_score: Optional[float] = None) -> None:
        """
        Add LLM tag to track message provenance and security status.

        Args:
            agent_id: Unique identifier for the agent creating/modifying this message
            agent_type: Type of agent (guard, preprocessor, policy_enforcer, etc.)
            trust_level: Trust level of the source agent (0.0 to 1.0)
            security_clearance: Security clearance level
            injection_score: Detected injection probability (if analyzed)
        """
        self.llm_tag = LLMTag(
            agent_id=agent_id,
            agent_type=agent_type,
            trust_level=trust_level,
            security_clearance=security_clearance,
            injection_score=injection_score
        )
        self.signature = self.llm_tag.compute_signature(self.content.utterance)
        logger.info("LLM tag added", agent_id=agent_id, signature=self.signature[:16])

    def increment_hop(self) -> None:
        """Increment hop count when message passes through an agent."""
        self.hop_count += 1
        if self.hop_count > 10:
            logger.warning("Message hop count exceeded threshold",
                         message_id=self.message_id,
                         hop_count=self.hop_count)

    def verify_signature(self) -> bool:
        """Verify message signature matches content."""
        if not self.llm_tag or not self.signature:
            return False
        expected_sig = self.llm_tag.compute_signature(self.content.utterance)
        return expected_sig == self.signature

    def is_safe(self, max_injection_threshold: float = 0.5,
                min_trust_threshold: float = 0.3) -> bool:
        """
        Determine if message is safe based on LLM tag metadata.

        Args:
            max_injection_threshold: Maximum allowed injection score
            min_trust_threshold: Minimum required trust level

        Returns:
            True if message passes safety checks
        """
        if not self.llm_tag:
            logger.warning("Message missing LLM tag", message_id=self.message_id)
            return False

        # Check trust level
        if self.llm_tag.trust_level < min_trust_threshold:
            logger.warning("Low trust level",
                         agent_id=self.llm_tag.agent_id,
                         trust_level=self.llm_tag.trust_level)
            return False

        # Check injection score if present
        if self.llm_tag.injection_score is not None:
            if self.llm_tag.injection_score > max_injection_threshold:
                logger.warning("High injection score detected",
                             agent_id=self.llm_tag.agent_id,
                             injection_score=self.llm_tag.injection_score)
                return False

        # Verify signature
        if not self.verify_signature():
            logger.error("Signature verification failed", message_id=self.message_id)
            return False

        return True

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.json()

    @classmethod
    def from_json(cls, json_str: str) -> "OVONMessage":
        """Deserialize from JSON string."""
        return cls.parse_raw(json_str)
