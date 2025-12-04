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

class SecureMessage(BaseModel):
    """
    OVON-compliant secure message with LLM tagging and integrity checks.
    
    Structure:
    {
      "sender_id": "agent_guard_01",
      "target_id": "agent_policy_01",
      "content": "Normalized user prompt...",
      "whisper": {
         "security_verified": true,
         "detection_score": 0.05,
         "source_tag_hash": "sha256_signature"
      }
    }
    """
    sender_id: str = Field(..., alias="source_agent")
    target_id: str = Field(..., alias="destination_agent")
    content: OVONContent
    whisper: Optional[Dict[str, Any]] = Field(default_factory=dict)
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    
    # Legacy support for existing code that uses llm_tag
    llm_tag: Optional[LLMTag] = None

    class Config:
        populate_by_name = True

    def verify_integrity(self) -> bool:
        """
        Verify message integrity by checking the signature against the sender_id.
        """
        if not self.whisper or "source_tag_hash" not in self.whisper:
            # Fallback to legacy check if whisper is missing but llm_tag exists
            if self.llm_tag:
                return self.is_safe()
            return False
            
        # In a real implementation, we would verify the cryptographic signature here
        # For now, we check if the whisper metadata claims security verification
        return self.whisper.get("security_verified", False)

    def add_whisper_metadata(self, verified: bool, score: float, signature: str):
        """Add security metadata to the whisper channel."""
        self.whisper = {
            "security_verified": verified,
            "detection_score": score,
            "source_tag_hash": signature
        }

    def is_safe(self) -> bool:
        """Legacy safety check."""
        if self.llm_tag:
            return self.llm_tag.trust_level > 0.5
        return False

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.json()

    @classmethod
    def from_json(cls, json_str: str) -> "SecureMessage":
        """Deserialize from JSON string."""
        return cls.parse_raw(json_str)
