from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
import uuid

class OVONContent(BaseModel):
    """Content payload for OVON messages."""
    utterance: str
    whisper: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata not visible to the user")

class OVONMessage(BaseModel):
    """
    Open Voice Interoperability Initiative (OVON) compatible message schema.
    """
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_agent: str
    destination_agent: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    content: OVONContent
    tags: List[str] = Field(default_factory=list)

    @validator('timestamp')
    def validate_timestamp(cls, v):
        if v > datetime.utcnow():
            # Allow small clock skew but generally shouldn't be in future
            pass 
        return v

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.json()

    @classmethod
    def from_json(cls, json_str: str) -> "OVONMessage":
        """Deserialize from JSON string."""
        return cls.parse_raw(json_str)
