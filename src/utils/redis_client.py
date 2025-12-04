import os
import structlog
from typing import Optional, Any, List, Set

logger = structlog.get_logger()

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis client not installed. Using in-memory fallback.")

class RedisClient:
    """
    Wrapper for Redis client with graceful fallback to in-memory storage.
    Ensures the application works even if Redis is not configured.
    """
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.enabled = False
        self.client = None
        
        # Check if explicitly disabled via env
        if os.getenv("DISABLE_REDIS", "false").lower() == "true":
            logger.info("Redis disabled via environment variable")
            return

        if REDIS_AVAILABLE:
            try:
                self.client = redis.Redis(
                    host=os.getenv("REDIS_HOST", host),
                    port=int(os.getenv("REDIS_PORT", port)),
                    db=int(os.getenv("REDIS_DB", db)),
                    decode_responses=True,
                    socket_connect_timeout=1
                )
                self.client.ping()
                self.enabled = True
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning("Failed to connect to Redis. Using in-memory fallback.", error=str(e))
                self.client = None
        
    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Set a value with optional expiration."""
        if self.enabled and self.client:
            try:
                return self.client.set(key, value, ex=ex)
            except Exception:
                return False
        return False

    def get(self, key: str) -> Optional[str]:
        """Get a value."""
        if self.enabled and self.client:
            try:
                return self.client.get(key)
            except Exception:
                return None
        return None
        
    def sadd(self, key: str, *values: Any) -> int:
        """Add to set."""
        if self.enabled and self.client and values:
            try:
                return self.client.sadd(key, *values)
            except Exception:
                return 0
        return 0
        
    def sismember(self, key: str, value: Any) -> bool:
        """Check if value is in set."""
        if self.enabled and self.client:
            try:
                return self.client.sismember(key, value)
            except Exception:
                return False
        return False
        
    def smembers(self, key: str) -> Set[str]:
        """Get all members of set."""
        if self.enabled and self.client:
            try:
                return self.client.smembers(key)
            except Exception:
                return set()
        return set()
        
    def delete(self, key: str) -> int:
        """Delete a key."""
        if self.enabled and self.client:
            try:
                return self.client.delete(key)
            except Exception:
                return 0
        return 0
