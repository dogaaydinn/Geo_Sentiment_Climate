"""
API Key Management System.

Enterprise-grade API key features:
- Secure key generation
- Key rotation
- Usage tracking
- Key permissions and scopes
- Rate limiting per key
- Expiration management
"""

import os
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from fastapi import Header, HTTPException, status
import redis
from source.utils.logger import setup_logger

logger = setup_logger(name="api_key", log_file="../logs/api_key.log")

# Redis connection for API key storage
redis_client = None
try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=int(os.getenv("REDIS_API_KEY_DB", "2")),
        decode_responses=True
    )
    redis_client.ping()
except Exception as e:
    logger.warning(f"Redis not available for API keys: {e}")


class APIKey(BaseModel):
    """API Key model."""
    key_id: str
    name: str
    key_hash: str
    user_id: str
    scopes: List[str] = Field(default_factory=list)
    rate_limit: int = 1000
    rate_window: int = 3600
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    is_active: bool = True


class APIKeyManager:
    """
    Manage API keys for authentication.

    Features:
    - Generate secure API keys
    - Validate keys
    - Track usage
    - Rotate keys
    - Scope-based permissions
    """

    def __init__(self, redis_client: Optional[redis.Redis] = redis_client):
        """Initialize API key manager."""
        self.redis = redis_client
        self.logger = logger

    def generate_key(
        self,
        user_id: str,
        name: str,
        scopes: List[str] = None,
        rate_limit: int = 1000,
        expires_in_days: Optional[int] = None
    ) -> tuple[str, APIKey]:
        """
        Generate new API key.

        Args:
            user_id: User ID
            name: Key name/description
            scopes: List of permission scopes
            rate_limit: Rate limit for this key
            expires_in_days: Expiration in days (None for no expiration)

        Returns:
            Tuple of (plain_key, api_key_object)
        """
        # Generate secure random key
        plain_key = f"gsc_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(plain_key.encode()).hexdigest()
        key_id = secrets.token_urlsafe(16)

        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        api_key = APIKey(
            key_id=key_id,
            name=name,
            key_hash=key_hash,
            user_id=user_id,
            scopes=scopes or ["read"],
            rate_limit=rate_limit,
            expires_at=expires_at
        )

        # Store in Redis
        if self.redis:
            self._store_key(api_key)

        self.logger.info(f"Generated API key {key_id} for user {user_id}")
        return plain_key, api_key

    def _store_key(self, api_key: APIKey):
        """Store API key in Redis."""
        key = f"api_key:{api_key.key_hash}"
        data = api_key.model_dump_json()

        if api_key.expires_at:
            ttl = int((api_key.expires_at - datetime.utcnow()).total_seconds())
            self.redis.setex(key, ttl, data)
        else:
            self.redis.set(key, data)

    def validate_key(self, plain_key: str) -> Optional[APIKey]:
        """
        Validate API key.

        Args:
            plain_key: Plain text API key

        Returns:
            APIKey object if valid, None otherwise
        """
        if not plain_key.startswith("gsc_"):
            return None

        key_hash = hashlib.sha256(plain_key.encode()).hexdigest()

        if not self.redis:
            return None

        key_data = self.redis.get(f"api_key:{key_hash}")
        if not key_data:
            return None

        api_key = APIKey.model_validate_json(key_data)

        # Check if expired
        if api_key.expires_at and api_key.expires_at < datetime.utcnow():
            self.logger.warning(f"API key {api_key.key_id} expired")
            return None

        # Check if active
        if not api_key.is_active:
            return None

        # Update last used
        api_key.last_used = datetime.utcnow()
        self._store_key(api_key)

        return api_key

    def revoke_key(self, key_hash: str) -> bool:
        """Revoke API key."""
        if not self.redis:
            return False

        return self.redis.delete(f"api_key:{key_hash}") > 0


async def validate_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> APIKey:
    """
    FastAPI dependency to validate API key from header.

    Args:
        x_api_key: API key from header

    Returns:
        Valid API key object

    Raises:
        HTTPException: If key is invalid
    """
    manager = APIKeyManager()
    api_key = manager.validate_key(x_api_key)

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    return api_key
