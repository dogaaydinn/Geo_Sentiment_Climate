"""
Authentication and Authorization Module.

Provides JWT-based authentication and API key management for the API.
Part of Phase 2: Enhancement & Integration - Security implementation.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from jose import JWTError, jwt
from passlib.context import CryptContext
import secrets
import os

# Security configurations
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
bearer_scheme = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class AuthManager:
    """Manages authentication and authorization."""

    def __init__(self):
        self.api_keys: Dict[str, Dict] = {}
        self._load_api_keys()

    def _load_api_keys(self):
        """Load API keys from environment or database."""
        # Example: Load from environment
        default_key = os.getenv("DEFAULT_API_KEY")
        if default_key:
            self.api_keys[default_key] = {
                "name": "default",
                "permissions": ["read", "write"],
                "created_at": datetime.now().isoformat()
            }

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)

    def hash_password(self, password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)

    def create_access_token(
        self,
        data: dict,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT access token.

        Args:
            data: Data to encode in token
            expires_delta: Token expiration time

        Returns:
            Encoded JWT token
        """
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

        return encoded_jwt

    def verify_token(self, token: str) -> Dict:
        """
        Verify and decode JWT token.

        Args:
            token: JWT token to verify

        Returns:
            Decoded token data

        Raises:
            HTTPException: If token is invalid
        """
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError as e:
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def verify_api_key(self, api_key: str) -> Dict:
        """
        Verify API key.

        Args:
            api_key: API key to verify

        Returns:
            API key information

        Raises:
            HTTPException: If API key is invalid
        """
        if api_key not in self.api_keys:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )

        return self.api_keys[api_key]

    def generate_api_key(self, name: str, permissions: list) -> str:
        """
        Generate a new API key.

        Args:
            name: Name/description for the API key
            permissions: List of permissions

        Returns:
            Generated API key
        """
        api_key = f"gsc_{secrets.token_urlsafe(32)}"

        self.api_keys[api_key] = {
            "name": name,
            "permissions": permissions,
            "created_at": datetime.now().isoformat()
        }

        return api_key


# Global auth manager instance
auth_manager = AuthManager()


# Dependency functions
async def get_current_user_jwt(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)
) -> Dict:
    """
    Dependency to get current user from JWT token.

    Args:
        credentials: Bearer token credentials

    Returns:
        User information from token
    """
    token = credentials.credentials
    return auth_manager.verify_token(token)


async def get_current_user_api_key(
    api_key: str = Security(api_key_header)
) -> Dict:
    """
    Dependency to get current user from API key.

    Args:
        api_key: API key from header

    Returns:
        API key information
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required"
        )

    return auth_manager.verify_api_key(api_key)


async def get_current_user(
    jwt_user: Optional[Dict] = Depends(get_current_user_jwt),
    api_key_user: Optional[Dict] = Depends(get_current_user_api_key)
) -> Dict:
    """
    Dependency to get current user from either JWT or API key.

    Returns:
        User information
    """
    # Try JWT first, then API key
    if jwt_user:
        return jwt_user
    if api_key_user:
        return api_key_user

    raise HTTPException(
        status_code=401,
        detail="Authentication required"
    )


# Optional authentication (for routes that can work with or without auth)
async def get_optional_user(
    api_key: Optional[str] = Security(api_key_header)
) -> Optional[Dict]:
    """
    Optional authentication dependency.

    Returns:
        User information if authenticated, None otherwise
    """
    if api_key:
        try:
            return auth_manager.verify_api_key(api_key)
        except HTTPException:
            return None
    return None
