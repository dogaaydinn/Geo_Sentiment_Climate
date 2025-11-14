"""
Enterprise-grade Authentication and Authorization Module.

Implements:
- OAuth2 with JWT tokens
- Password-based authentication
- API key authentication
- RBAC (Role-Based Access Control)
- User management
- Session management
- Audit logging

Part of Week 3-4: Security Implementation
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, List
from fastapi import HTTPException, Security, Depends, status, Request
from fastapi.security import (
    HTTPBearer,
    HTTPAuthorizationCredentials,
    APIKeyHeader,
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm
)
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
import secrets
import os
import structlog

from source.api.database import get_db, User, APIKey, AuditLog, Role, Permission

# Configure logger
logger = structlog.get_logger()

# ============================================================================
# Configuration
# ============================================================================

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
REFRESH_SECRET_KEY = os.getenv("JWT_REFRESH_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ============================================================================
# Password & Token Management
# ============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create JWT access token.

    Args:
        data: Payload data to encode
        expires_delta: Token expiration time

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({
        "exp": expire,
        "type": "access",
        "iat": datetime.utcnow()
    })

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """
    Create JWT refresh token.

    Args:
        data: Payload data to encode

    Returns:
        Encoded refresh token
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode.update({
        "exp": expire,
        "type": "refresh",
        "iat": datetime.utcnow()
    })

    encoded_jwt = jwt.encode(to_encode, REFRESH_SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str, token_type: str = "access") -> Dict:
    """
    Verify and decode JWT token.

    Args:
        token: JWT token to verify
        token_type: Type of token ("access" or "refresh")

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If token is invalid
    """
    try:
        # Select correct secret based on token type
        secret = SECRET_KEY if token_type == "access" else REFRESH_SECRET_KEY

        payload = jwt.decode(token, secret, algorithms=[ALGORITHM])

        # Verify token type
        if payload.get("type") != token_type:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token type. Expected {token_type}",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return payload

    except JWTError as e:
        logger.error("jwt_verification_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ============================================================================
# User Authentication
# ============================================================================

def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """
    Authenticate user with username and password.

    Args:
        db: Database session
        username: Username or email
        password: Plain text password

    Returns:
        User object if authentication successful, None otherwise
    """
    # Try to find user by username or email
    user = db.query(User).filter(
        (User.username == username) | (User.email == username)
    ).first()

    if not user:
        logger.warning("authentication_failed", reason="invalid_credentials")
        return None

    if not verify_password(password, user.hashed_password):
        logger.warning("authentication_failed", reason="invalid_credentials")
        return None

    if not user.is_active:
        logger.warning("authentication_failed", reason="invalid_credentials")
        return None

    # Update last login
    user.last_login_at = datetime.utcnow()
    db.commit()

    logger.info("user_authenticated", user_id=user.id, username=user.username)
    return user


def create_user_tokens(user: User) -> Dict[str, str]:
    """
    Create access and refresh tokens for user.

    Args:
        user: User object

    Returns:
        Dictionary with access_token and refresh_token
    """
    # Create token payload
    token_data = {
        "sub": user.username,
        "user_id": user.id,
        "email": user.email,
        "is_admin": user.is_admin,
        "tier": user.tier,
        "roles": [role.name for role in user.roles],
    }

    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token({"sub": user.username, "user_id": user.id})

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


# ============================================================================
# User Management
# ============================================================================

def create_user(
    db: Session,
    username: str,
    email: str,
    password: str,
    full_name: Optional[str] = None,
    tier: str = "free"
) -> User:
    """
    Create a new user.

    Args:
        db: Database session
        username: Username
        email: Email address
        password: Plain text password
        full_name: Full name (optional)
        tier: User tier (free, basic, pro, enterprise)

    Returns:
        Created user object

    Raises:
        HTTPException: If username or email already exists
    """
    # Check if username exists
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )

    # Check if email exists
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create user
    hashed_password = get_password_hash(password)

    user = User(
        username=username,
        email=email,
        full_name=full_name,
        hashed_password=hashed_password,
        tier=tier,
        is_active=True,
        is_verified=False
    )

    # Assign default "user" role
    default_role = db.query(Role).filter(Role.name == "user").first()
    if default_role:
        user.roles.append(default_role)

    db.add(user)
    db.commit()
    db.refresh(user)

    logger.info("user_created", user_id=user.id, username=username, tier=tier)
    return user


# ============================================================================
# API Key Management
# ============================================================================

def create_api_key(
    db: Session,
    user_id: int,
    name: str,
    scopes: List[str],
    expires_in_days: Optional[int] = None,
    rate_limit: int = 60,
    daily_quota: int = 1000
) -> str:
    """
    Create API key for user.

    Args:
        db: Database session
        user_id: User ID
        name: API key name/description
        scopes: List of allowed scopes
        expires_in_days: Expiration in days (None = never)
        rate_limit: Requests per minute
        daily_quota: Daily request quota

    Returns:
        Generated API key string
    """
    # Generate secure API key
    key = f"gsc_{secrets.token_urlsafe(32)}"

    # Calculate expiration
    expires_at = None
    if expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

    api_key = APIKey(
        key=key,
        name=name,
        user_id=user_id,
        scopes=scopes,
        expires_at=expires_at,
        rate_limit_per_minute=rate_limit,
        daily_quota=daily_quota,
        is_active=True
    )

    db.add(api_key)
    db.commit()

    logger.info("api_key_created", user_id=user_id, key_name=name)
    return key


def verify_api_key(db: Session, key: str) -> Optional[User]:
    """
    Verify API key and return associated user.

    Args:
        db: Database session
        key: API key string

    Returns:
        User object if key is valid, None otherwise
    """
    api_key = db.query(APIKey).filter(
        APIKey.key == key,
        APIKey.is_active == True
    ).first()

    if not api_key:
        logger.warning("api_key_invalid", key_prefix=key[:10])
        return None

    # Check expiration
    if api_key.expires_at and api_key.expires_at < datetime.utcnow():
        logger.warning("api_key_expired", key_id=api_key.id)
        return None

    # Update last used
    api_key.last_used_at = datetime.utcnow()
    db.commit()

    # Get user
    user = db.query(User).filter(User.id == api_key.user_id).first()

    if not user or not user.is_active:
        return None

    logger.info("api_key_authenticated", user_id=user.id, key_id=api_key.id)
    return user


# ============================================================================
# Permission Checking
# ============================================================================

def has_permission(user: User, resource: str, action: str) -> bool:
    """
    Check if user has permission for resource and action.

    Args:
        user: User object
        resource: Resource name (e.g., "model", "prediction")
        action: Action name (e.g., "read", "write", "delete")

    Returns:
        True if user has permission, False otherwise
    """
    # Admin has all permissions
    if user.is_admin:
        return True

    # Check permissions through roles
    for role in user.roles:
        for permission in role.permissions:
            if permission.resource == resource and permission.action == action:
                return True

    return False


def require_permission(resource: str, action: str):
    """
    Decorator to require specific permission.

    Usage:
        @app.post("/models/{model_id}/delete")
        @require_permission("model", "delete")
        async def delete_model(model_id: str, current_user: User = Depends(get_current_user)):
            ...
    """
    def permission_checker(current_user: User = Depends(get_current_user)):
        if not has_permission(current_user, resource, action):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {resource}:{action}"
            )
        return current_user
    return permission_checker


# ============================================================================
# Dependencies for FastAPI
# ============================================================================

async def get_current_user_from_token(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Get current user from JWT token.

    Args:
        token: JWT token from Authorization header
        db: Database session

    Returns:
        User object if token is valid
    """
    if not token:
        return None

    try:
        payload = verify_token(token, "access")
        user_id: int = payload.get("user_id")

        if user_id is None:
            return None

        user = db.query(User).filter(User.id == user_id).first()

        if not user or not user.is_active:
            return None

        return user

    except HTTPException:
        return None


async def get_current_user_from_api_key(
    api_key: str = Security(api_key_header),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Get current user from API key.

    Args:
        api_key: API key from X-API-Key header
        db: Database session

    Returns:
        User object if API key is valid
    """
    if not api_key:
        return None

    return verify_api_key(db, api_key)


async def get_current_user(
    token_user: Optional[User] = Depends(get_current_user_from_token),
    api_key_user: Optional[User] = Depends(get_current_user_from_api_key)
) -> User:
    """
    Get current user from either JWT token or API key.

    Args:
        token_user: User from JWT token
        api_key_user: User from API key

    Returns:
        User object

    Raises:
        HTTPException: If not authenticated
    """
    user = token_user or api_key_user

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Ensure user is active."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


async def get_current_admin_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Ensure user has admin privileges."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


# Optional authentication
async def get_optional_user(
    token_user: Optional[User] = Depends(get_current_user_from_token),
    api_key_user: Optional[User] = Depends(get_current_user_from_api_key)
) -> Optional[User]:
    """
    Get current user if authenticated, None otherwise.

    For endpoints that work with or without authentication.
    """
    return token_user or api_key_user


# ============================================================================
# Audit Logging
# ============================================================================

async def log_audit(
    db: Session,
    user: Optional[User],
    action: str,
    resource_type: Optional[str],
    resource_id: Optional[str],
    details: Optional[Dict],
    request: Request,
    success: bool = True,
    error_message: Optional[str] = None
):
    """
    Log audit event.

    Args:
        db: Database session
        user: User who performed action (optional)
        action: Action performed
        resource_type: Type of resource
        resource_id: ID of resource
        details: Additional details (JSON)
        request: FastAPI request object
        success: Whether action succeeded
        error_message: Error message if failed
    """
    audit_log = AuditLog(
        user_id=user.id if user else None,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        success=success,
        error_message=error_message
    )

    db.add(audit_log)
    db.commit()

    logger.info(
        "audit_log",
        user_id=user.id if user else None,
        action=action,
        success=success
    )
