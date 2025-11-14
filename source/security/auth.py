"""
JWT and OAuth2 Authentication Module.

Enterprise-grade authentication with:
- JWT tokens (access + refresh)
- Password hashing with bcrypt
- OAuth2 password flow
- Token refresh mechanism
- User session management
- Multi-factor authentication support
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
import redis
from source.utils.logger import setup_logger

logger = setup_logger(name="auth", log_file="../logs/auth.log")

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production-min-32-chars-long")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

# Redis connection for token blacklist
redis_client = None
try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=int(os.getenv("REDIS_DB", "0")),
        decode_responses=True
    )
    redis_client.ping()
    logger.info("Redis connection established for auth")
except Exception as e:
    logger.warning(f"Redis not available for token blacklist: {e}")


# Pydantic Models
class Token(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token payload data."""
    user_id: Optional[str] = None
    username: Optional[str] = None
    email: Optional[str] = None
    roles: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    exp: Optional[datetime] = None


class User(BaseModel):
    """User model."""
    id: str
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    disabled: bool = False
    roles: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    mfa_enabled: bool = False


class UserInDB(User):
    """User model with hashed password."""
    hashed_password: str


class UserCreate(BaseModel):
    """User creation model."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
    roles: List[str] = Field(default_factory=lambda: ["user"])


class UserLogin(BaseModel):
    """User login model."""
    username: str
    password: str
    mfa_code: Optional[str] = None


class JWTHandler:
    """
    JWT token handler with refresh token support.

    Features:
    - Access and refresh token generation
    - Token verification and validation
    - Token blacklisting (logout)
    - Automatic token refresh
    """

    def __init__(
        self,
        secret_key: str = SECRET_KEY,
        algorithm: str = ALGORITHM,
        access_token_expire_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES,
        refresh_token_expire_days: int = REFRESH_TOKEN_EXPIRE_DAYS
    ):
        """
        Initialize JWT handler.

        Args:
            secret_key: Secret key for encoding tokens
            algorithm: JWT algorithm (HS256, RS256, etc.)
            access_token_expire_minutes: Access token lifetime
            refresh_token_expire_days: Refresh token lifetime
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        self.logger = logger

    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT access token.

        Args:
            data: Token payload data
            expires_delta: Custom expiration time

        Returns:
            Encoded JWT token
        """
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)

        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })

        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def create_refresh_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT refresh token.

        Args:
            data: Token payload data
            expires_delta: Custom expiration time

        Returns:
            Encoded refresh token
        """
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)

        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        })

        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str) -> TokenData:
        """
        Verify and decode JWT token.

        Args:
            token: JWT token string

        Returns:
            Decoded token data

        Raises:
            HTTPException: If token is invalid or expired
        """
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

        try:
            # Check if token is blacklisted
            if redis_client and self._is_token_blacklisted(token):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )

            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            user_id: str = payload.get("user_id")
            username: str = payload.get("username")
            email: str = payload.get("email")
            roles: List[str] = payload.get("roles", [])
            permissions: List[str] = payload.get("permissions", [])

            if user_id is None or username is None:
                raise credentials_exception

            token_data = TokenData(
                user_id=user_id,
                username=username,
                email=email,
                roles=roles,
                permissions=permissions,
                exp=datetime.fromtimestamp(payload.get("exp"))
            )

            return token_data

        except JWTError as e:
            self.logger.error(f"JWT verification failed: {e}")
            raise credentials_exception

    def blacklist_token(self, token: str, exp: datetime) -> bool:
        """
        Blacklist a token (for logout).

        Args:
            token: JWT token to blacklist
            exp: Token expiration time

        Returns:
            True if successful, False otherwise
        """
        if not redis_client:
            self.logger.warning("Redis not available, cannot blacklist token")
            return False

        try:
            # Calculate TTL based on token expiration
            ttl = int((exp - datetime.utcnow()).total_seconds())
            if ttl > 0:
                redis_client.setex(f"blacklist:{token}", ttl, "1")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to blacklist token: {e}")
            return False

    def _is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted."""
        try:
            return redis_client.exists(f"blacklist:{token}") > 0
        except Exception:
            return False

    def refresh_access_token(self, refresh_token: str) -> Token:
        """
        Create new access token from refresh token.

        Args:
            refresh_token: Valid refresh token

        Returns:
            New token pair

        Raises:
            HTTPException: If refresh token is invalid
        """
        try:
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=[self.algorithm])

            if payload.get("type") != "refresh":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )

            # Create new tokens
            token_data = {
                "user_id": payload.get("user_id"),
                "username": payload.get("username"),
                "email": payload.get("email"),
                "roles": payload.get("roles", []),
                "permissions": payload.get("permissions", [])
            }

            access_token = self.create_access_token(token_data)
            new_refresh_token = self.create_refresh_token(token_data)

            return Token(
                access_token=access_token,
                refresh_token=new_refresh_token,
                token_type="bearer",
                expires_in=self.access_token_expire_minutes * 60
            )

        except JWTError as e:
            self.logger.error(f"Token refresh failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not refresh token"
            )


class OAuth2Handler:
    """
    OAuth2 authentication handler.

    Supports password flow and can be extended for
    authorization code flow, client credentials, etc.
    """

    def __init__(self):
        """Initialize OAuth2 handler."""
        self.jwt_handler = JWTHandler()
        self.logger = logger

    def authenticate_user(self, username: str, password: str) -> Optional[UserInDB]:
        """
        Authenticate user with username and password.

        Args:
            username: Username or email
            password: Plain text password

        Returns:
            User object if authentication successful, None otherwise
        """
        # TODO: Implement database lookup
        # This is a placeholder - integrate with your database
        user = self._get_user_from_db(username)

        if not user:
            return None

        if not self.verify_password(password, user.hashed_password):
            return None

        return user

    def _get_user_from_db(self, username: str) -> Optional[UserInDB]:
        """
        Get user from database (placeholder).

        Args:
            username: Username or email

        Returns:
            User object or None
        """
        # TODO: Implement actual database query
        # Placeholder for testing
        if username == "admin":
            return UserInDB(
                id="1",
                username="admin",
                email="admin@example.com",
                full_name="Admin User",
                disabled=False,
                roles=["admin", "user"],
                permissions=["read", "write", "admin"],
                hashed_password=self.get_password_hash("admin123"),
                created_at=datetime.utcnow()
            )
        return None

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify password against hash.

        Args:
            plain_password: Plain text password
            hashed_password: Hashed password from database

        Returns:
            True if password matches, False otherwise
        """
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """
        Hash password with bcrypt.

        Args:
            password: Plain text password

        Returns:
            Hashed password
        """
        return pwd_context.hash(password)

    async def login(self, form_data: OAuth2PasswordRequestForm) -> Token:
        """
        Authenticate user and return tokens.

        Args:
            form_data: OAuth2 form data with username and password

        Returns:
            Token pair (access + refresh)

        Raises:
            HTTPException: If authentication fails
        """
        user = self.authenticate_user(form_data.username, form_data.password)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if user.disabled:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is disabled"
            )

        # Create tokens
        token_data = {
            "user_id": user.id,
            "username": user.username,
            "email": user.email,
            "roles": user.roles,
            "permissions": user.permissions
        }

        access_token = self.jwt_handler.create_access_token(token_data)
        refresh_token = self.jwt_handler.create_refresh_token(token_data)

        self.logger.info(f"User {user.username} logged in successfully")

        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )

    async def logout(self, token: str) -> bool:
        """
        Logout user by blacklisting token.

        Args:
            token: Access token to blacklist

        Returns:
            True if successful
        """
        token_data = self.jwt_handler.verify_token(token)
        success = self.jwt_handler.blacklist_token(token, token_data.exp)

        if success:
            self.logger.info(f"User {token_data.username} logged out")

        return success


# Dependency functions for FastAPI
async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Get current authenticated user from token.

    Args:
        token: JWT access token

    Returns:
        Current user

    Raises:
        HTTPException: If token is invalid
    """
    jwt_handler = JWTHandler()
    token_data = jwt_handler.verify_token(token)

    # TODO: Get user from database
    # Placeholder implementation
    user = User(
        id=token_data.user_id,
        username=token_data.username,
        email=token_data.email,
        roles=token_data.roles,
        permissions=token_data.permissions
    )

    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Get current active user (not disabled).

    Args:
        current_user: Current user from token

    Returns:
        Active user

    Raises:
        HTTPException: If user is disabled
    """
    if current_user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    return current_user


def require_role(required_roles: List[str]):
    """
    Dependency to require specific roles.

    Args:
        required_roles: List of required roles

    Returns:
        Dependency function
    """
    async def role_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if not any(role in current_user.roles for role in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User does not have required role(s): {required_roles}"
            )
        return current_user

    return role_checker


def require_permission(required_permissions: List[str]):
    """
    Dependency to require specific permissions.

    Args:
        required_permissions: List of required permissions

    Returns:
        Dependency function
    """
    async def permission_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if not any(perm in current_user.permissions for perm in required_permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User does not have required permission(s): {required_permissions}"
            )
        return current_user

    return permission_checker
