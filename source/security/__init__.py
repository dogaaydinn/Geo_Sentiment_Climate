"""
Enterprise Security Module.

Provides comprehensive security features:
- JWT authentication with refresh tokens
- OAuth2/OIDC integration
- API key management
- Role-based access control (RBAC)
- Rate limiting
- Input validation and sanitization
- Audit logging
"""

from .auth import (
    JWTHandler,
    OAuth2Handler,
    get_current_user,
    get_current_active_user,
    require_role,
    require_permission
)
from .api_key import APIKeyManager, validate_api_key
from .rate_limiter import RateLimiter, rate_limit
from .rbac import RBACManager, Permission, Role
from .validation import InputValidator, sanitize_input
from .audit import AuditLogger, log_action

__all__ = [
    "JWTHandler",
    "OAuth2Handler",
    "get_current_user",
    "get_current_active_user",
    "require_role",
    "require_permission",
    "APIKeyManager",
    "validate_api_key",
    "RateLimiter",
    "rate_limit",
    "RBACManager",
    "Permission",
    "Role",
    "InputValidator",
    "sanitize_input",
    "AuditLogger",
    "log_action",
]
