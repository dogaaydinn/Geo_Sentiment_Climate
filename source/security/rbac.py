"""Role-Based Access Control (RBAC) system."""

from enum import Enum
from typing import List, Set
from pydantic import BaseModel


class Permission(str, Enum):
    """Permission types."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    MODEL_TRAIN = "model:train"
    MODEL_DEPLOY = "model:deploy"
    MODEL_DELETE = "model:delete"
    USER_MANAGE = "user:manage"
    API_KEY_MANAGE = "api_key:manage"


class Role(str, Enum):
    """User roles."""
    GUEST = "guest"
    USER = "user"
    DEVELOPER = "developer"
    DATA_SCIENTIST = "data_scientist"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class RBACManager:
    """Manage role-based access control."""

    ROLE_PERMISSIONS = {
        Role.GUEST: {Permission.READ},
        Role.USER: {Permission.READ, Permission.WRITE},
        Role.DEVELOPER: {Permission.READ, Permission.WRITE, Permission.MODEL_TRAIN},
        Role.DATA_SCIENTIST: {
            Permission.READ, Permission.WRITE,
            Permission.MODEL_TRAIN, Permission.MODEL_DEPLOY
        },
        Role.ADMIN: {
            Permission.READ, Permission.WRITE, Permission.DELETE,
            Permission.MODEL_TRAIN, Permission.MODEL_DEPLOY, Permission.MODEL_DELETE,
            Permission.USER_MANAGE, Permission.API_KEY_MANAGE
        },
        Role.SUPER_ADMIN: set(Permission),
    }

    @classmethod
    def get_permissions(cls, role: Role) -> Set[Permission]:
        """Get permissions for role."""
        return cls.ROLE_PERMISSIONS.get(role, set())

    @classmethod
    def has_permission(cls, role: Role, permission: Permission) -> bool:
        """Check if role has permission."""
        return permission in cls.get_permissions(role)
