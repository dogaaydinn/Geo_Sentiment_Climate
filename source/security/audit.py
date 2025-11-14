"""Audit logging for security and compliance."""

import json
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
from source.utils.logger import setup_logger

logger = setup_logger(name="audit", log_file="../logs/audit.log")


class AuditAction(str, Enum):
    """Audit action types."""
    LOGIN = "login"
    LOGOUT = "logout"
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    PREDICT = "predict"
    MODEL_TRAIN = "model_train"
    MODEL_DEPLOY = "model_deploy"


class AuditLogger:
    """Log security and compliance events."""

    @staticmethod
    def log_action(
        action: AuditAction,
        user_id: Optional[str],
        resource: str,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        success: bool = True
    ):
        """
        Log audit event.

        Args:
            action: Type of action
            user_id: User performing action
            resource: Resource being accessed
            details: Additional details
            ip_address: Client IP address
            success: Whether action was successful
        """
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action.value,
            "user_id": user_id,
            "resource": resource,
            "details": details or {},
            "ip_address": ip_address,
            "success": success
        }

        logger.info(f"AUDIT: {json.dumps(event)}")


def log_action(action: AuditAction, user_id: str, resource: str, **kwargs):
    """Convenience function for audit logging."""
    AuditLogger.log_action(action, user_id, resource, **kwargs)
