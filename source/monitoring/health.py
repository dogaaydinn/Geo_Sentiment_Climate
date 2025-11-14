"""Health check system."""

from enum import Enum
from typing import Dict, List
from pydantic import BaseModel
from datetime import datetime


class HealthStatus(str, Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Health status of a component."""
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    metadata: Dict = {}


class HealthChecker:
    """Aggregate health checker for all system components."""

    @staticmethod
    async def check_database() -> ComponentHealth:
        """Check database health."""
        try:
            from source.database.base import check_db_connection
            start = datetime.now()
            is_healthy = check_db_connection()
            latency = (datetime.now() - start).total_seconds() * 1000

            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
                message="PostgreSQL connection OK" if is_healthy else "Connection failed",
                latency_ms=latency
            )
        except Exception as e:
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Error: {str(e)}"
            )

    @staticmethod
    async def check_redis() -> ComponentHealth:
        """Check Redis health."""
        try:
            from source.cache.redis_cache import cache
            start = datetime.now()
            cache.client.ping() if cache.client else None
            latency = (datetime.now() - start).total_seconds() * 1000

            return ComponentHealth(
                name="redis",
                status=HealthStatus.HEALTHY,
                message="Redis connection OK",
                latency_ms=latency
            )
        except Exception as e:
            return ComponentHealth(
                name="redis",
                status=HealthStatus.DEGRADED,
                message=f"Redis unavailable: {str(e)}"
            )

    @staticmethod
    async def check_models() -> ComponentHealth:
        """Check model availability."""
        try:
            from source.ml.model_registry import ModelRegistry
            registry = ModelRegistry()
            models = registry.list_models(stage="production")

            return ComponentHealth(
                name="models",
                status=HealthStatus.HEALTHY if models else HealthStatus.DEGRADED,
                message=f"{len(models)} production models available",
                metadata={"count": len(models)}
            )
        except Exception as e:
            return ComponentHealth(
                name="models",
                status=HealthStatus.UNHEALTHY,
                message=f"Error: {str(e)}"
            )

    @staticmethod
    async def check_all() -> Dict[str, ComponentHealth]:
        """Check all components."""
        return {
            "database": await HealthChecker.check_database(),
            "redis": await HealthChecker.check_redis(),
            "models": await HealthChecker.check_models(),
        }
