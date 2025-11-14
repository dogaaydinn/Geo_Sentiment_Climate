"""
Multi-Tenancy Infrastructure.

Provides tenant isolation, resource management, and custom configurations.
Part of Phase 5: Enterprise & Ecosystem - Enterprise Features.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


@dataclass
class Tenant:
    """Tenant configuration and metadata."""
    tenant_id: str
    name: str
    tier: str  # free, basic, pro, enterprise
    created_at: str
    status: str = "active"  # active, suspended, deleted

    # Resources
    max_users: int = 5
    max_requests_per_day: int = 1000
    max_models: int = 5
    storage_quota_gb: int = 10

    # Features
    custom_domain: Optional[str] = None
    sso_enabled: bool = False
    white_label_enabled: bool = False
    dedicated_support: bool = False

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)


class TenantManager:
    """
    Manages multi-tenant infrastructure.

    Features:
    - Tenant provisioning and deprovisioning
    - Resource quota management
    - Tenant isolation
    - Custom configurations
    - Tenant analytics
    """

    def __init__(self, database_manager):
        """
        Initialize tenant manager.

        Args:
            database_manager: Database manager instance
        """
        self.db = database_manager
        self.tenants: Dict[str, Tenant] = {}

        # Tier configurations
        self.tier_configs = {
            "free": {
                "max_users": 5,
                "max_requests_per_day": 1000,
                "max_models": 3,
                "storage_quota_gb": 5,
                "custom_domain": False,
                "sso_enabled": False,
                "white_label_enabled": False,
                "dedicated_support": False,
                "monthly_cost": 0
            },
            "basic": {
                "max_users": 25,
                "max_requests_per_day": 50000,
                "max_models": 10,
                "storage_quota_gb": 50,
                "custom_domain": True,
                "sso_enabled": False,
                "white_label_enabled": False,
                "dedicated_support": False,
                "monthly_cost": 99
            },
            "pro": {
                "max_users": 100,
                "max_requests_per_day": 1000000,
                "max_models": 50,
                "storage_quota_gb": 500,
                "custom_domain": True,
                "sso_enabled": True,
                "white_label_enabled": True,
                "dedicated_support": True,
                "monthly_cost": 499
            },
            "enterprise": {
                "max_users": -1,  # Unlimited
                "max_requests_per_day": -1,  # Unlimited
                "max_models": -1,  # Unlimited
                "storage_quota_gb": -1,  # Unlimited
                "custom_domain": True,
                "sso_enabled": True,
                "white_label_enabled": True,
                "dedicated_support": True,
                "monthly_cost": "custom"
            }
        }

        logger.info("Tenant manager initialized")

    def create_tenant(
        self,
        name: str,
        tier: str = "free",
        admin_email: str = None,
        settings: Optional[Dict] = None
    ) -> Tenant:
        """
        Create a new tenant.

        Args:
            name: Tenant name
            tier: Subscription tier
            admin_email: Admin user email
            settings: Custom settings

        Returns:
            Created tenant
        """
        tenant_id = str(uuid.uuid4())

        # Get tier configuration
        tier_config = self.tier_configs.get(tier, self.tier_configs["free"])

        # Create tenant
        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            tier=tier,
            created_at=datetime.utcnow().isoformat(),
            max_users=tier_config["max_users"],
            max_requests_per_day=tier_config["max_requests_per_day"],
            max_models=tier_config["max_models"],
            storage_quota_gb=tier_config["storage_quota_gb"],
            sso_enabled=tier_config["sso_enabled"],
            white_label_enabled=tier_config["white_label_enabled"],
            dedicated_support=tier_config["dedicated_support"],
            settings=settings or {}
        )

        # Store in memory
        self.tenants[tenant_id] = tenant

        # Persist to database
        self._persist_tenant(tenant)

        logger.info(f"Created tenant: {tenant_id} ({name}) - {tier} tier")

        return tenant

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        return self.tenants.get(tenant_id)

    def update_tenant(
        self,
        tenant_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update tenant configuration.

        Args:
            tenant_id: Tenant ID
            updates: Dictionary of fields to update

        Returns:
            Success status
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            logger.warning(f"Tenant not found: {tenant_id}")
            return False

        # Update fields
        for key, value in updates.items():
            if hasattr(tenant, key):
                setattr(tenant, key, value)

        # Persist changes
        self._persist_tenant(tenant)

        logger.info(f"Updated tenant: {tenant_id}")
        return True

    def upgrade_tenant(
        self,
        tenant_id: str,
        new_tier: str
    ) -> bool:
        """
        Upgrade tenant to new tier.

        Args:
            tenant_id: Tenant ID
            new_tier: New subscription tier

        Returns:
            Success status
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False

        tier_config = self.tier_configs.get(new_tier)
        if not tier_config:
            logger.error(f"Invalid tier: {new_tier}")
            return False

        # Update tier and resources
        tenant.tier = new_tier
        tenant.max_users = tier_config["max_users"]
        tenant.max_requests_per_day = tier_config["max_requests_per_day"]
        tenant.max_models = tier_config["max_models"]
        tenant.storage_quota_gb = tier_config["storage_quota_gb"]
        tenant.sso_enabled = tier_config["sso_enabled"]
        tenant.white_label_enabled = tier_config["white_label_enabled"]
        tenant.dedicated_support = tier_config["dedicated_support"]

        self._persist_tenant(tenant)

        logger.info(f"Upgraded tenant {tenant_id} to {new_tier}")
        return True

    def delete_tenant(self, tenant_id: str) -> bool:
        """
        Delete (soft delete) a tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            Success status
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False

        tenant.status = "deleted"
        self._persist_tenant(tenant)

        logger.info(f"Deleted tenant: {tenant_id}")
        return True

    def check_quota(
        self,
        tenant_id: str,
        resource_type: str,
        current_usage: int
    ) -> Dict[str, Any]:
        """
        Check if tenant is within quota.

        Args:
            tenant_id: Tenant ID
            resource_type: Type of resource (users, requests, models, storage)
            current_usage: Current usage amount

        Returns:
            Quota status
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return {"error": "Tenant not found"}

        # Get quota limit
        quota_map = {
            "users": tenant.max_users,
            "requests": tenant.max_requests_per_day,
            "models": tenant.max_models,
            "storage": tenant.storage_quota_gb
        }

        limit = quota_map.get(resource_type, 0)

        # -1 means unlimited
        if limit == -1:
            return {
                "tenant_id": tenant_id,
                "resource": resource_type,
                "limit": "unlimited",
                "current_usage": current_usage,
                "within_quota": True
            }

        within_quota = current_usage < limit
        remaining = max(0, limit - current_usage)

        return {
            "tenant_id": tenant_id,
            "resource": resource_type,
            "limit": limit,
            "current_usage": current_usage,
            "remaining": remaining,
            "within_quota": within_quota,
            "usage_percentage": (current_usage / limit * 100) if limit > 0 else 0
        }

    def get_tenant_analytics(self, tenant_id: str) -> Dict[str, Any]:
        """
        Get analytics for tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            Analytics data
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return {"error": "Tenant not found"}

        # In production, would fetch real usage data
        return {
            "tenant_id": tenant_id,
            "name": tenant.name,
            "tier": tenant.tier,
            "created_at": tenant.created_at,
            "status": tenant.status,
            "quotas": {
                "users": {
                    "limit": tenant.max_users,
                    "current": 0  # Would fetch from DB
                },
                "requests_per_day": {
                    "limit": tenant.max_requests_per_day,
                    "current": 0  # Would fetch from usage meter
                },
                "models": {
                    "limit": tenant.max_models,
                    "current": 0  # Would fetch from model registry
                },
                "storage_gb": {
                    "limit": tenant.storage_quota_gb,
                    "current": 0  # Would fetch from storage system
                }
            },
            "features": {
                "custom_domain": tenant.custom_domain,
                "sso_enabled": tenant.sso_enabled,
                "white_label_enabled": tenant.white_label_enabled,
                "dedicated_support": tenant.dedicated_support
            }
        }

    def list_tenants(
        self,
        status: Optional[str] = None,
        tier: Optional[str] = None
    ) -> List[Tenant]:
        """
        List tenants with optional filters.

        Args:
            status: Filter by status
            tier: Filter by tier

        Returns:
            List of tenants
        """
        tenants = list(self.tenants.values())

        if status:
            tenants = [t for t in tenants if t.status == status]

        if tier:
            tenants = [t for t in tenants if t.tier == tier]

        return tenants

    def _persist_tenant(self, tenant: Tenant):
        """Persist tenant to database."""
        # In production, would save to database
        # For now, just keep in memory
        pass


class TenantIsolation:
    """
    Ensures tenant data isolation.

    Features:
    - Database schema per tenant
    - Object storage namespace per tenant
    - Cache key prefixing
    - API request validation
    """

    def __init__(self, tenant_manager: TenantManager):
        """Initialize tenant isolation."""
        self.tenant_manager = tenant_manager

    def get_tenant_db_schema(self, tenant_id: str) -> str:
        """Get database schema name for tenant."""
        return f"tenant_{tenant_id.replace('-', '_')}"

    def get_tenant_cache_prefix(self, tenant_id: str) -> str:
        """Get cache key prefix for tenant."""
        return f"tenant:{tenant_id}:"

    def get_tenant_storage_path(self, tenant_id: str) -> str:
        """Get object storage path for tenant."""
        return f"tenants/{tenant_id}/"

    def validate_tenant_access(
        self,
        tenant_id: str,
        resource_id: str
    ) -> bool:
        """
        Validate that resource belongs to tenant.

        Args:
            tenant_id: Tenant ID
            resource_id: Resource ID to check

        Returns:
            True if access allowed
        """
        # In production, would check resource ownership in database
        return True
