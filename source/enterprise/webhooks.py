"""
Webhook System for Event Notifications.

Enables real-time event notifications to external systems via webhooks.
Part of Phase 5: Enterprise & Ecosystem - Integrations.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
import hmac
import logging
import uuid
from collections import defaultdict
import asyncio
import aiohttp

logger = logging.getLogger(__name__)


class WebhookEvent(Enum):
    """Types of webhook events."""
    # Prediction events
    PREDICTION_CREATED = "prediction.created"
    PREDICTION_COMPLETED = "prediction.completed"
    PREDICTION_FAILED = "prediction.failed"

    # Model events
    MODEL_DEPLOYED = "model.deployed"
    MODEL_UPDATED = "model.updated"
    MODEL_RETIRED = "model.retired"

    # Data events
    DATA_INGESTED = "data.ingested"
    DATA_PROCESSED = "data.processed"
    DATA_EXPORTED = "data.exported"

    # User events
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"

    # SLA events
    SLA_BREACHED = "sla.breached"
    SLA_RECOVERED = "sla.recovered"

    # Alert events
    ALERT_TRIGGERED = "alert.triggered"
    ALERT_RESOLVED = "alert.resolved"

    # Quota events
    QUOTA_WARNING = "quota.warning"
    QUOTA_EXCEEDED = "quota.exceeded"

    # System events
    SYSTEM_MAINTENANCE = "system.maintenance"
    SYSTEM_INCIDENT = "system.incident"


class WebhookStatus(Enum):
    """Webhook delivery status."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class WebhookEndpoint:
    """Webhook endpoint configuration."""
    endpoint_id: str
    tenant_id: str
    url: str
    secret: str
    events: List[str]
    active: bool = True
    created_at: str = ""

    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: int = 60

    # Metadata
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WebhookDelivery:
    """Webhook delivery attempt."""
    delivery_id: str
    endpoint_id: str
    event_type: str
    payload: Dict[str, Any]
    status: str
    created_at: str

    # Delivery details
    attempt_count: int = 0
    last_attempt_at: Optional[str] = None
    next_retry_at: Optional[str] = None

    # Response details
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None


class WebhookManager:
    """
    Manages webhook endpoints and delivery.

    Features:
    - Event subscription management
    - Secure webhook delivery with HMAC signatures
    - Automatic retries with exponential backoff
    - Delivery tracking and analytics
    - Rate limiting per endpoint
    """

    def __init__(self, database_manager=None):
        """
        Initialize webhook manager.

        Args:
            database_manager: Optional database manager
        """
        self.db = database_manager
        self.endpoints: Dict[str, WebhookEndpoint] = {}
        self.deliveries: List[WebhookDelivery] = []

        # Event subscribers
        self.event_subscribers: Dict[str, List[str]] = defaultdict(list)

        # Delivery queue
        self.delivery_queue: List[WebhookDelivery] = []

        # Rate limiting
        self.rate_limits: Dict[str, List[datetime]] = defaultdict(list)
        self.max_requests_per_minute = 60

        logger.info("Webhook manager initialized")

    def create_endpoint(
        self,
        tenant_id: str,
        url: str,
        events: List[str],
        description: str = "",
        metadata: Optional[Dict] = None
    ) -> WebhookEndpoint:
        """
        Create a webhook endpoint.

        Args:
            tenant_id: Tenant ID
            url: Webhook URL
            events: List of event types to subscribe to
            description: Endpoint description
            metadata: Optional metadata

        Returns:
            Created webhook endpoint
        """
        endpoint_id = str(uuid.uuid4())
        secret = self._generate_secret()

        endpoint = WebhookEndpoint(
            endpoint_id=endpoint_id,
            tenant_id=tenant_id,
            url=url,
            secret=secret,
            events=events,
            created_at=datetime.utcnow().isoformat(),
            description=description,
            metadata=metadata or {}
        )

        self.endpoints[endpoint_id] = endpoint

        # Update subscribers
        for event in events:
            self.event_subscribers[event].append(endpoint_id)

        logger.info(
            f"Created webhook endpoint: {endpoint_id} for tenant {tenant_id}"
        )

        return endpoint

    def update_endpoint(
        self,
        endpoint_id: str,
        url: Optional[str] = None,
        events: Optional[List[str]] = None,
        active: Optional[bool] = None
    ) -> bool:
        """
        Update a webhook endpoint.

        Args:
            endpoint_id: Endpoint ID
            url: New URL
            events: New event list
            active: Active status

        Returns:
            Success status
        """
        endpoint = self.endpoints.get(endpoint_id)
        if not endpoint:
            logger.warning(f"Endpoint not found: {endpoint_id}")
            return False

        # Remove from old subscriptions
        if events and events != endpoint.events:
            for event in endpoint.events:
                if endpoint_id in self.event_subscribers[event]:
                    self.event_subscribers[event].remove(endpoint_id)

            # Add to new subscriptions
            for event in events:
                self.event_subscribers[event].append(endpoint_id)
            endpoint.events = events

        if url:
            endpoint.url = url

        if active is not None:
            endpoint.active = active

        logger.info(f"Updated webhook endpoint: {endpoint_id}")
        return True

    def delete_endpoint(self, endpoint_id: str) -> bool:
        """
        Delete a webhook endpoint.

        Args:
            endpoint_id: Endpoint ID

        Returns:
            Success status
        """
        endpoint = self.endpoints.get(endpoint_id)
        if not endpoint:
            return False

        # Remove from subscribers
        for event in endpoint.events:
            if endpoint_id in self.event_subscribers[event]:
                self.event_subscribers[event].remove(endpoint_id)

        del self.endpoints[endpoint_id]

        logger.info(f"Deleted webhook endpoint: {endpoint_id}")
        return True

    async def send_event(
        self,
        event_type: WebhookEvent,
        payload: Dict[str, Any],
        tenant_id: Optional[str] = None
    ):
        """
        Send an event to subscribed webhooks.

        Args:
            event_type: Event type
            payload: Event payload
            tenant_id: Optional tenant ID filter
        """
        # Get subscribed endpoints
        endpoint_ids = self.event_subscribers.get(event_type.value, [])

        if not endpoint_ids:
            logger.debug(f"No subscribers for event: {event_type.value}")
            return

        # Filter by tenant if specified
        endpoints = [
            self.endpoints[eid] for eid in endpoint_ids
            if eid in self.endpoints and
            self.endpoints[eid].active and
            (tenant_id is None or self.endpoints[eid].tenant_id == tenant_id)
        ]

        # Create delivery for each endpoint
        for endpoint in endpoints:
            await self._deliver_webhook(endpoint, event_type.value, payload)

    async def _deliver_webhook(
        self,
        endpoint: WebhookEndpoint,
        event_type: str,
        payload: Dict[str, Any]
    ):
        """
        Deliver webhook to an endpoint.

        Args:
            endpoint: Webhook endpoint
            event_type: Event type
            payload: Event payload
        """
        # Check rate limit
        if not self._check_rate_limit(endpoint.endpoint_id):
            logger.warning(
                f"Rate limit exceeded for endpoint: {endpoint.endpoint_id}"
            )
            return

        # Create delivery record
        delivery_id = str(uuid.uuid4())
        delivery = WebhookDelivery(
            delivery_id=delivery_id,
            endpoint_id=endpoint.endpoint_id,
            event_type=event_type,
            payload=payload,
            status=WebhookStatus.PENDING.value,
            created_at=datetime.utcnow().isoformat()
        )

        self.deliveries.append(delivery)

        # Attempt delivery
        await self._attempt_delivery(delivery, endpoint)

    async def _attempt_delivery(
        self,
        delivery: WebhookDelivery,
        endpoint: WebhookEndpoint
    ):
        """
        Attempt to deliver a webhook.

        Args:
            delivery: Delivery record
            endpoint: Webhook endpoint
        """
        delivery.attempt_count += 1
        delivery.last_attempt_at = datetime.utcnow().isoformat()

        # Prepare payload
        webhook_payload = {
            "id": delivery.delivery_id,
            "event": delivery.event_type,
            "timestamp": delivery.created_at,
            "data": delivery.payload
        }

        # Generate signature
        signature = self._generate_signature(
            json.dumps(webhook_payload),
            endpoint.secret
        )

        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Signature": signature,
            "X-Webhook-Event": delivery.event_type,
            "X-Webhook-Delivery": delivery.delivery_id
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint.url,
                    json=webhook_payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    delivery.response_code = response.status
                    delivery.response_body = await response.text()

                    if 200 <= response.status < 300:
                        delivery.status = WebhookStatus.DELIVERED.value
                        logger.info(
                            f"Webhook delivered: {delivery.delivery_id} "
                            f"to {endpoint.url}"
                        )
                    else:
                        raise Exception(f"HTTP {response.status}")

        except Exception as e:
            delivery.error_message = str(e)
            logger.error(
                f"Webhook delivery failed: {delivery.delivery_id} - {str(e)}"
            )

            # Schedule retry if within limits
            if delivery.attempt_count < endpoint.max_retries:
                delivery.status = WebhookStatus.RETRYING.value
                retry_delay = endpoint.retry_delay_seconds * (2 ** (delivery.attempt_count - 1))
                delivery.next_retry_at = (
                    datetime.utcnow() + timedelta(seconds=retry_delay)
                ).isoformat()

                # Add to retry queue
                self.delivery_queue.append(delivery)

                logger.info(
                    f"Webhook scheduled for retry: {delivery.delivery_id} "
                    f"(attempt {delivery.attempt_count}/{endpoint.max_retries})"
                )
            else:
                delivery.status = WebhookStatus.FAILED.value
                logger.error(
                    f"Webhook delivery permanently failed: {delivery.delivery_id}"
                )

    async def process_retry_queue(self):
        """Process pending webhook retries."""
        now = datetime.utcnow()

        retries = []
        remaining = []

        for delivery in self.delivery_queue:
            if delivery.next_retry_at and \
               datetime.fromisoformat(delivery.next_retry_at) <= now:
                retries.append(delivery)
            else:
                remaining.append(delivery)

        self.delivery_queue = remaining

        # Process retries
        for delivery in retries:
            endpoint = self.endpoints.get(delivery.endpoint_id)
            if endpoint:
                await self._attempt_delivery(delivery, endpoint)

    def _check_rate_limit(self, endpoint_id: str) -> bool:
        """
        Check if endpoint is within rate limit.

        Args:
            endpoint_id: Endpoint ID

        Returns:
            True if within limit
        """
        now = datetime.utcnow()
        one_minute_ago = now - timedelta(minutes=1)

        # Remove old timestamps
        self.rate_limits[endpoint_id] = [
            ts for ts in self.rate_limits[endpoint_id]
            if ts > one_minute_ago
        ]

        # Check limit
        if len(self.rate_limits[endpoint_id]) >= self.max_requests_per_minute:
            return False

        # Add current request
        self.rate_limits[endpoint_id].append(now)
        return True

    def _generate_secret(self) -> str:
        """Generate a webhook secret."""
        return hashlib.sha256(
            str(uuid.uuid4()).encode()
        ).hexdigest()

    def _generate_signature(self, payload: str, secret: str) -> str:
        """
        Generate HMAC signature for webhook payload.

        Args:
            payload: JSON payload
            secret: Webhook secret

        Returns:
            HMAC signature
        """
        return hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

    def verify_signature(
        self,
        payload: str,
        signature: str,
        secret: str
    ) -> bool:
        """
        Verify webhook signature.

        Args:
            payload: JSON payload
            signature: Provided signature
            secret: Webhook secret

        Returns:
            True if valid
        """
        expected_signature = self._generate_signature(payload, secret)
        return hmac.compare_digest(signature, expected_signature)

    def get_endpoint_analytics(
        self,
        endpoint_id: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get analytics for an endpoint.

        Args:
            endpoint_id: Endpoint ID
            days: Number of days to analyze

        Returns:
            Analytics data
        """
        start_date = datetime.utcnow() - timedelta(days=days)

        deliveries = [
            d for d in self.deliveries
            if d.endpoint_id == endpoint_id and
            datetime.fromisoformat(d.created_at) >= start_date
        ]

        if not deliveries:
            return {
                "endpoint_id": endpoint_id,
                "period_days": days,
                "total_deliveries": 0
            }

        # Calculate statistics
        delivered = sum(1 for d in deliveries if d.status == WebhookStatus.DELIVERED.value)
        failed = sum(1 for d in deliveries if d.status == WebhookStatus.FAILED.value)
        pending = sum(1 for d in deliveries if d.status in [
            WebhookStatus.PENDING.value,
            WebhookStatus.RETRYING.value
        ])

        # Event distribution
        event_counts = defaultdict(int)
        for d in deliveries:
            event_counts[d.event_type] += 1

        # Response times (if available)
        response_times = []
        for d in deliveries:
            if d.last_attempt_at and d.created_at:
                created = datetime.fromisoformat(d.created_at)
                attempted = datetime.fromisoformat(d.last_attempt_at)
                response_times.append(
                    (attempted - created).total_seconds()
                )

        avg_response_time = (
            sum(response_times) / len(response_times)
            if response_times else 0
        )

        return {
            "endpoint_id": endpoint_id,
            "period_days": days,
            "total_deliveries": len(deliveries),
            "delivered": delivered,
            "failed": failed,
            "pending": pending,
            "success_rate": (delivered / len(deliveries) * 100) if deliveries else 0,
            "event_distribution": dict(event_counts),
            "avg_response_time_seconds": avg_response_time,
            "avg_attempts": sum(d.attempt_count for d in deliveries) / len(deliveries)
        }

    def list_endpoints(
        self,
        tenant_id: Optional[str] = None,
        active_only: bool = False
    ) -> List[WebhookEndpoint]:
        """
        List webhook endpoints.

        Args:
            tenant_id: Optional tenant filter
            active_only: Only return active endpoints

        Returns:
            List of endpoints
        """
        endpoints = list(self.endpoints.values())

        if tenant_id:
            endpoints = [e for e in endpoints if e.tenant_id == tenant_id]

        if active_only:
            endpoints = [e for e in endpoints if e.active]

        return endpoints

    def get_delivery_history(
        self,
        endpoint_id: Optional[str] = None,
        event_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[WebhookDelivery]:
        """
        Get webhook delivery history.

        Args:
            endpoint_id: Optional endpoint filter
            event_type: Optional event type filter
            status: Optional status filter
            limit: Maximum results

        Returns:
            List of deliveries
        """
        deliveries = self.deliveries

        if endpoint_id:
            deliveries = [d for d in deliveries if d.endpoint_id == endpoint_id]

        if event_type:
            deliveries = [d for d in deliveries if d.event_type == event_type]

        if status:
            deliveries = [d for d in deliveries if d.status == status]

        # Sort by created_at descending
        deliveries.sort(
            key=lambda d: d.created_at,
            reverse=True
        )

        return deliveries[:limit]


# Webhook event helpers
async def emit_prediction_event(
    webhook_manager: WebhookManager,
    prediction_id: str,
    status: str,
    result: Optional[Dict] = None
):
    """Emit a prediction event."""
    if status == "completed":
        event = WebhookEvent.PREDICTION_COMPLETED
    elif status == "failed":
        event = WebhookEvent.PREDICTION_FAILED
    else:
        event = WebhookEvent.PREDICTION_CREATED

    payload = {
        "prediction_id": prediction_id,
        "status": status,
        "result": result
    }

    await webhook_manager.send_event(event, payload)


async def emit_sla_breach_event(
    webhook_manager: WebhookManager,
    tenant_id: str,
    metric: str,
    target: float,
    actual: float
):
    """Emit an SLA breach event."""
    payload = {
        "tenant_id": tenant_id,
        "metric": metric,
        "target_value": target,
        "actual_value": actual,
        "timestamp": datetime.utcnow().isoformat()
    }

    await webhook_manager.send_event(
        WebhookEvent.SLA_BREACHED,
        payload,
        tenant_id=tenant_id
    )
