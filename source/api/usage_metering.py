"""
API Usage Metering and Tracking.

Tracks API usage for billing, rate limiting, and analytics.
Part of Phase 4: Advanced Features - API Marketplace.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import redis
from dataclasses import dataclass, asdict
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class UsageRecord:
    """Usage record for API calls."""
    user_id: str
    api_key: str
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    timestamp: str
    request_size_bytes: int
    response_size_bytes: int
    model_id: Optional[str] = None
    cached: bool = False


class UsageMeter:
    """
    Track and meter API usage.

    Features:
    - Real-time usage tracking
    - Usage aggregation by time period
    - Quota management
    - Billing metrics
    - Usage analytics
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        key_prefix: str = "usage"
    ):
        """
        Initialize usage meter.

        Args:
            redis_client: Redis client for storage
            key_prefix: Prefix for Redis keys
        """
        self.redis = redis_client
        self.key_prefix = key_prefix

    def record_usage(self, record: UsageRecord):
        """
        Record API usage.

        Args:
            record: Usage record to store
        """
        try:
            # Store individual record
            record_key = f"{self.key_prefix}:records:{record.user_id}:{record.timestamp}"
            self.redis.setex(
                record_key,
                86400 * 7,  # Keep for 7 days
                json.dumps(asdict(record))
            )

            # Update counters
            self._update_counters(record)

            # Update quota tracking
            self._update_quota(record)

            logger.debug(f"Recorded usage for {record.user_id}: {record.endpoint}")

        except Exception as e:
            logger.error(f"Failed to record usage: {e}")

    def _update_counters(self, record: UsageRecord):
        """Update usage counters."""
        timestamp = datetime.fromisoformat(record.timestamp)

        # Hourly counters
        hour_key = timestamp.strftime("%Y-%m-%d-%H")
        self._increment_counter(f"hourly:{record.user_id}:{hour_key}")

        # Daily counters
        day_key = timestamp.strftime("%Y-%m-%d")
        self._increment_counter(f"daily:{record.user_id}:{day_key}")

        # Monthly counters
        month_key = timestamp.strftime("%Y-%m")
        self._increment_counter(f"monthly:{record.user_id}:{month_key}")

        # Endpoint-specific counters
        self._increment_counter(f"endpoint:{record.user_id}:{record.endpoint}")

        # Total lifetime counter
        self._increment_counter(f"total:{record.user_id}")

    def _increment_counter(self, key: str, amount: int = 1):
        """Increment a counter."""
        full_key = f"{self.key_prefix}:count:{key}"
        self.redis.incr(full_key, amount)
        # Set expiry for time-based keys
        if any(x in key for x in ['hourly', 'daily', 'monthly']):
            self.redis.expire(full_key, 86400 * 90)  # Keep for 90 days

    def _update_quota(self, record: UsageRecord):
        """Update quota tracking."""
        # Track requests in current period (minute)
        minute_key = datetime.fromisoformat(record.timestamp).strftime("%Y-%m-%d-%H-%M")
        quota_key = f"{self.key_prefix}:quota:{record.user_id}:minute:{minute_key}"

        self.redis.incr(quota_key)
        self.redis.expire(quota_key, 120)  # Keep for 2 minutes

    def get_usage(
        self,
        user_id: str,
        period: str = "day",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get usage statistics for user.

        Args:
            user_id: User ID
            period: Time period (hour/day/month)
            start_date: Start date for range query
            end_date: End date for range query

        Returns:
            Usage statistics
        """
        if period == "total":
            total_key = f"{self.key_prefix}:count:total:{user_id}"
            total = int(self.redis.get(total_key) or 0)
            return {
                "user_id": user_id,
                "total_requests": total
            }

        # Get usage for specific period
        if start_date is None:
            start_date = datetime.now() - timedelta(days=1 if period == "day" else 30)
        if end_date is None:
            end_date = datetime.now()

        usage_data = []
        current = start_date

        while current <= end_date:
            if period == "hour":
                key = current.strftime("%Y-%m-%d-%H")
                delta = timedelta(hours=1)
            elif period == "day":
                key = current.strftime("%Y-%m-%d")
                delta = timedelta(days=1)
            else:  # month
                key = current.strftime("%Y-%m")
                delta = timedelta(days=30)

            count_key = f"{self.key_prefix}:count:{period}ly:{user_id}:{key}"
            count = int(self.redis.get(count_key) or 0)

            usage_data.append({
                "period": key,
                "requests": count
            })

            current += delta

        return {
            "user_id": user_id,
            "period": period,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "usage": usage_data,
            "total": sum(u["requests"] for u in usage_data)
        }

    def check_quota(
        self,
        user_id: str,
        limit: int = 1000
    ) -> Dict[str, Any]:
        """
        Check if user is within quota.

        Args:
            user_id: User ID
            limit: Request limit per minute

        Returns:
            Quota status
        """
        current_minute = datetime.now().strftime("%Y-%m-%d-%H-%M")
        quota_key = f"{self.key_prefix}:quota:{user_id}:minute:{current_minute}"

        current_usage = int(self.redis.get(quota_key) or 0)
        remaining = max(0, limit - current_usage)

        return {
            "user_id": user_id,
            "limit": limit,
            "current_usage": current_usage,
            "remaining": remaining,
            "reset_at": (datetime.now() + timedelta(minutes=1)).isoformat(),
            "within_quota": current_usage < limit
        }

    def get_billing_metrics(
        self,
        user_id: str,
        month: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get billing metrics for user.

        Args:
            user_id: User ID
            month: Month in format YYYY-MM (default: current month)

        Returns:
            Billing metrics
        """
        if month is None:
            month = datetime.now().strftime("%Y-%m")

        # Get monthly usage
        monthly_key = f"{self.key_prefix}:count:monthly:{user_id}:{month}"
        total_requests = int(self.redis.get(monthly_key) or 0)

        # Calculate cost (example pricing: $0.001 per request)
        cost_per_request = 0.001
        total_cost = total_requests * cost_per_request

        # Get endpoint breakdown
        # This is simplified - in production, would aggregate from detailed records
        return {
            "user_id": user_id,
            "billing_period": month,
            "total_requests": total_requests,
            "cost_per_request": cost_per_request,
            "total_cost_usd": round(total_cost, 2),
            "breakdown": {
                "predictions": total_requests,
                "cached_requests": 0,  # Would track separately
                "explanation_requests": 0  # Would track separately
            }
        }

    def get_analytics(
        self,
        user_id: str,
        period: str = "day"
    ) -> Dict[str, Any]:
        """
        Get usage analytics.

        Args:
            user_id: User ID
            period: Aggregation period

        Returns:
            Analytics data
        """
        usage = self.get_usage(user_id, period)

        # Calculate stats
        requests = [u["requests"] for u in usage["usage"]]
        total = sum(requests)
        avg = total / len(requests) if requests else 0
        peak = max(requests) if requests else 0

        return {
            "user_id": user_id,
            "period": period,
            "total_requests": total,
            "average_per_period": round(avg, 2),
            "peak_requests": peak,
            "usage_trend": usage["usage"]
        }


class QuotaManager:
    """
    Manage user quotas and rate limits.

    Features:
    - Flexible quota tiers
    - Rate limiting
    - Quota alerts
    - Usage notifications
    """

    def __init__(self, redis_client: redis.Redis):
        """Initialize quota manager."""
        self.redis = redis_client
        self.usage_meter = UsageMeter(redis_client)

        # Default quota tiers
        self.quotas = {
            "free": {"requests_per_minute": 60, "requests_per_day": 1000},
            "basic": {"requests_per_minute": 600, "requests_per_day": 50000},
            "pro": {"requests_per_minute": 6000, "requests_per_day": 1000000},
            "enterprise": {"requests_per_minute": 60000, "requests_per_day": -1}  # Unlimited
        }

    def check_rate_limit(
        self,
        user_id: str,
        tier: str = "free"
    ) -> bool:
        """
        Check if user is within rate limit.

        Args:
            user_id: User ID
            tier: User tier (free/basic/pro/enterprise)

        Returns:
            True if within limit, False otherwise
        """
        quota_config = self.quotas.get(tier, self.quotas["free"])
        quota_status = self.usage_meter.check_quota(
            user_id,
            quota_config["requests_per_minute"]
        )

        return quota_status["within_quota"]

    def get_quota_status(
        self,
        user_id: str,
        tier: str = "free"
    ) -> Dict[str, Any]:
        """Get detailed quota status for user."""
        quota_config = self.quotas.get(tier, self.quotas["free"])

        # Check minute quota
        minute_status = self.usage_meter.check_quota(
            user_id,
            quota_config["requests_per_minute"]
        )

        # Check daily quota if applicable
        if quota_config["requests_per_day"] > 0:
            usage = self.usage_meter.get_usage(user_id, period="day")
            today_usage = usage["usage"][-1]["requests"] if usage["usage"] else 0
            daily_remaining = max(0, quota_config["requests_per_day"] - today_usage)
        else:
            today_usage = 0
            daily_remaining = -1  # Unlimited

        return {
            "user_id": user_id,
            "tier": tier,
            "rate_limit": {
                "per_minute": quota_config["requests_per_minute"],
                "current_minute_usage": minute_status["current_usage"],
                "minute_remaining": minute_status["remaining"]
            },
            "daily_limit": {
                "limit": quota_config["requests_per_day"],
                "today_usage": today_usage,
                "remaining": daily_remaining
            }
        }
