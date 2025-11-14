"""
SLA Monitoring and Guarantees System.

Tracks service level agreements, uptime, and performance metrics.
Part of Phase 5: Enterprise & Ecosystem - Enterprise Features.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class SLAMetric(Enum):
    """SLA metric types."""
    UPTIME = "uptime"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"
    LATENCY_P50 = "latency_p50"
    LATENCY_P95 = "latency_p95"
    LATENCY_P99 = "latency_p99"


class SLAStatus(Enum):
    """SLA compliance status."""
    COMPLIANT = "compliant"
    AT_RISK = "at_risk"
    BREACHED = "breached"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class SLATarget:
    """SLA target definition."""
    metric: str
    target_value: float
    comparison: str  # ">", "<", ">=", "<=", "=="
    measurement_period: str  # "hourly", "daily", "monthly"
    breach_threshold: float = 0.95  # 95% compliance required


@dataclass
class SLATier:
    """SLA tier configuration."""
    tier_name: str
    uptime_percentage: float
    max_response_time_ms: float
    max_error_rate: float
    support_response_time_hours: int
    targets: List[SLATarget] = field(default_factory=list)
    credits_on_breach: bool = True
    credit_percentage: float = 10.0  # % of monthly fee


@dataclass
class SLAMeasurement:
    """Single SLA measurement."""
    timestamp: str
    metric: str
    value: float
    target_value: float
    compliant: bool
    tenant_id: Optional[str] = None


@dataclass
class SLABreach:
    """SLA breach event."""
    breach_id: str
    tenant_id: str
    metric: str
    target_value: float
    actual_value: float
    breach_timestamp: str
    resolved: bool = False
    resolved_timestamp: Optional[str] = None
    credit_issued: bool = False
    credit_amount: float = 0.0


class SLAMonitor:
    """
    Monitors and enforces SLA guarantees.

    Features:
    - Real-time SLA tracking
    - Breach detection and alerting
    - Automated credit calculation
    - Performance reporting
    - Trend analysis
    """

    def __init__(self, database_manager=None):
        """
        Initialize SLA monitor.

        Args:
            database_manager: Optional database manager
        """
        self.db = database_manager
        self.measurements: List[SLAMeasurement] = []
        self.breaches: List[SLABreach] = []

        # SLA tiers
        self.tiers = self._initialize_tiers()

        # Tenant SLA assignments
        self.tenant_tiers: Dict[str, str] = {}

        # Metrics buffer for aggregation
        self.metrics_buffer: Dict[str, List[float]] = defaultdict(list)

        logger.info("SLA monitor initialized")

    def _initialize_tiers(self) -> Dict[str, SLATier]:
        """Initialize SLA tier definitions."""
        return {
            "free": SLATier(
                tier_name="free",
                uptime_percentage=99.0,
                max_response_time_ms=1000,
                max_error_rate=0.05,  # 5%
                support_response_time_hours=72,
                credits_on_breach=False,
                targets=[
                    SLATarget(
                        metric=SLAMetric.UPTIME.value,
                        target_value=99.0,
                        comparison=">=",
                        measurement_period="monthly"
                    )
                ]
            ),
            "basic": SLATier(
                tier_name="basic",
                uptime_percentage=99.5,
                max_response_time_ms=500,
                max_error_rate=0.02,  # 2%
                support_response_time_hours=24,
                credits_on_breach=True,
                credit_percentage=10.0,
                targets=[
                    SLATarget(
                        metric=SLAMetric.UPTIME.value,
                        target_value=99.5,
                        comparison=">=",
                        measurement_period="monthly"
                    ),
                    SLATarget(
                        metric=SLAMetric.RESPONSE_TIME.value,
                        target_value=500,
                        comparison="<=",
                        measurement_period="daily"
                    )
                ]
            ),
            "pro": SLATier(
                tier_name="pro",
                uptime_percentage=99.9,
                max_response_time_ms=200,
                max_error_rate=0.01,  # 1%
                support_response_time_hours=4,
                credits_on_breach=True,
                credit_percentage=25.0,
                targets=[
                    SLATarget(
                        metric=SLAMetric.UPTIME.value,
                        target_value=99.9,
                        comparison=">=",
                        measurement_period="monthly"
                    ),
                    SLATarget(
                        metric=SLAMetric.RESPONSE_TIME.value,
                        target_value=200,
                        comparison="<=",
                        measurement_period="daily"
                    ),
                    SLATarget(
                        metric=SLAMetric.LATENCY_P95.value,
                        target_value=300,
                        comparison="<=",
                        measurement_period="hourly"
                    )
                ]
            ),
            "enterprise": SLATier(
                tier_name="enterprise",
                uptime_percentage=99.99,
                max_response_time_ms=100,
                max_error_rate=0.005,  # 0.5%
                support_response_time_hours=1,
                credits_on_breach=True,
                credit_percentage=50.0,
                targets=[
                    SLATarget(
                        metric=SLAMetric.UPTIME.value,
                        target_value=99.99,
                        comparison=">=",
                        measurement_period="monthly"
                    ),
                    SLATarget(
                        metric=SLAMetric.RESPONSE_TIME.value,
                        target_value=100,
                        comparison="<=",
                        measurement_period="hourly"
                    ),
                    SLATarget(
                        metric=SLAMetric.LATENCY_P99.value,
                        target_value=150,
                        comparison="<=",
                        measurement_period="hourly"
                    ),
                    SLATarget(
                        metric=SLAMetric.ERROR_RATE.value,
                        target_value=0.005,
                        comparison="<=",
                        measurement_period="hourly"
                    )
                ]
            )
        }

    def assign_tenant_tier(
        self,
        tenant_id: str,
        tier_name: str
    ) -> bool:
        """
        Assign SLA tier to tenant.

        Args:
            tenant_id: Tenant ID
            tier_name: SLA tier name

        Returns:
            Success status
        """
        if tier_name not in self.tiers:
            logger.error(f"Invalid SLA tier: {tier_name}")
            return False

        self.tenant_tiers[tenant_id] = tier_name
        logger.info(f"Assigned tier '{tier_name}' to tenant {tenant_id}")
        return True

    def record_metric(
        self,
        metric: SLAMetric,
        value: float,
        tenant_id: Optional[str] = None
    ):
        """
        Record an SLA metric measurement.

        Args:
            metric: Metric type
            value: Measured value
            tenant_id: Optional tenant ID
        """
        self.metrics_buffer[metric.value].append(value)

        # Check if we should evaluate SLA
        if len(self.metrics_buffer[metric.value]) >= 100:
            self._evaluate_sla(metric, tenant_id)

    def _evaluate_sla(
        self,
        metric: SLAMetric,
        tenant_id: Optional[str] = None
    ):
        """
        Evaluate SLA compliance for a metric.

        Args:
            metric: Metric to evaluate
            tenant_id: Optional tenant ID
        """
        if not self.metrics_buffer[metric.value]:
            return

        # Calculate aggregate value
        values = self.metrics_buffer[metric.value]

        if metric == SLAMetric.UPTIME:
            aggregate_value = (sum(values) / len(values)) * 100
        elif metric in [SLAMetric.RESPONSE_TIME, SLAMetric.LATENCY_P50]:
            aggregate_value = sum(values) / len(values)
        elif metric == SLAMetric.LATENCY_P95:
            sorted_values = sorted(values)
            aggregate_value = sorted_values[int(len(sorted_values) * 0.95)]
        elif metric == SLAMetric.LATENCY_P99:
            sorted_values = sorted(values)
            aggregate_value = sorted_values[int(len(sorted_values) * 0.99)]
        elif metric == SLAMetric.ERROR_RATE:
            aggregate_value = sum(values) / len(values)
        else:
            aggregate_value = sum(values) / len(values)

        # Get target for tenant
        tier_name = self.tenant_tiers.get(tenant_id, "free")
        tier = self.tiers[tier_name]

        # Find matching target
        target = next(
            (t for t in tier.targets if t.metric == metric.value),
            None
        )

        if not target:
            return

        # Check compliance
        compliant = self._check_compliance(
            aggregate_value,
            target.target_value,
            target.comparison
        )

        # Record measurement
        measurement = SLAMeasurement(
            timestamp=datetime.utcnow().isoformat(),
            metric=metric.value,
            value=aggregate_value,
            target_value=target.target_value,
            compliant=compliant,
            tenant_id=tenant_id
        )
        self.measurements.append(measurement)

        # Check for breach
        if not compliant:
            self._record_breach(
                tenant_id,
                metric.value,
                target.target_value,
                aggregate_value
            )

        # Clear buffer
        self.metrics_buffer[metric.value] = []

    def _check_compliance(
        self,
        actual: float,
        target: float,
        comparison: str
    ) -> bool:
        """Check if actual value meets target."""
        if comparison == ">=":
            return actual >= target
        elif comparison == "<=":
            return actual <= target
        elif comparison == ">":
            return actual > target
        elif comparison == "<":
            return actual < target
        elif comparison == "==":
            return abs(actual - target) < 0.001
        return False

    def _record_breach(
        self,
        tenant_id: str,
        metric: str,
        target_value: float,
        actual_value: float
    ):
        """
        Record an SLA breach.

        Args:
            tenant_id: Tenant ID
            metric: Breached metric
            target_value: Target value
            actual_value: Actual value
        """
        import uuid

        breach = SLABreach(
            breach_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            metric=metric,
            target_value=target_value,
            actual_value=actual_value,
            breach_timestamp=datetime.utcnow().isoformat()
        )

        self.breaches.append(breach)

        # Calculate credit if applicable
        tier_name = self.tenant_tiers.get(tenant_id, "free")
        tier = self.tiers[tier_name]

        if tier.credits_on_breach:
            breach.credit_issued = True
            breach.credit_amount = tier.credit_percentage

        logger.warning(
            f"SLA BREACH: Tenant {tenant_id} - {metric} "
            f"(target: {target_value}, actual: {actual_value})"
        )

        # Trigger alert
        self._trigger_breach_alert(breach)

    def _trigger_breach_alert(self, breach: SLABreach):
        """
        Trigger alert for SLA breach.

        Args:
            breach: Breach event
        """
        # In production, would send:
        # - Email to customer
        # - Slack/PagerDuty to ops team
        # - Create support ticket
        logger.critical(
            f"SLA BREACH ALERT: {breach.breach_id} - "
            f"Tenant: {breach.tenant_id}, Metric: {breach.metric}"
        )

    def get_sla_status(
        self,
        tenant_id: str,
        period: str = "24h"
    ) -> Dict[str, Any]:
        """
        Get SLA status for a tenant.

        Args:
            tenant_id: Tenant ID
            period: Time period (24h, 7d, 30d)

        Returns:
            SLA status summary
        """
        # Parse period
        period_map = {
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30)
        }
        delta = period_map.get(period, timedelta(hours=24))
        start_time = datetime.utcnow() - delta

        # Get measurements for period
        measurements = [
            m for m in self.measurements
            if m.tenant_id == tenant_id and
            datetime.fromisoformat(m.timestamp) >= start_time
        ]

        # Get breaches for period
        breaches = [
            b for b in self.breaches
            if b.tenant_id == tenant_id and
            datetime.fromisoformat(b.breach_timestamp) >= start_time
        ]

        # Calculate compliance by metric
        compliance_by_metric = {}
        for metric in SLAMetric:
            metric_measurements = [
                m for m in measurements
                if m.metric == metric.value
            ]
            if metric_measurements:
                compliant_count = sum(1 for m in metric_measurements if m.compliant)
                compliance_by_metric[metric.value] = {
                    "compliance_rate": (compliant_count / len(metric_measurements)) * 100,
                    "measurements": len(metric_measurements),
                    "breaches": sum(1 for b in breaches if b.metric == metric.value)
                }

        # Overall status
        tier_name = self.tenant_tiers.get(tenant_id, "free")
        overall_compliant = len(breaches) == 0

        return {
            "tenant_id": tenant_id,
            "tier": tier_name,
            "period": period,
            "overall_status": SLAStatus.COMPLIANT.value if overall_compliant
            else SLAStatus.BREACHED.value,
            "compliance_by_metric": compliance_by_metric,
            "total_breaches": len(breaches),
            "unresolved_breaches": sum(1 for b in breaches if not b.resolved),
            "credits_owed": sum(b.credit_amount for b in breaches if b.credit_issued)
        }

    def get_breach_history(
        self,
        tenant_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[SLABreach]:
        """
        Get breach history.

        Args:
            tenant_id: Optional tenant filter
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            List of breaches
        """
        breaches = self.breaches

        if tenant_id:
            breaches = [b for b in breaches if b.tenant_id == tenant_id]

        if start_date:
            breaches = [
                b for b in breaches
                if datetime.fromisoformat(b.breach_timestamp) >= start_date
            ]

        if end_date:
            breaches = [
                b for b in breaches
                if datetime.fromisoformat(b.breach_timestamp) <= end_date
            ]

        return breaches

    def resolve_breach(
        self,
        breach_id: str
    ) -> bool:
        """
        Mark a breach as resolved.

        Args:
            breach_id: Breach ID

        Returns:
            Success status
        """
        breach = next((b for b in self.breaches if b.breach_id == breach_id), None)

        if not breach:
            logger.warning(f"Breach not found: {breach_id}")
            return False

        breach.resolved = True
        breach.resolved_timestamp = datetime.utcnow().isoformat()

        logger.info(f"Resolved breach: {breach_id}")
        return True

    def calculate_uptime(
        self,
        start_time: datetime,
        end_time: datetime,
        downtime_events: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate uptime percentage.

        Args:
            start_time: Period start
            end_time: Period end
            downtime_events: List of downtime events

        Returns:
            Uptime percentage
        """
        total_seconds = (end_time - start_time).total_seconds()

        downtime_seconds = 0
        for event in downtime_events:
            event_start = datetime.fromisoformat(event["start"])
            event_end = datetime.fromisoformat(event.get("end", datetime.utcnow().isoformat()))

            # Clip to period bounds
            event_start = max(event_start, start_time)
            event_end = min(event_end, end_time)

            if event_end > event_start:
                downtime_seconds += (event_end - event_start).total_seconds()

        uptime_seconds = total_seconds - downtime_seconds
        uptime_percentage = (uptime_seconds / total_seconds) * 100

        return uptime_percentage

    def generate_sla_report(
        self,
        tenant_id: str,
        month: int,
        year: int
    ) -> Dict[str, Any]:
        """
        Generate monthly SLA report.

        Args:
            tenant_id: Tenant ID
            month: Month (1-12)
            year: Year

        Returns:
            SLA report
        """
        # Calculate period
        from datetime import date
        start_date = datetime(year, month, 1)

        if month == 12:
            end_date = datetime(year + 1, 1, 1)
        else:
            end_date = datetime(year, month + 1, 1)

        # Get measurements and breaches
        measurements = [
            m for m in self.measurements
            if m.tenant_id == tenant_id and
            start_date <= datetime.fromisoformat(m.timestamp) < end_date
        ]

        breaches = [
            b for b in self.breaches
            if b.tenant_id == tenant_id and
            start_date <= datetime.fromisoformat(b.breach_timestamp) < end_date
        ]

        # Calculate metrics
        tier_name = self.tenant_tiers.get(tenant_id, "free")
        tier = self.tiers[tier_name]

        metric_summary = {}
        for target in tier.targets:
            metric_measurements = [
                m for m in measurements
                if m.metric == target.metric
            ]

            if metric_measurements:
                avg_value = sum(m.value for m in metric_measurements) / len(metric_measurements)
                compliant = sum(1 for m in metric_measurements if m.compliant)
                compliance_rate = (compliant / len(metric_measurements)) * 100

                metric_summary[target.metric] = {
                    "target": target.target_value,
                    "actual_avg": avg_value,
                    "compliance_rate": compliance_rate,
                    "breaches": sum(1 for b in breaches if b.metric == target.metric)
                }

        # Calculate credits
        total_credits = sum(b.credit_amount for b in breaches if b.credit_issued)

        return {
            "tenant_id": tenant_id,
            "tier": tier_name,
            "period": f"{year}-{month:02d}",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "sla_targets": {
                "uptime": tier.uptime_percentage,
                "response_time": tier.max_response_time_ms,
                "error_rate": tier.max_error_rate
            },
            "performance": metric_summary,
            "breaches": {
                "total": len(breaches),
                "by_metric": {
                    metric: sum(1 for b in breaches if b.metric == metric)
                    for metric in set(b.metric for b in breaches)
                }
            },
            "credits": {
                "total_percentage": total_credits,
                "applicable": tier.credits_on_breach
            },
            "overall_compliance": len(breaches) == 0
        }
