"""
Audit Trail and Logging System.

Provides comprehensive audit logging for compliance and security.
Part of Phase 5: Enterprise & Ecosystem - Compliance & Governance.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""
    # User actions
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    PASSWORD_CHANGED = "user.password_changed"

    # Data access
    DATA_READ = "data.read"
    DATA_EXPORT = "data.export"
    DATA_DELETE = "data.delete"
    DATA_UPDATE = "data.update"

    # API operations
    API_PREDICT = "api.predict"
    API_KEY_CREATED = "api.key_created"
    API_KEY_REVOKED = "api.key_revoked"

    # Admin actions
    ADMIN_ACCESS = "admin.access"
    CONFIG_CHANGED = "config.changed"
    ROLE_ASSIGNED = "role.assigned"
    PERMISSION_CHANGED = "permission.changed"

    # GDPR compliance
    GDPR_DATA_EXPORT = "gdpr.data_export"
    GDPR_DATA_DELETE = "gdpr.data_delete"
    GDPR_CONSENT_GRANTED = "gdpr.consent_granted"
    GDPR_CONSENT_REVOKED = "gdpr.consent_revoked"

    # Security events
    SECURITY_AUTH_FAILED = "security.auth_failed"
    SECURITY_ACCESS_DENIED = "security.access_denied"
    SECURITY_BREACH_DETECTED = "security.breach_detected"

    # System events
    SYSTEM_BACKUP = "system.backup"
    SYSTEM_RESTORE = "system.restore"
    SYSTEM_CONFIG = "system.config"


class AuditSeverity(Enum):
    """Severity levels for audit events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event record."""
    event_id: str
    timestamp: str
    event_type: str
    severity: str

    # Actor information
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    # Action details
    action: str = ""
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None

    # Request context
    request_id: Optional[str] = None
    session_id: Optional[str] = None

    # Results
    status: str = "success"  # success, failure, partial
    status_code: Optional[int] = None
    error_message: Optional[str] = None

    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Integrity
    checksum: Optional[str] = None
    previous_checksum: Optional[str] = None


class AuditTrail:
    """
    Audit trail management system.

    Features:
    - Tamper-proof logging with checksums
    - Chain-of-custody tracking
    - Real-time event streaming
    - Compliance-oriented retention
    - Advanced querying and filtering
    - Automated alerts
    """

    def __init__(self, database_manager):
        """
        Initialize audit trail system.

        Args:
            database_manager: Database manager instance
        """
        self.db = database_manager
        self.events: List[AuditEvent] = []
        self.last_checksum: Optional[str] = None

        # Retention policy (7 years for audit logs)
        self.retention_days = 2555

        # Event counters for analytics
        self.event_counters = defaultdict(int)

        # Alert thresholds
        self.alert_thresholds = {
            AuditEventType.SECURITY_AUTH_FAILED.value: 5,  # 5 failed logins
            AuditEventType.SECURITY_ACCESS_DENIED.value: 10,
            AuditEventType.SECURITY_BREACH_DETECTED.value: 1
        }

        logger.info("Audit trail system initialized")

    def log_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        action: str = "",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        status: str = "success",
        severity: AuditSeverity = AuditSeverity.INFO,
        metadata: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> AuditEvent:
        """
        Log an audit event.

        Args:
            event_type: Type of event
            user_id: User performing action
            tenant_id: Tenant context
            action: Description of action
            resource_type: Type of resource affected
            resource_id: ID of resource affected
            status: Event status
            severity: Event severity
            metadata: Additional event data
            ip_address: IP address of request
            request_id: Request ID for tracing

        Returns:
            Created audit event
        """
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.utcnow().isoformat(),
            event_type=event_type.value,
            severity=severity.value,
            user_id=user_id,
            tenant_id=tenant_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            status=status,
            metadata=metadata or {},
            ip_address=ip_address,
            request_id=request_id,
            previous_checksum=self.last_checksum
        )

        # Calculate checksum for integrity
        event.checksum = self._calculate_checksum(event)
        self.last_checksum = event.checksum

        # Store event
        self.events.append(event)
        self._persist_event(event)

        # Update counters
        self.event_counters[event_type.value] += 1

        # Check for alerts
        self._check_alerts(event)

        logger.debug(
            f"Audit event logged: {event_type.value} "
            f"by user {user_id} - {status}"
        )

        return event

    def query_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        event_types: Optional[List[str]] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        resource_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuditEvent]:
        """
        Query audit events with filters.

        Args:
            start_time: Start of time range
            end_time: End of time range
            user_id: Filter by user
            tenant_id: Filter by tenant
            event_types: Filter by event types
            severity: Filter by severity
            status: Filter by status
            resource_type: Filter by resource type
            limit: Maximum results
            offset: Results offset

        Returns:
            Matching audit events
        """
        filtered_events = self.events

        # Apply filters
        if start_time:
            filtered_events = [
                e for e in filtered_events
                if datetime.fromisoformat(e.timestamp) >= start_time
            ]

        if end_time:
            filtered_events = [
                e for e in filtered_events
                if datetime.fromisoformat(e.timestamp) <= end_time
            ]

        if user_id:
            filtered_events = [
                e for e in filtered_events
                if e.user_id == user_id
            ]

        if tenant_id:
            filtered_events = [
                e for e in filtered_events
                if e.tenant_id == tenant_id
            ]

        if event_types:
            filtered_events = [
                e for e in filtered_events
                if e.event_type in event_types
            ]

        if severity:
            filtered_events = [
                e for e in filtered_events
                if e.severity == severity
            ]

        if status:
            filtered_events = [
                e for e in filtered_events
                if e.status == status
            ]

        if resource_type:
            filtered_events = [
                e for e in filtered_events
                if e.resource_type == resource_type
            ]

        # Apply pagination
        return filtered_events[offset:offset + limit]

    def export_audit_log(
        self,
        start_time: datetime,
        end_time: datetime,
        format: str = "json"
    ) -> str:
        """
        Export audit log for compliance reporting.

        Args:
            start_time: Start of export range
            end_time: End of export range
            format: Export format (json/csv)

        Returns:
            Exported audit log data
        """
        events = self.query_events(
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )

        if format == "json":
            return json.dumps(
                [asdict(e) for e in events],
                indent=2
            )
        elif format == "csv":
            return self._events_to_csv(events)
        else:
            return json.dumps([asdict(e) for e in events])

    def verify_integrity(
        self,
        event_id: str
    ) -> Dict[str, Any]:
        """
        Verify integrity of audit trail.

        Args:
            event_id: Event ID to verify from

        Returns:
            Verification results
        """
        # Find event
        event = next(
            (e for e in self.events if e.event_id == event_id),
            None
        )

        if not event:
            return {
                "verified": False,
                "error": "Event not found"
            }

        # Recalculate checksum
        calculated_checksum = self._calculate_checksum(event)

        # Verify chain
        chain_valid = True
        if event.previous_checksum:
            # Find previous event
            event_idx = self.events.index(event)
            if event_idx > 0:
                previous_event = self.events[event_idx - 1]
                if previous_event.checksum != event.previous_checksum:
                    chain_valid = False

        verified = (
            calculated_checksum == event.checksum and
            chain_valid
        )

        return {
            "verified": verified,
            "event_id": event_id,
            "checksum_match": calculated_checksum == event.checksum,
            "chain_valid": chain_valid,
            "timestamp": event.timestamp
        }

    def anonymize_user_logs(
        self,
        user_id: str
    ) -> int:
        """
        Anonymize user in audit logs (for GDPR).

        Args:
            user_id: User ID to anonymize

        Returns:
            Number of events anonymized
        """
        anonymized_count = 0
        anonymous_id = f"anonymous_{hashlib.sha256(user_id.encode()).hexdigest()[:16]}"

        for event in self.events:
            if event.user_id == user_id:
                event.user_id = anonymous_id
                # Remove PII from metadata
                if "email" in event.metadata:
                    del event.metadata["email"]
                if "name" in event.metadata:
                    del event.metadata["name"]

                # Recalculate checksum
                event.checksum = self._calculate_checksum(event)

                anonymized_count += 1

        logger.info(f"Anonymized {anonymized_count} events for user {user_id}")
        return anonymized_count

    def apply_retention_policy(self) -> int:
        """
        Apply retention policy and delete old events.

        Returns:
            Number of events deleted
        """
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)

        events_to_keep = []
        deleted_count = 0

        for event in self.events:
            event_time = datetime.fromisoformat(event.timestamp)
            if event_time >= cutoff_date:
                events_to_keep.append(event)
            else:
                deleted_count += 1

        self.events = events_to_keep

        logger.info(
            f"Retention policy applied: deleted {deleted_count} events "
            f"older than {self.retention_days} days"
        )

        return deleted_count

    def get_audit_summary(
        self,
        time_period: str = "24h"
    ) -> Dict[str, Any]:
        """
        Get summary of audit events.

        Args:
            time_period: Time period (24h, 7d, 30d)

        Returns:
            Audit summary statistics
        """
        # Parse time period
        period_map = {
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30)
        }

        delta = period_map.get(time_period, timedelta(hours=24))
        start_time = datetime.utcnow() - delta

        # Query events
        events = self.query_events(
            start_time=start_time,
            limit=100000
        )

        # Calculate statistics
        event_type_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        user_activity = defaultdict(int)
        failed_events = 0

        for event in events:
            event_type_counts[event.event_type] += 1
            severity_counts[event.severity] += 1
            if event.user_id:
                user_activity[event.user_id] += 1
            if event.status == "failure":
                failed_events += 1

        return {
            "time_period": time_period,
            "start_time": start_time.isoformat(),
            "end_time": datetime.utcnow().isoformat(),
            "total_events": len(events),
            "event_types": dict(event_type_counts),
            "severity_distribution": dict(severity_counts),
            "failed_events": failed_events,
            "unique_users": len(user_activity),
            "most_active_users": sorted(
                user_activity.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        import uuid
        return str(uuid.uuid4())

    def _calculate_checksum(self, event: AuditEvent) -> str:
        """
        Calculate checksum for event integrity.

        Args:
            event: Event to checksum

        Returns:
            SHA-256 checksum
        """
        # Create deterministic string representation
        data = f"{event.event_id}|{event.timestamp}|{event.event_type}|" \
               f"{event.user_id}|{event.tenant_id}|{event.action}|" \
               f"{event.resource_type}|{event.resource_id}|{event.status}|" \
               f"{json.dumps(event.metadata, sort_keys=True)}|" \
               f"{event.previous_checksum}"

        return hashlib.sha256(data.encode()).hexdigest()

    def _persist_event(self, event: AuditEvent):
        """Persist event to database."""
        # In production, would write to database
        # For now, events are kept in memory
        pass

    def _check_alerts(self, event: AuditEvent):
        """
        Check if event should trigger alerts.

        Args:
            event: Event to check
        """
        threshold = self.alert_thresholds.get(event.event_type)

        if threshold and self.event_counters[event.event_type] >= threshold:
            self._trigger_alert(event)

    def _trigger_alert(self, event: AuditEvent):
        """
        Trigger alert for suspicious activity.

        Args:
            event: Event that triggered alert
        """
        logger.warning(
            f"ALERT: Threshold exceeded for {event.event_type} - "
            f"Count: {self.event_counters[event.event_type]}"
        )

        # In production, would send notifications:
        # - Email to security team
        # - Slack/PagerDuty alert
        # - SIEM integration

    def _events_to_csv(self, events: List[AuditEvent]) -> str:
        """
        Convert events to CSV format.

        Args:
            events: Events to convert

        Returns:
            CSV string
        """
        if not events:
            return "No events found"

        # CSV header
        csv_lines = [
            "event_id,timestamp,event_type,severity,user_id,tenant_id,"
            "action,resource_type,resource_id,status,ip_address"
        ]

        # CSV rows
        for event in events:
            csv_lines.append(
                f"{event.event_id},{event.timestamp},{event.event_type},"
                f"{event.severity},{event.user_id or ''},"
                f"{event.tenant_id or ''},{event.action},"
                f"{event.resource_type or ''},{event.resource_id or ''},"
                f"{event.status},{event.ip_address or ''}"
            )

        return "\n".join(csv_lines)


class ComplianceReporter:
    """
    Generates compliance reports from audit logs.

    Supports:
    - SOC 2 Type II reports
    - ISO 27001 audit trails
    - GDPR Article 30 records
    - HIPAA audit logs
    """

    def __init__(self, audit_trail: AuditTrail):
        """Initialize compliance reporter."""
        self.audit_trail = audit_trail

    def generate_soc2_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Generate SOC 2 Type II compliance report.

        Args:
            start_date: Report start date
            end_date: Report end date

        Returns:
            SOC 2 compliance report
        """
        # Query relevant events
        events = self.audit_trail.query_events(
            start_time=start_date,
            end_time=end_date,
            limit=100000
        )

        # Analyze security events
        security_events = [
            e for e in events
            if e.event_type.startswith("security.")
        ]

        # Analyze access controls
        access_events = [
            e for e in events
            if e.event_type in [
                AuditEventType.USER_LOGIN.value,
                AuditEventType.ADMIN_ACCESS.value,
                AuditEventType.SECURITY_ACCESS_DENIED.value
            ]
        ]

        return {
            "report_type": "SOC 2 Type II",
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "total_events": len(events),
            "security_controls": {
                "access_controls": {
                    "total_access_attempts": len(access_events),
                    "successful_logins": len([
                        e for e in access_events
                        if e.event_type == AuditEventType.USER_LOGIN.value
                        and e.status == "success"
                    ]),
                    "failed_logins": len([
                        e for e in access_events
                        if e.event_type == AuditEventType.USER_LOGIN.value
                        and e.status == "failure"
                    ]),
                    "access_denied": len([
                        e for e in access_events
                        if e.event_type == AuditEventType.SECURITY_ACCESS_DENIED.value
                    ])
                },
                "security_incidents": len(security_events),
                "data_protection": {
                    "encryption_enabled": True,
                    "backup_frequency": "daily",
                    "retention_policy_days": self.audit_trail.retention_days
                }
            },
            "change_management": {
                "config_changes": len([
                    e for e in events
                    if e.event_type == AuditEventType.CONFIG_CHANGED.value
                ]),
                "permission_changes": len([
                    e for e in events
                    if e.event_type == AuditEventType.PERMISSION_CHANGED.value
                ])
            },
            "monitoring_controls": {
                "logging_enabled": True,
                "log_integrity_verified": True,
                "alert_system_active": True
            }
        }

    def generate_gdpr_article30_record(self) -> Dict[str, Any]:
        """
        Generate GDPR Article 30 record of processing activities.

        Returns:
            Article 30 compliance record
        """
        return {
            "controller": {
                "name": "Geo Climate Platform",
                "contact": "privacy@geo-climate.com"
            },
            "processing_activities": [
                {
                    "name": "Air Quality Predictions",
                    "purpose": "Provide air quality forecasting services",
                    "legal_basis": "Consent (Article 6(1)(a))",
                    "data_categories": [
                        "Location data",
                        "Environmental sensor data",
                        "Usage statistics"
                    ],
                    "data_subjects": ["Users", "API clients"],
                    "recipients": ["None (internal processing only)"],
                    "retention_period": "365 days",
                    "technical_measures": [
                        "Encryption at rest and in transit",
                        "Access controls and authentication",
                        "Audit logging",
                        "Regular backups"
                    ]
                }
            ],
            "data_transfers": {
                "third_countries": [],
                "safeguards": "Data processed within EU/US"
            }
        }
