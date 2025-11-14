"""
Data Export Capabilities.

Provides flexible data export in multiple formats for analysis and integration.
Part of Phase 5: Enterprise & Ecosystem - Enterprise Features.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import csv
import io
import logging

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    PARQUET = "parquet"
    XML = "xml"
    YAML = "yaml"


class ExportStatus(Enum):
    """Export job status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExportJob:
    """Export job metadata."""
    job_id: str
    user_id: str
    tenant_id: str
    export_type: str
    format: str
    status: str
    created_at: str

    # Configuration
    filters: Dict[str, Any] = field(default_factory=dict)
    columns: Optional[List[str]] = None

    # Progress
    total_records: int = 0
    processed_records: int = 0
    progress_percentage: float = 0.0

    # Results
    file_url: Optional[str] = None
    file_size_bytes: int = 0
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


class DataExporter:
    """
    Handles data export in various formats.

    Features:
    - Multiple format support (JSON, CSV, Excel, Parquet, XML, YAML)
    - Large dataset streaming
    - Custom column selection
    - Filtering and transformation
    - Compression options
    """

    def __init__(self, database_manager=None):
        """
        Initialize data exporter.

        Args:
            database_manager: Optional database manager
        """
        self.db = database_manager
        self.export_jobs: Dict[str, ExportJob] = {}

        logger.info("Data exporter initialized")

    def create_export_job(
        self,
        user_id: str,
        tenant_id: str,
        export_type: str,
        format: ExportFormat,
        filters: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None
    ) -> ExportJob:
        """
        Create a new export job.

        Args:
            user_id: User requesting export
            tenant_id: Tenant ID
            export_type: Type of data to export
            format: Export format
            filters: Optional data filters
            columns: Optional column selection

        Returns:
            Created export job
        """
        import uuid

        job_id = str(uuid.uuid4())

        job = ExportJob(
            job_id=job_id,
            user_id=user_id,
            tenant_id=tenant_id,
            export_type=export_type,
            format=format.value,
            status=ExportStatus.PENDING.value,
            created_at=datetime.utcnow().isoformat(),
            filters=filters or {},
            columns=columns
        )

        self.export_jobs[job_id] = job

        logger.info(
            f"Created export job: {job_id} - "
            f"{export_type} as {format.value} for user {user_id}"
        )

        return job

    async def execute_export(
        self,
        job_id: str
    ) -> bool:
        """
        Execute an export job.

        Args:
            job_id: Export job ID

        Returns:
            Success status
        """
        job = self.export_jobs.get(job_id)

        if not job:
            logger.error(f"Export job not found: {job_id}")
            return False

        job.status = ExportStatus.PROCESSING.value

        try:
            # Fetch data based on export type
            data = await self._fetch_export_data(job)

            # Export in requested format
            if job.format == ExportFormat.JSON.value:
                output = self._export_json(data, job)
            elif job.format == ExportFormat.CSV.value:
                output = self._export_csv(data, job)
            elif job.format == ExportFormat.EXCEL.value:
                output = self._export_excel(data, job)
            elif job.format == ExportFormat.PARQUET.value:
                output = self._export_parquet(data, job)
            elif job.format == ExportFormat.XML.value:
                output = self._export_xml(data, job)
            elif job.format == ExportFormat.YAML.value:
                output = self._export_yaml(data, job)
            else:
                raise ValueError(f"Unsupported format: {job.format}")

            # Store output
            file_url = await self._store_export(job_id, output)

            job.status = ExportStatus.COMPLETED.value
            job.file_url = file_url
            job.file_size_bytes = len(output)
            job.completed_at = datetime.utcnow().isoformat()
            job.progress_percentage = 100.0

            logger.info(f"Export job completed: {job_id}")
            return True

        except Exception as e:
            job.status = ExportStatus.FAILED.value
            job.error_message = str(e)
            logger.error(f"Export job failed: {job_id} - {str(e)}")
            return False

    async def _fetch_export_data(
        self,
        job: ExportJob
    ) -> List[Dict[str, Any]]:
        """
        Fetch data for export.

        Args:
            job: Export job

        Returns:
            Data records
        """
        # Simulated data fetch based on export type
        if job.export_type == "predictions":
            return self._fetch_predictions(job)
        elif job.export_type == "models":
            return self._fetch_models(job)
        elif job.export_type == "users":
            return self._fetch_users(job)
        elif job.export_type == "metrics":
            return self._fetch_metrics(job)
        elif job.export_type == "audit_logs":
            return self._fetch_audit_logs(job)
        else:
            return []

    def _fetch_predictions(self, job: ExportJob) -> List[Dict[str, Any]]:
        """Fetch prediction data."""
        # Simulated prediction data
        return [
            {
                "prediction_id": f"pred_{i}",
                "timestamp": datetime.utcnow().isoformat(),
                "model_id": "model_v1",
                "input": {"location": f"40.{i},-74.{i}"},
                "output": {"aqi": 50 + i, "category": "Good"},
                "latency_ms": 150 + i
            }
            for i in range(100)
        ]

    def _fetch_models(self, job: ExportJob) -> List[Dict[str, Any]]:
        """Fetch model data."""
        return [
            {
                "model_id": f"model_v{i}",
                "name": f"AQI Predictor v{i}",
                "version": f"1.{i}.0",
                "status": "production",
                "accuracy": 0.85 + (i * 0.01),
                "created_at": datetime.utcnow().isoformat()
            }
            for i in range(10)
        ]

    def _fetch_users(self, job: ExportJob) -> List[Dict[str, Any]]:
        """Fetch user data (anonymized)."""
        return [
            {
                "user_id": f"user_{i}",
                "created_at": datetime.utcnow().isoformat(),
                "last_login": datetime.utcnow().isoformat(),
                "api_calls_count": 1000 + i,
                "tier": "pro"
            }
            for i in range(50)
        ]

    def _fetch_metrics(self, job: ExportJob) -> List[Dict[str, Any]]:
        """Fetch metrics data."""
        return [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "metric_name": "api_requests",
                "value": 1000 + i,
                "tags": {"endpoint": "/predict"}
            }
            for i in range(100)
        ]

    def _fetch_audit_logs(self, job: ExportJob) -> List[Dict[str, Any]]:
        """Fetch audit log data."""
        return [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "user.login",
                "user_id": f"user_{i}",
                "status": "success",
                "ip_address": f"192.168.1.{i}"
            }
            for i in range(100)
        ]

    def _export_json(
        self,
        data: List[Dict[str, Any]],
        job: ExportJob
    ) -> bytes:
        """Export data as JSON."""
        # Apply column filtering if specified
        if job.columns:
            data = [
                {k: v for k, v in record.items() if k in job.columns}
                for record in data
            ]

        json_str = json.dumps(data, indent=2)
        return json_str.encode('utf-8')

    def _export_csv(
        self,
        data: List[Dict[str, Any]],
        job: ExportJob
    ) -> bytes:
        """Export data as CSV."""
        if not data:
            return b"No data available"

        output = io.StringIO()

        # Determine columns
        columns = job.columns if job.columns else list(data[0].keys())

        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()

        for record in data:
            # Filter columns
            filtered_record = {
                k: v for k, v in record.items()
                if k in columns
            }
            writer.writerow(filtered_record)

        return output.getvalue().encode('utf-8')

    def _export_excel(
        self,
        data: List[Dict[str, Any]],
        job: ExportJob
    ) -> bytes:
        """Export data as Excel (XLSX)."""
        try:
            import openpyxl
            from openpyxl import Workbook

            wb = Workbook()
            ws = wb.active
            ws.title = job.export_type

            if not data:
                return b"No data available"

            # Determine columns
            columns = job.columns if job.columns else list(data[0].keys())

            # Write headers
            ws.append(columns)

            # Write data
            for record in data:
                row = [record.get(col, "") for col in columns]
                ws.append(row)

            # Save to bytes
            output = io.BytesIO()
            wb.save(output)
            return output.getvalue()

        except ImportError:
            logger.warning("openpyxl not available, falling back to CSV")
            return self._export_csv(data, job)

    def _export_parquet(
        self,
        data: List[Dict[str, Any]],
        job: ExportJob
    ) -> bytes:
        """Export data as Parquet."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            # Convert to PyArrow table
            table = pa.Table.from_pylist(data)

            # Apply column filtering
            if job.columns:
                table = table.select(job.columns)

            # Write to bytes
            output = io.BytesIO()
            pq.write_table(table, output)
            return output.getvalue()

        except ImportError:
            logger.warning("pyarrow not available, falling back to JSON")
            return self._export_json(data, job)

    def _export_xml(
        self,
        data: List[Dict[str, Any]],
        job: ExportJob
    ) -> bytes:
        """Export data as XML."""
        xml_lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml_lines.append(f'<{job.export_type}>')

        for record in data:
            xml_lines.append('  <record>')
            for key, value in record.items():
                if job.columns and key not in job.columns:
                    continue
                xml_lines.append(f'    <{key}>{value}</{key}>')
            xml_lines.append('  </record>')

        xml_lines.append(f'</{job.export_type}>')

        return '\n'.join(xml_lines).encode('utf-8')

    def _export_yaml(
        self,
        data: List[Dict[str, Any]],
        job: ExportJob
    ) -> bytes:
        """Export data as YAML."""
        try:
            import yaml

            # Apply column filtering
            if job.columns:
                data = [
                    {k: v for k, v in record.items() if k in job.columns}
                    for record in data
                ]

            yaml_str = yaml.dump(data, default_flow_style=False)
            return yaml_str.encode('utf-8')

        except ImportError:
            logger.warning("PyYAML not available, falling back to JSON")
            return self._export_json(data, job)

    async def _store_export(
        self,
        job_id: str,
        data: bytes
    ) -> str:
        """
        Store export file.

        Args:
            job_id: Export job ID
            data: Export data

        Returns:
            File URL
        """
        # In production, would upload to S3/GCS/Azure Blob Storage
        file_url = f"https://exports.geo-climate.com/{job_id}.export"

        logger.info(f"Stored export file: {file_url} ({len(data)} bytes)")

        return file_url

    def get_export_status(
        self,
        job_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get export job status.

        Args:
            job_id: Export job ID

        Returns:
            Job status or None
        """
        job = self.export_jobs.get(job_id)

        if not job:
            return None

        return {
            "job_id": job.job_id,
            "status": job.status,
            "export_type": job.export_type,
            "format": job.format,
            "created_at": job.created_at,
            "progress_percentage": job.progress_percentage,
            "total_records": job.total_records,
            "processed_records": job.processed_records,
            "file_url": job.file_url,
            "file_size_bytes": job.file_size_bytes,
            "completed_at": job.completed_at,
            "error_message": job.error_message
        }

    def list_export_jobs(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[ExportJob]:
        """
        List export jobs.

        Args:
            user_id: Optional user filter
            tenant_id: Optional tenant filter
            status: Optional status filter

        Returns:
            List of export jobs
        """
        jobs = list(self.export_jobs.values())

        if user_id:
            jobs = [j for j in jobs if j.user_id == user_id]

        if tenant_id:
            jobs = [j for j in jobs if j.tenant_id == tenant_id]

        if status:
            jobs = [j for j in jobs if j.status == status]

        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs

    def delete_export(
        self,
        job_id: str
    ) -> bool:
        """
        Delete an export job and its file.

        Args:
            job_id: Export job ID

        Returns:
            Success status
        """
        if job_id not in self.export_jobs:
            return False

        # In production, would delete file from storage
        del self.export_jobs[job_id]

        logger.info(f"Deleted export job: {job_id}")
        return True


class BulkDataExporter:
    """
    Handles large-scale bulk data exports.

    Features:
    - Streaming export for large datasets
    - Chunked processing
    - Compression
    - Scheduled exports
    """

    def __init__(self, data_exporter: DataExporter):
        """
        Initialize bulk data exporter.

        Args:
            data_exporter: Data exporter instance
        """
        self.exporter = data_exporter
        self.scheduled_exports: Dict[str, Dict[str, Any]] = {}

        logger.info("Bulk data exporter initialized")

    async def create_bulk_export(
        self,
        user_id: str,
        tenant_id: str,
        export_configs: List[Dict[str, Any]],
        compress: bool = True
    ) -> List[str]:
        """
        Create multiple exports in bulk.

        Args:
            user_id: User ID
            tenant_id: Tenant ID
            export_configs: List of export configurations
            compress: Whether to compress outputs

        Returns:
            List of job IDs
        """
        job_ids = []

        for config in export_configs:
            job = self.exporter.create_export_job(
                user_id=user_id,
                tenant_id=tenant_id,
                export_type=config["export_type"],
                format=ExportFormat(config["format"]),
                filters=config.get("filters"),
                columns=config.get("columns")
            )
            job_ids.append(job.job_id)

            # Execute export
            await self.exporter.execute_export(job.job_id)

        logger.info(f"Created {len(job_ids)} bulk export jobs")

        return job_ids

    def schedule_export(
        self,
        schedule_id: str,
        user_id: str,
        tenant_id: str,
        export_type: str,
        format: ExportFormat,
        cron_expression: str,
        filters: Optional[Dict[str, Any]] = None
    ):
        """
        Schedule a recurring export.

        Args:
            schedule_id: Schedule identifier
            user_id: User ID
            tenant_id: Tenant ID
            export_type: Type of data to export
            format: Export format
            cron_expression: Cron expression for schedule
            filters: Optional filters
        """
        self.scheduled_exports[schedule_id] = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "export_type": export_type,
            "format": format.value,
            "cron_expression": cron_expression,
            "filters": filters or {},
            "created_at": datetime.utcnow().isoformat(),
            "active": True
        }

        logger.info(f"Scheduled export: {schedule_id} ({cron_expression})")

    def get_scheduled_exports(
        self,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get scheduled exports.

        Args:
            user_id: Optional user filter

        Returns:
            List of scheduled exports
        """
        schedules = list(self.scheduled_exports.values())

        if user_id:
            schedules = [
                s for s in schedules
                if s["user_id"] == user_id
            ]

        return schedules
