"""
Disaster Recovery and Backup System.

Provides automated backup, restore, and disaster recovery capabilities.
Part of Phase 3: Scaling & Optimization - Global Infrastructure.
"""

import os
import subprocess
import tarfile
import boto3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import logging
import json

logger = logging.getLogger(__name__)


class BackupManager:
    """
    Manages backups for models, data, and database.

    Features:
    - Automated scheduled backups
    - S3 storage with versioning
    - Point-in-time recovery
    - Cross-region replication
    """

    def __init__(
        self,
        s3_bucket: str,
        backup_prefix: str = "backups",
        retention_days: int = 30,
        enable_cross_region: bool = True,
        replica_regions: Optional[List[str]] = None
    ):
        """
        Initialize backup manager.

        Args:
            s3_bucket: S3 bucket for backups
            backup_prefix: Prefix for backup objects
            retention_days: Days to retain backups
            enable_cross_region: Enable cross-region replication
            replica_regions: List of replica regions
        """
        self.s3_bucket = s3_bucket
        self.backup_prefix = backup_prefix
        self.retention_days = retention_days
        self.enable_cross_region = enable_cross_region
        self.replica_regions = replica_regions or ["us-west-2", "eu-west-1"]

        # AWS clients
        self.s3_client = boto3.client('s3')
        self.rds_client = boto3.client('rds')

        logger.info(f"Backup manager initialized: bucket={s3_bucket}")

    def backup_database(
        self,
        db_identifier: str,
        snapshot_identifier: Optional[str] = None
    ) -> str:
        """
        Create RDS database snapshot.

        Args:
            db_identifier: RDS instance identifier
            snapshot_identifier: Custom snapshot name

        Returns:
            Snapshot identifier
        """
        if snapshot_identifier is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            snapshot_identifier = f"{db_identifier}-backup-{timestamp}"

        try:
            logger.info(f"Creating DB snapshot: {snapshot_identifier}")

            response = self.rds_client.create_db_snapshot(
                DBSnapshotIdentifier=snapshot_identifier,
                DBInstanceIdentifier=db_identifier,
                Tags=[
                    {'Key': 'BackupType', 'Value': 'Automated'},
                    {'Key': 'CreatedAt', 'Value': datetime.now().isoformat()}
                ]
            )

            snapshot_arn = response['DBSnapshot']['DBSnapshotArn']
            logger.info(f"DB snapshot created: {snapshot_arn}")

            # Copy to replica regions if enabled
            if self.enable_cross_region:
                self._replicate_db_snapshot(snapshot_identifier)

            return snapshot_identifier

        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            raise

    def _replicate_db_snapshot(self, snapshot_identifier: str):
        """Replicate DB snapshot to other regions."""
        for region in self.replica_regions:
            try:
                regional_client = boto3.client('rds', region_name=region)

                target_snapshot_id = f"{snapshot_identifier}-{region}"

                logger.info(
                    f"Replicating snapshot to {region}: {target_snapshot_id}"
                )

                regional_client.copy_db_snapshot(
                    SourceDBSnapshotIdentifier=snapshot_identifier,
                    TargetDBSnapshotIdentifier=target_snapshot_id,
                    CopyTags=True
                )

            except Exception as e:
                logger.error(f"Snapshot replication to {region} failed: {e}")

    def backup_models(self, models_dir: str) -> str:
        """
        Backup model files to S3.

        Args:
            models_dir: Directory containing models

        Returns:
            S3 object key
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        archive_name = f"models-backup-{timestamp}.tar.gz"
        local_archive = f"/tmp/{archive_name}"

        try:
            # Create tar archive
            logger.info(f"Creating models archive: {archive_name}")

            with tarfile.open(local_archive, "w:gz") as tar:
                tar.add(models_dir, arcname="models")

            # Upload to S3
            s3_key = f"{self.backup_prefix}/models/{archive_name}"

            logger.info(f"Uploading to S3: {s3_key}")

            self.s3_client.upload_file(
                local_archive,
                self.s3_bucket,
                s3_key,
                ExtraArgs={
                    'Metadata': {
                        'backup-type': 'models',
                        'created-at': datetime.now().isoformat()
                    }
                }
            )

            # Clean up local archive
            os.remove(local_archive)

            logger.info(f"Models backup completed: s3://{self.s3_bucket}/{s3_key}")

            return s3_key

        except Exception as e:
            logger.error(f"Models backup failed: {e}")
            raise

    def backup_data(self, data_dir: str) -> str:
        """
        Backup data files to S3.

        Args:
            data_dir: Directory containing data

        Returns:
            S3 object key
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        archive_name = f"data-backup-{timestamp}.tar.gz"
        local_archive = f"/tmp/{archive_name}"

        try:
            logger.info(f"Creating data archive: {archive_name}")

            with tarfile.open(local_archive, "w:gz") as tar:
                tar.add(data_dir, arcname="data")

            s3_key = f"{self.backup_prefix}/data/{archive_name}"

            logger.info(f"Uploading to S3: {s3_key}")

            self.s3_client.upload_file(
                local_archive,
                self.s3_bucket,
                s3_key
            )

            os.remove(local_archive)

            logger.info(f"Data backup completed: s3://{self.s3_bucket}/{s3_key}")

            return s3_key

        except Exception as e:
            logger.error(f"Data backup failed: {e}")
            raise

    def restore_models(self, s3_key: str, target_dir: str):
        """
        Restore models from S3 backup.

        Args:
            s3_key: S3 object key
            target_dir: Target directory for restoration
        """
        local_archive = "/tmp/models-restore.tar.gz"

        try:
            logger.info(f"Downloading from S3: {s3_key}")

            self.s3_client.download_file(
                self.s3_bucket,
                s3_key,
                local_archive
            )

            logger.info(f"Extracting to: {target_dir}")

            with tarfile.open(local_archive, "r:gz") as tar:
                tar.extractall(target_dir)

            os.remove(local_archive)

            logger.info("Models restore completed")

        except Exception as e:
            logger.error(f"Models restore failed: {e}")
            raise

    def restore_database(
        self,
        snapshot_identifier: str,
        target_db_identifier: str
    ):
        """
        Restore database from snapshot.

        Args:
            snapshot_identifier: Snapshot to restore from
            target_db_identifier: New DB instance identifier
        """
        try:
            logger.info(
                f"Restoring DB from snapshot: {snapshot_identifier}"
            )

            response = self.rds_client.restore_db_instance_from_db_snapshot(
                DBInstanceIdentifier=target_db_identifier,
                DBSnapshotIdentifier=snapshot_identifier,
                DBInstanceClass='db.t3.medium',  # Adjust as needed
                PubliclyAccessible=False,
                Tags=[
                    {'Key': 'RestoredFrom', 'Value': snapshot_identifier},
                    {'Key': 'RestoredAt', 'Value': datetime.now().isoformat()}
                ]
            )

            logger.info(f"Database restore initiated: {target_db_identifier}")

            return response['DBInstance']['DBInstanceIdentifier']

        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            raise

    def cleanup_old_backups(self):
        """Delete backups older than retention period."""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        try:
            # Cleanup S3 backups
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=self.backup_prefix
            )

            deleted_count = 0

            for obj in response.get('Contents', []):
                if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                    self.s3_client.delete_object(
                        Bucket=self.s3_bucket,
                        Key=obj['Key']
                    )
                    deleted_count += 1

            logger.info(f"Cleaned up {deleted_count} old backups from S3")

            # Cleanup old DB snapshots
            snapshots = self.rds_client.describe_db_snapshots(
                SnapshotType='manual'
            )

            for snapshot in snapshots['DBSnapshots']:
                snapshot_time = snapshot['SnapshotCreateTime'].replace(tzinfo=None)
                if snapshot_time < cutoff_date:
                    self.rds_client.delete_db_snapshot(
                        DBSnapshotIdentifier=snapshot['DBSnapshotIdentifier']
                    )
                    deleted_count += 1

            logger.info(f"Cleaned up old DB snapshots")

        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")

    def create_disaster_recovery_plan(self) -> dict:
        """
        Create disaster recovery plan document.

        Returns:
            DR plan as dictionary
        """
        plan = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "rpo_hours": 1,  # Recovery Point Objective
            "rto_hours": 4,  # Recovery Time Objective
            "backup_schedule": {
                "database": "Every 6 hours",
                "models": "Daily at 02:00 UTC",
                "data": "Daily at 03:00 UTC"
            },
            "recovery_procedures": {
                "database": [
                    "1. Identify latest snapshot",
                    "2. Restore to new RDS instance",
                    "3. Update application connection strings",
                    "4. Validate data integrity",
                    "5. Redirect traffic"
                ],
                "models": [
                    "1. Download latest backup from S3",
                    "2. Extract to models directory",
                    "3. Restart model server",
                    "4. Verify model loading",
                    "5. Run health checks"
                ],
                "complete_outage": [
                    "1. Activate secondary region",
                    "2. Update DNS to failover endpoint",
                    "3. Restore database from cross-region snapshot",
                    "4. Restore models and data",
                    "5. Run comprehensive tests",
                    "6. Monitor metrics"
                ]
            },
            "contacts": {
                "primary": "ops-team@geo-climate.com",
                "escalation": "engineering-lead@geo-climate.com"
            }
        }

        # Save plan to S3
        plan_key = f"{self.backup_prefix}/dr-plan.json"
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=plan_key,
            Body=json.dumps(plan, indent=2),
            ContentType='application/json'
        )

        logger.info(f"DR plan saved to s3://{self.s3_bucket}/{plan_key}")

        return plan
