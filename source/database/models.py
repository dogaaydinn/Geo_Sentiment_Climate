"""
SQLAlchemy ORM Models.

Enterprise database schema with:
- User management
- API keys
- Prediction logging
- Model metadata
- Audit trails
"""

from datetime import datetime
from typing import List
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Float, JSON, Text,
    ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
import uuid
from .base import Base


class User(Base):
    """User model for authentication and authorization."""

    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))

    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)

    # Roles and permissions
    roles = Column(ARRAY(String), default=["user"], nullable=False)
    permissions = Column(ARRAY(String), default=["read"], nullable=False)

    # Security
    mfa_enabled = Column(Boolean, default=False, nullable=False)
    mfa_secret = Column(String(32))
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    last_activity = Column(DateTime)

    # Settings
    settings = Column(JSONB, default={})

    # Relationships
    api_keys = relationship("APIKeyModel", back_populates="user", cascade="all, delete-orphan")
    predictions = relationship("PredictionLog", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")

    # Indexes
    __table_args__ = (
        Index("idx_user_username", "username"),
        Index("idx_user_email", "email"),
        Index("idx_user_created", "created_at"),
    )

    def __repr__(self):
        return f"<User(id={self.id}, username={self.username})>"


class APIKeyModel(Base):
    """API key model for programmatic access."""

    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key_id = Column(String(32), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    key_hash = Column(String(64), unique=True, nullable=False, index=True)

    # Owner
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Permissions
    scopes = Column(ARRAY(String), default=["read"], nullable=False)

    # Rate limiting
    rate_limit = Column(Integer, default=1000, nullable=False)
    rate_window = Column(Integer, default=3600, nullable=False)

    # Status
    is_active = Column(Boolean, default=True, nullable=False)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime)
    last_used = Column(DateTime)
    usage_count = Column(Integer, default=0, nullable=False)

    # Relationships
    user = relationship("User", back_populates="api_keys")

    __table_args__ = (
        Index("idx_apikey_user", "user_id"),
        Index("idx_apikey_hash", "key_hash"),
    )

    def __repr__(self):
        return f"<APIKey(id={self.id}, name={self.name})>"


class PredictionLog(Base):
    """Log of all predictions made through the API."""

    __tablename__ = "prediction_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # User context
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    api_key_id = Column(UUID(as_uuid=True), ForeignKey("api_keys.id", ondelete="SET NULL"))
    ip_address = Column(String(45))

    # Model information
    model_id = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50))

    # Input/Output
    input_features = Column(JSONB, nullable=False)
    predictions = Column(JSONB, nullable=False)
    probabilities = Column(JSONB)

    # Performance
    inference_time_ms = Column(Float, nullable=False)
    batch_size = Column(Integer, default=1)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    request_id = Column(String(100), index=True)

    # Quality tracking
    feedback_score = Column(Float)
    actual_value = Column(Float)
    error = Column(Float)

    # Relationships
    user = relationship("User", back_populates="predictions")

    __table_args__ = (
        Index("idx_prediction_model", "model_id"),
        Index("idx_prediction_created", "created_at"),
        Index("idx_prediction_user", "user_id"),
    )

    def __repr__(self):
        return f"<PredictionLog(id={self.id}, model={self.model_id})>"


class ModelMetadata(Base):
    """Metadata for trained ML models."""

    __tablename__ = "model_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(String(100), unique=True, nullable=False, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    version = Column(String(50), nullable=False)

    # Model details
    model_type = Column(String(50), nullable=False)  # xgboost, lightgbm, etc.
    task_type = Column(String(50), nullable=False)  # regression, classification
    framework = Column(String(50))  # sklearn, tensorflow, pytorch

    # Training information
    trained_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    training_dataset = Column(String(255))
    training_samples = Column(Integer)
    training_duration_seconds = Column(Float)

    # Performance metrics
    metrics = Column(JSONB, nullable=False)  # {"rmse": 0.5, "r2": 0.95, ...}
    validation_metrics = Column(JSONB)
    test_metrics = Column(JSONB)

    # Hyperparameters
    hyperparameters = Column(JSONB)

    # Feature information
    feature_names = Column(ARRAY(String))
    feature_importance = Column(JSONB)
    num_features = Column(Integer)

    # Lifecycle
    stage = Column(String(20), default="dev", nullable=False)  # dev, staging, production
    is_active = Column(Boolean, default=True, nullable=False)

    # Storage
    artifact_path = Column(String(500))
    model_size_bytes = Column(Integer)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deployed_at = Column(DateTime)
    deprecated_at = Column(DateTime)

    # Additional info
    description = Column(Text)
    tags = Column(ARRAY(String))
    config = Column(JSONB)

    __table_args__ = (
        Index("idx_model_name_version", "model_name", "version"),
        Index("idx_model_stage", "stage"),
        Index("idx_model_created", "created_at"),
    )

    def __repr__(self):
        return f"<ModelMetadata(id={self.model_id}, name={self.model_name})>"


class AuditLog(Base):
    """Audit trail for security and compliance."""

    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Actor
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    username = Column(String(50))
    ip_address = Column(String(45))
    user_agent = Column(String(500))

    # Action
    action = Column(String(50), nullable=False, index=True)  # login, create, update, delete
    resource_type = Column(String(50), nullable=False)  # user, model, prediction
    resource_id = Column(String(100))

    # Details
    details = Column(JSONB)
    success = Column(Boolean, default=True, nullable=False)
    error_message = Column(Text)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    request_id = Column(String(100))

    # Relationships
    user = relationship("User", back_populates="audit_logs")

    __table_args__ = (
        Index("idx_audit_user", "user_id"),
        Index("idx_audit_action", "action"),
        Index("idx_audit_resource", "resource_type", "resource_id"),
        Index("idx_audit_created", "created_at"),
    )

    def __repr__(self):
        return f"<AuditLog(id={self.id}, action={self.action})>"


class DataQualityMetric(Base):
    """Track data quality metrics over time."""

    __tablename__ = "data_quality_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Dataset information
    dataset_name = Column(String(100), nullable=False, index=True)
    dataset_version = Column(String(50))

    # Quality metrics
    total_rows = Column(Integer, nullable=False)
    null_count = Column(Integer)
    duplicate_count = Column(Integer)
    outlier_count = Column(Integer)

    # Distribution metrics
    column_stats = Column(JSONB)  # Per-column statistics

    # Validation results
    validation_passed = Column(Boolean, nullable=False)
    validation_errors = Column(JSONB)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))

    __table_args__ = (
        Index("idx_dq_dataset", "dataset_name"),
        Index("idx_dq_created", "created_at"),
    )
