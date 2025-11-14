"""
Database models and connection management.

SQLAlchemy models for:
- User management
- Roles and permissions
- API keys
- Usage tracking
- Audit logs
"""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    ForeignKey,
    Table,
    Float,
    JSON,
    Text,
    Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.pool import StaticPool
import os

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./geo_climate.db"
)

# Create engine
if "sqlite" in DATABASE_URL:
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
else:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20
    )

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# ============================================================================
# Association Tables (Many-to-Many)
# ============================================================================

user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id', ondelete='CASCADE')),
    Column('role_id', Integer, ForeignKey('roles.id', ondelete='CASCADE')),
    Index('idx_user_roles_user_id', 'user_id'),
    Index('idx_user_roles_role_id', 'role_id'),
)

role_permissions = Table(
    'role_permissions',
    Base.metadata,
    Column('role_id', Integer, ForeignKey('roles.id', ondelete='CASCADE')),
    Column('permission_id', Integer, ForeignKey('permissions.id', ondelete='CASCADE')),
    Index('idx_role_permissions_role_id', 'role_id'),
    Index('idx_role_permissions_permission_id', 'permission_id'),
)


# ============================================================================
# User Management Models
# ============================================================================

class User(Base):
    """User model for authentication and authorization."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255))
    hashed_password = Column(String(255), nullable=False)

    # Status flags
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_admin = Column(Boolean, default=False)

    # User tier for quotas
    tier = Column(String(20), default="free")  # free, basic, pro, enterprise

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime)
    email_verified_at = Column(DateTime)

    # Relationships
    roles = relationship("Role", secondary=user_roles, back_populates="users")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    usage_records = relationship("UsageRecord", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_users_email', 'email'),
        Index('idx_users_username', 'username'),
        Index('idx_users_tier', 'tier'),
    )

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', tier='{self.tier}')>"


class Role(Base):
    """Role model for RBAC."""

    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False, index=True)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    users = relationship("User", secondary=user_roles, back_populates="roles")
    permissions = relationship("Permission", secondary=role_permissions, back_populates="roles")

    def __repr__(self):
        return f"<Role(id={self.id}, name='{self.name}')>"


class Permission(Base):
    """Permission model for fine-grained access control."""

    __tablename__ = "permissions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    resource = Column(String(50), nullable=False)  # e.g., "model", "prediction"
    action = Column(String(50), nullable=False)  # e.g., "read", "write", "delete"
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    roles = relationship("Role", secondary=role_permissions, back_populates="permissions")

    # Indexes
    __table_args__ = (
        Index('idx_permissions_resource_action', 'resource', 'action'),
    )

    def __repr__(self):
        return f"<Permission(name='{self.name}', resource='{self.resource}', action='{self.action}')>"


# ============================================================================
# API Key Management
# ============================================================================

class APIKey(Base):
    """API Key model for programmatic access."""

    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(100))
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Key properties
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime)
    last_used_at = Column(DateTime)

    # Scopes/permissions
    scopes = Column(JSON, default=list)  # List of allowed scopes

    # Rate limiting
    rate_limit_per_minute = Column(Integer, default=60)
    daily_quota = Column(Integer, default=1000)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="api_keys")

    # Indexes
    __table_args__ = (
        Index('idx_api_keys_key', 'key'),
        Index('idx_api_keys_user_id', 'user_id'),
    )

    def __repr__(self):
        return f"<APIKey(id={self.id}, name='{self.name}', user_id={self.user_id})>"


# ============================================================================
# Usage Tracking
# ============================================================================

class UsageRecord(Base):
    """Usage tracking for API quota management."""

    __tablename__ = "usage_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    api_key_id = Column(Integer, ForeignKey("api_keys.id", ondelete="SET NULL"))

    # Request details
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer)
    response_time_ms = Column(Float)

    # Resource usage
    prediction_count = Column(Integer, default=0)
    tokens_used = Column(Integer, default=0)

    # Timestamps
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    user = relationship("User", back_populates="usage_records")

    # Indexes
    __table_args__ = (
        Index('idx_usage_records_user_id_timestamp', 'user_id', 'timestamp'),
        Index('idx_usage_records_timestamp', 'timestamp'),
    )

    def __repr__(self):
        return f"<UsageRecord(user_id={self.user_id}, endpoint='{self.endpoint}')>"


# ============================================================================
# Audit Logs
# ============================================================================

class AuditLog(Base):
    """Audit log for security and compliance."""

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))

    # Action details
    action = Column(String(100), nullable=False)  # e.g., "login", "model_promote"
    resource_type = Column(String(50))  # e.g., "user", "model"
    resource_id = Column(String(255))
    details = Column(JSON)

    # Request context
    ip_address = Column(String(45))
    user_agent = Column(String(255))

    # Status
    success = Column(Boolean, default=True)
    error_message = Column(Text)

    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    user = relationship("User", back_populates="audit_logs")

    # Indexes
    __table_args__ = (
        Index('idx_audit_logs_user_id_timestamp', 'user_id', 'timestamp'),
        Index('idx_audit_logs_action', 'action'),
        Index('idx_audit_logs_timestamp', 'timestamp'),
    )

    def __repr__(self):
        return f"<AuditLog(action='{self.action}', user_id={self.user_id})>"


# ============================================================================
# Database Session Dependency
# ============================================================================

def get_db() -> Session:
    """
    Dependency to get database session.

    Usage in FastAPI:
        @app.get("/users")
        def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================================
# Database Initialization
# ============================================================================

def init_db():
    """
    Initialize database with tables and default data.

    Creates all tables and adds default roles/permissions.
    """
    # Create all tables
    Base.metadata.create_all(bind=engine)

    # Create session
    db = SessionLocal()

    try:
        # Check if roles exist
        if db.query(Role).count() == 0:
            # Create default roles
            roles = [
                Role(name="admin", description="Full system access"),
                Role(name="user", description="Standard user access"),
                Role(name="api_user", description="API-only access"),
                Role(name="viewer", description="Read-only access"),
            ]
            db.add_all(roles)
            db.commit()

        # Check if permissions exist
        if db.query(Permission).count() == 0:
            # Create default permissions
            permissions = [
                # Model permissions
                Permission(name="model:read", resource="model", action="read", description="View models"),
                Permission(name="model:write", resource="model", action="write", description="Create/update models"),
                Permission(name="model:delete", resource="model", action="delete", description="Delete models"),
                Permission(name="model:promote", resource="model", action="promote", description="Promote models"),

                # Prediction permissions
                Permission(name="prediction:read", resource="prediction", action="read", description="View predictions"),
                Permission(name="prediction:write", resource="prediction", action="write", description="Make predictions"),

                # User permissions
                Permission(name="user:read", resource="user", action="read", description="View users"),
                Permission(name="user:write", resource="user", action="write", description="Create/update users"),
                Permission(name="user:delete", resource="user", action="delete", description="Delete users"),

                # System permissions
                Permission(name="system:admin", resource="system", action="admin", description="System administration"),
            ]
            db.add_all(permissions)
            db.commit()

            # Assign permissions to roles
            admin_role = db.query(Role).filter(Role.name == "admin").first()
            user_role = db.query(Role).filter(Role.name == "user").first()
            viewer_role = db.query(Role).filter(Role.name == "viewer").first()

            # Admin gets all permissions
            all_permissions = db.query(Permission).all()
            admin_role.permissions = all_permissions

            # User gets read/write permissions (not delete/admin)
            user_permissions = db.query(Permission).filter(
                Permission.action.in_(["read", "write", "promote"])
            ).all()
            user_role.permissions = user_permissions

            # Viewer gets only read permissions
            viewer_permissions = db.query(Permission).filter(
                Permission.action == "read"
            ).all()
            viewer_role.permissions = viewer_permissions

            db.commit()

        print("✅ Database initialized successfully!")

    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        db.rollback()
    finally:
        db.close()


# ============================================================================
# Helper Functions
# ============================================================================

def create_admin_user(
    username: str,
    email: str,
    password: str,
    db: Session
) -> User:
    """
    Create an admin user.

    Args:
        username: Username
        email: Email address
        password: Plain text password (will be hashed)
        db: Database session

    Returns:
        Created user object
    """
    from passlib.context import CryptContext

    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    hashed_password = pwd_context.hash(password)

    user = User(
        username=username,
        email=email,
        hashed_password=hashed_password,
        is_admin=True,
        is_active=True,
        is_verified=True,
        tier="enterprise"
    )

    # Assign admin role
    admin_role = db.query(Role).filter(Role.name == "admin").first()
    if admin_role:
        user.roles.append(admin_role)

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


if __name__ == "__main__":
    init_db()
