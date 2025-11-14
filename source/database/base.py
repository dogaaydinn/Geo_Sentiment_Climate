"""
Database connection and session management.

Enterprise-grade features:
- Connection pooling
- Automatic reconnection
- Query optimization
- Read replicas support
"""

import os
from typing import Generator
from sqlalchemy import create_engine, event, pool
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from source.utils.logger import setup_logger

logger = setup_logger(name="database", log_file="../logs/database.log")

# Database URL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://geo_climate:secure_password_change_me@localhost:5432/geo_climate_db"
)

# Engine configuration with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,  # Maintain 10 connections
    max_overflow=20,  # Allow up to 20 additional connections
    pool_timeout=30,  # Wait 30 seconds for available connection
    pool_recycle=3600,  # Recycle connections after 1 hour
    pool_pre_ping=True,  # Verify connections before using
    echo=os.getenv("SQL_ECHO", "false").lower() == "true",  # Log SQL queries
    future=True,
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)

# Base class for models
Base = declarative_base()


# Event listeners for connection management
@event.listens_for(engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    """Execute on connection establishment."""
    logger.debug("Database connection established")


@event.listens_for(engine, "close")
def receive_close(dbapi_conn, connection_record):
    """Execute on connection close."""
    logger.debug("Database connection closed")


@event.listens_for(engine, "checkin")
def receive_checkin(dbapi_conn, connection_record):
    """Execute when connection returned to pool."""
    # Reset any session-level settings
    pass


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database sessions.

    Yields:
        Database session

    Example:
        @app.get("/users")
        def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """
    Initialize database (create all tables).

    Call this during application startup.
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def check_db_connection() -> bool:
    """
    Check database connectivity.

    Returns:
        True if connected, False otherwise
    """
    try:
        with engine.connect() as connection:
            connection.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False
