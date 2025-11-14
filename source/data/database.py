"""
Database Connection Pool and Optimization.

Provides optimized database access with connection pooling, query optimization,
and automatic retry logic.
Part of Phase 3: Scaling & Optimization - Database Optimization.
"""

import os
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from sqlalchemy import create_engine, pool, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import QueuePool
import logging

from source.utils.circuit_breaker import db_circuit_breaker, default_retry

logger = logging.getLogger(__name__)

# SQLAlchemy base
Base = declarative_base()


class DatabaseManager:
    """
    Database manager with connection pooling and optimization.

    Features:
    - Connection pooling with QueuePool
    - Automatic reconnection
    - Query optimization
    - Circuit breaker protection
    - Prepared statements
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        echo: bool = False
    ):
        """
        Initialize database manager.

        Args:
            db_url: Database URL (postgresql://user:pass@host:port/db)
            pool_size: Number of connections to maintain
            max_overflow: Max overflow connections
            pool_timeout: Timeout for getting connection
            pool_recycle: Recycle connections after seconds
            echo: Echo SQL queries (debug)
        """
        # Build database URL
        if db_url is None:
            db_url = self._build_db_url()

        # Create engine with connection pooling
        self.engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
            pool_pre_ping=True,  # Verify connections before using
            echo=echo
        )

        # Session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

        # Setup event listeners for optimization
        self._setup_event_listeners()

        logger.info(
            f"Database initialized with pool_size={pool_size}, "
            f"max_overflow={max_overflow}"
        )

    def _build_db_url(self) -> str:
        """Build database URL from environment variables."""
        user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD", "postgres")
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        database = os.getenv("DB_NAME", "geo_climate")

        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

    def _setup_event_listeners(self):
        """Setup SQLAlchemy event listeners for optimization."""
        # Log slow queries
        @event.listens_for(self.engine, "before_cursor_execute")
        def receive_before_cursor_execute(
            conn, cursor, statement, params, context, executemany
        ):
            context._query_start_time = time.time()

        @event.listens_for(self.engine, "after_cursor_execute")
        def receive_after_cursor_execute(
            conn, cursor, statement, params, context, executemany
        ):
            import time
            total = time.time() - context._query_start_time
            if total > 1.0:  # Log queries taking > 1 second
                logger.warning(
                    f"Slow query ({total:.2f}s): {statement[:100]}..."
                )

    @contextmanager
    def get_session(self) -> Session:
        """
        Get database session with automatic cleanup.

        Usage:
            with db_manager.get_session() as session:
                session.query(Model).all()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()

    @db_circuit_breaker
    def execute_with_retry(
        self,
        query: str,
        params: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Execute query with circuit breaker and retry logic.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            Query results as list of dictionaries
        """
        def _execute():
            with self.get_session() as session:
                result = session.execute(query, params or {})
                if result.returns_rows:
                    return [dict(row) for row in result]
                return []

        return default_retry.execute(_execute)

    def bulk_insert(
        self,
        table_name: str,
        data: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> int:
        """
        Bulk insert data in batches.

        Args:
            table_name: Table name
            data: List of dictionaries to insert
            batch_size: Batch size for insertion

        Returns:
            Number of rows inserted
        """
        total_inserted = 0

        with self.get_session() as session:
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]

                try:
                    # Use bulk insert for performance
                    session.bulk_insert_mappings(
                        table_name,
                        batch
                    )
                    session.commit()
                    total_inserted += len(batch)

                    logger.debug(
                        f"Inserted batch {i // batch_size + 1}: "
                        f"{len(batch)} rows"
                    )

                except Exception as e:
                    session.rollback()
                    logger.error(f"Bulk insert error: {e}")
                    raise

        logger.info(f"Bulk inserted {total_inserted} rows into {table_name}")
        return total_inserted

    def get_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status."""
        pool = self.engine.pool

        return {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "total_connections": pool.size() + pool.overflow()
        }

    def health_check(self) -> bool:
        """Check database health."""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    def create_tables(self):
        """Create all tables defined in models."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise

    def close(self):
        """Close all database connections."""
        self.engine.dispose()
        logger.info("Database connections closed")


# Models for metadata storage
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text
from datetime import datetime


class ModelMetadata(Base):
    """Model metadata storage."""
    __tablename__ = "models"

    id = Column(Integer, primary_key=True)
    model_id = Column(String(100), unique=True, index=True, nullable=False)
    model_name = Column(String(100), index=True)
    version = Column(String(50))
    model_type = Column(String(50))
    task_type = Column(String(50))
    stage = Column(String(20), index=True, default="dev")
    metrics = Column(JSON)
    hyperparameters = Column(JSON)
    file_path = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PredictionLog(Base):
    """Prediction logging for analytics."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    model_id = Column(String(100), index=True)
    prediction_value = Column(Float)
    input_data = Column(JSON)
    inference_time_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class SystemMetrics(Base):
    """System performance metrics."""
    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True)
    metric_name = Column(String(100), index=True)
    metric_value = Column(Float)
    metric_type = Column(String(50))
    tags = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


# Global database manager
db_manager = DatabaseManager()
