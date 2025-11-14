"""
Global pytest configuration and fixtures.

Enterprise-grade test infrastructure with comprehensive fixtures for:
- Async testing
- API testing
- Database testing
- Redis/cache testing
- ML model testing
- Mock data generation
- Performance monitoring
"""
import pytest
import os
import sys
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator, AsyncGenerator
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from httpx import AsyncClient
from tests.helpers.api_helpers import APITestHelper


# ============================================================================
# Event Loop Configuration for Async Tests
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """
    Create event loop for async tests.

    Uses session scope to reuse the same event loop across all tests
    for better performance.
    """
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# API Test Clients
# ============================================================================

@pytest.fixture(scope="session")
def test_client():
    """
    Create FastAPI synchronous test client.

    For synchronous API tests using requests-like interface.
    """
    from source.api.main import app
    return TestClient(app)


@pytest.fixture(scope="session")
async def async_client():
    """
    Create FastAPI asynchronous test client.

    For async API tests with better performance.
    """
    from source.api.main import app
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture(scope="session")
def api_helper(test_client):
    """Create API test helper with utility methods."""
    return APITestHelper(test_client)


# ============================================================================
# Environment & Configuration
# ============================================================================

@pytest.fixture(autouse=True)
def reset_test_environment(monkeypatch):
    """
    Reset environment before each test.

    Sets test-specific environment variables and cleans up afterward.
    """
    # Set test environment
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/15")  # Test DB

    yield

    # Cleanup is automatic with monkeypatch


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Test configuration dictionary."""
    return {
        "api_version": "v1",
        "test_timeout": 30,
        "max_retries": 3,
        "batch_size": 100,
        "cache_ttl": 300,
    }


# ============================================================================
# Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_model_dir(tmp_path) -> Path:
    """Create temporary directory for models."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


@pytest.fixture
def temp_data_dir(tmp_path) -> Path:
    """Create temporary directory structure for data."""
    data_dir = tmp_path / "data"
    (data_dir / "raw").mkdir(parents=True, exist_ok=True)
    (data_dir / "interim").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed").mkdir(parents=True, exist_ok=True)
    (data_dir / "external").mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def temp_cache_dir(tmp_path) -> Path:
    """Create temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


# ============================================================================
# Mock Data Generators
# ============================================================================

@pytest.fixture
def sample_features() -> Dict[str, float]:
    """Generate sample feature dictionary for predictions."""
    return {
        "temperature": 25.5,
        "humidity": 65.0,
        "wind_speed": 12.3,
        "pressure": 1013.25,
        "co": 0.8,
        "no2": 45.2,
        "o3": 68.4,
        "pm25": 35.6,
    }


@pytest.fixture
def batch_features(sample_features) -> list:
    """Generate batch of feature dictionaries."""
    np.random.seed(42)  # Reproducible

    batch = []
    for i in range(100):
        features = {
            "temperature": np.random.uniform(15, 35),
            "humidity": np.random.uniform(30, 90),
            "wind_speed": np.random.uniform(0, 20),
            "pressure": np.random.uniform(980, 1030),
            "co": np.random.uniform(0, 5),
            "no2": np.random.uniform(0, 200),
            "o3": np.random.uniform(0, 150),
            "pm25": np.random.uniform(0, 200),
        }
        batch.append(features)

    return batch


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Generate sample pandas DataFrame for testing."""
    np.random.seed(42)

    n_samples = 1000
    data = {
        "timestamp": pd.date_range(start="2024-01-01", periods=n_samples, freq="H"),
        "temperature": np.random.uniform(15, 35, n_samples),
        "humidity": np.random.uniform(30, 90, n_samples),
        "wind_speed": np.random.uniform(0, 20, n_samples),
        "pressure": np.random.uniform(980, 1030, n_samples),
        "co": np.random.uniform(0, 5, n_samples),
        "no2": np.random.uniform(0, 200, n_samples),
        "o3": np.random.uniform(0, 150, n_samples),
        "pm25": np.random.uniform(0, 200, n_samples),
        "target": np.random.randint(0, 6, n_samples),  # AQI categories
    }

    return pd.DataFrame(data)


# ============================================================================
# Mock Services
# ============================================================================

@pytest.fixture
def mock_model_registry():
    """Mock model registry for testing."""
    mock_registry = MagicMock()
    mock_registry.models = {}
    mock_registry.list_models.return_value = []
    mock_registry.get_model.return_value = None
    mock_registry.register_model.return_value = "test-model-id"
    return mock_registry


@pytest.fixture
def mock_inference_engine():
    """Mock inference engine for testing."""
    mock_engine = MagicMock()
    mock_engine.predict.return_value = MagicMock(
        predictions=[0.75],
        model_id="test-model",
        inference_time_ms=15.5,
        timestamp=datetime.now().isoformat(),
        input_shape=(1, 8)
    )
    return mock_engine


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock_client = MagicMock()
    mock_client.get.return_value = None
    mock_client.set.return_value = True
    mock_client.delete.return_value = 1
    mock_client.exists.return_value = False
    mock_client.ttl.return_value = -1
    return mock_client


# ============================================================================
# Performance Monitoring
# ============================================================================

@pytest.fixture
def performance_monitor():
    """
    Performance monitoring fixture.

    Tracks test execution time and resource usage.
    """
    import time
    import psutil

    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.metrics = {}

        def start(self):
            self.start_time = time.time()
            self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        def stop(self):
            if self.start_time:
                duration = time.time() - self.start_time
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_delta = current_memory - self.start_memory

                self.metrics = {
                    "duration_seconds": duration,
                    "memory_mb": current_memory,
                    "memory_delta_mb": memory_delta,
                }
                return self.metrics
            return {}

    monitor = PerformanceMonitor()
    monitor.start()
    yield monitor
    metrics = monitor.stop()

    # Log if test is slow or memory-intensive
    if metrics.get("duration_seconds", 0) > 5:
        print(f"\n⚠️  Slow test: {metrics['duration_seconds']:.2f}s")
    if metrics.get("memory_delta_mb", 0) > 100:
        print(f"\n⚠️  High memory usage: {metrics['memory_delta_mb']:.2f}MB")


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def db_engine():
    """Create test database engine."""
    from sqlalchemy import create_engine
    engine = create_engine("sqlite:///:memory:")
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    """Create database session for testing."""
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


# ============================================================================
# Import Additional Fixtures
# ============================================================================

pytest_plugins = [
    "tests.fixtures.data_fixtures",
]


# ============================================================================
# Pytest Hooks
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    markers = [
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
        "integration: marks tests as integration tests",
        "unit: marks tests as unit tests",
        "e2e: marks tests as end-to-end tests",
        "smoke: marks tests as smoke tests",
        "database: tests that require database",
        "redis: tests that require redis",
        "api: tests for API endpoints",
        "ml: tests for ML models",
        "performance: marks tests as performance tests",
        "benchmark: marks tests as benchmark tests",
        "security: marks tests as security tests",
    ]

    for marker in markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """
    Modify test items during collection.

    Automatically adds markers based on file location and test characteristics.
    """
    for item in items:
        # Add markers based on file path
        item_path = str(item.fspath)

        if "integration" in item_path:
            item.add_marker(pytest.mark.integration)
        if "e2e" in item_path:
            item.add_marker(pytest.mark.e2e)
        if "performance" in item_path or "load" in item_path:
            item.add_marker(pytest.mark.performance)
        if "security" in item_path:
            item.add_marker(pytest.mark.security)
        if "benchmark" in item_path:
            item.add_marker(pytest.mark.benchmark)

        # Add slow marker to tests with specific names
        if "slow" in item.name or "endurance" in item.name:
            item.add_marker(pytest.mark.slow)


def pytest_runtest_setup(item):
    """Run before each test."""
    # Skip slow tests unless explicitly requested
    if "slow" in item.keywords and not item.config.getoption("-m"):
        pytest.skip("Skipping slow test (use -m slow to run)")


def pytest_report_header(config):
    """Add custom header to pytest output."""
    return [
        "=" * 80,
        "Geo_Sentiment_Climate Test Suite",
        "Enterprise-Grade Air Quality Prediction Platform",
        "=" * 80,
    ]


# ============================================================================
# Custom Assertions
# ============================================================================

@pytest.fixture
def assert_response_time():
    """Custom assertion for API response time."""
    def _assert(duration_ms: float, max_ms: float = 1000):
        assert duration_ms < max_ms, (
            f"Response time {duration_ms}ms exceeds maximum {max_ms}ms"
        )
    return _assert


@pytest.fixture
def assert_data_quality():
    """Custom assertion for data quality checks."""
    def _assert(df: pd.DataFrame, min_rows: int = 1, max_null_pct: float = 0.1):
        assert len(df) >= min_rows, f"DataFrame has only {len(df)} rows"

        null_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        assert null_pct <= max_null_pct, (
            f"Null percentage {null_pct:.2%} exceeds maximum {max_null_pct:.2%}"
        )
    return _assert
