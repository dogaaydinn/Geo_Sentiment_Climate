"""Global pytest configuration and fixtures."""
import pytest
import os
import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from tests.helpers.api_helpers import APITestHelper


@pytest.fixture(scope="session")
def test_client():
    """Create FastAPI test client."""
    from source.api.main import app
    return TestClient(app)


@pytest.fixture(scope="session")
def api_helper(test_client):
    """Create API test helper."""
    return APITestHelper(test_client)


@pytest.fixture(autouse=True)
def reset_test_environment():
    """Reset environment before each test."""
    # Clear any cached data
    yield
    # Cleanup after test


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create temporary directory for models."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary directory for data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "raw").mkdir()
    (data_dir / "interim").mkdir()
    (data_dir / "processed").mkdir()
    return data_dir


# Import fixtures from other modules
pytest_plugins = [
    "tests.fixtures.data_fixtures",
]


# Hooks for custom test behavior
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test items during collection."""
    # Add markers automatically based on file location
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        if "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        if "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)
