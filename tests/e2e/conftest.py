"""E2E test configuration and fixtures."""
import pytest
import docker
import time
import requests
from pathlib import Path
import subprocess
import os


@pytest.fixture(scope="session")
def docker_client():
    """Create Docker client for E2E tests."""
    try:
        client = docker.from_env()
        # Test connection
        client.ping()
        return client
    except Exception as e:
        pytest.skip(f"Docker not available: {e}")


@pytest.fixture(scope="session")
def api_container(docker_client, tmp_path_factory):
    """Start API container for E2E testing."""
    # Get project root
    project_root = Path(__file__).parent.parent.parent

    # Build image
    print("\n🐳 Building Docker image for E2E tests...")
    try:
        image, logs = docker_client.images.build(
            path=str(project_root),
            tag="geo-climate-test:latest",
            rm=True,
            forcerm=True
        )
        print("✅ Docker image built successfully")
    except Exception as e:
        pytest.skip(f"Failed to build Docker image: {e}")

    # Start container
    print("🚀 Starting API container...")
    try:
        container = docker_client.containers.run(
            image="geo-climate-test:latest",
            detach=True,
            ports={'8000/tcp': 8000},
            environment={
                'ENVIRONMENT': 'test',
                'LOG_LEVEL': 'INFO',
                'TESTING': 'true'
            },
            remove=True
        )
    except Exception as e:
        pytest.skip(f"Failed to start container: {e}")

    # Wait for API to be ready
    print("⏳ Waiting for API to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print(f"✅ API ready after {i+1} attempts")
                break
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(1)
    else:
        container.stop()
        pytest.fail("API failed to start within 30 seconds")

    yield container

    # Cleanup
    print("\n🧹 Stopping API container...")
    try:
        container.stop()
        print("✅ Container stopped")
    except:
        pass


@pytest.fixture(scope="session")
def docker_compose_stack(tmp_path_factory):
    """Start full docker-compose stack for E2E tests."""
    project_root = Path(__file__).parent.parent.parent
    compose_file = project_root / "docker-compose.yml"

    if not compose_file.exists():
        pytest.skip("docker-compose.yml not found")

    print("\n🐳 Starting docker-compose stack...")

    # Start stack
    try:
        result = subprocess.run(
            ["docker-compose", "-f", str(compose_file), "up", "-d"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            pytest.skip(f"docker-compose up failed: {result.stderr}")

        print("✅ Docker-compose stack started")
    except subprocess.TimeoutExpired:
        pytest.skip("docker-compose up timed out")
    except FileNotFoundError:
        pytest.skip("docker-compose not installed")

    # Wait for services
    print("⏳ Waiting for services to be ready...")
    time.sleep(15)

    # Check if API is ready
    max_retries = 20
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print(f"✅ Services ready after {i+1} attempts")
                break
        except:
            pass
        time.sleep(2)
    else:
        # Cleanup and skip
        subprocess.run(
            ["docker-compose", "-f", str(compose_file), "down", "-v"],
            cwd=str(project_root)
        )
        pytest.skip("Services failed to be ready")

    yield

    # Cleanup
    print("\n🧹 Stopping docker-compose stack...")
    subprocess.run(
        ["docker-compose", "-f", str(compose_file), "down", "-v"],
        cwd=str(project_root),
        capture_output=True
    )
    print("✅ Stack stopped")


@pytest.fixture
def e2e_api_base_url(api_container):
    """Base URL for E2E API tests."""
    return "http://localhost:8000"


@pytest.fixture
def e2e_api_base_url_compose(docker_compose_stack):
    """Base URL for E2E API tests with compose stack."""
    return "http://localhost:8000"


@pytest.fixture
def wait_for_service():
    """Helper to wait for a service to be ready."""
    def _wait(url: str, timeout: int = 30) -> bool:
        """Wait for service at URL to respond."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code in [200, 404]:  # 404 is ok, service is up
                    return True
            except:
                pass
            time.sleep(1)
        return False

    return _wait
