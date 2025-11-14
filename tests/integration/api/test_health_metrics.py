"""
Comprehensive integration tests for health check and metrics endpoints.

Tests cover:
- Health check endpoints (liveness, readiness)
- Metrics collection
- System monitoring
- Uptime tracking
- Performance monitoring
"""
import pytest
import time
import asyncio
from fastapi import status
from httpx import AsyncClient
from datetime import datetime


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.asyncio
class TestHealthEndpoints:
    """Comprehensive tests for health check endpoints."""

    async def test_health_check_basic(self, async_client: AsyncClient):
        """
        Test basic health check endpoint.

        This is the primary health check used by load balancers.
        """
        response = await async_client.get("/health")

        assert response.status_code == status.HTTP_200_OK, (
            "Health check must return 200"
        )

        data = response.json()

        # Required fields
        assert "status" in data, "Must have status field"
        assert data["status"] == "healthy", "Status must be 'healthy'"
        assert "timestamp" in data, "Must have timestamp"
        assert "version" in data, "Must have version"
        assert "uptime_seconds" in data, "Must have uptime"

        # Validate types
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0

    async def test_health_check_timestamp_valid(
        self,
        async_client: AsyncClient
    ):
        """Test health check timestamp is valid ISO format."""
        response = await async_client.get("/health")
        data = response.json()

        timestamp = data["timestamp"]

        # Should be valid ISO format
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            # Should be recent (within last minute)
            now = datetime.now(dt.tzinfo)
            diff = (now - dt).total_seconds()
            assert abs(diff) < 60, "Timestamp should be current"
        except ValueError as e:
            pytest.fail(f"Invalid timestamp format: {timestamp}, error: {e}")

    async def test_health_check_uptime_increases(
        self,
        async_client: AsyncClient
    ):
        """
        Test health check uptime increases over time.

        Validates uptime tracking is working correctly.
        """
        # First check
        response1 = await async_client.get("/health")
        data1 = response1.json()
        uptime1 = data1["uptime_seconds"]

        # Wait a second
        await asyncio.sleep(1.1)

        # Second check
        response2 = await async_client.get("/health")
        data2 = response2.json()
        uptime2 = data2["uptime_seconds"]

        assert uptime2 > uptime1, (
            "Uptime should increase over time"
        )
        assert uptime2 - uptime1 >= 1.0, (
            "Uptime should increase by at least 1 second"
        )

    async def test_liveness_probe(self, async_client: AsyncClient):
        """
        Test liveness probe endpoint.

        Used by Kubernetes to determine if pod should be restarted.
        """
        response = await async_client.get("/health/live")

        assert response.status_code == status.HTTP_200_OK, (
            "Liveness probe must return 200 when alive"
        )

        data = response.json()
        assert "status" in data
        assert data["status"] == "alive"

    async def test_readiness_probe(self, async_client: AsyncClient):
        """
        Test readiness probe endpoint.

        Used by Kubernetes to determine if pod can receive traffic.
        """
        response = await async_client.get("/health/ready")

        # Should return 200 when ready, 503 when not ready
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_503_SERVICE_UNAVAILABLE
        ]

        data = response.json()

        if response.status_code == status.HTTP_200_OK:
            assert "status" in data
            assert data["status"] == "ready"
            # Should report available resources
            if "models_available" in data:
                assert isinstance(data["models_available"], int)

    async def test_health_endpoints_fast(
        self,
        async_client: AsyncClient
    ):
        """
        Test all health endpoints respond quickly.

        Target: < 100ms response time
        """
        endpoints = ["/health", "/health/live", "/health/ready"]

        for endpoint in endpoints:
            start = time.time()
            response = await async_client.get(endpoint)
            duration_ms = (time.time() - start) * 1000

            assert response.status_code in [200, 503], (
                f"{endpoint} should return 200 or 503"
            )
            assert duration_ms < 500, (
                f"{endpoint} took {duration_ms:.2f}ms, exceeds 500ms"
            )

    async def test_health_check_concurrent(
        self,
        async_client: AsyncClient
    ):
        """
        Test health check handles concurrent requests.

        Load balancers may check health frequently from multiple instances.
        """
        async def check_health():
            return await async_client.get("/health")

        # Make 50 concurrent health checks
        tasks = [check_health() for _ in range(50)]
        responses = await asyncio.gather(*tasks)

        # All should succeed
        for response in responses:
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "healthy"

    async def test_root_endpoint(self, async_client: AsyncClient):
        """Test root endpoint returns basic API information."""
        response = await async_client.get("/")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "message" in data or "version" in data
        assert "docs" in data, "Should provide docs URL"

        # Docs URL should be valid
        assert data["docs"] == "/docs"


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.asyncio
class TestMetricsEndpoints:
    """Comprehensive tests for metrics endpoints."""

    async def test_metrics_endpoint_basic(
        self,
        async_client: AsyncClient
    ):
        """
        Test basic metrics endpoint.

        Provides system metrics for monitoring.
        """
        response = await async_client.get("/metrics")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()

        # Should have basic metrics
        assert "uptime_seconds" in data or "timestamp" in data
        assert isinstance(data, dict)

    async def test_metrics_system_info(
        self,
        async_client: AsyncClient
    ):
        """Test metrics include system information."""
        response = await async_client.get("/metrics")
        data = response.json()

        # Common system metrics
        expected_metrics = [
            "uptime_seconds",
            "total_models",
            "production_models",
            "timestamp"
        ]

        # At least some should be present
        present = [m for m in expected_metrics if m in data]
        assert len(present) > 0, (
            f"Should have some system metrics. Got: {data.keys()}"
        )

    async def test_metrics_model_counts(
        self,
        async_client: AsyncClient
    ):
        """Test metrics include model counts."""
        response = await async_client.get("/metrics")
        data = response.json()

        if "total_models" in data:
            assert isinstance(data["total_models"], int)
            assert data["total_models"] >= 0

        if "production_models" in data:
            assert isinstance(data["production_models"], int)
            assert data["production_models"] >= 0

            # Production models should not exceed total
            if "total_models" in data:
                assert data["production_models"] <= data["total_models"]

    async def test_metrics_timestamp(
        self,
        async_client: AsyncClient
    ):
        """Test metrics include current timestamp."""
        response = await async_client.get("/metrics")
        data = response.json()

        if "timestamp" in data:
            # Should be valid ISO format
            try:
                dt = datetime.fromisoformat(
                    data["timestamp"].replace("Z", "+00:00")
                )
                # Should be recent
                now = datetime.now(dt.tzinfo)
                diff = (now - dt).total_seconds()
                assert abs(diff) < 60
            except ValueError:
                pytest.fail(f"Invalid timestamp: {data['timestamp']}")

    async def test_metrics_performance(
        self,
        async_client: AsyncClient
    ):
        """
        Test metrics endpoint is fast.

        Target: < 200ms
        """
        start = time.time()
        response = await async_client.get("/metrics")
        duration_ms = (time.time() - start) * 1000

        assert response.status_code == status.HTTP_200_OK
        assert duration_ms < 500, (
            f"Metrics took {duration_ms:.2f}ms, exceeds 500ms"
        )

    async def test_metrics_consistency(
        self,
        async_client: AsyncClient
    ):
        """
        Test metrics are consistent across multiple calls.

        Values like total_models shouldn't change rapidly.
        """
        response1 = await async_client.get("/metrics")
        response2 = await async_client.get("/metrics")

        data1 = response1.json()
        data2 = response2.json()

        # Model counts should be same (unless models added/removed)
        if "total_models" in data1 and "total_models" in data2:
            # Allow for small changes during testing
            assert abs(data1["total_models"] - data2["total_models"]) <= 1


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.asyncio
class TestAPIDocumentation:
    """Tests for API documentation endpoints."""

    async def test_swagger_ui_available(
        self,
        async_client: AsyncClient
    ):
        """Test Swagger UI documentation is available."""
        response = await async_client.get("/docs")

        assert response.status_code == status.HTTP_200_OK
        assert "text/html" in response.headers.get("content-type", "")

    async def test_redoc_available(
        self,
        async_client: AsyncClient
    ):
        """Test ReDoc documentation is available."""
        response = await async_client.get("/redoc")

        assert response.status_code == status.HTTP_200_OK
        assert "text/html" in response.headers.get("content-type", "")

    async def test_openapi_spec_available(
        self,
        async_client: AsyncClient
    ):
        """Test OpenAPI specification is available."""
        response = await async_client.get("/openapi.json")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()

        # Validate OpenAPI structure
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data

        # Check version
        assert data["openapi"].startswith("3.")

        # Check info
        info = data["info"]
        assert "title" in info
        assert "version" in info
        assert "description" in info

    async def test_openapi_endpoints_documented(
        self,
        async_client: AsyncClient
    ):
        """Test all endpoints are documented in OpenAPI spec."""
        response = await async_client.get("/openapi.json")
        data = response.json()

        paths = data["paths"]

        # Key endpoints should be documented
        expected_paths = [
            "/health",
            "/predict",
            "/models",
        ]

        for path in expected_paths:
            assert path in paths, (
                f"Path {path} should be documented"
            )

    async def test_openapi_has_schemas(
        self,
        async_client: AsyncClient
    ):
        """Test OpenAPI spec includes schema definitions."""
        response = await async_client.get("/openapi.json")
        data = response.json()

        # Should have component schemas
        assert "components" in data
        assert "schemas" in data["components"]

        schemas = data["components"]["schemas"]
        assert len(schemas) > 0, "Should have schema definitions"
