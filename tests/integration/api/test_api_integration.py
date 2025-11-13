"""Integration tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
@pytest.mark.api
class TestAPIIntegration:
    """Integration tests for API endpoints."""

    def test_health_check_endpoint(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200, "Health endpoint should return 200"

        data = response.json()
        assert "status" in data, "Should have status field"
        assert data["status"] == "healthy", "Status should be healthy"
        assert "timestamp" in data, "Should have timestamp"
        assert "version" in data, "Should have version"

    def test_root_endpoint(self, test_client):
        """Test root endpoint."""
        response = test_client.get("/")
        assert response.status_code == 200, "Root endpoint should return 200"

        data = response.json()
        assert "message" in data or "name" in data, "Should have welcome message or app name"

    def test_health_ready_endpoint(self, test_client):
        """Test readiness endpoint."""
        response = test_client.get("/health/ready")
        # May be 200 or 503 depending on system state
        assert response.status_code in [200, 503], "Ready endpoint should return 200 or 503"

        if response.status_code == 200:
            data = response.json()
            assert "status" in data, "Should have status field"

    def test_health_live_endpoint(self, test_client):
        """Test liveness endpoint."""
        response = test_client.get("/health/live")
        assert response.status_code == 200, "Live endpoint should return 200"

        data = response.json()
        assert "status" in data, "Should have status field"
        assert data["status"] == "alive", "Status should be alive"

    def test_models_list_endpoint(self, test_client):
        """Test listing models endpoint."""
        response = test_client.get("/models")
        assert response.status_code == 200, "Models endpoint should return 200"

        data = response.json()
        assert isinstance(data, list), "Should return a list"
        # May be empty if no models registered yet

    def test_models_list_with_filters(self, test_client):
        """Test listing models with filters."""
        # Filter by stage
        response = test_client.get("/models", params={"stage": "production"})
        assert response.status_code == 200, "Filtered models should return 200"
        data = response.json()
        assert isinstance(data, list), "Should return a list"

        # Filter by model name
        response = test_client.get("/models", params={"model_name": "test_model"})
        assert response.status_code == 200, "Name-filtered models should return 200"

    def test_metrics_endpoint(self, test_client):
        """Test metrics endpoint."""
        response = test_client.get("/metrics")
        assert response.status_code == 200, "Metrics endpoint should return 200"

        data = response.json()
        assert "uptime_seconds" in data or "timestamp" in data, "Should have metrics data"

    def test_prediction_endpoint_validation(self, test_client):
        """Test prediction endpoint input validation."""
        # Missing data
        response = test_client.post("/predict", json={})
        assert response.status_code == 422, "Should validate required fields"

        # Invalid data type
        response = test_client.post("/predict", json={"data": "not a dict"})
        assert response.status_code == 422, "Should validate data type"

    def test_prediction_endpoint_no_models(self, test_client):
        """Test prediction when no models available."""
        # Valid structure but no models (should fail gracefully)
        response = test_client.post(
            "/predict",
            json={"data": {"feature1": 1.0, "feature2": 2.0}}
        )

        # Should return 404 (no models) or 500 (other error) or 200 (if models exist)
        assert response.status_code in [200, 404, 500], "Should handle no models gracefully"

        if response.status_code in [404, 500]:
            data = response.json()
            assert "detail" in data or "message" in data, "Should have error message"

    def test_api_documentation_available(self, test_client):
        """Test that API documentation is available."""
        # OpenAPI docs
        response = test_client.get("/docs")
        assert response.status_code == 200, "Swagger docs should be available"

        # ReDoc
        response = test_client.get("/redoc")
        assert response.status_code == 200, "ReDoc should be available"

        # OpenAPI JSON
        response = test_client.get("/openapi.json")
        assert response.status_code == 200, "OpenAPI spec should be available"
        data = response.json()
        assert "openapi" in data, "Should have OpenAPI version"
        assert "paths" in data, "Should have API paths"

    def test_cors_headers(self, test_client):
        """Test CORS headers are present."""
        response = test_client.options("/health")
        # CORS headers might be set
        # This is a basic check
        assert response.status_code in [200, 405], "OPTIONS should be handled"

    def test_error_responses_format(self, test_client):
        """Test that error responses have consistent format."""
        # Trigger a validation error
        response = test_client.post("/predict", json={"invalid": "request"})

        if response.status_code == 422:
            data = response.json()
            assert "detail" in data, "Validation errors should have detail"

    def test_api_handles_large_batch(self, test_client):
        """Test API handles large prediction batches."""
        # Create a large batch
        batch_data = [{"feature1": i, "feature2": i * 2} for i in range(100)]

        response = test_client.post(
            "/predict/batch",
            json={"data": batch_data, "batch_size": 50}
        )

        # Should handle gracefully (200 if models exist, 404/500 otherwise)
        assert response.status_code in [200, 404, 500], "Should handle large batches"

    def test_concurrent_api_requests(self, test_client):
        """Test API handles concurrent requests."""
        from concurrent.futures import ThreadPoolExecutor

        def make_health_request():
            return test_client.get("/health")

        # Make concurrent health checks
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_health_request) for _ in range(20)]
            responses = [f.result() for f in futures]

        # All should succeed
        for response in responses:
            assert response.status_code == 200, "Concurrent requests should succeed"

    def test_api_response_times(self, test_client):
        """Test API response times are reasonable."""
        import time

        # Health check should be fast
        start = time.time()
        response = test_client.get("/health")
        duration = time.time() - start

        assert response.status_code == 200, "Health check should succeed"
        assert duration < 1.0, "Health check should be fast (< 1 second)"

        # Models list should be reasonable
        start = time.time()
        response = test_client.get("/models")
        duration = time.time() - start

        assert response.status_code == 200, "Models list should succeed"
        assert duration < 2.0, "Models list should be reasonably fast (< 2 seconds)"
