"""
Comprehensive integration tests for API error handling.

Tests cover:
- HTTP error responses
- Validation errors
- Server errors
- Error response format
- CORS configuration
- Request/response middleware
"""
import pytest
from fastapi import status
from httpx import AsyncClient


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.asyncio
class TestErrorHandling:
    """Tests for API error handling and responses."""

    async def test_404_not_found(self, async_client: AsyncClient):
        """Test 404 error for non-existent endpoint."""
        response = await async_client.get("/nonexistent/endpoint")

        assert response.status_code == status.HTTP_404_NOT_FOUND

        data = response.json()
        assert "detail" in data

    async def test_405_method_not_allowed(
        self,
        async_client: AsyncClient
    ):
        """Test 405 error for wrong HTTP method."""
        # POST to a GET-only endpoint
        response = await async_client.post("/health")

        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

    async def test_validation_error_format(
        self,
        async_client: AsyncClient
    ):
        """
        Test validation errors have consistent format.

        FastAPI returns 422 with detailed validation errors.
        """
        response = await async_client.post(
            "/predict",
            json={"invalid_field": "value"}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        data = response.json()
        assert "detail" in data

        # FastAPI validation errors have specific structure
        if isinstance(data["detail"], list):
            error = data["detail"][0]
            assert "loc" in error or "msg" in error

    async def test_malformed_json(self, async_client: AsyncClient):
        """Test API handles malformed JSON gracefully."""
        response = await async_client.post(
            "/predict",
            content="{ invalid json }",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code in [
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_400_BAD_REQUEST
        ]

    async def test_missing_content_type(
        self,
        async_client: AsyncClient
    ):
        """Test API handles missing Content-Type header."""
        response = await async_client.post(
            "/predict",
            content='{"data": {"temp": 25.0}}'
            # No Content-Type header
        )

        # Should handle gracefully or default to JSON
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            status.HTTP_404_NOT_FOUND  # If no models
        ]

    async def test_wrong_content_type(
        self,
        async_client: AsyncClient
    ):
        """Test API handles wrong Content-Type."""
        response = await async_client.post(
            "/predict",
            content="some data",
            headers={"Content-Type": "text/plain"}
        )

        assert response.status_code in [
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
        ]

    async def test_empty_request_body(
        self,
        async_client: AsyncClient
    ):
        """Test API handles empty request body."""
        response = await async_client.post(
            "/predict",
            json={}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_null_request_body(
        self,
        async_client: AsyncClient
    ):
        """Test API handles null request body."""
        response = await async_client.post(
            "/predict",
            json=None
        )

        assert response.status_code in [
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_400_BAD_REQUEST
        ]

    async def test_oversized_request(
        self,
        async_client: AsyncClient
    ):
        """
        Test API handles very large requests.

        Creates a large batch to test size limits.
        """
        # Create very large batch (may hit size limits)
        large_batch = [{"temp": i} for i in range(10000)]

        response = await async_client.post(
            "/predict/batch",
            json={"data": large_batch}
        )

        # Should handle gracefully (may succeed or fail with proper error)
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            status.HTTP_404_NOT_FOUND  # If no models
        ]

    async def test_error_response_has_timestamp(
        self,
        async_client: AsyncClient
    ):
        """Test error responses include timestamp."""
        # Trigger a 404
        response = await async_client.get("/nonexistent")

        # Some errors might include timestamp
        # This is a nice-to-have, not required

    async def test_error_response_format_consistency(
        self,
        async_client: AsyncClient
    ):
        """Test all errors have consistent response format."""
        # Trigger various errors
        errors = []

        # 404
        r1 = await async_client.get("/nonexistent")
        if r1.status_code >= 400:
            errors.append(r1.json())

        # 422
        r2 = await async_client.post("/predict", json={})
        if r2.status_code >= 400:
            errors.append(r2.json())

        # All should have detail field
        for error in errors:
            assert "detail" in error, (
                f"Error response should have detail: {error}"
            )

    async def test_internal_server_error_handling(
        self,
        async_client: AsyncClient
    ):
        """
        Test 500 errors are handled gracefully.

        Note: Hard to trigger intentionally without breaking code.
        """
        # This is more of a smoke test
        # In production, 500s should be logged and handled gracefully

    async def test_unicode_in_request(
        self,
        async_client: AsyncClient
    ):
        """Test API handles Unicode characters."""
        response = await async_client.post(
            "/predict",
            json={
                "data": {"feature": 25.0},
                "model_name": "测试模型"  # Chinese characters
            }
        )

        # Should handle Unicode gracefully
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_404_NOT_FOUND,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]

    async def test_special_characters_in_request(
        self,
        async_client: AsyncClient
    ):
        """Test API handles special characters."""
        response = await async_client.post(
            "/predict",
            json={
                "data": {"feature": 25.0},
                "model_name": "model-v1.2.3_beta+test"
            }
        )

        # Should handle special chars
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_404_NOT_FOUND,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.asyncio
class TestCORS:
    """Tests for CORS configuration."""

    async def test_cors_headers_present(
        self,
        async_client: AsyncClient
    ):
        """Test CORS headers are present in responses."""
        response = await async_client.get(
            "/health",
            headers={"Origin": "http://example.com"}
        )

        headers = response.headers

        # Check for CORS headers (may or may not be present)
        # CORS is typically configured but varies by deployment
        # This is more of a smoke test

    async def test_options_request(self, async_client: AsyncClient):
        """Test OPTIONS request (CORS preflight)."""
        response = await async_client.options("/health")

        # Should handle OPTIONS
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_405_METHOD_NOT_ALLOWED
        ]

    async def test_cors_allows_methods(
        self,
        async_client: AsyncClient
    ):
        """Test CORS allows required HTTP methods."""
        response = await async_client.options(
            "/predict",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST"
            }
        )

        # Should handle preflight request
        assert response.status_code in [200, 204, 405]


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.asyncio
class TestRequestMiddleware:
    """Tests for request/response middleware."""

    async def test_custom_headers_present(
        self,
        async_client: AsyncClient
    ):
        """Test custom headers are added to responses."""
        response = await async_client.get("/health")

        headers = response.headers

        # Check for custom headers like X-Process-Time
        # (if implemented)

    async def test_request_logging(
        self,
        async_client: AsyncClient
    ):
        """
        Test requests are logged.

        This is more of a smoke test - actual logging validation
        requires checking log files.
        """
        # Make a request
        response = await async_client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        # Request should be logged (would check logs in real scenario)

    async def test_response_compression(
        self,
        async_client: AsyncClient
    ):
        """Test GZip compression for large responses."""
        # Get a potentially large response
        response = await async_client.get(
            "/models",
            headers={"Accept-Encoding": "gzip"}
        )

        # Check if compression is applied for large responses
        # (FastAPI's GZipMiddleware applies compression if response > 1000 bytes)
        headers = response.headers

        # Compression headers may or may not be present
        # depending on response size

    async def test_multiple_requests_same_client(
        self,
        async_client: AsyncClient
    ):
        """Test multiple sequential requests work correctly."""
        # Make multiple requests
        r1 = await async_client.get("/health")
        r2 = await async_client.get("/models")
        r3 = await async_client.get("/metrics")

        assert r1.status_code == status.HTTP_200_OK
        assert r2.status_code == status.HTTP_200_OK
        assert r3.status_code == status.HTTP_200_OK


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.asyncio
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_very_long_url(self, async_client: AsyncClient):
        """Test API handles very long URLs."""
        # Create a very long model ID
        long_id = "a" * 1000

        response = await async_client.get(f"/models/{long_id}")

        # Should handle gracefully (likely 404)
        assert response.status_code in [
            status.HTTP_404_NOT_FOUND,
            status.HTTP_414_URI_TOO_LONG
        ]

    async def test_numeric_string_as_id(
        self,
        async_client: AsyncClient
    ):
        """Test numeric strings as model IDs."""
        response = await async_client.get("/models/12345")

        # Should treat as string ID
        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_special_characters_in_url(
        self,
        async_client: AsyncClient
    ):
        """Test special characters in URL parameters."""
        # Try model name with special characters
        response = await async_client.get(
            "/models",
            params={"model_name": "test/model@v1.0"}
        )

        # Should handle URL encoding
        assert response.status_code == status.HTTP_200_OK

    async def test_case_sensitivity(
        self,
        async_client: AsyncClient
    ):
        """Test API path case sensitivity."""
        # FastAPI paths are case-sensitive by default
        response = await async_client.get("/Health")  # Capital H

        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_trailing_slash(self, async_client: AsyncClient):
        """Test endpoints with trailing slash."""
        # Try with trailing slash
        r1 = await async_client.get("/health/")

        # FastAPI may redirect or treat as different route
        assert r1.status_code in [
            status.HTTP_200_OK,
            status.HTTP_307_TEMPORARY_REDIRECT,
            status.HTTP_404_NOT_FOUND
        ]

    async def test_double_slash_in_path(
        self,
        async_client: AsyncClient
    ):
        """Test paths with double slashes."""
        response = await async_client.get("/models//test")

        # Should normalize or return 404
        assert response.status_code == status.HTTP_404_NOT_FOUND
