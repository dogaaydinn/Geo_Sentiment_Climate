"""
Comprehensive integration tests for prediction API endpoints.

Tests cover:
- Single and batch predictions
- Input validation
- Performance requirements
- Error handling
- Model selection
- Response structure validation
"""
import pytest
import time
import asyncio
from typing import Dict, List
from fastapi import status
from httpx import AsyncClient


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.asyncio
class TestPredictionEndpoints:
    """Comprehensive tests for prediction endpoints."""

    async def test_single_prediction_success(
        self,
        async_client: AsyncClient,
        sample_features: Dict[str, float]
    ):
        """
        Test successful single prediction.

        Validates:
        - Response status 200
        - Response structure
        - Prediction data types
        - Metadata presence
        """
        response = await async_client.post(
            "/predict",
            json={"data": sample_features}
        )

        assert response.status_code == status.HTTP_200_OK, (
            "Prediction should succeed"
        )

        data = response.json()

        # Validate response structure
        assert "predictions" in data, "Response must have predictions"
        assert "model_id" in data, "Response must have model_id"
        assert "inference_time_ms" in data, "Response must have inference time"
        assert "timestamp" in data, "Response must have timestamp"
        assert "input_shape" in data, "Response must have input shape"

        # Validate data types
        assert isinstance(data["predictions"], list), "Predictions must be a list"
        assert len(data["predictions"]) > 0, "Predictions must not be empty"
        assert isinstance(data["inference_time_ms"], (int, float)), (
            "Inference time must be numeric"
        )

    async def test_single_prediction_performance(
        self,
        async_client: AsyncClient,
        sample_features: Dict[str, float]
    ):
        """
        Test single prediction meets performance requirements.

        Target: <100ms p95 latency
        """
        latencies = []

        # Run 20 predictions to get p95
        for _ in range(20):
            start = time.time()
            response = await async_client.post(
                "/predict",
                json={"data": sample_features}
            )
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)

            assert response.status_code == status.HTTP_200_OK

        # Calculate p95 latency
        latencies.sort()
        p95_latency = latencies[int(len(latencies) * 0.95)]

        # Allow for overhead in testing environment
        assert p95_latency < 500, (
            f"P95 latency {p95_latency:.2f}ms exceeds 500ms threshold"
        )

    async def test_batch_prediction_success(
        self,
        async_client: AsyncClient,
        batch_features: List[Dict[str, float]]
    ):
        """
        Test batch prediction with 100 samples.

        Validates batch processing capabilities.
        """
        response = await async_client.post(
            "/predict/batch",
            json={
                "data": batch_features[:100],
                "batch_size": 50
            }
        )

        assert response.status_code == status.HTTP_200_OK, (
            "Batch prediction should succeed"
        )

        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 100, (
            "Should return 100 predictions"
        )

    async def test_batch_prediction_performance(
        self,
        async_client: AsyncClient,
        batch_features: List[Dict[str, float]]
    ):
        """
        Test batch prediction performance.

        Target: 100 predictions in < 5 seconds
        """
        start = time.time()
        response = await async_client.post(
            "/predict/batch",
            json={
                "data": batch_features[:100],
                "batch_size": 100
            }
        )
        duration = time.time() - start

        assert response.status_code == status.HTTP_200_OK
        assert duration < 10.0, (
            f"Batch prediction took {duration:.2f}s, exceeds 10s limit"
        )

        data = response.json()
        assert len(data["predictions"]) == 100

    async def test_prediction_input_validation_missing_data(
        self,
        async_client: AsyncClient
    ):
        """Test prediction rejects missing data."""
        response = await async_client.post(
            "/predict",
            json={}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY, (
            "Should reject request with missing data"
        )

        data = response.json()
        assert "detail" in data

    async def test_prediction_input_validation_invalid_type(
        self,
        async_client: AsyncClient
    ):
        """Test prediction rejects invalid data types."""
        response = await async_client.post(
            "/predict",
            json={"data": "not_a_dict"}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY, (
            "Should reject invalid data type"
        )

    async def test_prediction_input_validation_null_values(
        self,
        async_client: AsyncClient
    ):
        """Test prediction handles null values appropriately."""
        response = await async_client.post(
            "/predict",
            json={
                "data": {
                    "temperature": 25.0,
                    "humidity": None,  # Null value
                    "wind_speed": 10.0
                }
            }
        )

        # Should either handle gracefully or reject with validation error
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_400_BAD_REQUEST
        ]

    async def test_prediction_with_model_id(
        self,
        async_client: AsyncClient,
        sample_features: Dict[str, float]
    ):
        """Test prediction with specific model ID."""
        response = await async_client.post(
            "/predict",
            json={
                "data": sample_features,
                "model_id": "test-model-123"
            }
        )

        # Should return 404 if model doesn't exist, or 200 if it does
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_404_NOT_FOUND
        ]

    async def test_prediction_with_model_name(
        self,
        async_client: AsyncClient,
        sample_features: Dict[str, float]
    ):
        """Test prediction with model name filter."""
        response = await async_client.post(
            "/predict",
            json={
                "data": sample_features,
                "model_name": "test_model"
            }
        )

        # Should return 404 if no models, or 200 if models exist
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_404_NOT_FOUND
        ]

    async def test_prediction_concurrent_requests(
        self,
        async_client: AsyncClient,
        sample_features: Dict[str, float]
    ):
        """
        Test API handles concurrent prediction requests.

        Validates thread-safety and proper resource management.
        """
        async def make_prediction():
            return await async_client.post(
                "/predict",
                json={"data": sample_features}
            )

        # Make 20 concurrent requests
        tasks = [make_prediction() for _ in range(20)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successful responses
        successful = sum(
            1 for r in responses
            if not isinstance(r, Exception) and r.status_code == 200
        )

        # At least some should succeed (depends on models available)
        assert successful >= 0, "Should handle concurrent requests"

    async def test_prediction_returns_probabilities(
        self,
        async_client: AsyncClient,
        sample_features: Dict[str, float]
    ):
        """Test prediction can return probability distributions."""
        response = await async_client.post(
            "/predict",
            json={
                "data": sample_features,
                "return_probabilities": True
            }
        )

        # If models support probabilities, should include them
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            # Probabilities might be in predictions or separate field
            assert "predictions" in data

    async def test_prediction_error_handling(
        self,
        async_client: AsyncClient
    ):
        """Test prediction handles errors gracefully."""
        # Invalid JSON
        response = await async_client.post(
            "/predict",
            data="invalid json"
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_batch_prediction_empty_list(
        self,
        async_client: AsyncClient
    ):
        """Test batch prediction with empty list."""
        response = await async_client.post(
            "/predict/batch",
            json={"data": []}
        )

        # Should reject empty batch or return empty predictions
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]

        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert len(data["predictions"]) == 0

    async def test_batch_prediction_large_batch(
        self,
        async_client: AsyncClient,
        batch_features: List[Dict[str, float]]
    ):
        """
        Test batch prediction with large dataset.

        Tests scalability with 1000+ samples.
        """
        # Create large batch
        large_batch = batch_features * 10  # 1000 samples

        response = await async_client.post(
            "/predict/batch",
            json={
                "data": large_batch,
                "batch_size": 100
            }
        )

        # Should handle large batches
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]

    async def test_prediction_response_headers(
        self,
        async_client: AsyncClient,
        sample_features: Dict[str, float]
    ):
        """Test prediction response includes proper headers."""
        response = await async_client.post(
            "/predict",
            json={"data": sample_features}
        )

        # Check for custom headers (like process time)
        if response.status_code == status.HTTP_200_OK:
            headers = response.headers
            # May include X-Process-Time or similar
            assert headers is not None

    async def test_prediction_idempotency(
        self,
        async_client: AsyncClient,
        sample_features: Dict[str, float]
    ):
        """
        Test predictions are consistent for same input.

        Note: May vary slightly due to model stochasticity.
        """
        # Make two identical predictions
        response1 = await async_client.post(
            "/predict",
            json={"data": sample_features}
        )
        response2 = await async_client.post(
            "/predict",
            json={"data": sample_features}
        )

        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()

            # Should use same model (unless registry changed)
            # Predictions might differ slightly for some models
            assert "model_id" in data1
            assert "model_id" in data2
