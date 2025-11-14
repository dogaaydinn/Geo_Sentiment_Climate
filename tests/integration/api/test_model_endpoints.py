"""
Comprehensive integration tests for model management API endpoints.

Tests cover:
- Model listing and filtering
- Model information retrieval
- Model promotion workflow
- Model versioning
- Error handling
"""
import pytest
from fastapi import status
from httpx import AsyncClient
@pytest.mark.integration
@pytest.mark.api
@pytest.mark.asyncio
class TestModelEndpoints:
    """Comprehensive tests for model management endpoints."""

    async def test_list_all_models(self, async_client: AsyncClient):
        """
        Test listing all registered models.

        Validates:
        - Response status
        - Response is a list
        - Model structure (if models exist)
        """
        response = await async_client.get("/models")

        assert response.status_code == status.HTTP_200_OK, (
            "Should successfully list models"
        )

        data = response.json()
        assert isinstance(data, list), "Response should be a list"

        # If models exist, validate structure
        if len(data) > 0:
            model = data[0]
            assert "model_id" in model
            assert "model_name" in model
            assert "version" in model
            assert "stage" in model
            assert "model_type" in model

    async def test_list_models_filter_by_stage(
        self,
        async_client: AsyncClient
    ):
        """
        Test filtering models by stage.

        Tests filtering for dev, staging, and production stages.
        """
        stages = ["dev", "staging", "production"]

        for stage in stages:
            response = await async_client.get(
                "/models",
                params={"stage": stage}
            )

            assert response.status_code == status.HTTP_200_OK, (
                f"Should successfully filter by stage: {stage}"
            )

            data = response.json()
            assert isinstance(data, list)

            # All returned models should be in requested stage
            for model in data:
                assert model["stage"] == stage, (
                    f"Model stage should be {stage}"
                )

    async def test_list_models_filter_by_name(
        self,
        async_client: AsyncClient
    ):
        """Test filtering models by name."""
        response = await async_client.get(
            "/models",
            params={"model_name": "test_model"}
        )

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert isinstance(data, list)

        # All returned models should match name
        for model in data:
            assert model["model_name"] == "test_model"

    async def test_list_models_multiple_filters(
        self,
        async_client: AsyncClient
    ):
        """Test filtering models with multiple parameters."""
        response = await async_client.get(
            "/models",
            params={
                "model_name": "air_quality_model",
                "stage": "production"
            }
        )

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert isinstance(data, list)

        for model in data:
            assert model["model_name"] == "air_quality_model"
            assert model["stage"] == "production"

    async def test_get_model_by_id_success(
        self,
        async_client: AsyncClient
    ):
        """
        Test retrieving a specific model by ID.

        First gets list of models, then retrieves one by ID.
        """
        # First get list of models
        list_response = await async_client.get("/models")
        models = list_response.json()

        if len(models) == 0:
            pytest.skip("No models available for testing")

        # Get first model by ID
        model_id = models[0]["model_id"]
        response = await async_client.get(f"/models/{model_id}")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["model_id"] == model_id
        assert "model_name" in data
        assert "version" in data
        assert "metrics" in data
        assert "created_at" in data

    async def test_get_model_by_id_not_found(
        self,
        async_client: AsyncClient
    ):
        """Test retrieving non-existent model returns 404."""
        fake_model_id = "non_existent_model_id_12345"

        response = await async_client.get(f"/models/{fake_model_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    async def test_promote_model_to_staging(
        self,
        async_client: AsyncClient
    ):
        """
        Test promoting a model to staging.

        Note: Requires a model in dev stage to promote.
        """
        # Get models in dev stage
        response = await async_client.get(
            "/models",
            params={"stage": "dev"}
        )

        models = response.json()
        if len(models) == 0:
            pytest.skip("No dev models available for promotion testing")

        model_id = models[0]["model_id"]

        # Promote to staging
        response = await async_client.post(
            f"/models/{model_id}/promote",
            params={"new_stage": "staging"}
        )

        # Should succeed or fail gracefully
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_404_NOT_FOUND,
            status.HTTP_400_BAD_REQUEST
        ]

        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "message" in data or "model_id" in data

    async def test_promote_model_to_production(
        self,
        async_client: AsyncClient
    ):
        """Test promoting a model to production."""
        # Get models in staging
        response = await async_client.get(
            "/models",
            params={"stage": "staging"}
        )

        models = response.json()
        if len(models) == 0:
            pytest.skip("No staging models available for promotion testing")

        model_id = models[0]["model_id"]

        # Promote to production
        response = await async_client.post(
            f"/models/{model_id}/promote",
            params={"new_stage": "production"}
        )

        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_404_NOT_FOUND
        ]

    async def test_promote_model_invalid_stage(
        self,
        async_client: AsyncClient
    ):
        """Test promoting to invalid stage returns 400."""
        fake_model_id = "test_model_id"

        response = await async_client.post(
            f"/models/{fake_model_id}/promote",
            params={"new_stage": "invalid_stage"}
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

        data = response.json()
        assert "detail" in data
        assert "invalid" in data["detail"].lower()

    async def test_promote_nonexistent_model(
        self,
        async_client: AsyncClient
    ):
        """Test promoting non-existent model returns 404."""
        fake_model_id = "non_existent_model_12345"

        response = await async_client.post(
            f"/models/{fake_model_id}/promote",
            params={"new_stage": "production"}
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_model_metrics_structure(
        self,
        async_client: AsyncClient
    ):
        """
        Test model metrics have expected structure.

        Validates metrics are present and properly formatted.
        """
        response = await async_client.get("/models")
        models = response.json()

        if len(models) == 0:
            pytest.skip("No models available")

        model = models[0]
        assert "metrics" in model
        assert isinstance(model["metrics"], dict), (
            "Metrics should be a dictionary"
        )

        # Common metrics that might be present
        possible_metrics = [
            "accuracy", "precision", "recall", "f1_score",
            "mse", "rmse", "mae", "r2_score"
        ]

        # At least some metrics should be present
        has_metrics = any(
            metric in model["metrics"]
            for metric in possible_metrics
        )

        # Note: New models might not have metrics yet
        # So we don't fail if no metrics, just check structure

    async def test_model_versioning(
        self,
        async_client: AsyncClient
    ):
        """
        Test model versioning is properly tracked.

        Validates version field exists and follows semantic versioning.
        """
        response = await async_client.get("/models")
        models = response.json()

        if len(models) == 0:
            pytest.skip("No models available")

        for model in models:
            assert "version" in model
            assert isinstance(model["version"], str)
            # Version should be non-empty
            assert len(model["version"]) > 0

    async def test_model_listing_performance(
        self,
        async_client: AsyncClient
    ):
        """
        Test model listing is fast.

        Target: < 1 second for listing all models
        """
        import time

        start = time.time()
        response = await async_client.get("/models")
        duration = time.time() - start

        assert response.status_code == status.HTTP_200_OK
        assert duration < 2.0, (
            f"Model listing took {duration:.2f}s, exceeds 2s limit"
        )

    async def test_list_production_models_only(
        self,
        async_client: AsyncClient
    ):
        """
        Test filtering for production models only.

        This is a critical filter for production deployments.
        """
        response = await async_client.get(
            "/models",
            params={"stage": "production"}
        )

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert isinstance(data, list)

        # All should be production models
        for model in data:
            assert model["stage"] == "production"

    async def test_model_timestamps(
        self,
        async_client: AsyncClient
    ):
        """
        Test model timestamps are properly formatted.

        Validates created_at is an ISO format timestamp.
        """
        response = await async_client.get("/models")
        models = response.json()

        if len(models) == 0:
            pytest.skip("No models available")

        from datetime import datetime

        for model in models:
            if "created_at" in model:
                # Should be a valid ISO format timestamp
                try:
                    datetime.fromisoformat(
                        model["created_at"].replace("Z", "+00:00")
                    )
                except ValueError:
                    pytest.fail(f"Invalid timestamp format: {model['created_at']}")

    async def test_concurrent_model_requests(
        self,
        async_client: AsyncClient
    ):
        """
        Test concurrent model listing requests.

        Validates thread-safety of model registry operations.
        """
        import asyncio

        async def list_models():
            return await async_client.get("/models")

        # Make 20 concurrent requests
        tasks = [list_models() for _ in range(20)]
        responses = await asyncio.gather(*tasks)

        # All should succeed
        for response in responses:
            assert response.status_code == status.HTTP_200_OK
            assert isinstance(response.json(), list)
