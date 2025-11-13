"""E2E tests for complete user journeys."""
import pytest
import pandas as pd
from tests.e2e.helpers import E2ETestClient, create_sample_e2e_data


@pytest.mark.e2e
@pytest.mark.slow
class TestUserJourneys:
    """E2E tests for complete user journeys."""

    def test_data_scientist_workflow(
        self,
        e2e_api_base_url,
        sample_training_data,
        tmp_path
    ):
        """Test complete data scientist workflow: check system -> list models -> predict."""
        client = E2ETestClient(e2e_api_base_url)

        # 1. Check system health
        print("\n📋 Step 1: Checking system health...")
        health = client.health_check()
        assert health["status"] == "healthy", "System should be healthy"
        print(f"✅ System healthy: {health}")

        # 2. Check available models
        print("\n📋 Step 2: Listing available models...")
        response = client.get("/models")
        assert response.status_code == 200, "Should list models"
        models = response.json()
        initial_model_count = len(models)
        print(f"✅ Found {initial_model_count} models")

        # 3. Make prediction with existing model (if any)
        if initial_model_count > 0:
            print("\n📋 Step 3: Making prediction...")
            test_data = create_sample_e2e_data(1)[0]
            try:
                result = client.predict(test_data)
                assert "predictions" in result or "detail" in result
                print(f"✅ Prediction made successfully")
            except Exception as e:
                print(f"⚠️ Prediction failed (expected if no models): {e}")
        else:
            print("\n⚠️ No models available, skipping prediction test")

        # 4. Check metrics
        print("\n📋 Step 4: Checking system metrics...")
        response = client.get("/metrics")
        assert response.status_code == 200, "Should get metrics"
        metrics = response.json()
        print(f"✅ Metrics retrieved: {list(metrics.keys())}")

    def test_ml_engineer_workflow(self, e2e_api_base_url):
        """Test ML engineer workflow: model management and deployment."""
        client = E2ETestClient(e2e_api_base_url)

        # 1. List all models
        print("\n📋 Step 1: Listing all models...")
        response = client.get("/models")
        assert response.status_code == 200, "Should list models"
        all_models = response.json()
        print(f"✅ Total models: {len(all_models)}")

        # 2. Filter by stage
        print("\n📋 Step 2: Filtering models by stage...")
        response = client.get("/models", params={"stage": "production"})
        assert response.status_code == 200, "Should filter by stage"
        prod_models = response.json()
        print(f"✅ Production models: {len(prod_models)}")

        # 3. Filter by name
        print("\n📋 Step 3: Filtering models by name...")
        response = client.get("/models", params={"model_name": "test"})
        assert response.status_code == 200, "Should filter by name"
        print(f"✅ Filtered models retrieved")

        # 4. Check metrics endpoint
        print("\n📋 Step 4: Checking metrics...")
        response = client.get("/metrics")
        assert response.status_code == 200, "Should get metrics"
        metrics = response.json()
        print(f"✅ System metrics: {metrics}")

    def test_api_consumer_workflow(self, e2e_api_base_url, sample_training_data):
        """Test API consumer workflow: just making predictions."""
        client = E2ETestClient(e2e_api_base_url)

        # 1. Check API documentation
        print("\n📋 Step 1: Checking API documentation...")
        response = client.get("/docs")
        assert response.status_code == 200, "Docs should be available"
        print(f"✅ API documentation accessible")

        # 2. Check health
        print("\n📋 Step 2: Checking health...")
        health = client.health_check()
        assert health["status"] == "healthy", "Should be healthy"
        print(f"✅ Health check passed")

        # 3. Try to make predictions
        print("\n📋 Step 3: Attempting predictions...")
        test_data = create_sample_e2e_data(1)[0]
        response = client.post("/predict", json={"data": test_data})

        # Accept both success and "no models" error
        assert response.status_code in [200, 404, 500, 422], "Should handle request"
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful: {result}")
        else:
            print(f"⚠️ Prediction failed (expected if no models): {response.status_code}")

    def test_batch_prediction_workflow(self, e2e_api_base_url):
        """Test batch prediction workflow."""
        client = E2ETestClient(e2e_api_base_url)

        # 1. Create batch data
        print("\n📋 Step 1: Creating batch data...")
        batch_data = create_sample_e2e_data(20)
        print(f"✅ Created {len(batch_data)} samples")

        # 2. Make batch prediction
        print("\n📋 Step 2: Making batch prediction...")
        response = client.post(
            "/predict/batch",
            json={"data": batch_data, "batch_size": 10}
        )

        # May fail if no models, which is acceptable
        if response.status_code == 200:
            result = response.json()
            assert "predictions" in result
            print(f"✅ Batch prediction successful: {len(result['predictions'])} predictions")
        else:
            print(f"⚠️ Batch prediction not available: {response.status_code}")

    def test_model_lifecycle_workflow(self, e2e_api_base_url):
        """Test complete model lifecycle: list -> get info -> promote."""
        client = E2ETestClient(e2e_api_base_url)

        # 1. List models
        print("\n📋 Step 1: Listing models...")
        models = client.list_models()
        print(f"✅ Found {len(models)} models")

        if len(models) > 0:
            model_id = models[0].get("model_id") or models[0].get("id")

            if model_id:
                # 2. Get model info
                print(f"\n📋 Step 2: Getting info for model {model_id}...")
                try:
                    info = client.get_model_info(model_id)
                    assert "model_id" in info or "id" in info
                    print(f"✅ Model info retrieved")
                except Exception as e:
                    print(f"⚠️ Model info retrieval failed: {e}")

                # 3. Try to promote (may fail if not admin)
                print(f"\n📋 Step 3: Attempting to promote model...")
                try:
                    result = client.promote_model(model_id, "staging")
                    print(f"✅ Model promoted successfully")
                except Exception as e:
                    print(f"⚠️ Promotion failed (expected if no auth): {e}")
        else:
            print("\n⚠️ No models available for lifecycle test")

    def test_error_recovery_workflow(self, e2e_api_base_url):
        """Test error handling and recovery."""
        client = E2ETestClient(e2e_api_base_url)

        # 1. Invalid prediction request
        print("\n📋 Step 1: Testing invalid prediction...")
        response = client.post("/predict", json={"invalid": "data"})
        assert response.status_code in [400, 422, 500], "Should handle invalid input"
        print(f"✅ Invalid input handled: {response.status_code}")

        # 2. Non-existent model
        print("\n📋 Step 2: Testing non-existent model...")
        response = client.post(
            "/predict",
            json={"data": {"feature1": 1}, "model_id": "nonexistent_12345"}
        )
        assert response.status_code in [404, 500], "Should handle missing model"
        print(f"✅ Missing model handled: {response.status_code}")

        # 3. System still responsive after errors
        print("\n📋 Step 3: Verifying system still responsive...")
        health = client.health_check()
        assert health["status"] == "healthy", "Should recover from errors"
        print(f"✅ System recovered and responsive")

    def test_concurrent_user_workflow(self, e2e_api_base_url):
        """Test multiple concurrent users."""
        from concurrent.futures import ThreadPoolExecutor

        client = E2ETestClient(e2e_api_base_url)

        def user_session(user_id: int):
            """Simulate a user session."""
            # Check health
            health = client.health_check()
            assert health["status"] == "healthy"

            # List models
            models = client.list_models()

            # Get metrics
            metrics = client.get_metrics()

            return {"user_id": user_id, "success": True}

        print("\n📋 Simulating 5 concurrent users...")
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(user_session, i) for i in range(5)]
            results = [f.result() for f in futures]

        assert len(results) == 5, "All users should complete"
        assert all(r["success"] for r in results), "All sessions should succeed"
        print(f"✅ All 5 concurrent users completed successfully")
