"""Integration tests for model registry."""
import pytest
import joblib
from pathlib import Path
from datetime import datetime


@pytest.mark.integration
@pytest.mark.ml
class TestModelRegistryIntegration:
    """Integration tests for model registry."""

    def test_register_and_retrieve_model(self, temp_model_dir, tmp_path):
        """Test registering and retrieving a model."""
        from source.ml.model_registry import ModelRegistry
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.datasets import make_regression

        # Train a simple model
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Save model temporarily
        model_path = tmp_path / "test_model.joblib"
        joblib.dump(model, model_path)

        # Initialize registry
        registry = ModelRegistry(registry_dir=temp_model_dir)

        # Register model
        model_id = registry.register_model(
            model_path=model_path,
            model_name="test_rf_model",
            model_type="randomforest",
            task_type="regression",
            metrics={"r2": 0.85, "rmse": 10.5},
            hyperparameters={"n_estimators": 10, "max_depth": 5},
            feature_columns=["feature_1", "feature_2"],
            tags=["test", "rf"],
            description="Test random forest model",
            stage="dev"
        )

        assert model_id is not None, "Should return model ID"
        assert model_id in registry.models, "Model should be in registry"

        # Retrieve model
        loaded_model = registry.get_model(model_id)
        assert loaded_model is not None, "Should load model"

        # Make predictions with loaded model
        predictions = loaded_model.predict(X[:10])
        assert len(predictions) == 10, "Should make predictions"

    def test_model_versioning(self, temp_model_dir, tmp_path):
        """Test model versioning system."""
        from source.ml.model_registry import ModelRegistry
        from sklearn.ensemble import RandomForestRegressor

        registry = ModelRegistry(registry_dir=temp_model_dir)

        # Register multiple versions
        for version in range(1, 4):
            model = RandomForestRegressor(n_estimators=version * 10, random_state=42)
            model_path = tmp_path / f"model_v{version}.joblib"
            joblib.dump(model, model_path)

            model_id = registry.register_model(
                model_path=model_path,
                model_name="versioned_model",
                model_type="randomforest",
                task_type="regression",
                metrics={"r2": 0.8 + version * 0.02},
                hyperparameters={"n_estimators": version * 10},
                stage="dev"
            )

            assert model_id is not None, f"Version {version} should register"

        # List all versions
        models = registry.list_models(model_name="versioned_model")
        assert len(models) == 3, "Should have 3 versions"

        # Verify versions are different
        versions = [m.version for m in models]
        assert len(set(versions)) == 3, "Versions should be unique"

    def test_model_promotion_workflow(self, temp_model_dir, tmp_path):
        """Test model promotion through stages."""
        from source.ml.model_registry import ModelRegistry
        from sklearn.ensemble import RandomForestRegressor

        registry = ModelRegistry(registry_dir=temp_model_dir)

        # Register model in dev
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model_path = tmp_path / "model.joblib"
        joblib.dump(model, model_path)

        model_id = registry.register_model(
            model_path=model_path,
            model_name="promotion_test",
            model_type="randomforest",
            task_type="regression",
            metrics={"r2": 0.9},
            hyperparameters={},
            stage="dev"
        )

        # Verify in dev
        assert registry.models[model_id].stage == "dev", "Should start in dev"

        # Promote to staging
        success = registry.promote_model(model_id, "staging")
        assert success, "Promotion to staging should succeed"
        assert registry.models[model_id].stage == "staging", "Should be in staging"

        # Promote to production
        success = registry.promote_model(model_id, "production")
        assert success, "Promotion to production should succeed"
        assert registry.models[model_id].stage == "production", "Should be in production"

        # Verify only production models
        prod_models = registry.list_models(stage="production")
        assert len(prod_models) == 1, "Should have one production model"
        assert prod_models[0].model_id == model_id, "Should be the right model"

    def test_list_models_with_filters(self, temp_model_dir, tmp_path):
        """Test listing models with various filters."""
        from source.ml.model_registry import ModelRegistry
        from sklearn.ensemble import RandomForestRegressor

        registry = ModelRegistry(registry_dir=temp_model_dir)

        # Register multiple models
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model_path = tmp_path / "model1.joblib"
        joblib.dump(model, model_path)

        # Model 1: XGBoost dev
        registry.register_model(
            model_path=model_path,
            model_name="model_a",
            model_type="xgboost",
            task_type="regression",
            metrics={},
            hyperparameters={},
            stage="dev"
        )

        # Model 2: XGBoost production
        registry.register_model(
            model_path=model_path,
            model_name="model_a",
            model_type="xgboost",
            task_type="regression",
            metrics={},
            hyperparameters={},
            stage="production"
        )

        # Model 3: LightGBM dev
        registry.register_model(
            model_path=model_path,
            model_name="model_b",
            model_type="lightgbm",
            task_type="regression",
            metrics={},
            hyperparameters={},
            stage="dev"
        )

        # Test filters
        all_models = registry.list_models()
        assert len(all_models) >= 3, "Should have at least 3 models"

        model_a_models = registry.list_models(model_name="model_a")
        assert len(model_a_models) == 2, "Should have 2 model_a models"

        prod_models = registry.list_models(stage="production")
        assert len(prod_models) == 1, "Should have 1 production model"

        dev_models = registry.list_models(stage="dev")
        assert len(dev_models) >= 2, "Should have at least 2 dev models"

    def test_get_latest_model(self, temp_model_dir, tmp_path):
        """Test getting the latest model."""
        from source.ml.model_registry import ModelRegistry
        from sklearn.ensemble import RandomForestRegressor
        import time

        registry = ModelRegistry(registry_dir=temp_model_dir)

        model = RandomForestRegressor(n_estimators=10, random_state=42)

        # Register models with time delay
        for i in range(3):
            model_path = tmp_path / f"model_{i}.joblib"
            joblib.dump(model, model_path)

            registry.register_model(
                model_path=model_path,
                model_name="latest_test",
                model_type="randomforest",
                task_type="regression",
                metrics={"iteration": i},
                hyperparameters={},
                stage="dev"
            )

            time.sleep(0.1)  # Small delay to ensure different timestamps

        # Get latest
        latest = registry.get_latest_model(model_name="latest_test")
        assert latest is not None, "Should get latest model"
        # Latest should have highest iteration
        assert latest.metrics.get("iteration") == 2, "Should be the last registered model"

    def test_model_metadata_persistence(self, temp_model_dir, tmp_path):
        """Test that model metadata persists."""
        from source.ml.model_registry import ModelRegistry
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model_path = tmp_path / "model.joblib"
        joblib.dump(model, model_path)

        # Create first registry and register model
        registry1 = ModelRegistry(registry_dir=temp_model_dir)
        model_id = registry1.register_model(
            model_path=model_path,
            model_name="persist_test",
            model_type="randomforest",
            task_type="regression",
            metrics={"r2": 0.95},
            hyperparameters={"n_estimators": 10},
            stage="production"
        )

        # Save metadata
        registry1.save_metadata()

        # Create new registry instance (simulating restart)
        registry2 = ModelRegistry(registry_dir=temp_model_dir)
        registry2.load_metadata()

        # Check that model is still there
        assert model_id in registry2.models, "Model should persist"
        assert registry2.models[model_id].stage == "production", "Stage should persist"
        assert registry2.models[model_id].metrics["r2"] == 0.95, "Metrics should persist"
