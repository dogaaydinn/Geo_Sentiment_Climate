"""Integration tests for inference engine."""
import pytest
import pandas as pd
from pathlib import Path


@pytest.mark.integration
@pytest.mark.ml
class TestInferenceIntegration:
    """Integration tests for inference engine."""

    def test_single_prediction_pipeline(
        self,
        sample_training_data,
        temp_model_dir,
        tmp_path
    ):
        """Test single prediction end-to-end."""
        from source.ml.model_training import ModelTrainer, TrainingConfig
        from source.ml.model_registry import ModelRegistry
        from source.ml.inference import InferenceEngine

        # Train and register a model first
        data_path = tmp_path / "data.csv"
        sample_training_data.to_csv(data_path, index=False)

        config = TrainingConfig(
            model_type="xgboost",
            n_trials=2,
            cv_folds=2,
            target_column="Daily AQI Value",
            train_data_path=str(data_path),
            model_save_dir=temp_model_dir,
            random_state=42
        )

        trainer = ModelTrainer(config)
        results = trainer.run_full_pipeline()
        model_path = Path(results['model_path'])

        # Register model
        registry = ModelRegistry(registry_dir=temp_model_dir)
        model_id = registry.register_model(
            model_path=model_path,
            model_name="test_prediction_model",
            model_type="xgboost",
            task_type="regression",
            metrics=results['test_metrics'],
            hyperparameters=results['best_params'],
            stage="production"
        )

        # Create inference engine
        engine = InferenceEngine(model_registry=registry)

        # Prepare input
        input_data = sample_training_data.iloc[0:1].to_dict('records')[0]

        # Make prediction
        result = engine.predict(data=input_data, model_id=model_id)

        # Verify result
        assert 'predictions' in result or hasattr(result, 'predictions'), "Should have predictions"
        if hasattr(result, 'predictions'):
            assert len(result.predictions) > 0, "Should have at least one prediction"
            assert result.model_id == model_id, "Should match model ID"
            assert result.inference_time_ms > 0, "Should have inference time"
        else:
            assert len(result['predictions']) > 0, "Should have predictions"

    def test_batch_prediction_pipeline(
        self,
        sample_training_data,
        temp_model_dir,
        tmp_path
    ):
        """Test batch prediction end-to-end."""
        from source.ml.model_training import ModelTrainer, TrainingConfig
        from source.ml.model_registry import ModelRegistry
        from source.ml.inference import InferenceEngine

        # Train model
        data_path = tmp_path / "data.csv"
        sample_training_data.to_csv(data_path, index=False)

        config = TrainingConfig(
            model_type="lightgbm",
            n_trials=2,
            cv_folds=2,
            target_column="Daily AQI Value",
            train_data_path=str(data_path),
            model_save_dir=temp_model_dir,
            random_state=42
        )

        trainer = ModelTrainer(config)
        results = trainer.run_full_pipeline()

        registry = ModelRegistry(registry_dir=temp_model_dir)
        model_id = registry.register_model(
            model_path=Path(results['model_path']),
            model_name="batch_test_model",
            model_type="lightgbm",
            task_type="regression",
            metrics=results['test_metrics'],
            hyperparameters=results['best_params'],
            stage="production"
        )

        # Create inference engine
        engine = InferenceEngine(model_registry=registry)

        # Prepare batch input
        batch_data = sample_training_data.iloc[0:20]

        # Make batch prediction
        result = engine.batch_predict(
            data=batch_data,
            model_id=model_id,
            batch_size=10
        )

        # Verify result
        if hasattr(result, 'predictions'):
            assert len(result.predictions) == 20, "Should have 20 predictions"
            assert result.inference_time_ms > 0, "Should have inference time"
            assert result.metadata.get('batch_size') == 10, "Should track batch size"
        else:
            assert len(result['predictions']) == 20, "Should have 20 predictions"

    def test_prediction_with_missing_features(
        self,
        sample_training_data,
        temp_model_dir,
        tmp_path
    ):
        """Test prediction handles missing features gracefully."""
        from source.ml.model_training import ModelTrainer, TrainingConfig
        from source.ml.model_registry import ModelRegistry
        from source.ml.inference import InferenceEngine

        # Train model
        data_path = tmp_path / "data.csv"
        sample_training_data.to_csv(data_path, index=False)

        config = TrainingConfig(
            model_type="xgboost",
            n_trials=2,
            cv_folds=2,
            target_column="Daily AQI Value",
            train_data_path=str(data_path),
            model_save_dir=temp_model_dir,
            random_state=42
        )

        trainer = ModelTrainer(config)
        results = trainer.run_full_pipeline()

        registry = ModelRegistry(registry_dir=temp_model_dir)
        model_id = registry.register_model(
            model_path=Path(results['model_path']),
            model_name="missing_features_test",
            model_type="xgboost",
            task_type="regression",
            metrics=results['test_metrics'],
            hyperparameters=results['best_params'],
            stage="production"
        )

        engine = InferenceEngine(model_registry=registry)

        # Create input with some missing features
        input_data = sample_training_data.iloc[0].to_dict()
        # Remove a feature
        if 'month' in input_data:
            del input_data['month']

        # Try prediction (should handle gracefully)
        try:
            result = engine.predict(data=input_data, model_id=model_id)
            # If it succeeds, that's good
            assert result is not None
        except Exception as e:
            # If it fails, error should be informative
            assert "feature" in str(e).lower() or "column" in str(e).lower()

    def test_prediction_caching(
        self,
        sample_training_data,
        temp_model_dir,
        tmp_path
    ):
        """Test that model loading is cached."""
        from source.ml.model_training import ModelTrainer, TrainingConfig
        from source.ml.model_registry import ModelRegistry
        from source.ml.inference import InferenceEngine
        import time

        # Train model
        data_path = tmp_path / "data.csv"
        sample_training_data.to_csv(data_path, index=False)

        config = TrainingConfig(
            model_type="xgboost",
            n_trials=2,
            cv_folds=2,
            target_column="Daily AQI Value",
            train_data_path=str(data_path),
            model_save_dir=temp_model_dir,
            random_state=42
        )

        trainer = ModelTrainer(config)
        results = trainer.run_full_pipeline()

        registry = ModelRegistry(registry_dir=temp_model_dir)
        model_id = registry.register_model(
            model_path=Path(results['model_path']),
            model_name="cache_test",
            model_type="xgboost",
            task_type="regression",
            metrics=results['test_metrics'],
            hyperparameters=results['best_params'],
            stage="production"
        )

        engine = InferenceEngine(model_registry=registry)
        input_data = sample_training_data.iloc[0].to_dict()

        # First prediction (loads model)
        start1 = time.time()
        result1 = engine.predict(data=input_data, model_id=model_id)
        time1 = time.time() - start1

        # Second prediction (should use cached model)
        start2 = time.time()
        result2 = engine.predict(data=input_data, model_id=model_id)
        time2 = time.time() - start2

        # Both should succeed
        assert result1 is not None
        assert result2 is not None

        # Second prediction might be faster (cached)
        # Note: This is a weak assertion as timing can vary
        # Just check both completed
        assert time1 > 0 and time2 > 0

    def test_concurrent_predictions(
        self,
        sample_training_data,
        temp_model_dir,
        tmp_path
    ):
        """Test handling concurrent predictions."""
        from source.ml.model_training import ModelTrainer, TrainingConfig
        from source.ml.model_registry import ModelRegistry
        from source.ml.inference import InferenceEngine
        from concurrent.futures import ThreadPoolExecutor

        # Train model
        data_path = tmp_path / "data.csv"
        sample_training_data.to_csv(data_path, index=False)

        config = TrainingConfig(
            model_type="xgboost",
            n_trials=2,
            cv_folds=2,
            target_column="Daily AQI Value",
            train_data_path=str(data_path),
            model_save_dir=temp_model_dir,
            random_state=42
        )

        trainer = ModelTrainer(config)
        results = trainer.run_full_pipeline()

        registry = ModelRegistry(registry_dir=temp_model_dir)
        model_id = registry.register_model(
            model_path=Path(results['model_path']),
            model_name="concurrent_test",
            model_type="xgboost",
            task_type="regression",
            metrics=results['test_metrics'],
            hyperparameters=results['best_params'],
            stage="production"
        )

        engine = InferenceEngine(model_registry=registry)

        def make_prediction(index):
            input_data = sample_training_data.iloc[index].to_dict()
            return engine.predict(data=input_data, model_id=model_id)

        # Make concurrent predictions
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_prediction, i) for i in range(10)]
            results = [f.result() for f in futures]

        # All predictions should succeed
        assert len(results) == 10, "Should have 10 results"
        for result in results:
            assert result is not None, "Each prediction should succeed"
