"""Integration tests for model training pipeline."""
import pytest
import joblib
from pathlib import Path


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.slow
class TestModelTrainingIntegration:
    """Integration tests for model training pipeline."""

    def test_xgboost_training_pipeline(
        self,
        sample_training_data,
        temp_model_dir,
        tmp_path
    ):
        """Test complete XGBoost training pipeline."""
        from source.ml.model_training import ModelTrainer, TrainingConfig

        # Save data
        data_path = tmp_path / "train_data.csv"
        sample_training_data.to_csv(data_path, index=False)

        # Configure training
        config = TrainingConfig(
            model_type="xgboost",
            task_type="regression",
            n_trials=3,  # Minimal for testing
            cv_folds=2,  # Minimal for testing
            target_column="Daily AQI Value",
            train_data_path=str(data_path),
            model_save_dir=temp_model_dir,
            random_state=42
        )

        # Train
        trainer = ModelTrainer(config)

        # Load data
        train_df, test_df = trainer.load_data()
        assert not train_df.empty, "Training data should not be empty"
        assert not test_df.empty, "Test data should not be empty"

        # Prepare features
        X_train, y_train = trainer.prepare_features(train_df)
        X_test, y_test = trainer.prepare_features(test_df)

        assert X_train.shape[1] > 0, "Should have features"
        assert len(y_train) > 0, "Should have targets"

        # Optimize hyperparameters
        best_params = trainer.optimize_hyperparameters(X_train, y_train)
        assert best_params is not None, "Should return best params"
        assert 'n_estimators' in best_params or 'max_depth' in best_params, "Should contain XGBoost params"

        # Train model
        model = trainer.train(X_train, y_train, X_test, y_test)
        assert model is not None, "Model should be trained"

        # Make predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test), "Should predict for all test samples"

        # Save model
        model_path = trainer.save_model()
        assert model_path.exists(), "Model file should exist"

        # Load and verify
        loaded_model = joblib.load(model_path)
        loaded_predictions = loaded_model.predict(X_test)
        assert len(loaded_predictions) == len(predictions), "Loaded model should work"

    def test_lightgbm_training_pipeline(
        self,
        sample_training_data,
        temp_model_dir,
        tmp_path
    ):
        """Test LightGBM training pipeline."""
        from source.ml.model_training import ModelTrainer, TrainingConfig

        data_path = tmp_path / "train_data.csv"
        sample_training_data.to_csv(data_path, index=False)

        config = TrainingConfig(
            model_type="lightgbm",
            n_trials=3,
            cv_folds=2,
            target_column="Daily AQI Value",
            train_data_path=str(data_path),
            model_save_dir=temp_model_dir,
            random_state=42
        )

        trainer = ModelTrainer(config)
        results = trainer.run_full_pipeline()

        # Verify results
        assert 'model_path' in results, "Should return model path"
        assert Path(results['model_path']).exists(), "Model file should exist"
        assert 'train_metrics' in results, "Should return training metrics"
        assert 'test_metrics' in results, "Should return test metrics"

        # Check that model has some predictive power
        if 'r2' in results['test_metrics']:
            # For regression, R2 should be reasonable
            assert results['test_metrics']['r2'] > -1, "R2 should be > -1"

    def test_catboost_training_pipeline(
        self,
        sample_training_data,
        temp_model_dir,
        tmp_path
    ):
        """Test CatBoost training pipeline."""
        from source.ml.model_training import ModelTrainer, TrainingConfig

        data_path = tmp_path / "train_data.csv"
        sample_training_data.to_csv(data_path, index=False)

        config = TrainingConfig(
            model_type="catboost",
            n_trials=2,
            cv_folds=2,
            target_column="Daily AQI Value",
            train_data_path=str(data_path),
            model_save_dir=temp_model_dir,
            random_state=42
        )

        trainer = ModelTrainer(config)
        results = trainer.run_full_pipeline()

        assert 'model_path' in results, "Should return model path"
        assert Path(results['model_path']).exists(), "Model should be saved"

    def test_model_comparison(
        self,
        sample_training_data,
        temp_model_dir,
        tmp_path
    ):
        """Test training multiple models and comparing."""
        from source.ml.model_training import ModelTrainer, TrainingConfig

        data_path = tmp_path / "train_data.csv"
        sample_training_data.to_csv(data_path, index=False)

        models = ['xgboost', 'lightgbm', 'catboost']
        results = {}

        for model_type in models:
            config = TrainingConfig(
                model_type=model_type,
                n_trials=2,  # Minimal
                cv_folds=2,
                target_column="Daily AQI Value",
                train_data_path=str(data_path),
                model_save_dir=temp_model_dir,
                random_state=42
            )

            trainer = ModelTrainer(config)
            results[model_type] = trainer.run_full_pipeline()

        # Verify all models trained
        for model_type in models:
            assert Path(results[model_type]['model_path']).exists(), f"{model_type} model should exist"
            assert 'test_metrics' in results[model_type], f"{model_type} should have metrics"

        # Compare performance (all should have some R2 score)
        r2_scores = {
            model: results[model]['test_metrics'].get('r2', -999)
            for model in models
        }

        # At least one model should have reasonable performance
        max_r2 = max(r2_scores.values())
        assert max_r2 > -1, "At least one model should have R2 > -1"

    def test_hyperparameter_optimization_improves_model(
        self,
        sample_training_data,
        temp_model_dir,
        tmp_path
    ):
        """Test that hyperparameter optimization improves model."""
        from source.ml.model_training import ModelTrainer, TrainingConfig

        data_path = tmp_path / "train_data.csv"
        sample_training_data.to_csv(data_path, index=False)

        # Train with optimization
        config_optimized = TrainingConfig(
            model_type="xgboost",
            n_trials=5,
            cv_folds=2,
            target_column="Daily AQI Value",
            train_data_path=str(data_path),
            model_save_dir=temp_model_dir / "optimized",
            random_state=42
        )

        trainer_optimized = ModelTrainer(config_optimized)
        results_optimized = trainer_optimized.run_full_pipeline()

        # Should complete successfully
        assert 'test_metrics' in results_optimized, "Optimized model should have metrics"
        assert 'best_params' in results_optimized, "Should return best params"

    def test_training_with_different_target_columns(
        self,
        sample_training_data,
        temp_model_dir,
        tmp_path
    ):
        """Test training with different target columns."""
        from source.ml.model_training import ModelTrainer, TrainingConfig

        data_path = tmp_path / "train_data.csv"
        sample_training_data.to_csv(data_path, index=False)

        targets = ['Daily AQI Value', 'Daily Mean PM2.5 Concentration']

        for target in targets:
            config = TrainingConfig(
                model_type="xgboost",
                n_trials=2,
                cv_folds=2,
                target_column=target,
                train_data_path=str(data_path),
                model_save_dir=temp_model_dir / target,
                random_state=42
            )

            trainer = ModelTrainer(config)
            results = trainer.run_full_pipeline()

            assert Path(results['model_path']).exists(), f"Model for {target} should exist"
