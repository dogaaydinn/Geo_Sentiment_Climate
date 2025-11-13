"""Integration tests for data preprocessing pipeline."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.mark.integration
class TestPreprocessingIntegration:
    """Integration tests for data preprocessing."""

    def test_full_preprocessing_pipeline(self, sample_training_data, tmp_path):
        """Test complete preprocessing pipeline."""
        from source.data.data_preprocessing.data_preprocessor import DataPreprocessor

        # Initialize preprocessor
        preprocessor = DataPreprocessor(plots_dir=tmp_path)

        # Add some missing values
        df = sample_training_data.copy()
        df.loc[0:5, 'Daily AQI Value'] = np.nan
        df.loc[10:15, 'Daily Mean PM2.5 Concentration'] = np.nan

        # Run preprocessing
        result = preprocessor.process(
            df,
            handle_missing=True,
            missing_method='mean',
            remove_outliers=True,
            outlier_method='iqr',
            scale_features=False  # Don't scale for this test
        )

        # Verify
        assert not result.empty, "Result should not be empty"
        assert result['Daily AQI Value'].isnull().sum() == 0, "Missing values should be filled"
        assert len(result) <= len(df), "Outliers should be removed"

    def test_missing_value_imputation_mean(self, sample_data_with_missing, concentration_columns):
        """Test mean imputation."""
        from source.data.data_preprocessing.data_preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()

        df = sample_data_with_missing.copy()
        initial_missing = df[concentration_columns].isnull().sum().sum()

        result = preprocessor.fill_missing_values(
            df,
            method='mean',
            columns=concentration_columns
        )

        final_missing = result[concentration_columns].isnull().sum().sum()
        assert final_missing < initial_missing, "Missing values should be reduced"

    def test_missing_value_imputation_knn(self, sample_data_with_missing, concentration_columns):
        """Test KNN imputation."""
        from source.missing_handle import knn_imputation

        df = sample_data_with_missing.copy()
        initial_missing = df[concentration_columns].isnull().sum().sum()

        result = knn_imputation(df, columns=concentration_columns, n_neighbors=5)

        final_missing = result[concentration_columns].isnull().sum().sum()
        assert final_missing < initial_missing, "KNN should reduce missing values"
        assert result.shape == df.shape, "Shape should remain same"

    def test_missing_value_imputation_mice(self, sample_multivariate_data):
        """Test MICE imputation."""
        from source.missing_handle import mice_imputation

        df = sample_multivariate_data.copy()
        initial_missing = df.isnull().sum().sum()

        result = mice_imputation(df, max_iter=5, random_state=42)

        final_missing = result.isnull().sum().sum()
        assert final_missing < initial_missing, "MICE should reduce missing values"

    def test_outlier_removal_iqr(self, sample_data_with_outliers):
        """Test IQR-based outlier removal."""
        from source.data.data_preprocessing.data_preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()

        df = sample_data_with_outliers.copy()
        initial_count = len(df)

        result = preprocessor.remove_outliers(
            df,
            method='iqr',
            columns=['Daily AQI Value', 'Daily Mean PM2.5 Concentration']
        )

        assert len(result) < initial_count, "Outliers should be removed"
        assert not result.empty, "Some data should remain"

    def test_outlier_removal_zscore(self, sample_data_with_outliers):
        """Test Z-score based outlier removal."""
        from source.data.data_preprocessing.data_preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()

        df = sample_data_with_outliers.copy()
        initial_count = len(df)

        result = preprocessor.remove_outliers(
            df,
            method='zscore',
            threshold=3.0,
            columns=['Daily AQI Value']
        )

        assert len(result) <= initial_count, "Outliers should be removed or same"

    def test_feature_scaling_standard(self, sample_training_data):
        """Test standard scaling."""
        from source.data.data_preprocessing.data_preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()

        df = sample_training_data.copy()
        columns_to_scale = ['Daily AQI Value', 'Daily Mean PM2.5 Concentration']

        result = preprocessor.scale_features(
            df,
            method='standard',
            columns=columns_to_scale
        )

        # Check that scaled values have mean ~0 and std ~1
        for col in columns_to_scale:
            assert abs(result[col].mean()) < 0.1, f"{col} should have mean near 0"
            assert abs(result[col].std() - 1.0) < 0.1, f"{col} should have std near 1"

    def test_feature_scaling_minmax(self, sample_training_data):
        """Test min-max scaling."""
        from source.data.data_preprocessing.data_preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()

        df = sample_training_data.copy()
        columns_to_scale = ['Daily AQI Value', 'Daily Mean PM2.5 Concentration']

        result = preprocessor.scale_features(
            df,
            method='minmax',
            columns=columns_to_scale
        )

        # Check that scaled values are in [0, 1]
        for col in columns_to_scale:
            assert result[col].min() >= 0, f"{col} min should be >= 0"
            assert result[col].max() <= 1, f"{col} max should be <= 1"

    def test_concentration_pipeline_end_to_end(self, sample_data_with_missing):
        """Test advanced concentration pipeline."""
        from source.missing_handle import advanced_concentration_pipeline

        df = sample_data_with_missing.copy()
        total_missing_before = df.isnull().sum().sum()

        result = advanced_concentration_pipeline(
            df,
            imputation_method='mean',
            scale_method='standard'
        )

        total_missing_after = result.isnull().sum().sum()
        assert total_missing_after < total_missing_before, "Pipeline should reduce missing values"
        assert not result.empty, "Result should not be empty"

    def test_preprocessing_preserves_data_types(self, sample_training_data):
        """Test that preprocessing preserves appropriate data types."""
        from source.data.data_preprocessing.data_preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()

        df = sample_training_data.copy()

        result = preprocessor.process(
            df,
            handle_missing=True,
            remove_outliers=False,
            scale_features=False
        )

        # Numeric columns should remain numeric
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in result.columns:
                assert pd.api.types.is_numeric_dtype(result[col]), f"{col} should remain numeric"
