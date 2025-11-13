"""Test fixtures for data-related tests."""
import pandas as pd
import numpy as np
from pathlib import Path
import pytest
from datetime import datetime, timedelta


@pytest.fixture
def sample_air_quality_data():
    """Generate sample air quality data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    data = {
        'Date': dates,
        'Site ID': ['SITE001'] * 100,
        'State': ['California'] * 100,
        'County': ['Los Angeles'] * 100,
        'Site Latitude': [34.0522] * 100,
        'Site Longitude': [-118.2437] * 100,
        'Daily Max 8-hour CO Concentration': np.random.uniform(0.1, 5.0, 100),
        'Daily Max 1-hour NO2 Concentration': np.random.uniform(10, 80, 100),
        'Daily Max 8-hour Ozone Concentration': np.random.uniform(20, 100, 100),
        'Daily Mean PM2.5 Concentration': np.random.uniform(5, 50, 100),
        'Daily Max 1-hour SO2 Concentration': np.random.uniform(1, 20, 100),
        'Daily AQI Value': np.random.randint(0, 200, 100),
        'Daily Obs Count': np.random.randint(20, 24, 100),
        'Percent Complete': np.random.uniform(90, 100, 100),
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_processed_data(tmp_path, sample_air_quality_data):
    """Save sample data to temporary file."""
    file_path = tmp_path / "test_data.csv"
    sample_air_quality_data.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def sample_training_data(sample_air_quality_data):
    """Prepare data for model training."""
    df = sample_air_quality_data.copy()

    # Add some features
    df['month'] = pd.to_datetime(df['Date']).dt.month
    df['day_of_week'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['season'] = df['month'].apply(lambda x: (x % 12 + 3) // 3)

    return df


@pytest.fixture
def sample_data_with_missing(sample_air_quality_data):
    """Generate sample data with missing values."""
    df = sample_air_quality_data.copy()

    # Add missing values to various columns
    df.loc[0:10, 'Daily AQI Value'] = np.nan
    df.loc[20:25, 'Daily Mean PM2.5 Concentration'] = np.nan
    df.loc[40:45, 'Daily Max 8-hour CO Concentration'] = np.nan

    return df


@pytest.fixture
def sample_data_with_outliers(sample_air_quality_data):
    """Generate sample data with outliers."""
    df = sample_air_quality_data.copy()

    # Add extreme outliers
    df.loc[0, 'Daily AQI Value'] = 999
    df.loc[1, 'Daily Mean PM2.5 Concentration'] = 500
    df.loc[2, 'Daily Max 8-hour CO Concentration'] = 100

    return df


@pytest.fixture
def mock_model_config():
    """Mock training configuration."""
    from source.ml.model_training import TrainingConfig

    return TrainingConfig(
        model_type="xgboost",
        task_type="regression",
        n_trials=5,  # Reduced for testing
        cv_folds=3,  # Reduced for testing
        target_column="Daily AQI Value",
        random_state=42
    )


@pytest.fixture
def concentration_columns():
    """List of concentration columns."""
    return [
        'Daily Max 8-hour CO Concentration',
        'Daily Max 1-hour NO2 Concentration',
        'Daily Max 8-hour Ozone Concentration',
        'Daily Mean PM2.5 Concentration',
        'Daily Max 1-hour SO2 Concentration'
    ]


@pytest.fixture
def sample_multivariate_data():
    """Generate multivariate data for complex imputation tests."""
    n_samples = 200

    data = {
        'temperature': np.random.normal(20, 5, n_samples),
        'humidity': np.random.normal(60, 15, n_samples),
        'wind_speed': np.random.gamma(2, 2, n_samples),
        'pm25': np.random.lognormal(3, 0.5, n_samples),
        'aqi': np.random.randint(0, 200, n_samples)
    }

    df = pd.DataFrame(data)

    # Add correlations
    df['pm25'] = df['pm25'] + df['temperature'] * 0.5 - df['wind_speed'] * 2
    df['aqi'] = (df['pm25'] * 2 + df['temperature'] * 1.5).clip(0, 500).astype(int)

    # Add missing values
    missing_indices = np.random.choice(n_samples, size=30, replace=False)
    df.loc[missing_indices, 'pm25'] = np.nan

    missing_indices = np.random.choice(n_samples, size=20, replace=False)
    df.loc[missing_indices, 'temperature'] = np.nan

    return df
