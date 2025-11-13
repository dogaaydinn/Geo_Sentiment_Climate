"""
Advanced Missing Value Handling Module.

Provides sophisticated imputation strategies including:
- MICE (Multivariate Imputation by Chained Equations)
- KNN Imputation
- Regression-based Imputation
- Time-series forward/backward fill
- Advanced concentration pipeline
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge

from source.utils.logger import setup_logger

logger = setup_logger(
    name="missing_handle",
    log_file="../logs/missing_handle.log",
    log_level="INFO"
)


def mice_imputation(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    max_iter: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Perform MICE (Multivariate Imputation by Chained Equations) imputation.

    MICE is an iterative imputation method that models each feature with
    missing values as a function of other features.

    Args:
        df: Input DataFrame
        columns: Columns to impute (default: all numeric columns)
        max_iter: Maximum number of imputation iterations
        random_state: Random state for reproducibility

    Returns:
        DataFrame with imputed values
    """
    logger.info("Starting MICE imputation")

    df_imputed = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Filter to existing columns
    available_cols = [col for col in columns if col in df.columns]

    if not available_cols:
        logger.warning("No numeric columns found for imputation")
        return df_imputed

    missing_before = df[available_cols].isnull().sum().sum()
    logger.info(f"Missing values before MICE: {missing_before}")

    try:
        # Use BayesianRidge as the estimator
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=max_iter,
            random_state=random_state,
            verbose=0
        )

        df_imputed[available_cols] = imputer.fit_transform(df[available_cols])

        missing_after = df_imputed[available_cols].isnull().sum().sum()
        logger.info(f"Missing values after MICE: {missing_after}")
        logger.info(f"MICE imputation completed successfully")

    except Exception as e:
        logger.error(f"MICE imputation failed: {e}")
        logger.info("Falling back to mean imputation")
        imputer = SimpleImputer(strategy='mean')
        df_imputed[available_cols] = imputer.fit_transform(df[available_cols])

    return df_imputed


def knn_imputation(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    n_neighbors: int = 5,
    weights: str = "uniform"
) -> pd.DataFrame:
    """
    Perform KNN-based imputation.

    KNN imputation fills missing values using the mean of k-nearest neighbors.

    Args:
        df: Input DataFrame
        columns: Columns to impute (default: all numeric columns)
        n_neighbors: Number of neighbors to use
        weights: Weight function ('uniform' or 'distance')

    Returns:
        DataFrame with imputed values
    """
    logger.info(f"Starting KNN imputation with {n_neighbors} neighbors")

    df_imputed = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Filter to existing columns
    available_cols = [col for col in columns if col in df.columns]

    if not available_cols:
        logger.warning("No numeric columns found for imputation")
        return df_imputed

    missing_before = df[available_cols].isnull().sum().sum()
    logger.info(f"Missing values before KNN: {missing_before}")

    try:
        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        df_imputed[available_cols] = imputer.fit_transform(df[available_cols])

        missing_after = df_imputed[available_cols].isnull().sum().sum()
        logger.info(f"Missing values after KNN: {missing_after}")
        logger.info("KNN imputation completed successfully")

    except Exception as e:
        logger.error(f"KNN imputation failed: {e}")
        logger.info("Falling back to median imputation")
        imputer = SimpleImputer(strategy='median')
        df_imputed[available_cols] = imputer.fit_transform(df[available_cols])

    return df_imputed


def regression_imputation(
    df: pd.DataFrame,
    target_column: str,
    predictor_columns: Optional[List[str]] = None,
    model_type: str = "random_forest"
) -> pd.DataFrame:
    """
    Perform regression-based imputation for a specific column.

    Uses machine learning models to predict missing values based on
    other features.

    Args:
        df: Input DataFrame
        target_column: Column to impute
        predictor_columns: Columns to use as predictors
        model_type: Type of model ('random_forest', 'gradient_boosting')

    Returns:
        DataFrame with imputed target column
    """
    logger.info(f"Starting regression imputation for {target_column}")

    df_imputed = df.copy()

    if target_column not in df.columns:
        logger.error(f"Target column {target_column} not found")
        return df_imputed

    if predictor_columns is None:
        predictor_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in predictor_columns:
            predictor_columns.remove(target_column)

    # Filter to existing columns
    available_predictors = [col for col in predictor_columns if col in df.columns]

    if not available_predictors:
        logger.warning("No predictor columns available")
        return df_imputed

    # Split into missing and non-missing
    missing_mask = df[target_column].isnull()
    train_df = df[~missing_mask]
    predict_df = df[missing_mask]

    if len(predict_df) == 0:
        logger.info(f"No missing values in {target_column}")
        return df_imputed

    if len(train_df) == 0:
        logger.warning(f"No non-missing values in {target_column}")
        return df_imputed

    try:
        # Select model
        if model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == "gradient_boosting":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Train model
        X_train = train_df[available_predictors].fillna(train_df[available_predictors].mean())
        y_train = train_df[target_column]

        model.fit(X_train, y_train)

        # Predict missing values
        X_predict = predict_df[available_predictors].fillna(train_df[available_predictors].mean())
        predictions = model.predict(X_predict)

        # Fill missing values
        df_imputed.loc[missing_mask, target_column] = predictions

        logger.info(f"Regression imputation completed for {target_column}")
        logger.info(f"Imputed {len(predictions)} missing values")

    except Exception as e:
        logger.error(f"Regression imputation failed: {e}")
        logger.info("Falling back to mean imputation")
        df_imputed[target_column].fillna(df[target_column].mean(), inplace=True)

    return df_imputed


def time_series_imputation(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "linear",
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Perform time-series specific imputation.

    Uses interpolation methods suitable for time series data.

    Args:
        df: Input DataFrame (should have datetime index or be sorted by time)
        columns: Columns to impute
        method: Interpolation method ('linear', 'time', 'polynomial', 'spline')
        limit: Maximum number of consecutive NaNs to fill

    Returns:
        DataFrame with imputed values
    """
    logger.info(f"Starting time-series imputation with method: {method}")

    df_imputed = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Filter to existing columns
    available_cols = [col for col in columns if col in df.columns]

    if not available_cols:
        logger.warning("No numeric columns found for imputation")
        return df_imputed

    for col in available_cols:
        missing_before = df[col].isnull().sum()

        try:
            df_imputed[col] = df[col].interpolate(
                method=method,
                limit=limit,
                limit_direction='both'
            )

            missing_after = df_imputed[col].isnull().sum()
            logger.info(f"{col}: {missing_before} -> {missing_after} missing values")

        except Exception as e:
            logger.error(f"Interpolation failed for {col}: {e}")
            # Fallback to forward fill then backward fill
            df_imputed[col] = df[col].fillna(method='ffill').fillna(method='bfill')

    logger.info("Time-series imputation completed")
    return df_imputed


def advanced_concentration_pipeline(
    df: pd.DataFrame,
    concentration_cols: Optional[List[str]] = None,
    imputation_method: str = "mice",
    random_state: int = 42,
    n_neighbors: int = 5
) -> pd.DataFrame:
    """
    Advanced pipeline for handling missing values in concentration data.

    This is the main function called by data_preprocessing.py.
    It applies sophisticated imputation strategies specifically designed
    for air quality concentration data.

    Args:
        df: Input DataFrame
        concentration_cols: List of concentration columns to impute
        imputation_method: Method to use ('mice', 'knn', 'regression', 'time_series')
        random_state: Random state for reproducibility
        n_neighbors: Number of neighbors for KNN (if using KNN)

    Returns:
        DataFrame with imputed concentration values
    """
    logger.info("=" * 80)
    logger.info("Starting Advanced Concentration Pipeline")
    logger.info("=" * 80)

    df_processed = df.copy()

    # Identify concentration columns if not provided
    if concentration_cols is None:
        concentration_patterns = ['concentration', 'conc', 'aqi', 'pm2.5', 'pm25', 'co', 'no2', 'o3', 'so2']
        concentration_cols = [
            col for col in df.columns
            if any(pattern in col.lower() for pattern in concentration_patterns)
            and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]

    if not concentration_cols:
        logger.warning("No concentration columns identified")
        return df_processed

    logger.info(f"Identified {len(concentration_cols)} concentration columns: {concentration_cols}")

    # Log missing value statistics before imputation
    missing_stats_before = df[concentration_cols].isnull().sum()
    logger.info("\nMissing values before imputation:")
    for col, count in missing_stats_before.items():
        if count > 0:
            pct = (count / len(df)) * 100
            logger.info(f"  {col}: {count} ({pct:.2f}%)")

    # Apply selected imputation method
    if imputation_method == "mice":
        df_processed = mice_imputation(
            df_processed,
            columns=concentration_cols,
            random_state=random_state
        )
    elif imputation_method == "knn":
        df_processed = knn_imputation(
            df_processed,
            columns=concentration_cols,
            n_neighbors=n_neighbors
        )
    elif imputation_method == "time_series":
        df_processed = time_series_imputation(
            df_processed,
            columns=concentration_cols
        )
    elif imputation_method == "regression":
        # Apply regression imputation to each column
        for col in concentration_cols:
            df_processed = regression_imputation(
                df_processed,
                target_column=col,
                model_type="random_forest"
            )
    else:
        logger.warning(f"Unknown imputation method: {imputation_method}. Using MICE.")
        df_processed = mice_imputation(
            df_processed,
            columns=concentration_cols,
            random_state=random_state
        )

    # Log missing value statistics after imputation
    missing_stats_after = df_processed[concentration_cols].isnull().sum()
    logger.info("\nMissing values after imputation:")
    for col, count in missing_stats_after.items():
        if count > 0:
            pct = (count / len(df_processed)) * 100
            logger.info(f"  {col}: {count} ({pct:.2f}%)")

    # Calculate improvement
    total_missing_before = missing_stats_before.sum()
    total_missing_after = missing_stats_after.sum()
    improvement = total_missing_before - total_missing_after

    logger.info("\n" + "=" * 80)
    logger.info(f"Pipeline Summary:")
    logger.info(f"  Total missing values before: {total_missing_before}")
    logger.info(f"  Total missing values after: {total_missing_after}")
    logger.info(f"  Values imputed: {improvement}")
    logger.info(f"  Imputation method: {imputation_method}")
    logger.info("=" * 80)

    return df_processed


def hybrid_imputation(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    methods: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Apply different imputation methods to different columns.

    Args:
        df: Input DataFrame
        columns: Columns to impute
        methods: Dictionary mapping columns to imputation methods

    Returns:
        DataFrame with imputed values
    """
    logger.info("Starting hybrid imputation")

    df_imputed = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if methods is None:
        # Default: use KNN for all
        methods = {col: 'knn' for col in columns}

    for col, method in methods.items():
        if col not in df.columns:
            continue

        logger.info(f"Imputing {col} using {method}")

        if method == 'mice':
            df_imputed = mice_imputation(df_imputed, columns=[col])
        elif method == 'knn':
            df_imputed = knn_imputation(df_imputed, columns=[col])
        elif method == 'regression':
            df_imputed = regression_imputation(df_imputed, target_column=col)
        elif method == 'time_series':
            df_imputed = time_series_imputation(df_imputed, columns=[col])
        else:
            logger.warning(f"Unknown method {method} for {col}")

    logger.info("Hybrid imputation completed")
    return df_imputed
