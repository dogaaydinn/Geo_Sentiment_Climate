"""
Base Data Processor Module.

Provides the foundational class for all data processing operations
with enterprise-level features including logging, error handling,
and performance monitoring.
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Callable
from abc import ABC, abstractmethod
from functools import wraps
from datetime import datetime

from source.utils.logger import setup_logger


def timer_decorator(func: Callable) -> Callable:
    """
    Decorator to measure execution time of methods.

    Args:
        func: Function to be timed

    Returns:
        Wrapped function with timing capability
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time

        if hasattr(self, 'logger'):
            self.logger.info(
                f"{func.__name__} completed in {duration:.2f} seconds"
            )

        return result
    return wrapper


def validate_dataframe(func: Callable) -> Callable:
    """
    Decorator to validate DataFrame inputs.

    Args:
        func: Function to be validated

    Returns:
        Wrapped function with validation
    """
    @wraps(func)
    def wrapper(self, df: pd.DataFrame, *args, **kwargs):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df)}")

        if df.empty:
            if hasattr(self, 'logger'):
                self.logger.warning(f"{func.__name__} received empty DataFrame")

        return func(self, df, *args, **kwargs)
    return wrapper


class BaseDataProcessor(ABC):
    """
    Abstract base class for data processing operations.

    This class provides common functionality for all data processors including:
    - Logging setup
    - Data loading/saving
    - Basic statistics
    - Error handling
    - Performance monitoring

    Attributes:
        logger: Configured logger instance
        _processing_stats: Dictionary storing processing statistics
    """

    def __init__(
        self,
        logger: Optional[Any] = None,
        log_file: Optional[str] = None,
        log_level: str = "INFO"
    ):
        """
        Initialize the BaseDataProcessor.

        Args:
            logger: Pre-configured logger instance (optional)
            log_file: Path to log file (optional)
            log_level: Logging level (default: INFO)
        """
        if logger:
            self.logger = logger
        else:
            self.logger = setup_logger(
                name=self.__class__.__name__,
                log_file=log_file or f"logs/{self.__class__.__name__.lower()}.log",
                log_level=log_level
            )

        self._processing_stats: Dict[str, Any] = {
            "initialized_at": datetime.now().isoformat(),
            "operations_count": 0,
            "total_rows_processed": 0,
            "errors_count": 0
        }

        self.logger.info(f"{self.__class__.__name__} initialized successfully")

    @timer_decorator
    @validate_dataframe
    def load_data(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from various file formats.

        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments for pandas read functions

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
        """
        file_path = Path(file_path)

        if not file_path.exists():
            self._processing_stats["errors_count"] += 1
            raise FileNotFoundError(f"File not found: {file_path}")

        self.logger.info(f"Loading data from {file_path}")

        # Determine file format and load accordingly
        suffix = file_path.suffix.lower()

        try:
            if suffix == '.csv':
                df = pd.read_csv(file_path, **kwargs)
            elif suffix == '.parquet':
                df = pd.read_parquet(file_path, **kwargs)
            elif suffix == '.feather':
                df = pd.read_feather(file_path, **kwargs)
            elif suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, **kwargs)
            elif suffix == '.json':
                df = pd.read_json(file_path, **kwargs)
            elif suffix == '.h5' or suffix == '.hdf5':
                df = pd.read_hdf(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")

            self._processing_stats["operations_count"] += 1
            self._processing_stats["total_rows_processed"] += len(df)

            self.logger.info(
                f"Successfully loaded {len(df)} rows and {len(df.columns)} columns"
            )

            return df

        except Exception as e:
            self._processing_stats["errors_count"] += 1
            self.logger.error(f"Error loading data from {file_path}: {e}")
            raise

    @timer_decorator
    def save_data(
        self,
        df: pd.DataFrame,
        file_path: Union[str, Path],
        **kwargs
    ) -> None:
        """
        Save DataFrame to various file formats.

        Args:
            df: DataFrame to save
            file_path: Path to save the file
            **kwargs: Additional arguments for pandas save functions
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Saving data to {file_path}")

        suffix = file_path.suffix.lower()

        try:
            if suffix == '.csv':
                df.to_csv(file_path, index=False, **kwargs)
            elif suffix == '.parquet':
                df.to_parquet(file_path, index=False, **kwargs)
            elif suffix == '.feather':
                df.to_feather(file_path, **kwargs)
            elif suffix in ['.xlsx', '.xls']:
                df.to_excel(file_path, index=False, **kwargs)
            elif suffix == '.json':
                df.to_json(file_path, **kwargs)
            elif suffix == '.h5' or suffix == '.hdf5':
                df.to_hdf(file_path, key='data', mode='w', **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")

            self._processing_stats["operations_count"] += 1
            self.logger.info(f"Successfully saved data to {file_path}")

        except Exception as e:
            self._processing_stats["errors_count"] += 1
            self.logger.error(f"Error saving data to {file_path}: {e}")
            raise

    @validate_dataframe
    def basic_info(
        self,
        df: pd.DataFrame,
        stage: str = "Current"
    ) -> Dict[str, Any]:
        """
        Generate basic information about the DataFrame.

        Args:
            df: DataFrame to analyze
            stage: Processing stage name for logging

        Returns:
            Dictionary containing basic statistics
        """
        info = {
            "stage": stage,
            "shape": df.shape,
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "dtypes": df.dtypes.value_counts().to_dict(),
            "missing_values": df.isnull().sum().sum(),
            "missing_percentage": (df.isnull().sum().sum() / df.size) * 100,
            "duplicate_rows": df.duplicated().sum(),
        }

        self.logger.info(f"=== DataFrame Info ({stage}) ===")
        self.logger.info(f"Shape: {info['shape']}")
        self.logger.info(f"Memory Usage: {info['memory_usage_mb']:.2f} MB")
        self.logger.info(f"Missing Values: {info['missing_values']} ({info['missing_percentage']:.2f}%)")
        self.logger.info(f"Duplicate Rows: {info['duplicate_rows']}")

        return info

    @validate_dataframe
    def get_column_info(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get detailed information about each column.

        Args:
            df: DataFrame to analyze

        Returns:
            DataFrame with column statistics
        """
        column_info = pd.DataFrame({
            'dtype': df.dtypes,
            'missing_count': df.isnull().sum(),
            'missing_pct': (df.isnull().sum() / len(df)) * 100,
            'unique_count': df.nunique(),
            'unique_pct': (df.nunique() / len(df)) * 100
        })

        # Add min/max for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        column_info.loc[numeric_cols, 'min'] = df[numeric_cols].min()
        column_info.loc[numeric_cols, 'max'] = df[numeric_cols].max()
        column_info.loc[numeric_cols, 'mean'] = df[numeric_cols].mean()
        column_info.loc[numeric_cols, 'std'] = df[numeric_cols].std()

        return column_info

    @validate_dataframe
    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = 'first'
    ) -> pd.DataFrame:
        """
        Remove duplicate rows from DataFrame.

        Args:
            df: Input DataFrame
            subset: Column labels to consider for identifying duplicates
            keep: Which duplicates to keep ('first', 'last', False)

        Returns:
            DataFrame with duplicates removed
        """
        initial_rows = len(df)
        df_clean = df.drop_duplicates(subset=subset, keep=keep)
        removed_rows = initial_rows - len(df_clean)

        self.logger.info(f"Removed {removed_rows} duplicate rows")
        self._processing_stats["operations_count"] += 1

        return df_clean

    @validate_dataframe
    def filter_columns(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """
        Filter DataFrame to keep only specified columns.

        Args:
            df: Input DataFrame
            columns: List of column names to keep

        Returns:
            Filtered DataFrame
        """
        missing_cols = set(columns) - set(df.columns)
        if missing_cols:
            self.logger.warning(f"Columns not found: {missing_cols}")

        available_cols = [col for col in columns if col in df.columns]
        df_filtered = df[available_cols].copy()

        self.logger.info(f"Filtered to {len(available_cols)} columns")
        self._processing_stats["operations_count"] += 1

        return df_filtered

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Dictionary of processing statistics
        """
        return self._processing_stats.copy()

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self._processing_stats = {
            "initialized_at": datetime.now().isoformat(),
            "operations_count": 0,
            "total_rows_processed": 0,
            "errors_count": 0
        }
        self.logger.info("Processing statistics reset")

    @abstractmethod
    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Abstract method for main processing logic.

        This method must be implemented by all subclasses.

        Args:
            df: Input DataFrame
            **kwargs: Additional processing arguments

        Returns:
            Processed DataFrame
        """
        pass
