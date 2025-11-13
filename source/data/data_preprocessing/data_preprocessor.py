"""
Data Preprocessor Module.

Provides comprehensive data preprocessing capabilities including:
- Missing value handling
- Feature scaling
- Outlier detection and removal
- Data visualization
- Correlation analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats

from source.data.data_preprocessing.base_data_processor import (
    BaseDataProcessor,
    timer_decorator,
    validate_dataframe
)
from source.config.config_utils import config


class DataPreprocessor(BaseDataProcessor):
    """
    Comprehensive data preprocessing class for the Geo_Sentiment_Climate project.

    Handles all preprocessing tasks including cleaning, transformation,
    visualization, and quality checks.
    """

    def __init__(
        self,
        logger: Optional[Any] = None,
        plots_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """
        Initialize the DataPreprocessor.

        Args:
            logger: Pre-configured logger instance
            plots_dir: Directory to save plots
            **kwargs: Additional arguments for BaseDataProcessor
        """
        super().__init__(logger=logger, **kwargs)

        self.plots_dir = Path(plots_dir or config.get("paths", {}).get("plots_dir", "../plots"))
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Set style for plots
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10

        self.logger.info(f"Plots will be saved to: {self.plots_dir}")

    @timer_decorator
    @validate_dataframe
    def process(
        self,
        df: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:
        """
        Main processing pipeline.

        Args:
            df: Input DataFrame
            **kwargs: Processing configuration

        Returns:
            Processed DataFrame
        """
        self.logger.info("Starting data preprocessing pipeline")

        # Basic info
        self.basic_info(df, stage="Initial")

        # Remove duplicates
        df = self.remove_duplicates(df)

        # Handle missing values if requested
        if kwargs.get('handle_missing', True):
            df = self.fill_missing_values(df, method=kwargs.get('missing_method', 'mean'))

        # Remove outliers if requested
        if kwargs.get('remove_outliers', False):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            df = self.remove_outliers(
                df,
                columns=numeric_cols,
                method=kwargs.get('outlier_method', 'iqr')
            )

        # Scale features if requested
        if kwargs.get('scale_features', False):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            df = self.scale_features(
                df,
                columns=numeric_cols,
                method=kwargs.get('scaling_method', 'standard')
            )

        # Final info
        self.basic_info(df, stage="Final")

        self.logger.info("Data preprocessing pipeline completed")
        return df

    @timer_decorator
    @validate_dataframe
    def fill_missing_values(
        self,
        df: pd.DataFrame,
        method: str = "mean",
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fill missing values using various strategies.

        Args:
            df: Input DataFrame
            method: Imputation method ('mean', 'median', 'mode', 'ffill', 'bfill', 'knn')
            columns: Columns to fill (default: all numeric columns)

        Returns:
            DataFrame with missing values filled
        """
        self.logger.info(f"Filling missing values using method: {method}")

        df_filled = df.copy()

        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = numeric_cols

        missing_before = df[columns].isnull().sum().sum()

        if method == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            df_filled[columns] = imputer.fit_transform(df[columns])
        elif method in ['mean', 'median', 'most_frequent']:
            imputer = SimpleImputer(strategy=method)
            df_filled[columns] = imputer.fit_transform(df[columns])
        elif method == 'ffill':
            df_filled[columns] = df[columns].fillna(method='ffill')
        elif method == 'bfill':
            df_filled[columns] = df[columns].fillna(method='bfill')
        elif method == 'zero':
            df_filled[columns] = df[columns].fillna(0)
        else:
            raise ValueError(f"Unsupported imputation method: {method}")

        missing_after = df_filled[columns].isnull().sum().sum()

        self.logger.info(
            f"Missing values reduced from {missing_before} to {missing_after}"
        )

        self._processing_stats["operations_count"] += 1
        return df_filled

    @timer_decorator
    @validate_dataframe
    def scale_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "standard"
    ) -> pd.DataFrame:
        """
        Scale features using various scaling methods.

        Args:
            df: Input DataFrame
            columns: Columns to scale
            method: Scaling method ('standard', 'minmax', 'robust')

        Returns:
            DataFrame with scaled features
        """
        self.logger.info(f"Scaling features using method: {method}")

        df_scaled = df.copy()

        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {method}")

        # Filter to existing columns
        available_cols = [col for col in columns if col in df.columns]

        if available_cols:
            df_scaled[available_cols] = scaler.fit_transform(df[available_cols])
            self.logger.info(f"Scaled {len(available_cols)} features")

        self._processing_stats["operations_count"] += 1
        return df_scaled

    @timer_decorator
    @validate_dataframe
    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Remove outliers using various methods.

        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            method: Detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with outliers removed
        """
        self.logger.info(f"Removing outliers using method: {method}")

        df_clean = df.copy()
        initial_rows = len(df_clean)

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if method == "iqr":
            for col in columns:
                if col not in df_clean.columns:
                    continue

                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                df_clean = df_clean[
                    (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                ]

        elif method == "zscore":
            for col in columns:
                if col not in df_clean.columns:
                    continue

                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                df_clean = df_clean[z_scores < threshold]

        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")

        removed_rows = initial_rows - len(df_clean)
        self.logger.info(f"Removed {removed_rows} outlier rows")

        self._processing_stats["operations_count"] += 1
        return df_clean

    @timer_decorator
    @validate_dataframe
    def visualize_missing_values(
        self,
        df: pd.DataFrame,
        save: bool = True,
        filename: str = "missing_values.png"
    ) -> None:
        """
        Visualize missing values in the DataFrame.

        Args:
            df: Input DataFrame
            save: Whether to save the plot
            filename: Filename for saving the plot
        """
        self.logger.info("Visualizing missing values")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Matrix plot
        try:
            msno.matrix(df, ax=axes[0, 0], fontsize=8)
            axes[0, 0].set_title("Missing Values Matrix", fontsize=12, fontweight='bold')
        except Exception as e:
            self.logger.warning(f"Failed to create matrix plot: {e}")
            axes[0, 0].text(0.5, 0.5, 'Matrix plot failed', ha='center', va='center')

        # Bar plot
        try:
            msno.bar(df, ax=axes[0, 1], fontsize=8)
            axes[0, 1].set_title("Missing Values Bar Chart", fontsize=12, fontweight='bold')
        except Exception as e:
            self.logger.warning(f"Failed to create bar plot: {e}")
            axes[0, 1].text(0.5, 0.5, 'Bar plot failed', ha='center', va='center')

        # Heatmap
        try:
            msno.heatmap(df, ax=axes[1, 0], fontsize=8)
            axes[1, 0].set_title("Missing Values Heatmap", fontsize=12, fontweight='bold')
        except Exception as e:
            self.logger.warning(f"Failed to create heatmap: {e}")
            axes[1, 0].text(0.5, 0.5, 'Heatmap failed', ha='center', va='center')

        # Dendrogram
        try:
            msno.dendrogram(df, ax=axes[1, 1], fontsize=8)
            axes[1, 1].set_title("Missing Values Dendrogram", fontsize=12, fontweight='bold')
        except Exception as e:
            self.logger.warning(f"Failed to create dendrogram: {e}")
            axes[1, 1].text(0.5, 0.5, 'Dendrogram failed', ha='center', va='center')

        plt.tight_layout()

        if save:
            save_path = self.plots_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Missing values plot saved to {save_path}")

        plt.close()

    @timer_decorator
    @validate_dataframe
    def correlation_analysis(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        save: bool = True,
        filename: str = "correlation_heatmap.png"
    ) -> pd.DataFrame:
        """
        Perform correlation analysis and create heatmap.

        Args:
            df: Input DataFrame
            columns: Columns to include in correlation analysis
            save: Whether to save the plot
            filename: Filename for saving the plot

        Returns:
            Correlation matrix
        """
        self.logger.info("Performing correlation analysis")

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Filter to existing columns
        available_cols = [col for col in columns if col in df.columns]

        if not available_cols:
            self.logger.warning("No numeric columns found for correlation analysis")
            return pd.DataFrame()

        corr_matrix = df[available_cols].corr()

        # Create heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8}
        )
        plt.title("Correlation Heatmap", fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()

        if save:
            save_path = self.plots_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Correlation heatmap saved to {save_path}")

        plt.close()

        return corr_matrix

    @timer_decorator
    @validate_dataframe
    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        save: bool = True,
        filename: str = "outliers_boxplot.png"
    ) -> None:
        """
        Detect and visualize outliers using boxplots.

        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            save: Whether to save the plot
            filename: Filename for saving the plot
        """
        self.logger.info("Detecting outliers")

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Limit to first 20 columns for visualization
        columns = columns[:20]

        # Filter to existing columns
        available_cols = [col for col in columns if col in df.columns]

        if not available_cols:
            self.logger.warning("No numeric columns found for outlier detection")
            return

        # Create subplots
        n_cols = min(4, len(available_cols))
        n_rows = (len(available_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for idx, col in enumerate(available_cols):
            if idx < len(axes):
                df.boxplot(column=col, ax=axes[idx])
                axes[idx].set_title(col, fontsize=10, fontweight='bold')
                axes[idx].set_ylabel('Value')

        # Hide unused subplots
        for idx in range(len(available_cols), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle("Outlier Detection - Boxplots", fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save:
            save_path = self.plots_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Outliers boxplot saved to {save_path}")

        plt.close()

    @timer_decorator
    @validate_dataframe
    def distribution_analysis(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        save: bool = True,
        filename: str = "distributions.png"
    ) -> None:
        """
        Analyze and visualize distributions of numeric features.

        Args:
            df: Input DataFrame
            columns: Columns to analyze
            save: Whether to save the plot
            filename: Filename for saving the plot
        """
        self.logger.info("Analyzing feature distributions")

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Limit to first 12 columns for visualization
        columns = columns[:12]

        # Filter to existing columns
        available_cols = [col for col in columns if col in df.columns]

        if not available_cols:
            self.logger.warning("No numeric columns found for distribution analysis")
            return

        # Create subplots
        n_cols = min(3, len(available_cols))
        n_rows = (len(available_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for idx, col in enumerate(available_cols):
            if idx < len(axes):
                # Histogram with KDE
                df[col].hist(ax=axes[idx], bins=30, alpha=0.7, edgecolor='black')
                axes[idx].set_title(f"{col} Distribution", fontsize=10, fontweight='bold')
                axes[idx].set_xlabel('Value')
                axes[idx].set_ylabel('Frequency')

                # Add KDE if possible
                try:
                    df[col].plot(kind='kde', ax=axes[idx], secondary_y=True, color='red')
                except:
                    pass

        # Hide unused subplots
        for idx in range(len(available_cols), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle("Feature Distributions", fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save:
            save_path = self.plots_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Distribution plots saved to {save_path}")

        plt.close()

    @timer_decorator
    def save_preprocessed_data(
        self,
        df: pd.DataFrame,
        filename: str,
        save_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Save preprocessed data to file.

        Args:
            df: DataFrame to save
            filename: Output filename
            save_dir: Directory to save the file
        """
        if save_dir is None:
            save_dir = Path(config.get("paths", {}).get("processed_dir", "../data/processed"))
        else:
            save_dir = Path(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename

        self.save_data(df, save_path)
        self.logger.info(f"Preprocessed data saved to {save_path}")
