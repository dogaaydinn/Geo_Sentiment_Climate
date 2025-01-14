import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import matplotlib.axes as maxes
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Enable IterativeImputer
from sklearn.experimental import enable_iterative_imputer
# Import the imputer classes
from sklearn.impute import IterativeImputer, KNNImputer
from source.missing_handle import advanced_concentration_pipeline
from source.missing_value_comparison import compare_missing_values
from source.utils.config_loader import load_config
from source.utils.logger import setup_logger
from source.utils.path_utils import add_source_to_sys_path


add_source_to_sys_path()

# Load config
CONFIG_PATH = Path("../config/settings.yml").resolve()
config = load_config(CONFIG_PATH)

if config is None:
    raise ValueError("Config file could not be loaded or returned empty.")

# Define directories using pathlib
RAW_DIR = Path(config["paths"]["raw_dir"]).resolve()
PROCESSED_DIR = Path(config["paths"]["processed_dir"]).resolve()
PLOTS_DIR = Path(config["paths"].get("plots_dir", "../plots")).resolve()
LOG_DIR = Path(config["paths"].get("logs_dir", "../logs")).resolve()

# Create necessary directories
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Setup logger
logger = setup_logger(
    name="data_preprocessing",
    log_file=str(LOG_DIR / "data_preprocessing.log"),
    log_level=config.get("logging", {}).get("level", "INFO").upper()
)

logger.info("Config file and directories loaded successfully.")


# Store the original function reference
_original_tick_params = maxes.Axes.tick_params

def _patched_tick_params(self, *args, **kwargs):
    # Remove 'grid_b' keyword if it exists
    if "grid_b" in kwargs:
        kwargs.pop("grid_b")
    # Call the original tick_params function
    return _original_tick_params(self, *args, **kwargs)

# Apply the patch only once
if not hasattr(maxes.Axes, "_already_patched_grid_b"):
    maxes.Axes._already_patched_grid_b = True
    maxes.Axes.tick_params = _patched_tick_params



def load_data(file_path: Path) -> pd.DataFrame:

    logger.info(f"Loading data from {file_path}")
    try:
        data = pd.read_csv(file_path)
        logger.info("Data loaded successfully")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise


def save_plot(fig: plt.Figure, filename: str) -> None:

    filepath = PLOTS_DIR / filename
    try:
        fig.savefig(filepath)
        logger.info(f"Plot saved to {filepath}")
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error saving plot {filename}: {e}")
        raise


def basic_info(df: pd.DataFrame, stage: str = "Initial") -> Dict[str, Any]:

    logger.info(f"Generating basic info of the dataset at {stage} stage")
    info = {
        "Shape": df.shape,
        "Columns": df.columns.tolist(),
        "Data Types": df.dtypes.to_dict(),
        "Missing Values": df.isnull().sum().to_dict()
    }
    logger.info(f"Dataset Info ({stage}): {info}")
    return info


def visualize_missing_values(df: pd.DataFrame, save: bool = True) -> None:

    logger.info("Visualizing missing values.")
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        msno.matrix(df, ax=ax)
        if save:
            save_plot(fig, "missing_values_matrix.png")
        else:
            plt.show()
    except Exception as e:
        logger.error(f"Error in missing values visualization: {e}")
        raise


def fill_missing_values(df: pd.DataFrame, method: str = "mean") -> pd.DataFrame:

    logger.info(f"Filling missing values using method: {method}")
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if method == "mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif method == "median":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif method == "mode":
            df = df.fillna(df.mode().iloc[0])
        else:
            raise ValueError("Invalid method for missing value handling")
        logger.info("Missing values filled successfully.")
        return df
    except Exception as e:
        logger.error(f"Error filling missing values: {e}")
        raise


def distribution_analysis(df: pd.DataFrame, numeric_cols: List[str], save: bool = True) -> None:

    logger.info("Performing distribution analysis for numeric columns")
    try:
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[col], kde=True, bins=30, color="blue", ax=ax)
            ax.set_title(f"Distribution of {col}")
            if save:
                save_plot(fig, f"distribution_{col}.png")
            else:
                plt.show()
    except Exception as e:
        logger.error(f"Error in distribution analysis: {e}")
        raise


def scale_features(df: pd.DataFrame, numeric_cols: List[str], method: str = "standard") -> pd.DataFrame:

    logger.info(f"Scaling features: {numeric_cols} using method: {method}")
    try:
        scaler = StandardScaler() if method == "standard" else MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        logger.info("Features scaled successfully.")
        return df
    except Exception as e:
        logger.error(f"Error scaling features: {e}")
        raise


def correlation_analysis(df: pd.DataFrame, numeric_cols: List[str], save: bool = True) -> None:

    logger.info("Performing correlation analysis.")
    try:
        corr_matrix = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Matrix")
        if save:
            save_plot(fig, "correlation_matrix.png")
        else:
            plt.show()
    except Exception as e:
        logger.error(f"Error in correlation analysis: {e}")
        raise


def detect_outliers(df: pd.DataFrame, numeric_cols: List[str], save: bool = True) -> None:

    logger.info("Detecting outliers in numeric columns.")
    try:
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f"Outliers in {col}")
            if save:
                save_plot(fig, f"outliers_{col}.png")
            else:
                plt.show()
    except Exception as e:
        logger.error(f"Error in outlier detection: {e}")
        raise


def save_preprocessed_data(df: pd.DataFrame, filename: str = "epa_preprocessed.csv") -> None:

    processed_file_path = PROCESSED_DIR / filename
    try:
        df.to_csv(processed_file_path, index=False)
        logger.info(f"Preprocessed data saved successfully at: {processed_file_path}")
    except Exception as e:
        logger.critical(f"Data preprocessing failed while saving: {e}")
        raise


def compare_missing_values_original_vs_processed(original_file_path: Path, processed_df: pd.DataFrame) -> None:

    logger.info("Comparing missing values between original and processed data.")
    try:
        original_df = load_data(original_file_path)
        compare_missing_values(original_df, processed_df)
    except Exception as e:
        logger.error(f"Error comparing missing values: {e}")
        raise


def preprocess_data() -> None:

    try:
        # Load data
        file_path = PROCESSED_DIR / "epa_long_preprocessed.csv"
        df = load_data(file_path)
        basic_info(df, stage="Initial")

        # Visualization before filling missing values
        visualize_missing_values(df, save=True)

        # Fill missing values
        missing_method = config["parameters"].get("missing_value_method", "mean")
        df = fill_missing_values(df, method=missing_method)
        basic_info(df, stage="After Filling Missing Values")

        # Advanced Missing Value Handling (MICE, KNN, Regression)
        df = advanced_concentration_pipeline(
            df,
            concentration_cols=config["data_check"].get("required_columns", []),
            imputation_method=config["parameters"].get("imputation_method", "mice"),
            random_state=config["parameters"].get("random_state", 42),
            n_neighbors=config["parameters"].get("n_neighbors", 5)
        )

        # Scale features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        scaling_method = config["parameters"].get("scaling_method", "standard")
        df = scale_features(df, numeric_cols, method=scaling_method)

        # Correlation analysis
        correlation_analysis(df, numeric_cols, save=True)

        # Detect outliers
        detect_outliers(df, numeric_cols, save=True)

        # Distribution analysis
        distribution_analysis(df, numeric_cols, save=True)

        # Save preprocessed data
        save_preprocessed_data(df)

        # Generate basic info after preprocessing
        basic_info(df, stage="Final")

        # Missing Value Comparison (Original vs Processed)
        original_file_path = RAW_DIR / "epa_long_preprocessed.csv"  # Adjust as needed
        compare_missing_values_original_vs_processed(original_file_path, df)

    except Exception as e:
        logger.critical(f"Data preprocessing failed with a critical error: {e}")
        raise


if __name__ == "__main__":
    preprocess_data()
