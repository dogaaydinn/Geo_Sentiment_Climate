import os
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from source.utils.path_utils import add_source_to_sys_path
from utils.config_loader import load_config
from utils.logger import setup_logger
from missing_handle import advanced_concentration_pipeline
from missing_value_comparison import compare_missing_values

# Config and Logger Settings
CONFIG_PATH = "../config/settings.yml"

# Add source to sys.path
add_source_to_sys_path()

try:
    # Load config file
    config = load_config(CONFIG_PATH)
    if config is None:
        raise ValueError("Config file could not be loaded or returned empty.")

    # Get directories from config and convert to absolute paths
    RAW_DIR = os.path.abspath(config["paths"]["raw_dir"])
    PROCESSED_DIR = os.path.abspath(config["paths"]["processed_dir"])
    PLOTS_DIR = os.path.abspath(config["paths"].get("plots_dir", "../plots"))
    LOG_DIR = os.path.abspath(config["paths"].get("logs_dir", "../logs"))

    # Check and create directories if they don't exist
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Create logger
    logger = setup_logger(
        name="data_preprocessing",
        log_file=os.path.join(LOG_DIR, "data_preprocessing.log"),
        log_level="INFO"
    )
    logger.info("Config file and directories loaded successfully.")
except KeyError as e:
    raise ValueError(f"Missing key in config file: {e}")
except Exception as e:
    raise RuntimeError(f"Error loading config: {e}")


def load_data(file_path):
    logger.info(f"Loading data from {file_path}")
    try:
        data = pd.read_csv(file_path)
        logger.info("Data loaded successfully")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise


def save_plot(fig, filename):
    filepath = os.path.join(PLOTS_DIR, filename)
    fig.savefig(filepath)
    logger.info(f"Plot saved to {filepath}")
    plt.close(fig)


def basic_info(df, stage="Initial"):
    logger.info(f"Generating basic info of the dataset at {stage} stage")
    info = {
        "Shape": df.shape,
        "Columns": df.columns.tolist(),
        "Data Types": df.dtypes.to_dict(),
        "Missing Values": df.isnull().sum().to_dict()
    }
    logger.info(f"Dataset Info ({stage}): {info}")
    return info


def visualize_missing_values(df, save=True):
    logger.info("Visualizing missing values.")
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        msno.matrix(df, ax=ax)
        if save:
            plot_path = os.path.join(PLOTS_DIR, "missing_values_matrix.png")
            save_plot(fig, "missing_values_matrix.png")
        else:
            plt.show()
    except Exception as e:
        logger.error(f"Error in missing values visualization: {e}")
        raise


def fill_missing_values(df, method="mean"):
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


def distribution_analysis(df, numeric_cols, save=True):
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


def scale_features(df, numeric_cols, method="standard"):
    logger.info(f"Scaling features: {numeric_cols} using method: {method}")
    try:
        scaler = StandardScaler() if method == "standard" else MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        logger.info("Features scaled successfully.")
        return df
    except Exception as e:
        logger.error(f"Error scaling features: {e}")
        raise


def correlation_analysis(df, numeric_cols, save=True):
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


def detect_outliers(df, numeric_cols, save=True):
    """Detects outliers in numeric columns."""
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


def remove_null_values(df, subset_cols):
    logger.info("Removing rows with null values in subset columns.")
    try:
        df_clean = df.dropna(subset=subset_cols)
        logger.info("Rows with null values removed successfully.")
        return df_clean
    except Exception as e:
        logger.error(f"Error in removing null values: {e}")
        raise


if __name__ == "__main__":
    try:
        # Load data
        file_path = os.path.join(PROCESSED_DIR, "epa_long_preprocessed.csv")
        df = load_data(file_path)
        basic_info(df, stage="Initial")

        # Visualization before filling missing values
        visualize_missing_values(df, save=True)

        # Fill missing values
        missing_method = config["parameters"].get("missing_value_method", "mean")
        df = fill_missing_values(df, method=missing_method)
        basic_info(df, stage="After Filling Missing Values")

        # Advanced Missing Value Handling (MICE, KNN, Regression)
        concentration_cols = config["data_check"].get("required_columns", [])
        imputation_method = config["parameters"].get("imputation_method", "mice")
        df = advanced_concentration_pipeline(
            df,
            concentration_cols=concentration_cols,
            imputation_method=imputation_method,
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
        processed_file_path = os.path.join(PROCESSED_DIR, "epa_preprocessed.csv")
        try:
            df.to_csv(processed_file_path, index=False)
            logger.info(f"Preprocessed data saved successfully at: {processed_file_path}")
        except Exception as e:
            logger.critical(f"Data preprocessing failed while saving: {e}")
            raise

        # Generate basic info after preprocessing
        basic_info(df, stage="Final")

        # Missing Value Comparison (Original vs Processed)
        # Assuming you have the original data loaded somewhere or accessible
        original_file_path = os.path.join(RAW_DIR, "epa_long_preprocessed.csv")  # Adjust as needed
        original_df = load_data(original_file_path)
        compare_missing_values(original_df, df)

    except Exception as e:
        logger.critical(f"Data preprocessing failed with a critical error: {e}")
        raise
