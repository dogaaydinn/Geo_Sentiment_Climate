# source/eda_exploration.py

import logging

import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any

from source.utils.path_utils import add_source_to_sys_path
from source.utils.logger import setup_logger

def add_source_to_sys_path_if_needed():
    """
    Adds the 'source' directory to the system path if it's not already included.
    """
    import sys
    source_path = Path(__file__).resolve().parent
    if str(source_path) not in sys.path:
        sys.path.append(str(source_path))

# Add source to sys.path
add_source_to_sys_path_if_needed()

# Setup logger
logger = setup_logger(
    name="eda_exploration",
    log_file=Path("../04-logs") / "eda_exploration.log",
    log_level="INFO"
)

def load_data(file_path: Path) -> pd.DataFrame:
    """
    Loads data from a CSV file.

    Args:
        file_path (Path): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    logger.info(f"Loading data from {file_path}")
    try:
        data = pd.read_csv(file_path)
        logger.info("Data loaded successfully")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def basic_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates basic information about the dataset.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        Dict[str, Any]: Dictionary containing basic information.
    """
    info = {
        "Shape": df.shape,
        "Columns": df.columns.tolist(),
        "Data Types": df.dtypes.to_dict(),
        "Missing Values": df.isnull().sum().to_dict()
    }
    logger.info("Basic info generated.")
    logger.info(f"Dataset Info: {info}")
    return info

def missing_values(df: pd.DataFrame, save: bool = False, save_path: Path = None) -> None:
    """
    Analyzes and visualizes missing values in the dataset.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        save (bool, optional): Whether to save the plot. Defaults to False.
        save_path (Path, optional): Path to save the plot if save is True. Defaults to None.
    """
    logger.info("Analyzing missing values in the dataset.")
    try:
        missing_percentages = df.isnull().mean() * 100
        logger.info("Missing Value Percentages:")
        logger.info(missing_percentages)

        print("[INFO] Missing Value Percentages:")
        print(missing_percentages)

        fig, ax = plt.subplots(figsize=(10, 6))
        msno.matrix(df, ax=ax)
        plt.title("Missing Values Matrix")
        if save and save_path:
            fig.savefig(save_path)
            logger.info(f"Missing values matrix saved at: {save_path}")
            plt.close(fig)
        else:
            plt.show()
    except Exception as e:
        logger.error(f"Error analyzing missing values: {e}")
        raise

def distribution_analysis(df: pd.DataFrame, numeric_cols: List[str], save: bool = False, save_dir: Path = None) -> None:
    """
    Performs distribution analysis for numeric columns and visualizes them.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        numeric_cols (List[str]): List of numeric columns.
        save (bool, optional): Whether to save the plots. Defaults to False.
        save_dir (Path, optional): Directory to save the plots if save is True. Defaults to None.
    """
    logger.info("Performing distribution analysis for numeric columns.")
    try:
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[col], kde=True, bins=30, color="blue", ax=ax)
            ax.set_title(f"Distribution of {col}")
            if save and save_dir:
                plot_path = save_dir / f"distribution_{col}.png"
                fig.savefig(plot_path)
                logger.info(f"Distribution plot for {col} saved at: {plot_path}")
                plt.close(fig)
            else:
                plt.show()
    except Exception as e:
        logger.error(f"Error in distribution analysis: {e}")
        raise

def correlation_analysis(df: pd.DataFrame, numeric_cols: List[str], save: bool = False, save_dir: Path = None) -> None:
    """
    Creates and visualizes a correlation matrix for numeric columns.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        numeric_cols (List[str]): List of numeric columns.
        save (bool, optional): Whether to save the plot. Defaults to False.
        save_dir (Path, optional): Directory to save the plot if save is True. Defaults to None.
    """
    logger.info("Creating correlation matrix for numeric columns.")
    try:
        corr_matrix = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Matrix")
        if save and save_dir:
            plot_path = save_dir / "correlation_matrix.png"
            fig.savefig(plot_path)
            logger.info(f"Correlation matrix saved at: {plot_path}")
            plt.close(fig)
        else:
            plt.show()
    except Exception as e:
        logger.error(f"Error in correlation analysis: {e}")
        raise

def detect_outliers(df: pd.DataFrame, numeric_cols: List[str], save: bool = False, save_dir: Path = None) -> None:
    """
    Detects and visualizes outliers in numeric columns using boxplots.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        numeric_cols (List[str]): List of numeric columns.
        save (bool, optional): Whether to save the plots. Defaults to False.
        save_dir (Path, optional): Directory to save the plots if save is True. Defaults to None.
    """
    logger.info("Detecting outliers in numeric columns.")
    try:
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f"Outliers in {col}")
            if save and save_dir:
                plot_path = save_dir / f"outliers_{col}.png"
                fig.savefig(plot_path)
                logger.info(f"Outlier plot for {col} saved at: {plot_path}")
                plt.close(fig)
            else:
                plt.show()
    except Exception as e:
        logger.error(f"Error in outlier detection: {e}")
        raise
