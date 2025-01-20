import logging
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional

class EDAExploration:

    def __init__(self, logger: logging.Logger, plots_dir: Optional[Path] = None):
        self.logger = logger
        self.plots_dir = plots_dir if plots_dir else Path("../plots").resolve()
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, file_path: Path) -> pd.DataFrame:
        self.logger.info(f"Loading data from {file_path}")
        try:
            data = pd.read_csv(file_path)
            self.logger.info("Data loaded successfully")
            return data
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {e}")
            raise

    def basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        info = {
            "Shape": df.shape,
            "Columns": df.columns.tolist(),
            "Data Types": df.dtypes.to_dict(),
            "Missing Values": df.isnull().sum().to_dict()
        }
        self.logger.info("Basic info generated:")
        self.logger.info(info)
        return info

    def save_or_show_plot(self, fig: plt.Figure, save: bool, save_dir: Optional[Path], plot_name: str) -> None:
        if save:
            if not save_dir:
                save_dir = self.plots_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            plot_path = save_dir / plot_name
            fig.savefig(plot_path)
            self.logger.info(f"Plot saved at: {plot_path}")
            plt.close(fig)
        else:
            plt.show()

    def missing_values(self, df: pd.DataFrame, save: bool = False, save_path: Optional[Path] = None) -> None:
        self.logger.info("Analyzing missing values in the dataset.")
        try:
            missing_percentages = df.isnull().mean() * 100
            self.logger.info("Missing Value Percentages:")
            self.logger.info(missing_percentages)
            print("[INFO] Missing Value Percentages:")
            print(missing_percentages)

            fig, ax = plt.subplots(figsize=(10, 6))
            msno.matrix(df, ax=ax, sparkline=False)
            plt.title("Missing Values Matrix")

            self.save_or_show_plot(fig, save, save_path, "missing_values_matrix.png")
        except Exception as e:
            self.logger.error(f"Error analyzing missing values: {e}")
            raise

    def distribution_analysis(self, df: pd.DataFrame, numeric_cols: List[str],
                              save: bool = False, save_dir: Optional[Path] = None) -> None:
        self.logger.info("Performing distribution analysis for numeric columns.")
        try:
            for col in numeric_cols:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(df[col], kde=True, bins=30, color="blue", ax=ax)
                ax.set_title(f"Distribution of {col}")
                self.save_or_show_plot(fig, save, save_dir, f"distribution_{col}.png")
        except Exception as e:
            self.logger.error(f"Error in distribution analysis: {e}")
            raise

    def correlation_analysis(self, df: pd.DataFrame, numeric_cols: List[str],
                             save: bool = False, save_dir: Optional[Path] = None) -> None:
        self.logger.info("Creating correlation matrix for numeric columns.")
        try:
            corr_matrix = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Correlation Matrix")
            self.save_or_show_plot(fig, save, save_dir, "correlation_matrix.png")
        except Exception as e:
            self.logger.error(f"Error in correlation analysis: {e}")
            raise

    def detect_outliers(self, df: pd.DataFrame, numeric_cols: List[str],
                        save: bool = False, save_dir: Optional[Path] = None) -> None:
        self.logger.info("Detecting outliers in numeric columns.")
        try:
            for col in numeric_cols:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.boxplot(x=df[col], ax=ax)
                ax.set_title(f"Outliers in {col}")
                self.save_or_show_plot(fig, save, save_dir, f"outliers_{col}.png")
        except Exception as e:
            self.logger.error(f"Error in outlier detection: {e}")
            raise