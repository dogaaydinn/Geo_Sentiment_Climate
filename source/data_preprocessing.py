import numpy as np
import matplotlib.axes as maxes
from pathlib import Path
from source.utils.logger import setup_logger
from source.config.config_utils import config
from source.utils.project_paths import ProjectPaths
from source.config.config_loader import check_required_keys
from source.missing_handle import advanced_concentration_pipeline
from source.missing_value_comparison import compare_missing_values
from source.utils.path_initializer import add_source_to_sys_path
from source.data.data_preprocessing.data_preprocessor import DataPreprocessor

add_source_to_sys_path()

# Setup logger
logger = setup_logger(
    name="data_preprocessing",
    log_file=str(Path(config["paths"]["logs_dir"]) / "data_preprocessing.log"),
    log_level=config.get("logging", {}).get("level", "INFO").upper()
)

REQUIRED_KEYS = ["raw_dir", "processed_dir", "plots_dir", "logs_dir"]
check_required_keys(config, REQUIRED_KEYS)

paths = ProjectPaths.from_config(config)
paths.ensure_directories()

logger.info("Config file and directories loaded successfully.")

# --- Monkey patch for missingno issue with grid_b ---
_original_tick_params = maxes.Axes.tick_params

def _patched_tick_params(self, *args, **kwargs):
    if "grid_b" in kwargs:
        kwargs.pop("grid_b")

    return _original_tick_params(self, *args, **kwargs)

if not hasattr(maxes.Axes, "_already_patched_grid_b"):
    maxes.Axes._already_patched_grid_b = True
    maxes.Axes.tick_params = _patched_tick_params

# Initialize DataPreprocessor
data_preprocessor = DataPreprocessor(logger)

def preprocess_data() -> None:
    try:
        # Load data
        file_path = Path(config["paths"]["processed_dir"]) / "epa_long_preprocessed.csv"
        df = data_preprocessor.load_data(file_path)
        data_preprocessor.basic_info(df, stage="Initial")

        # Visualization before filling missing values
        data_preprocessor.visualize_missing_values(df, save=True)

        # Fill missing values
        missing_method = config["parameters"].get("missing_value_method", "mean")
        df = data_preprocessor.fill_missing_values(df, method=missing_method)
        data_preprocessor.basic_info(df, stage="After Filling Missing Values")

        # Advanced Missing Value Handling (MICE, KNN, Regression)
        df = advanced_concentration_pipeline(
            df,
            concentration_cols=config["data_check"].get("required_columns", []),
            imputation_method=config["parameters"].get("imputation_method", "mice"),
            random_state=config["parameters"].get("random_state", 42),
            n_neighbors=config["parameters"].get("n_neighbors", 5)
        )

        from source.advanced_cleanup import advanced_cleanup
        logger.info(f"Data preprocessing completed. Final shape before advanced cleanup: {df.shape}")

        df = advanced_cleanup(
            df,
            pollutant="mixed",
            remove_rarely_used=True,
            drop_optional=False,
            correlation_threshold=0.95,
            outlier_method="iqr",
            save_path=Path(config["paths"]["processed_dir"]) / "epa_long_preprocessed_ADV.csv"
        )

        logger.info(f"Advanced cleanup completed. Final shape after advanced cleanup: {df.shape}")

        # Save preprocessed data
        save_dir = Path(config["paths"]["processed_dir"])
        data_preprocessor.save_preprocessed_data(df, filename="epa_long_preprocessed_ADV.csv", save_dir=save_dir)

        # Scale features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        scaling_method = config["parameters"].get("scaling_method", "standard")
        df = data_preprocessor.scale_features(df, numeric_cols, method=scaling_method)

        # Correlation analysis
        data_preprocessor.correlation_analysis(df, numeric_cols, save=True)

        # Detect outliers
        data_preprocessor.detect_outliers(df, numeric_cols, save=True)

        # Distribution analysis
        data_preprocessor.distribution_analysis(df, numeric_cols, save=True)

        # Save preprocessed data
        save_dir = Path(config["paths"]["processed_dir"])
        filename = "epa_long_preprocessed_ADV.csv"
        data_preprocessor.save_preprocessed_data(df, filename=filename, save_dir=save_dir)

        # Generate basic info after preprocessing
        data_preprocessor.basic_info(df, stage="Final")

        # Missing Value Comparison (Original vs Processed)
        original_file_path = Path(config["paths"]["raw_dir"]) / "epa_long_preprocessed.csv"
        original_df = data_preprocessor.load_data(original_file_path)
        compare_missing_values(original_df, df)

    except Exception as e:
        logger.critical(f"Data preprocessing failed with a critical error: {e}")
        raise

if __name__ == "__main__":
    preprocess_data()