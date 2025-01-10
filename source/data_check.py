import glob
import os
from datetime import datetime
from typing import List, Tuple, Dict, Any

import pandas as pd

from source.utils.config_loader import load_config
from source.utils.logger import setup_logger
from source.utils.path_utils import add_source_to_sys_path

MAX_ROWS_READ = 5000

# Load config
CONFIG_PATH = os.path.join(os.path.abspath("../config"), "settings.yml")
config = load_config(CONFIG_PATH)

# Add source to sys.path
add_source_to_sys_path()

# Validate required config keys
REQUIRED_KEYS = ["raw_dir", "interim_dir", "processed_dir", "archive_dir", "metadata_dir", "logs_dir"]
missing_keys = [key for key in REQUIRED_KEYS if key not in config.get("paths", {})]
if missing_keys:
    raise KeyError(f"Missing required config keys: {missing_keys}")

RAW_DIR = os.path.abspath(config["paths"]["raw_dir"])

# Logger setup
logger = setup_logger(
    name="data_check",
    log_file=os.path.join(config["paths"]["logs_dir"], "data_check.log"),
    log_level=config.get("logging", {}).get("level", "INFO").upper()
)


def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Validates if the DataFrame contains all required columns.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        required_columns (List[str]): List of required column names.

    Returns:
        Tuple[bool, List[str]]: A tuple indicating if validation passed and missing columns.
    """
    missing_cols = [col for col in required_columns if col not in df.columns]
    return len(missing_cols) == 0, missing_cols


def find_csv_files(directory: str) -> List[str]:
    """
    Finds all CSV files within a directory and its subdirectories.

    Args:
        directory (str): The directory to search.

    Returns:
        List[str]: List of CSV file paths.
    """
    csv_files = glob.glob(os.path.join(directory, "**/*.csv"), recursive=True)
    logger.debug(f"Found {len(csv_files)} CSV files in {directory}")
    return csv_files


def load_csv(filepath: str, nrows: int = MAX_ROWS_READ) -> pd.DataFrame:
    """
    Loads a CSV file into a DataFrame.

    Args:
        filepath (str): Path to the CSV file.
        nrows (int, optional): Number of rows to read. Defaults to MAX_ROWS_READ.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(filepath, nrows=nrows)


def generate_report(report_rows: List[Dict[str, Any]], processed_dir: str) -> str:
    """
    Generates a CSV report from report rows.

    Args:
        report_rows (List[Dict[str, Any]]): List of report data.
        processed_dir (str): Directory to save the report.

    Returns:
        str: Path to the generated report.
    """
    df_report = pd.DataFrame(report_rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"data_check_report_{timestamp}.csv"
    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, report_name)
    df_report.to_csv(output_path, index=False)
    logger.info(f"Data check report created at {output_path}")
    return output_path


def check_raw_data(raw_dir: str = RAW_DIR) -> None:
    """
    Performs data validation on raw CSV files.

    Args:
        raw_dir (str, optional): Directory containing raw CSV files. Defaults to RAW_DIR.
    """
    logger.info("=== Starting data check ===")

    csv_files = find_csv_files(raw_dir)
    if not csv_files:
        logger.warning("No CSV files found in the specified directory. Terminating data check.")
        return

    config_data_check = config.get("data_check", {})
    required_columns = config_data_check.get("required_columns", [])
    quick_check_limit = config_data_check.get("quick_check_limit", 5)

    report_rows = []

    for idx, csv_path in enumerate(csv_files, start=1):
        row_info = {
            "file_path": csv_path,
            "row_count": None,
            "col_count": None,
            "columns": None,
            "empty_file": False,
            "test_pass": True,
            "notes": "",
        }

        try:
            df = load_csv(csv_path)
            if df.empty:
                logger.warning(f"[Empty] The file is empty: {csv_path}")
                row_info["empty_file"] = True
                row_info["test_pass"] = False
                row_info["notes"] = "File is completely empty."
            else:
                row_info["row_count"] = df.shape[0]
                row_info["col_count"] = df.shape[1]
                row_info["columns"] = ", ".join(df.columns.tolist())

                cols_ok, missing_cols = validate_columns(df, required_columns)
                if not cols_ok:
                    row_info["test_pass"] = False
                    row_info["notes"] += f" Missing columns: {missing_cols}."

                if idx <= quick_check_limit:
                    missing_dict = df.isnull().sum().to_dict()
                    logger.info(f"[QuickCheck] File={csv_path}, Shape={df.shape}, Missing={missing_dict}")

        except pd.errors.EmptyDataError:
            logger.error(f"[EmptyDataError] The file is empty: {csv_path}")
            row_info["empty_file"] = True
            row_info["test_pass"] = False
            row_info["notes"] = "EmptyDataError."
        except pd.errors.ParserError as pe:
            logger.error(f"[ParserError] Error parsing {csv_path}: {pe}")
            row_info["test_pass"] = False
            row_info["notes"] = "ParserError."
        except Exception as e:
            logger.error(f"[UnknownError] Error processing {csv_path}: {e}")
            row_info["test_pass"] = False
            row_info["notes"] = f"Unknown error: {e}"

        report_rows.append(row_info)
        logger.debug(f"Processed file {idx}/{len(csv_files)}: {csv_path}")

    report_path = generate_report(report_rows, config["paths"]["processed_dir"])

    logger.info("=== Data check completed ===")


if __name__ == "__main__":
    check_raw_data()
