import glob
import os
from datetime import datetime

import pandas as pd

from source.utils.config_loader import load_config
from source.utils.logger import setup_logger

# Maximum number of rows to read (to speed up processing for large files)
MAX_ROWS_READ = 5000

# Logger setup (to write logs to a file)
logger = setup_logger(
    name="data_check",
    log_file="../logs/data_check.log",  # Can be changed as needed
    log_level="INFO"
)

def validate_columns(df, required_columns):
    missing_cols = [col for col in required_columns if col not in df.columns]
    return len(missing_cols) == 0, missing_cols

def check_raw_data() -> None:
    logger.info("=== Starting data check... ===")

    # 1) Load config
    config = load_config()
    if config is None:
        logger.error("Failed to load config file or returned None, terminating process.")
        return

    # Is 'data_check' or 'required_columns' defined?
    if "data_check" not in config or "required_columns" not in config["data_check"]:
        logger.warning("'data_check' / 'required_columns' not defined in config file. Continuing.")
        required_columns = []
    else:
        required_columns = config["data_check"]["required_columns"]

    # paths -> raw_dir, processed_dir
    raw_dir = config["paths"].get("raw_dir", "../data/raw")
    processed_dir = config["paths"].get("processed_dir", "../data/processed")

    # 2) Find all .csv files (including subfolders)
    csv_files = glob.glob(os.path.join(raw_dir, "**/*.csv"), recursive=True)
    logger.info(f"Number of CSV files found: {len(csv_files)} (directory: {raw_dir})")

    if not csv_files:
        logger.warning("No CSV files found! check_raw_data is terminating.")
        return

    # Collect report rows in a list
    report_rows = []

    # 3) Check each file
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

        # Read the file (first MAX_ROWS_READ rows)
        try:
            df = pd.read_csv(csv_path, nrows=MAX_ROWS_READ)

            if df.empty:
                # Empty file
                logger.warning(f"[Empty] File is completely empty: {csv_path}")
                row_info["empty_file"] = True
                row_info["test_pass"] = False
                row_info["notes"] = "File is completely empty."
            else:
                # File row/column info
                row_info["row_count"] = df.shape[0]
                row_info["col_count"] = df.shape[1]
                row_info["columns"] = ", ".join(df.columns.tolist())

                # Required column check
                cols_ok, missing_cols = validate_columns(df, required_columns)
                if not cols_ok:
                    row_info["test_pass"] = False
                    row_info["notes"] += f" Missing columns: {missing_cols}."

                # Log missing count for only the first 'quick_check_limit' files
                if idx <= config.get("data_check", {}).get("quick_check_limit", 5):
                    missing_dict = df.isnull().sum().to_dict()
                    logger.info(f"[QuickCheck] File={csv_path}, Shape={df.shape}, Missing={missing_dict}")

        except pd.errors.EmptyDataError:
            logger.error(f"[EmptyDataError] File is completely empty: {csv_path}")
            row_info["empty_file"] = True
            row_info["test_pass"] = False
            row_info["notes"] = "EmptyDataError."
        except pd.errors.ParserError as pe:
            logger.error(f"[ParserError] {csv_path}, Error: {pe}")
            row_info["test_pass"] = False
            row_info["notes"] = "ParserError."
        except Exception as e:
            logger.error(f"[UnknownError] {csv_path}, Error: {e}")
            row_info["test_pass"] = False
            row_info["notes"] = f"Unknown error: {e}"

        report_rows.append(row_info)

    # 4) Report DataFrame
    df_report = pd.DataFrame(report_rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"data_check_report_{timestamp}.csv"

    # Create processed_dir if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, report_name)
    df_report.to_csv(output_path, index=False)
    logger.info(f"Data check report created -> {output_path}")

    # Log each row if desired:
    for _, row in df_report.iterrows():
        logger.info(f"[Report] {row.to_dict()}")

    logger.info("=== Data check process completed. ===")

def main():
    try:
        check_raw_data()  # default config_path "../config/settings.yml"
    except Exception as e:
        logger.critical(f"[CRITICAL] Program terminated with a critical error: {e}")

if __name__ == "__main__":
    main()