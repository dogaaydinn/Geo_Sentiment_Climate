import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from source.utils.config_loader import load_config
from source.utils.hash_utils import compute_md5
from source.utils.logger import setup_logger
from source.utils.metadata_manager import (
    load_processed_files,
    is_file_processed,
    mark_file_as_processed
)


def add_source_to_sys_path():
    """
    Adds the 'source' directory to the system path.
    """
    import sys
    source_path = Path(__file__).resolve().parent
    if str(source_path) not in sys.path:
        sys.path.append(str(source_path))


# Add source to sys.path
add_source_to_sys_path()

# Load config

CONFIG_PATH = Path("../config/settings.yml").resolve()
config = load_config(CONFIG_PATH)

# Validate required config keys
REQUIRED_KEYS = [
    "raw_dir", "interim_dir", "processed_dir",
    "archive_dir", "metadata_dir", "logs_dir"
]
missing_keys = [key for key in REQUIRED_KEYS if key not in config.get("paths", {})]
if missing_keys:
    raise KeyError(f"Missing required config keys: {missing_keys}")

# Define directories using pathlib
RAW_DIR = Path(config["paths"]["raw_dir"]).resolve()
INTERIM_DIR = Path(config["paths"].get("interim_dir", "01-data/interim")).resolve()
PROCESSED_DIR = Path(config["paths"]["processed_dir"]).resolve()
ARCHIVE_DIR = Path(config["paths"]["archive_dir"]).resolve()
METADATA_DIR = Path(config["paths"]["metadata_dir"]).resolve()
METADATA_PATH = METADATA_DIR / "processed_files.json"
MAX_ROWS_READ = config.get("ingestion", {}).get("max_rows_read", None)  # None means no limit

# Setup logger
logger = setup_logger(
    name="data_ingestion",
    log_file=Path(config["paths"]["logs_dir"]) / "data_ingestion.log",
    log_level=config.get("logging", {}).get("level", "INFO").upper()
)


def find_csv_files(directory: Path) -> List[Path]:
    """
    Finds all CSV files within a directory and its subdirectories.

    Args:
        directory (Path): The directory to search.

    Returns:
        List[Path]: List of CSV file paths.
    """
    csv_files = list(directory.rglob("*.csv"))
    logger.debug(f"Found {len(csv_files)} CSV files in {directory}")
    return csv_files


def load_metadata(metadata_path: Path) -> Dict[str, Any]:
    """
    Loads metadata from a JSON file.

    Args:
        metadata_path (Path): Path to the metadata JSON file.

    Returns:
        Dict[str, Any]: Metadata dictionary.
    """
    return load_processed_files(metadata_path)


def save_metadata(metadata: Dict[str, Any], metadata_path: Path) -> None:
    """
    Saves the updated metadata to the JSON file.

    Args:
        metadata (Dict[str, Any]): Metadata dictionary.
        metadata_path (Path): Path to the metadata JSON file.
    """
    try:
        mark_file_as_processed(
            file_name="metadata_update",
            file_hash="N/A",
            rows_count=0,
            metadata=metadata,
            metadata_path=metadata_path
        )
        logger.debug(f"Metadata saved to {metadata_path}.")
    except Exception as e:
        logger.error(f"Error saving metadata to {metadata_path}: {e}")



def save_metadata(metadata: Dict[str, Any], metadata_path: Path) -> None:
    """
    Saves the updated metadata to the JSON file.

    Args:
        metadata (Dict[str, Any]): Metadata dictionary.
        metadata_path (Path): Path to the metadata JSON file.
    """
    try:
        mark_file_as_processed(metadata=metadata, metadata_path=metadata_path)
        logger.debug(f"Metadata saved to {metadata_path}.")
    except Exception as e:
        logger.error(f"Error saving metadata to {metadata_path}: {e}")


def process_file(
    csv_path: Path,
    required_columns: List[str],
    metadata: Dict[str, Any],
    max_rows_read: int = None
) -> pd.DataFrame:
    """
    Processes a single CSV file: deduplication, column validation, etc.

    Args:
        csv_path (Path): Path to the CSV file.
        required_columns (List[str]): List of required columns.
        metadata (Dict[str, Any]): Metadata dictionary.
        max_rows_read (int, optional): Maximum number of rows to read. Defaults to None.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    file_name = csv_path.name
    logger.info(f"Processing file: {file_name}")

    # Compute file hash
    try:
        file_hash = compute_md5(csv_path)
        logger.debug(f"Computed MD5 for {file_name}: {file_hash}")
    except Exception as e:
        logger.error(f"Failed to compute MD5 for {file_name}: {e}")
        return pd.DataFrame()  # Return empty DataFrame to skip processing

    # Check if file has been processed
    if is_file_processed(file_hash, metadata):
        logger.info(f"File {file_name} has already been processed. Skipping.")
        return pd.DataFrame()

    # Read CSV
    try:
        df = pd.read_csv(csv_path, nrows=max_rows_read) if max_rows_read else pd.read_csv(csv_path)
        logger.debug(f"Read {len(df)} rows from {file_name}")
        if df.empty:
            logger.warning(f"File {file_name} is empty. Marking as processed with 0 rows.")
            mark_file_as_processed(file_name, file_hash, 0, metadata, METADATA_PATH)
            return pd.DataFrame()
    except pd.errors.EmptyDataError:
        logger.error(f"EmptyDataError: File {file_name} is empty.")
        mark_file_as_processed(file_name, file_hash, 0, metadata, METADATA_PATH)
        return pd.DataFrame()
    except pd.errors.ParserError as pe:
        logger.error(f"ParserError while reading {file_name}: {pe}")
        mark_file_as_processed(file_name, file_hash, 0, metadata, METADATA_PATH)
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error while reading {file_name}: {e}")
        mark_file_as_processed(file_name, file_hash, 0, metadata, METADATA_PATH)
        return pd.DataFrame()

    # Deduplication
    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    final_shape = df.shape
    logger.info(f"Deduplicated {file_name}: {initial_shape} -> {final_shape}")

    # Validate and align columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"File {file_name} is missing columns: {missing_columns}. Adding with NaN values.")
        for col in missing_columns:
            df[col] = pd.NA
    df = df[required_columns]  # Reorder columns

    # Update metadata
    rows_count = final_shape[0]
    mark_file_as_processed(file_name, file_hash, rows_count, metadata, METADATA_PATH)
    logger.info(f"File {file_name} marked as processed with {rows_count} rows.")

    return df



def archive_file(csv_path: Path, archive_dir: Path) -> None:
    """
    Archives the processed CSV file by moving it to the archive directory with a timestamp.

    Args:
        csv_path (Path): Path to the CSV file.
        archive_dir (Path): Path to the archive directory.
    """
    try:
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file_name = f"{timestamp}_{csv_path.name}"
        archive_path = archive_dir / archive_file_name
        shutil.move(str(csv_path), str(archive_path))
        logger.info(f"Archived file {csv_path.name} to {archive_path}")
    except Exception as e:
        logger.error(f"Failed to archive file {csv_path.name}: {e}")


def ingest_data(raw_dir: Path = RAW_DIR) -> None:
    """
    Performs data ingestion: reads raw CSV files, processes them, and archives the raw files.

    Args:
        raw_dir (Path, optional): Directory containing raw CSV files. Defaults to RAW_DIR.
    """
    logger.info("=== Starting data ingestion process ===")

    # Load metadata
    metadata = load_metadata(METADATA_PATH)

    # Find all CSV files
    csv_files = find_csv_files(raw_dir)
    logger.info(f"Found {len(csv_files)} CSV files in {raw_dir}")

    if not csv_files:
        logger.warning("No CSV files found. Terminating data ingestion process.")
        return

    # Get required columns from config
    required_columns = config["data_check"].get("required_columns", [])
    if not required_columns:
        logger.warning("No required columns specified in config. Proceeding without column validation.")

    # Initialize combined DataFrame
    combined_df = pd.DataFrame()

    for csv_path in csv_files:
        processed_df = process_file(
            csv_path=csv_path,
            required_columns=required_columns,
            metadata=metadata,
            max_rows_read=MAX_ROWS_READ
        )

        if not processed_df.empty:
            combined_df = pd.concat([combined_df, processed_df], ignore_index=True)
            logger.info(f"Appended data from {csv_path.name} to combined DataFrame.")

            # Save to interim directory
            try:
                INTERIM_DIR.mkdir(parents=True, exist_ok=True)
                interim_file = INTERIM_DIR / f"interim_{csv_path.name}"
                processed_df.to_csv(interim_file, index=False)
                logger.info(f"Saved interim data to {interim_file}")
            except Exception as e:
                logger.error(f"Failed to save interim data for {csv_path.name}: {e}")
                continue

            # Move raw file to archive
            archive_file(csv_path, ARCHIVE_DIR)

    # Final Deduplication on Combined DataFrame
    if not combined_df.empty:
        initial_shape = combined_df.shape
        combined_df.drop_duplicates(inplace=True)
        final_shape = combined_df.shape
        logger.info(f"Combined DataFrame deduplication: {initial_shape} -> {final_shape}")

        # Save the combined DataFrame to processed directory
        output_file = PROCESSED_DIR / "epa_long_preprocessed.csv"
        try:
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            combined_df.to_csv(output_file, index=False)
            logger.info(f"Saved combined DataFrame to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save combined DataFrame: {e}")
    else:
        logger.warning("No data to combine. Skipping saving combined DataFrame.")

    # Save updated metadata
    save_metadata(metadata, METADATA_PATH)

    logger.info("=== Data ingestion process completed ===")



if __name__ == "__main__":
    ingest_data()
