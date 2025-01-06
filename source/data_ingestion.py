import glob
import os
import shutil
import pandas as pd
from datetime import datetime
from source.utils.path_utils import add_source_to_sys_path
from source.utils.config_loader import load_config
from source.utils.hash_utils import compute_md5
from source.utils.logger import setup_logger
from source.utils.metadata_manager import load_processed_files, is_file_processed, mark_file_as_processed

# Add source to sys.path
add_source_to_sys_path()

# Load config
config_path = os.path.join(os.path.abspath("../config"), "settings.yml")
config = load_config(config_path)

# Check if all required config keys are present
required_keys = ["raw_dir", "interim_dir", "processed_dir", "archive_dir", "metadata_dir", "logs_dir"]
for key in required_keys:
    if key not in config["paths"]:
        raise KeyError(f"Key 'paths.{key}' not found in config file.")

RAW_DIR = os.path.abspath(config["paths"]["raw_dir"])
INTERIM_DIR = os.path.abspath(config["paths"].get("interim_dir", "data/interim"))
PROCESSED_DIR = os.path.abspath(config["paths"]["processed_dir"])
ARCHIVE_DIR = os.path.abspath(config["paths"]["archive_dir"])
METADATA_DIR = os.path.abspath(config["paths"]["metadata_dir"])
METADATA_PATH = os.path.abspath(os.path.join(METADATA_DIR, "processed_files.json"))
MAX_ROWS_READ = config.get("ingestion", {}).get("max_rows_read", 5000)

# Logger Setup
logger = setup_logger(
    name="data_ingestion_advanced",
    log_file=os.path.join(config["paths"]["logs_dir"], "data_ingestion_advanced.log"),
    log_level="INFO"
)


def ingest_data(raw_dir=RAW_DIR):
    logger.info("Starting data ingestion process.")

    # 1. Load metadata
    try:
        metadata = load_processed_files(METADATA_PATH)
        logger.debug(f"Loaded metadata: {metadata}")
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        return

    # 2. Find all CSV files
    csv_files = glob.glob(os.path.join(raw_dir, "**/*.csv"), recursive=True)
    logger.info(f"Number of CSV files found: {len(csv_files)}")

    if not csv_files:
        logger.warning("No CSV files found. Terminating ingestion process.")
        return

    combined_df = pd.DataFrame()

    for csv_path in csv_files:
        file_name = os.path.basename(csv_path)
        logger.info(f"Processing: {file_name}")

        # 3. Compute file hash
        try:
            file_hash = compute_md5(csv_path)
            logger.info(f"File hash: {file_hash}")
        except Exception as e:
            logger.error(f"Failed to compute hash for {file_name}: {e}")
            continue

        # 4. Check if file has been processed before
        if is_file_processed(file_hash, metadata):
            logger.info(f"{file_name} has already been processed. Skipping.")
            continue

        # 5. Read CSV into DataFrame (limited number of rows)
        try:
            df = pd.read_csv(csv_path, nrows=MAX_ROWS_READ)
            logger.debug(f"Number of rows read from {file_name}: {len(df)}")
            if df.empty:
                logger.warning(f"{file_name} is an empty file. Not processing.")
                mark_file_as_processed(file_name, file_hash, 0, metadata, METADATA_PATH)
                continue
        except Exception as e:
            logger.error(f"Failed to read {file_name}: {e}")
            mark_file_as_processed(file_name, file_hash, 0, metadata, METADATA_PATH)
            continue

        # 6. Perform deduplication
        initial_shape = df.shape
        df.drop_duplicates(inplace=True)
        final_shape = df.shape
        logger.info(f"Deduplication: {initial_shape} -> {final_shape}")

        combined_df = pd.concat([combined_df, df], ignore_index=True)

        # 7. Save processed data to interim
        try:
            os.makedirs(INTERIM_DIR, exist_ok=True)
            interim_file = os.path.join(INTERIM_DIR, f"interim_{file_name}")
            df.to_csv(interim_file, index=False)
            logger.info(f"Processed data saved to interim: {interim_file}")
        except Exception as e:
            logger.error(f"Error saving processed data to interim: {e}")
            mark_file_as_processed(file_name, file_hash, final_shape[0], metadata, METADATA_PATH)
            continue

        # 8. Move processed data to processed directory
        try:
            os.makedirs(PROCESSED_DIR, exist_ok=True)
            processed_file = os.path.join(PROCESSED_DIR, f"processed_{file_name}")
            shutil.move(interim_file, processed_file)
            logger.info(f"Processed data moved to processed directory: {processed_file}")
        except Exception as e:
            logger.error(f"Error moving processed data to processed directory: {e}")
            mark_file_as_processed(file_name, file_hash, final_shape[0], metadata, METADATA_PATH)
            continue

        # 9. Move file to archive
        try:
            os.makedirs(ARCHIVE_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_file_name = f"{timestamp}_{file_name}"
            archive_path = os.path.join(ARCHIVE_DIR, archive_file_name)
            shutil.move(csv_path, archive_path)
            logger.info(f"File moved to archive: {archive_path}")
        except Exception as e:
            logger.error(f"Error moving file to archive: {e}")
            # Continue processing
            pass

        # 10. Update metadata
        rows_count = final_shape[0]
        try:
            mark_file_as_processed(file_name, file_hash, rows_count, metadata, METADATA_PATH)
            logger.info(f"{file_name} added to metadata.")
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")

    logger.info(f"Combined DataFrame size: {combined_df.shape}")

    # Save the combined DataFrame to a CSV file
    output_file = os.path.join(PROCESSED_DIR, "combined_data.csv")
    try:
        combined_df.to_csv(output_file, index=False)
        logger.info(f"Combined DataFrame saved: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save combined DataFrame: {e}")

    logger.info("Data ingestion process completed.")


if __name__ == "__main__":
    ingest_data()
