import shutil
import sys
from datetime import datetime
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from source.utils.config_loader import load_config
from source.utils.logger import setup_logger
from source.utils.metadata_manager import (
    load_processed_files,
    is_file_processed,
    mark_file_as_processed
)
from source.utils.hash_utils import compute_md5

def add_source_to_sys_path():
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.append(str(here))

add_source_to_sys_path()

# Load config
CONFIG_PATH = Path("../config/settings.yml").resolve()
config = load_config(CONFIG_PATH)

RAW_DIR = Path(config["paths"]["raw_dir"]).resolve()
INTERIM_DIR = Path(config["paths"]["interim_dir"]).resolve()
PROCESSED_DIR = Path(config["paths"]["processed_dir"]).resolve()
ARCHIVE_DIR = Path(config["paths"]["archive_dir"]).resolve()
METADATA_DIR = Path(config["paths"]["metadata_dir"]).resolve()
METADATA_PATH = METADATA_DIR / "processed_files.json"

logger = setup_logger(
    name="data_ingestion_o3",
    log_file=Path(config["paths"]["logs_dir"]) / "data_ingestion_o3.log",
    log_level=config.get("logging", {}).get("level", "INFO").upper()
)

def load_metadata(path: Path) -> Dict[str, Any]:
    return load_processed_files(path)

def save_metadata(metadata: Dict[str, Any], path: Path):
    # Burada mark_file_as_processed fonksiyonu parametrelerini doğru vermeliyiz!
    # Ama metadata güncelleme "global" parametre gibi kullanacaksak:
    logger.debug(f"Saving metadata with {len(metadata.get('processed_files',[]))} entries")
    # Örneğin:
    #   mark_file_as_processed(
    #       file_name="metadata_update",
    #       file_hash="N/A",
    #       rows_count=0,
    #       metadata=metadata,
    #       metadata_path=path
    #   )
    # Veya "dosya bazında" ekleniyorsa, process_file da parametre gönderiyor.
    pass

def find_o3_csv_files(raw_dir: Path) -> list[Path]:

    csv_files = list(raw_dir.rglob("*.csv"))
    o3_files = [f for f in csv_files if "o3" in f.name.lower()]
    return o3_files

def process_o3_file(csv_path: Path, metadata: Dict[str, Any], max_rows_read: Optional[int] = None) -> pd.DataFrame:
    file_name = csv_path.name
    logger.info(f"Processing o3 file: {file_name}")

    # MD5
    try:
        file_hash = compute_md5(csv_path)
    except Exception as e:
        logger.error(f"Failed to compute MD5 for {file_name}: {e}")
        return pd.DataFrame()

    # Already processed?
    if is_file_processed(file_hash, metadata):
        logger.info(f"File {file_name} has been processed. Skipping.")
        return pd.DataFrame()

    # Read CSV
    try:
        if max_rows_read:
            df = pd.read_csv(csv_path, nrows=max_rows_read)
        else:
            df = pd.read_csv(csv_path)

        if df.empty:
            logger.warning(f"{file_name} is empty => marking processed with 0 rows.")
            mark_file_as_processed(file_name, file_hash, 0, metadata, METADATA_PATH)
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error reading {file_name}: {e}")
        mark_file_as_processed(file_name, file_hash, 0, metadata, METADATA_PATH)
        return pd.DataFrame()

    # Minimal dedup
    init_shape = df.shape
    df.drop_duplicates(inplace=True)
    final_shape = df.shape
    logger.info(f"Deduplicated {file_name}: {init_shape} => {final_shape}")


    rows_count = df.shape[0]
    mark_file_as_processed(file_name, file_hash, rows_count, metadata, METADATA_PATH)
    logger.info(f"File {file_name} => {rows_count} rows processed for o3")

    return df

def archive_file(csv_path: Path):
    try:
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        arc_name = f"{ts}_{csv_path.name}"
        arc_path = ARCHIVE_DIR / arc_name
        shutil.move(str(csv_path), str(arc_path))
        logger.info(f"Archived {csv_path.name} => {arc_path}")
    except Exception as e:
        logger.error(f"Failed to archive {csv_path.name}: {e}")


def ingest_o3_data():

    logger.info("=== Starting o3 ingestion ===")
    metadata = load_metadata(METADATA_PATH)

    o3_files = find_o3_csv_files(RAW_DIR)
    logger.info(f"Found {len(o3_files)} o3files in raw_dir")

    ingestion_cfg = config.get("ingestion", {})
    max_rows_read = ingestion_cfg.get("max_rows_read", None)

    combined_o3 = pd.DataFrame()
    for path in o3_files:
        df_o3 = process_o3_file(path, metadata, max_rows_read)
        if not df_o3.empty:
            combined_o3 = pd.concat([combined_o3, df_o3], ignore_index=True)
            # Save interim
            try:
                INTERIM_DIR.mkdir(parents=True, exist_ok=True)
                interim_file = INTERIM_DIR / f"interim_{path.name}"
                df_o3.to_csv(interim_file, index=False)
                logger.info(f"Saved interim => {interim_file}")
            except Exception as e:
                logger.error(f"Failed to save interim for {path.name}: {e}")
            # Archive
            archive_file(path)

    if not combined_o3.empty:
        # final deduplicate
        init_shape = combined_o3.shape
        combined_o3.drop_duplicates(inplace=True)
        final_shape = combined_o3.shape
        logger.info(f"Combined o3 dedup: {init_shape} => {final_shape}")

        # save o3_preprocessed
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_file = PROCESSED_DIR / "o3_preprocessed.csv"
        combined_o3.to_csv(out_file, index=False)
        logger.info(f"Saved o3 preprocessed => {out_file}")
    else:
        logger.warning("No o3 data combined => skipping final save.")

    # update metadata? e.g. save_metadata(metadata, METADATA_PATH)
    logger.info("=== o3 ingestion completed ===")

if __name__ == "__main__":
    ingest_o3_data()
