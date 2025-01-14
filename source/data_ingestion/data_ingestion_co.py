import shutil
import sys
import os
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


def add_source_to_sys_path_if_needed():

    import sys
    source_path = Path(__file__).resolve().parent.parent / "source"
    if str(source_path) not in sys.path:
        sys.path.append(str(source_path))

# Add source to sys.path
add_source_to_sys_path_if_needed()

# Load config
CONFIG_PATH = Path("../config/settings.yml").resolve()
config = load_config(CONFIG_PATH)

if config is None:
    raise ValueError("Config file could not be loaded or returned empty.")


RAW_DIR = Path(config["paths"]["raw_dir"]).resolve()
INTERIM_DIR = Path(config["paths"]["interim_dir"]).resolve()
PROCESSED_DIR = Path(config["paths"]["processed_dir"]).resolve()
ARCHIVE_DIR = Path(config["paths"]["archive_dir"]).resolve()
METADATA_DIR = Path(config["paths"]["metadata_dir"]).resolve()
METADATA_PATH = METADATA_DIR / "processed_files.json"

logger = setup_logger(
    name="data_ingestion_no2",
    log_file=Path(config["paths"]["logs_dir"]) / "data_ingestion_no2.log",
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

def find_co_csv_files(raw_dir: Path) -> list[Path]:
    # Yalnız co dosyalarını (co) barındıran dosya adlarını bulmak
    # Örneğin => "epa-co-2022", "epa_co_arkansas_2023.csv" gibi
    csv_files = list(raw_dir.rglob("*.csv"))
    co_files = [f for f in csv_files if "co" in f.name.lower()]
    return co_files

def process_co_file(csv_path: Path, metadata: Dict[str, Any], max_rows_read: Optional[int] = None) -> pd.DataFrame:
    file_name = csv_path.name
    logger.info(f"Processing CO file: {file_name}")

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

    # Kolonları vs. -> Mesela biz CO'da "Daily Max 8-hour CO Concentration" + common alanları tutalım
    # Örnek:
    # keep_cols = ["Date","Site ID","County",...,"Daily Max 8-hour CO Concentration"]
    # df = df[keep_cols if in df...]

    rows_count = df.shape[0]
    mark_file_as_processed(file_name, file_hash, rows_count, metadata, METADATA_PATH)
    logger.info(f"File {file_name} => {rows_count} rows processed for CO")

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


def ingest_co_data():

    logger.info("=== Starting CO ingestion ===")
    metadata = load_metadata(METADATA_PATH)

    co_files = find_co_csv_files(RAW_DIR)
    logger.info(f"Found {len(co_files)} CO files in raw_dir")

    ingestion_cfg = config.get("ingestion", {})
    max_rows_read = ingestion_cfg.get("max_rows_read", None)

    combined_co = pd.DataFrame()
    for path in co_files:
        df_co = process_co_file(path, metadata, max_rows_read)
        if not df_co.empty:
            combined_co = pd.concat([combined_co, df_co], ignore_index=True)
            # Save interim
            try:
                INTERIM_DIR.mkdir(parents=True, exist_ok=True)
                interim_file = INTERIM_DIR / f"interim_{path.name}"
                df_co.to_csv(interim_file, index=False)
                logger.info(f"Saved interim => {interim_file}")
            except Exception as e:
                logger.error(f"Failed to save interim for {path.name}: {e}")
            # Archive
            archive_file(path)

    if not combined_co.empty:
        # final deduplicate
        init_shape = combined_co.shape
        combined_co.drop_duplicates(inplace=True)
        final_shape = combined_co.shape
        logger.info(f"Combined CO dedup: {init_shape} => {final_shape}")

        # save co_preprocessed
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_file = PROCESSED_DIR / "co_preprocessed.csv"
        combined_co.to_csv(out_file, index=False)
        logger.info(f"Saved CO preprocessed => {out_file}")
    else:
        logger.warning("No CO data combined => skipping final save.")

    # update metadata? e.g. save_metadata(metadata, METADATA_PATH)
    logger.info("=== CO ingestion completed ===")

if __name__ == "__main__":
    ingest_co_data()
