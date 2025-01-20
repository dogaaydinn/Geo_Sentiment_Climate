import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime
from source.config.config_utils import config
from typing import Dict, Any, Optional, List
from source.utils.logger import setup_logger
from source.utils.hash_utils import compute_md5
from source.utils.project_paths import ProjectPaths
from source.config.config_loader import check_required_keys
from source.utils.metadata_manager import load_processed_files
from source.utils.path_initializer import add_source_to_sys_path

add_source_to_sys_path()

logger = setup_logger(
    name="data_ingestion",
    log_file=str(Path(config["paths"]["logs_dir"]) / "data_ingestion.log"),
    log_level=config.get("logging", {}).get("level", "INFO").upper()
)

REQUIRED_KEYS = ["raw_dir","interim_dir","processed_dir","archive_dir","metadata_dir","logs_dir"]
check_required_keys(config, REQUIRED_KEYS)

paths = ProjectPaths.from_config(config)

paths.ensure_directories()

print(paths.raw_dir)


########################
# Some parameters from config
########################
ingestion_cfg = config.get("ingestion", {})
max_rows_read = ingestion_cfg.get("max_rows_read", None)

########################
# Pollutant detection
########################
def get_pollutant_from_filename(file_name: str) -> str:
    fname = file_name.lower()
    pollutants = {
        "so2": "so2",
        "co": "co",
        "no2": "no2",
        "o3": "o3",
        "pm2.5": "pm25",
        "pm25": "pm25"
    }
    for key, value in pollutants.items():
        if key in fname:
            return value
    return "unknown"

########################
# find_csv_files
########################
def find_csv_files(directory: str) -> List[Path]:
    directory_path = Path(directory)
    csv_files = list(directory_path.rglob("*.csv"))
    logger.info(f"Found {len(csv_files)} .csv files in {directory}")
    return csv_files

########################
# Fix for mark_file_as_processed signature:
########################

def mark_file_as_processed(
    file_name: str,
    file_hash: str,
    rows_count: int,
    metadata: Dict[str, Any],
    metadata_path: Path
):

    metadata.setdefault("processed_files", [])
    metadata["processed_files"].append({
        "file_name": file_name,
        "file_hash": file_hash,
        "rows_count": rows_count,
        "timestamp": datetime.now().isoformat()
    })

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

########################
# Process single file
########################
def process_file(
    csv_path: Path,
    metadata: Dict[str, Any],
    max_rows: Optional[int] = None
) -> pd.DataFrame:

    file_name = csv_path.name
    logger.info(f"Processing file {file_name}")

    # 1) MD5
    try:
        file_hash = compute_md5(str(csv_path))
    except Exception as e:
        logger.error(f"Failed to compute MD5 => {file_name}: {e}")
        return pd.DataFrame()

    # 2) Check if processed
    # we load processed_files from metadata
    if "processed_files" not in metadata:
        metadata["processed_files"] = []
    # quick check
    from source.utils.metadata_manager import is_file_processed
    if is_file_processed(file_hash, metadata):
        logger.info(f"Already processed => {file_name}")
        return pd.DataFrame()

    # 3) read
    try:
        if max_rows:
            df = pd.read_csv(csv_path, nrows=max_rows)
        else:
            df = pd.read_csv(csv_path)
        if df.empty:
            logger.warning(f"{file_name} is empty => mark processed 0 rows")
            mark_file_as_processed(
                file_name=file_name,
                file_hash=file_hash,
                rows_count=0,
                metadata=metadata,
                metadata_path=paths.metadata_dir / "metadata.json"
            )
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error reading {file_name}: {e}")
        mark_file_as_processed(
            file_name=file_name,
            file_hash=file_hash,
            rows_count=0,
            metadata=metadata,
            metadata_path=paths.metadata_dir / "metadata.json"
        )
        return pd.DataFrame()

    # 4) deduplicate
    init_shape = df.shape
    df.drop_duplicates(inplace=True)
    final_shape = df.shape
    logger.info(f"Deduplicated => {file_name}: {init_shape} => {final_shape}")

    # 5) figure out pollutant
    poll = get_pollutant_from_filename(file_name)
    # Depending on poll, we keep certain columns
    # (Could read from config["columns"][poll] -> "common_columns" + "all_columns")
    poll_config = config.get("columns", {}).get(poll, {})
    all_cols = poll_config.get("all_columns", [])
    # For ingestion => we keep "common_columns" + that poll's unique columns (like "Daily Max 8-hour CO" if co)
    # But let's do a simpler approach => keep all columns from "all_columns" if they exist
    keep = [c for c in all_cols if c in df.columns]
    if not keep:
        # fallback => keep df as is, or keep minimal
        logger.info(f"No keep columns found for {poll} => we keep all df columns for now.")
        keep = df.columns.tolist()
    df = df[keep]


    # 6) mark processed
    rows_count = df.shape[0]
    mark_file_as_processed(
        file_name=file_name,
        file_hash=file_hash,
        rows_count=rows_count,
        metadata=metadata,
        metadata_path=paths.metadata_dir / "metadata.json"
    )
    logger.info(f"File {file_name} => poll={poll}, {rows_count} rows, done.")
    return df

def archive_file(csv_path: Path):
    try:
        paths.archive_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        arc_name = f"{ts}_{csv_path.name}"
        arc_path = paths.archive_dir / arc_name
        shutil.move(str(csv_path), str(arc_path))
        logger.info(f"Archived => {arc_path}")
    except Exception as e:
        logger.error(f"Failed to archive {csv_path.name}: {e}")

########################
# Outer join final
########################
def merge_pollutants(dict_of_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:

    join_keys = ["Date","Site ID"]  # example; you can add "County" if you want
    # start from one
    pollutants = list(dict_of_dfs.keys())
    if not pollutants:
        return pd.DataFrame()

    final_df = dict_of_dfs[pollutants[0]]
    for p in pollutants[1:]:
        final_df = final_df.merge(dict_of_dfs[p], on=join_keys, how="outer", suffixes=("", f"_{p}"))

    return final_df

########################
# ingest_data => main
########################

def ingest_data(raw_dir: Path = paths.raw_dir):

    logger.info("=== Starting multi-pollutant ingestion with outer join ===")

    metadata = load_processed_files(paths.metadata_dir / "metadata.json")
    csv_files = find_csv_files(str(raw_dir))
    if not csv_files:
        logger.warning("No CSV to ingest => abort.")
        return

    # keep partial data in lists
    dict_of_lists = {
        "co": [],
        "so2": [],
        "no2": [],
        "o3": [],
        "pm25": [],
        "unknown": []
    }

    # read/ingest
    for path in csv_files:
        df_proc = process_file(path, metadata, max_rows_read)
        if not df_proc.empty:
            # find poll again or do inside process_file
            poll = get_pollutant_from_filename(path.name)
            dict_of_lists.setdefault(poll, [])
            dict_of_lists[poll].append(df_proc)

            # Save interim
            try:
                paths.interim_dir.mkdir(parents=True, exist_ok=True)
                interim_file = paths.interim_dir / f"interim_{path.name}"
                df_proc.to_csv(interim_file, index=False)
                logger.info(f"Saved interim => {interim_file}")
            except Exception as e:
                logger.error(f"Failed to save interim for {path.name}: {e}")
            # Archive
            archive_file(path)

    # after reading all => concat each poll
    dict_of_dfs = {}
    for poll, df_list in dict_of_lists.items():
        if df_list:
            combined = pd.concat(df_list, ignore_index=True)
            # dedup
            init_shape = combined.shape
            combined.drop_duplicates(inplace=True)
            final_shape = combined.shape
            logger.info(f"[{poll}] => {init_shape} => {final_shape}")
            dict_of_dfs[poll] = combined
        else:
            logger.warning(f"No data for {poll}")

    # Now outer join among [co, so2, no2, o3, pm25]
    # skip "unknown"
    polls = ["so2","co","no2","o3","pm25"]
    dict_join = {p: dict_of_dfs[p] for p in polls if p in dict_of_dfs}
    if not dict_join:
        logger.warning("No known pollutant data => skip final merge.")
        return

    df_final = merge_pollutants(dict_join)
    if not df_final.empty:
        # Save final
        paths.processed_dir.mkdir(parents=True, exist_ok=True)
        out_file = paths.processed_dir / "epa_long_5pollutants.csv"
        df_final.to_csv(out_file, index=False)
        logger.info(f"Saved final outer-joined => {out_file}")
    else:
        logger.warning("Final outer join is empty => skipping save.")

    # finally save metadata
    # we can do that here or inside process_file
    with open(paths.metadata_dir / "metadata.json", "r") as f:
        logger.info(f"Final metadata => {f.read()}")

    logger.info("=== Multi-pollutant ingestion completed ===")

if __name__=="__main__":
    ingest_data()
