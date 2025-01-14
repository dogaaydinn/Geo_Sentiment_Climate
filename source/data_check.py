import os
import glob
import pandas as pd
from datetime import datetime
from source.utils.logger import setup_logger
from source.utils.config_loader import load_config
from typing import List, Tuple, Dict, Any, Optional
from source.utils.path_utils import add_source_to_sys_path

CONFIG_PATH = os.path.join(os.path.abspath("../config"), "settings.yml")
config = load_config(CONFIG_PATH)

add_source_to_sys_path()

REQUIRED_KEYS = ["raw_dir", "interim_dir", "processed_dir", "archive_dir", "metadata_dir", "logs_dir"]
missing_keys = [key for key in REQUIRED_KEYS if key not in config.get("paths", {})]
if missing_keys:
    raise KeyError(f"Missing required config keys: {missing_keys}")

RAW_DIR = os.path.abspath(config["paths"]["raw_dir"])
logger = setup_logger(
    name="merge_pollutants",
    log_file=os.path.join(config["paths"]["logs_dir"], "merge_pollutants.log"),
    log_level=config.get("logging", {}).get("level", "INFO").upper()
)

# -----------------------------------------------------------------------------
# methods
# -----------------------------------------------------------------------------

def get_pollutant_from_filename(file_name: str) -> str:
    fname = file_name.lower()
    if "so2" in fname:
        return "so2"
    elif "co" in fname:
        return "co"
    elif "no2" in fname:
        return "no2"
    elif "o3" in fname and "no2" not in fname:
        return "o3"
    elif "pm2.5" in fname or "pm25" in fname:
        return "pm25"
    else:
        return "unknown"

def find_csv_files(directory: str) -> List[str]:

    pattern = os.path.join(directory, "**", "*.csv")
    files = glob.glob(pattern, recursive=True)
    return files

def load_csv(filepath: str, nrows: Optional[int] = None, sampling_ratio: float = 0) -> pd.DataFrame:

    if nrows is not None and nrows > 0:
        df = pd.read_csv(filepath, nrows=nrows)
        return df

    df = pd.read_csv(filepath)
    if sampling_ratio > 0 and sampling_ratio < 1:
        df = df.sample(frac=sampling_ratio, random_state=42)
        logger.debug(f"Sampled {len(df)} rows from {filepath} using ratio={sampling_ratio}")
    return df

def load_csv_in_chunks(filepath: str, chunk_size: int):

    for chunk_df in pd.read_csv(filepath, chunksize=chunk_size):
        yield chunk_df

def generate_report(report_rows: List[Dict[str, Any]], processed_dir: str) -> str:

    df_report = pd.DataFrame(report_rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"data_check_report_{timestamp}.csv"
    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, report_name)

    # Sözlük alanları (ör: missing_values) için string'e dönüştürme
    if "missing_values" in df_report.columns:
        df_report["missing_values"] = df_report["missing_values"].apply(lambda x: str(x) if isinstance(x, dict) else x)

    df_report.to_csv(output_path, index=False)
    logger.info(f"Data check report created at {output_path}")
    return output_path


def validate_required_optional_columns(
    df: pd.DataFrame,
    required_cols: List[str],
    optional_cols: List[str]
) -> Tuple[bool, List[str], List[str]]:

    missing_req = [c for c in required_cols if c not in df.columns]
    missing_opt = [c for c in optional_cols if c not in df.columns]

    req_ok = (len(missing_req) == 0)
    return req_ok, missing_req, missing_opt

def validate_column_types(df, column_types_map: Dict[str, List[str]]):
    errors = {}
    for dtype_name, cols_list in column_types_map.items():
        # dtype_name => "datetime", "float", "int", "string"
        # cols_list => ["Date", "Daily AQI Value", ...]
        for col in cols_list:
            if col not in df.columns:
                continue
            try:
                if dtype_name == "datetime":
                    pd.to_datetime(df[col], errors="raise")
                elif dtype_name == "float":
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        errors[col] = "Should be numeric(float)"
                elif dtype_name == "int":
                    if not pd.api.types.is_integer_dtype(df[col]):
                        errors[col] = "Should be integer(int)"
                elif dtype_name == "string":
                    # opsiyonel
                    pass
            except Exception as e:
                errors[col] = f"Type check failed: {e}"
    return (len(errors) == 0, errors)

# -----------------------------------------------------------------------------
# check_raw_data (Ana Fonksiyon)
# -----------------------------------------------------------------------------

def check_raw_data(raw_dir: str = RAW_DIR) -> None:

    logger.info("=== Starting advanced data check ===")

    # data_check altındaki numeric parametreler
    data_check_cfg = config.get("data_check", {})
    chunk_size_threshold = data_check_cfg.get("chunk_size_threshold", 0)
    chunk_size = data_check_cfg.get("chunk_size", 100000)
    sampling_ratio = data_check_cfg.get("sampling_ratio", 0.0)
    quick_check_limit = data_check_cfg.get("quick_check_limit", 5)

    # "column_types" -> dict, sadece "Date": "datetime" gibi
    column_types_map = data_check_cfg.get("column_types", {})
    # pollutant bazlı columns
    columns_config = config.get("columns", {})

    # max_rows_read -> int
    max_rows_read = data_check_cfg.get("max_rows_read", 5000)

    # CSV'leri bul
    csv_files = find_csv_files(raw_dir)
    if not csv_files:
        logger.warning("No CSV files found in the specified directory. Terminating data check.")
        return

    report_rows = []

    for idx, csv_path in enumerate(csv_files, start=1):
        file_name = os.path.basename(csv_path)
        pollutant = get_pollutant_from_filename(file_name)

        # Pollutant config
        poll_config = columns_config.get(pollutant, {})
        all_cols = poll_config.get("all_columns", [])
        common_cols = poll_config.get("common_columns", [])

        required_cols = list(set(common_cols))
        optional_cols = list(set(all_cols) - set(common_cols))

        row_info = {
            "file_path": csv_path,
            "pollutant": pollutant,
            "row_count": 0,
            "col_count": 0,
            "columns": None,
            "missing_req_cols": None,
            "missing_opt_cols": None,
            "dtype_issues": None,
            "empty_file": False,
            "test_pass": True,
            "notes": "",
            "missing_values": None,
        }

        file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
        use_chunk = (file_size_mb > 5) and (chunk_size_threshold > 0) and (file_size_mb > (chunk_size_threshold/(1024*1024)))

        try:
            if use_chunk:
                logger.info(f"[DataCheck] Reading large file in chunks: {csv_path} (size={file_size_mb:.2f} MB)")
                total_rows = 0
                total_missing = None
                chunk_idx = 0

                for chunk_df in load_csv_in_chunks(csv_path, chunk_size=chunk_size):
                    chunk_idx += 1
                    if chunk_idx == 1:
                        # Required/optional
                        req_ok, missing_req, missing_opt = validate_required_optional_columns(
                            chunk_df, required_cols, optional_cols
                        )
                        if not req_ok:
                            row_info["test_pass"] = False
                            row_info["missing_req_cols"] = missing_req
                            row_info["notes"] += f"Missing required columns: {missing_req}. "
                        if missing_opt:
                            row_info["missing_opt_cols"] = missing_opt
                            logger.info(f"[OptionalMissing] {file_name} missing optional: {missing_opt}")

                        # Tip check
                        type_ok, dtype_err = validate_column_types(chunk_df, column_types_map)
                        if not type_ok:
                            row_info["test_pass"] = False
                            row_info["dtype_issues"] = dtype_err
                            row_info["notes"] += f"Dtype issues: {dtype_err}. "

                        row_info["col_count"] = len(chunk_df.columns)
                        row_info["columns"] = ", ".join(chunk_df.columns.tolist())

                    # Eksik değer hesapla
                    chunk_missing = chunk_df.isnull().sum()
                    if total_missing is None:
                        total_missing = chunk_missing
                    else:
                        total_missing = total_missing.add(chunk_missing, fill_value=0)

                    total_rows += chunk_df.shape[0]

                row_info["row_count"] = total_rows
                if total_missing is not None:
                    row_info["missing_values"] = total_missing.to_dict()

            else:
                # Tek seferde ( örnekleme / max_rows_read ) okuma
                if max_rows_read and max_rows_read > 0:
                    df = load_csv(csv_path, nrows=max_rows_read)
                else:
                    df = load_csv(csv_path, sampling_ratio=sampling_ratio)

                if df.empty:
                    logger.warning(f"[Empty] {csv_path} is empty.")
                    row_info["empty_file"] = True
                    row_info["test_pass"] = False
                    row_info["notes"] = "File is completely empty."
                else:
                    req_ok, missing_req, missing_opt = validate_required_optional_columns(
                        df, required_cols, optional_cols
                    )
                    if not req_ok:
                        row_info["test_pass"] = False
                        row_info["missing_req_cols"] = missing_req
                        row_info["notes"] += f"Missing required columns: {missing_req}. "

                    if missing_opt:
                        row_info["missing_opt_cols"] = missing_opt
                        logger.info(f"[OptionalMissing] {file_name} missing optional: {missing_opt}")

                    type_ok, dtype_err = validate_column_types(df, column_types_map)
                    if not type_ok:
                        row_info["test_pass"] = False
                        row_info["dtype_issues"] = dtype_err
                        row_info["notes"] += f"Dtype issues: {dtype_err}. "

                    row_info["row_count"] = df.shape[0]
                    row_info["col_count"] = df.shape[1]
                    row_info["columns"] = ", ".join(df.columns.tolist())

                    missing_dict = df.isnull().sum().to_dict()
                    row_info["missing_values"] = missing_dict
                    logger.info(f"[MissingValues] {csv_path}, shape={df.shape}, missing={missing_dict}")

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
    logger.info(f"Data check completed. Report => {report_path}")

if __name__ == "__main__":
    check_raw_data()

