import glob
import os
from datetime import datetime
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional

# Proje içi yardımcı fonksiyonlar (uygunsa yollara dikkat edin)
from source.utils.config_loader import load_config
from source.utils.logger import setup_logger
from source.utils.path_utils import add_source_to_sys_path

# 1) Config yükleme
CONFIG_PATH = os.path.join(os.path.abspath("../config"), "settings.yml")
config = load_config(CONFIG_PATH)

# 2) sys.path'e source ekleme (ihtiyaç varsa)
add_source_to_sys_path()

# 3) Gerekli yol key'lerini kontrol
REQUIRED_KEYS = ["raw_dir", "interim_dir", "processed_dir", "archive_dir", "metadata_dir", "logs_dir"]
missing_keys = [key for key in REQUIRED_KEYS if key not in config.get("paths", {})]
if missing_keys:
    raise KeyError(f"Missing required config keys: {missing_keys}")

# 4) Path ve Logger ayarları
RAW_DIR = os.path.abspath(config["paths"]["raw_dir"])
logger = setup_logger(
    name="data_check",
    log_file=os.path.join(config["paths"]["logs_dir"], "data_check.log"),
    log_level=config.get("logging", {}).get("level", "INFO").upper()
)

# -----------------------------------------------------------------------------
# Yardımcı Fonksiyonlar
# -----------------------------------------------------------------------------

def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    'required_columns' listesinde olup, DataFrame'de eksik olan kolonları yakalar.
    Dönüş: (bool: tümü var mı, List: eksik kolonlar)
    """
    missing_cols = [col for col in required_columns if col not in df.columns]
    return len(missing_cols) == 0, missing_cols

def validate_column_types(df: pd.DataFrame, column_types: Dict[str, str]) -> Tuple[bool, Dict[str, str]]:
    """
    Kolonların veri tiplerini, configte belirtilen 'column_types' ile karşılaştırır.
    Örnek 'column_types':
      {"Date": "datetime", "Daily AQI Value": "float", "State FIPS Code": "string"}
    """
    errors = {}
    for col, expected_type in column_types.items():
        # Kolon yoksa bu fonksiyonun işi değil (validate_columns ile kontrol ediliyor).
        if col not in df.columns:
            continue
        try:
            if expected_type == "datetime":
                pd.to_datetime(df[col], errors="raise")
            elif expected_type == "float":
                if not pd.api.types.is_numeric_dtype(df[col]):
                    errors[col] = "Should be numeric(float)"
            elif expected_type == "int":
                if not pd.api.types.is_integer_dtype(df[col]):
                    errors[col] = "Should be integer(int)"
            elif expected_type == "string":
                # Daha katı kontrol isterseniz dtype='object' vs. bakabilirsiniz
                pass
            else:
                # Yeni veri tipleri ekleyebilirsiniz
                pass
        except Exception as e:
            errors[col] = f"Type check failed: {e}"
    return (len(errors) == 0, errors)

def find_csv_files(directory: str) -> List[str]:
    """
    Verilen dizin (directory) ve alt dizinlerdeki tüm CSV dosyalarının path listesini döndürür.
    """
    csv_files = glob.glob(os.path.join(directory, "**/*.csv"), recursive=True)
    logger.debug(f"Found {len(csv_files)} CSV files in {directory}")
    return csv_files

def load_csv(filepath: str, nrows: Optional[int] = None, sampling_ratio: float = 0) -> pd.DataFrame:
    """
    CSV dosyasını DataFrame'e yükler.
    - nrows: Belirli sayıda satır okumak için.
    - sampling_ratio: Oransal örnekleme (0 < ratio < 1) şeklinde.
    """
    if nrows is not None and nrows > 0:
        df = pd.read_csv(filepath, nrows=nrows)
        return df

    df = pd.read_csv(filepath)
    if sampling_ratio > 0 and sampling_ratio < 1:
        df = df.sample(frac=sampling_ratio, random_state=42)
        logger.debug(f"Sampled {len(df)} rows from {filepath} using ratio={sampling_ratio}")
    return df

def load_csv_in_chunks(filepath: str, chunk_size: int):
    """
    CSV'yi chunk bazlı (parçalar halinde) yüklemek için bir generator fonksiyonu.
    """
    for chunk_df in pd.read_csv(filepath, chunksize=chunk_size):
        yield chunk_df

def generate_report(report_rows: List[Dict[str, Any]], processed_dir: str) -> str:
    """
    Rapor sonuçlarını DataFrame'e çevirir ve CSV olarak kaydeder.
    """
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

# -----------------------------------------------------------------------------
# check_raw_data (Ana Fonksiyon)
# -----------------------------------------------------------------------------

def check_raw_data(raw_dir: str = RAW_DIR) -> None:
    """
    Gelişmiş data check fonksiyonu:
    1) CSV dosyalarını bul,
    2) Kolonları ve tiplerini doğrula,
    3) Eksik değer sayısını hesapla,
    4) Raporu CSV formatında kaydet.
    """

    logger.info("=== Starting advanced data check ===")

    # settings.yml içerisindeki data_check ayarlarını okuyalım
    config_data_check = config.get("data_check", {})
    required_columns = config_data_check.get("required_columns", [])
    column_types = config_data_check.get("column_types", {})
    # max_rows_read -> sadece belirli sayıda satır oku
    max_rows_read = config.get("parameters", {}).get("max_rows", 5000)
    # Büyük dosyalar için chunk bazlı okuma eşiği
    chunk_size_threshold = config_data_check.get("chunk_size_threshold", 0)
    chunk_size = config_data_check.get("chunk_size", 100000)
    sampling_ratio = config_data_check.get("sampling_ratio", 0.0)

    # CSV dosyalarını bul
    csv_files = find_csv_files(raw_dir)
    if not csv_files:
        logger.warning("No CSV files found in the specified directory. Terminating data check.")
        return

    report_rows = []

    for idx, csv_path in enumerate(csv_files, start=1):
        row_info = {
            "file_path": csv_path,
            "row_count": 0,
            "col_count": 0,
            "columns": None,
            "missing_cols": None,
            "dtype_issues": None,
            "empty_file": False,
            "test_pass": True,
            "notes": "",
            "missing_values": None,  # <-- Eksik değerler için
        }

        try:
            # 1 satır okuyarak dosyanın kolon sayısı ve boyutu hakkında fikir alıyoruz
            quick_df = pd.read_csv(csv_path, nrows=1)
            file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
            use_chunk = file_size_mb > 5  # Örnek kriter: 5 MB üstü chunk

            # -----------------------------------------------------------------
            # CHUNK BAZLI OKUMA
            # -----------------------------------------------------------------
            if use_chunk and chunk_size_threshold > 0:
                logger.info(f"Reading large file in chunks: {csv_path} (size={file_size_mb:.2f} MB)")

                total_rows = 0
                total_missing = None
                chunk_idx = 0

                for chunk_df in load_csv_in_chunks(csv_path, chunk_size=chunk_size):
                    chunk_idx += 1

                    # İlk chunk'ta kolon ve tip kontrolü
                    if chunk_idx == 1:
                        cols_ok, missing_cols = validate_columns(chunk_df, required_columns)
                        if not cols_ok:
                            row_info["test_pass"] = False
                            row_info["missing_cols"] = missing_cols
                            row_info["notes"] += f"Missing required columns: {missing_cols}. "

                        types_ok, dtype_errors = validate_column_types(chunk_df, column_types)
                        if not types_ok:
                            row_info["test_pass"] = False
                            row_info["dtype_issues"] = dtype_errors
                            row_info["notes"] += f"Dtype issues: {dtype_errors}. "

                        row_info["col_count"] = len(chunk_df.columns)
                        row_info["columns"] = ", ".join(chunk_df.columns.tolist())

                    # Eksik değerleri chunk bazında topla
                    chunk_missing = chunk_df.isnull().sum()
                    if total_missing is None:
                        total_missing = chunk_missing
                    else:
                        total_missing = total_missing.add(chunk_missing, fill_value=0)

                    total_rows += chunk_df.shape[0]

                # Tüm chunk'lar bitti
                row_info["row_count"] = total_rows
                if total_missing is not None:
                    row_info["missing_values"] = total_missing.to_dict()

            # -----------------------------------------------------------------
            # TEK SEFERDE / ÖRNEKLEMELİ OKUMA
            # -----------------------------------------------------------------
            else:
                if max_rows_read is not None and max_rows_read > 0:
                    df = load_csv(csv_path, nrows=max_rows_read)
                else:
                    df = load_csv(csv_path, nrows=None, sampling_ratio=sampling_ratio)

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
                        row_info["missing_cols"] = missing_cols
                        row_info["notes"] += f"Missing required columns: {missing_cols}. "

                    types_ok, dtype_errors = validate_column_types(df, column_types)
                    if not types_ok:
                        row_info["test_pass"] = False
                        row_info["dtype_issues"] = dtype_errors
                        row_info["notes"] += f"Dtype issues: {dtype_errors}. "

                    # Eksik değerler (her dosya için)
                    missing_dict = df.isnull().sum().to_dict()
                    row_info["missing_values"] = missing_dict
                    logger.info(f"[MissingValues] File={csv_path}, Shape={df.shape}, Missing={missing_dict}")

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

        # Her dosya işlenince rapor satırlarına ekle
        report_rows.append(row_info)
        logger.debug(f"Processed file {idx}/{len(csv_files)}: {csv_path}")

    # Son olarak rapor CSV'sini üret
    report_path = generate_report(report_rows, config["paths"]["processed_dir"])
    logger.info("=== Advanced data check completed ===")

# -----------------------------------------------------------------------------
# Ana çalıştırma
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    check_raw_data()

# -----------------------------------------------------------------------------
# Notlar:
'''
Önemli Notlar

use_chunk: 5 MB üzeri (veya chunk_size_thresholde göre) dosyalarda parça parça okuma yapar.
Eksik değer bilgisi her dosya için row_info["missing_values"]’te saklanır ve CSV raporunda string olarak kaydedilir.
“Kolon yokluğu” ve “kolon tip hatası” durumları, test_pass=False ve notes alanında toplanır.

'''