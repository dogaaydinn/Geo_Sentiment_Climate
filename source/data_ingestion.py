import os
import glob
import pandas as pd
from utils.config_loader import load_config
from utils.logger import setup_logger

# Configure logger
logger = setup_logger(name="data_ingestion", log_file="../logs/data_ingestion.log", log_level="INFO")


def ingest_data(raw_dir, max_rows=None):
    """
    Ham verileri birleştirerek tek bir DataFrame oluşturur.

    Args:
        raw_dir (str): Ham verilerin bulunduğu dizin.
        max_rows (int, optional): Maksimum okunacak satır sayısı.

    Returns:
        pd.DataFrame: Birleştirilmiş DataFrame.
    """
    logger.info("Veri birleştirme işlemi başlıyor.")
    csv_files = glob.glob(os.path.join(raw_dir, "**/*.csv"), recursive=True)
    print(f"[INFO] Bulunan CSV dosya sayısı: {len(csv_files)}")

    all_data = []

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, nrows=max_rows)
            df["source_file"] = os.path.basename(csv_path)  # Kaynak dosya bilgisi
            logger.info(f"{csv_path} dosyası başarıyla okundu.")
            all_data.append(df)
        except Exception as e:
            print(f"[ERROR] {csv_path} okunamadı: {str(e)}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Birleştirilen DataFrame boyutu: {combined_df.shape}")
        return combined_df
    else:
        logger.warning("Hiçbir veri birleştirilemedi!")
        return pd.DataFrame()
