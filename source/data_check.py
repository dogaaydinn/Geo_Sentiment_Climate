import os
import glob
import pandas as pd
import yaml
from datetime import datetime
from utils.config_loader import load_config
from utils.logger import setup_logger

# Configure logger
logger = setup_logger(name="data_check", log_file="../logs/data_check.log", log_level="INFO")


MAX_ROWS_READ = 5000  # Her CSV için en fazla kaç satır okuyalım


def check_raw_data(config_path="../config/settings.yml"):
    """
    data/raw dizinindeki tüm CSV dosyalarını tarar ve temel kontrol raporu oluşturur.

    Args:
        config_path (str): Ayar dosyasının yolu.
    """
    logger.info("Data check işlemi başlatılıyor...")

    # Konfigürasyonu yükle
    try:
        config = load_config(config_path)
        RAW_DIR = config["paths"]["raw_dir"]
        PROCESSED_DIR = config["paths"]["processed_dir"]
    except Exception as e:
        logger.error(f"Ayar dosyası yüklenirken hata oluştu: {e}")
        raise

    # Taranacak dosyaları bul
    csv_files = glob.glob(os.path.join(RAW_DIR, "**/*.csv"), recursive=True)
    logger.info(f"Bulunan CSV dosya sayısı: {len(csv_files)}")

    if not csv_files:
        logger.warning("Hiç CSV dosyası bulunamadı!")
        return

    report_rows = []

    # Dosya bazlı işlem
    for csv_path in csv_files:
        logger.info(f"Dosya kontrol ediliyor: {csv_path}")
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
            df = pd.read_csv(csv_path, nrows=MAX_ROWS_READ)

            # Dosya boş mu?
            if df.empty:
                logger.warning(f"Dosya boş: {csv_path}")
                row_info["empty_file"] = True
                row_info["test_pass"] = False
                row_info["notes"] = "Dosya tamamen boş."
            else:
                row_info["row_count"] = df.shape[0]
                row_info["col_count"] = df.shape[1]
                row_info["columns"] = ", ".join(df.columns.tolist())

                # Örnek mini testler
                if not any("Date" in col or "date" in col.lower() for col in df.columns):
                    row_info["test_pass"] = False
                    row_info["notes"] += "Date kolonu bulunamadı. "
                if not any("Arithmetic Mean" in col or "value" in col.lower() for col in df.columns):
                    row_info["test_pass"] = False
                    row_info["notes"] += "Arithmetic Mean kolonu bulunamadı. "

        except pd.errors.EmptyDataError:
            logger.error(f"Dosya tamamen boş: {csv_path}")
            row_info["test_pass"] = False
            row_info["notes"] = "EmptyDataError."
        except pd.errors.ParserError as e:
            logger.error(f"CSV parse hatası: {csv_path}, Hata: {e}")
            row_info["test_pass"] = False
            row_info["notes"] = "ParserError."
        except Exception as e:
            logger.error(f"Bilinmeyen bir hata oluştu: {csv_path}, Hata: {e}")
            row_info["test_pass"] = False
            row_info["notes"] = f"Bilinmeyen hata: {e}"

        report_rows.append(row_info)

        # Rapor oluşturma
        df_report = pd.DataFrame(report_rows)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_csv_name = f"data_check_report_{timestamp}.csv"
        output_path = os.path.join(PROCESSED_DIR, report_csv_name)

        os.makedirs(PROCESSED_DIR, exist_ok=True)
        df_report.to_csv(output_path, index=False)
        logger.info(f"Data check raporu oluşturuldu: {output_path}")

    if __name__ == "__main__":
        try:
            check_raw_data()
        except Exception as e:
            logger.critical(f"Program kritik bir hata ile sonlandı: {e}")