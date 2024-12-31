import os
import glob
import pandas as pd
from datetime import datetime
from utils.config_loader import load_config
from utils.logger import setup_logger

# Logger yapılandırması
logger = setup_logger(name="data_check", log_file="../logs/data_check.log", log_level="INFO")

# Maksimum okunacak satır sayısı
MAX_ROWS_READ = 5000


def validate_columns(df, required_columns):

    missing_columns = [col for col in required_columns if col not in df.columns]
    return len(missing_columns) == 0, missing_columns


def check_raw_data(config_path="../config/settings.yml"):

    logger.info("Data check işlemi başlatılıyor...")

    # Konfigürasyonu yükle
    try:
        config = load_config(config_path)
        raw_dir = config["paths"]["raw_dir"]
        processed_dir = config["paths"]["processed_dir"]
        required_columns = config["data_check"]["required_columns"]
    except KeyError as e:
        logger.error(f"Ayar dosyasındaki eksik anahtar: {e}")
        raise
    except Exception as e:
        logger.error(f"Ayar dosyası yüklenirken hata oluştu: {e}")
        raise

    # CSV dosyalarını bul
    csv_files = glob.glob(os.path.join(raw_dir, "**/*.csv"), recursive=True)
    logger.info(f"Bulunan CSV dosya sayısı: {len(csv_files)}")

    if not csv_files:
        logger.warning("Hiç CSV dosyası bulunamadı! İşlem sonlandırılıyor.")
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
            # Dosyayı oku
            df = pd.read_csv(csv_path, nrows=MAX_ROWS_READ)

            # Dosya boş mu?
            if df.empty:
                logger.warning(f"Dosya boş: {csv_path}")
                row_info["empty_file"] = True
                row_info["test_pass"] = False
                row_info["notes"] = "Dosya tamamen boş."
            else:
                # Veri çerçevesi bilgilerini al
                row_info["row_count"] = df.shape[0]
                row_info["col_count"] = df.shape[1]
                row_info["columns"] = ", ".join(df.columns.tolist())

                # Sütun doğrulama
                columns_ok, missing_cols = validate_columns(df, required_columns)
                if not columns_ok:
                    row_info["test_pass"] = False
                    row_info["notes"] += f"Eksik sütunlar: {missing_cols}."

                logger.info(f"Dosya bilgisi: {row_info}")

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

    # Rapor oluştur
    df_report = pd.DataFrame(report_rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_csv_name = f"data_check_report_{timestamp}.csv"
    output_path = os.path.join(processed_dir, report_csv_name)

    os.makedirs(processed_dir, exist_ok=True)
    df_report.to_csv(output_path, index=False)
    logger.info(f"Data check raporu oluşturuldu: {output_path}")

    # Log dosyasına rapor özeti yaz
    for index, row in df_report.iterrows():
        logger.info(f"Rapor Satırı: {row.to_dict()}")


if __name__ == "__main__":
    try:
        check_raw_data()
    except Exception as e:
        logger.critical(f"Program kritik bir hata ile sonlandı: {e}")
