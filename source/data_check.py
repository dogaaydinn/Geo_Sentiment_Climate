import os
import glob
import pandas as pd
from datetime import datetime
from source.utils.config_loader import load_config
from source.utils.logger import setup_logger

# En fazla okunacak satır (büyük dosyalarda hız kazanmak için)
MAX_ROWS_READ = 5000

# Logger ayarı (dosyaya log yazmak için)
logger = setup_logger(
    name="data_check",
    log_file="../logs/data_check.log",  # İsteğe göre değiştirilebilir
    log_level="INFO"
)

def validate_columns(df, required_columns):

    missing_cols = [col for col in required_columns if col not in df.columns]
    return len(missing_cols) == 0, missing_cols

def check_raw_data() -> None:

    logger.info("=== Data check başlatılıyor... ===")

    # 1) Config yükle
    config = load_config()
    if config is None:
        logger.error("Config dosyası yüklenemedi veya None döndü, işlem sonlanıyor.")
        return

    # 'data_check' veya 'required_columns' tanımlı mı?
    if "data_check" not in config or "required_columns" not in config["data_check"]:
        logger.warning("Config dosyasında 'data_check' / 'required_columns' tanımlı değil. Devam ediliyor.")
        required_columns = []
    else:
        required_columns = config["data_check"]["required_columns"]

    # paths -> raw_dir, processed_dir
    raw_dir = config["paths"].get("raw_dir", "../data/raw")
    processed_dir = config["paths"].get("processed_dir", "../data/processed")

    # 2) Tüm .csv dosyalarını bul (alt klasörler dahil)
    csv_files = glob.glob(os.path.join(raw_dir, "**/*.csv"), recursive=True)
    logger.info(f"Bulunan CSV dosya sayısı: {len(csv_files)} (dizin: {raw_dir})")

    if not csv_files:
        logger.warning("Hiç CSV dosyası bulunamadı! check_raw_data sonlandırılıyor.")
        return

    # Rapor satırlarını bir listeye toplayalım
    report_rows = []

    # 3) Her dosyada kontrol
    for idx, csv_path in enumerate(csv_files, start=1):
        row_info = {
            "file_path": csv_path,
            "row_count": None,
            "col_count": None,
            "columns": None,
            "empty_file": False,
            "test_pass": True,
            "notes": "",
        }

        # Dosyayı oku (ilk MAX_ROWS_READ satır)
        try:
            df = pd.read_csv(csv_path, nrows=MAX_ROWS_READ)

            if df.empty:
                # Boş dosya
                logger.warning(f"[Empty] Dosya tamamen boş: {csv_path}")
                row_info["empty_file"] = True
                row_info["test_pass"] = False
                row_info["notes"] = "Dosya tamamen boş."
            else:
                # Dosyanın satır/sütun bilgisi
                row_info["row_count"] = df.shape[0]
                row_info["col_count"] = df.shape[1]
                row_info["columns"] = ", ".join(df.columns.tolist())

                # Zorunlu sütun kontrolü
                cols_ok, missing_cols = validate_columns(df, required_columns)
                if not cols_ok:
                    row_info["test_pass"] = False
                    row_info["notes"] += f" Eksik sütunlar: {missing_cols}."

                # Sadece ilk 'quick_check_limit' dosyada missing count loglayalım
                if idx <= config.get("data_check", {}).get("quick_check_limit", 5):
                    missing_dict = df.isnull().sum().to_dict()
                    logger.info(f"[QuickCheck] File={csv_path}, Shape={df.shape}, Missing={missing_dict}")

        except pd.errors.EmptyDataError:
            logger.error(f"[EmptyDataError] Dosya tamamen boş: {csv_path}")
            row_info["empty_file"] = True
            row_info["test_pass"] = False
            row_info["notes"] = "EmptyDataError."
        except pd.errors.ParserError as pe:
            logger.error(f"[ParserError] {csv_path}, Hata: {pe}")
            row_info["test_pass"] = False
            row_info["notes"] = "ParserError."
        except Exception as e:
            logger.error(f"[UnknownError] {csv_path}, Hata: {e}")
            row_info["test_pass"] = False
            row_info["notes"] = f"Bilinmeyen hata: {e}"

        report_rows.append(row_info)

    # 4) Rapor DataFrame
    df_report = pd.DataFrame(report_rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"data_check_report_{timestamp}.csv"

    # processed_dir yoksa oluştur
    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, report_name)
    df_report.to_csv(output_path, index=False)
    logger.info(f"Data check raporu oluşturuldu -> {output_path}")

    # Her satırı log’a yazmak isterseniz:
    for _, row in df_report.iterrows():
        logger.info(f"[Report] {row.to_dict()}")

    logger.info("=== Data check işlemi tamamlandı. ===")


def main():

    try:
        check_raw_data()  # config_path varsayılan "../config/settings.yml"
    except Exception as e:
        logger.critical(f"[CRITICAL] Program kritik hata ile sonlandı: {e}")


if __name__ == "__main__":
    main()
