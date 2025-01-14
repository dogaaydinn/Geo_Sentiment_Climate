import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)
# Proje içi yardımcı fonksiyonlar
from source.utils.config_loader import load_config
from source.utils.logger import setup_logger
from source.utils.path_utils import add_source_to_sys_path

# 1) Config yükleme
CONFIG_PATH = os.path.join(os.path.abspath("../config"), "settings.yml")
config = load_config(CONFIG_PATH)

# 2) sys.path'e source ekleme
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

def merge_pollutants():
    # read each preprocessed
    df_co = pd.read_csv("../data/processed/co_preprocessed.csv")
    df_so2 = pd.read_csv("../data/processed/so2_preprocessed.csv")
    df_no2 = pd.read_csv("../data/processed/no2_preprocessed.csv")
    df_o3 = pd.read_csv("../data/processed/o3_preprocessed.csv")
    df_pm25 = pd.read_csv("../data/processed/pm25_preprocessed.csv")

    join_keys = ["Date","Site ID","County"]  # etc. or "State, Date, Site ID", up to you

    logging.info("Merging co + so2 => df_merged")
    df_merged = df_co.merge(df_so2, on=join_keys, how="outer", suffixes=("", "_so2"))

    logging.info("Merging df_merged + no2 => df_merged2")
    df_merged2 = df_merged.merge(df_no2, on=join_keys, how="outer", suffixes=("", "_no2"))

    logging.info("Merging df_merged2 + o3 => df_merged3")
    df_merged3 = df_merged2.merge(df_o3, on=join_keys, how="outer", suffixes=("", "_o3"))

    logging.info("Merging df_merged3 + pm25 => final")
    df_final = df_merged3.merge(df_pm25, on=join_keys, how="outer", suffixes=("", "_pm25"))

    # Now df_final has columns from all 5 data sets => if a row is in one set but not the other => NaN
    logging.info(f"Final shape after merges: {df_final.shape}")
    df_final.to_csv("../data/processed/epa_long_5pollutants.csv", index=False)
    logging.info("Saved epa_long_5pollutants.csv")

if __name__ == "__main__":
    merge_pollutants()
