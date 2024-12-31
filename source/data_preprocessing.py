import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from utils.logger import setup_logger
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.config_loader import load_config


# Config ve Logger Ayarları
CONFIG_PATH = "../config/settings.yml"

try:
    # Config dosyasını yükle
    config = load_config(CONFIG_PATH)

    # Dizinleri config'den al
    RAW_DIR = config["paths"]["raw_dir"]
    PROCESSED_DIR = config["paths"]["processed_dir"]
    PLOTS_DIR = config["paths"].get("plots_dir", "../plots")
    LOG_DIR = config["paths"].get("logs_dir", "../logs")

    # Dizinlerin varlığını kontrol et ve oluştur
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Logger oluştur
    logger = setup_logger(name="data_preprocessing", log_file=os.path.join(LOG_DIR, "data_preprocessing.log"), log_level="INFO")
    logger.info("Config dosyası ve dizinler başarıyla yüklendi.")
except KeyError as e:
    raise ValueError(f"Config dosyasındaki bir anahtar eksik: {e}")
except Exception as e:
    raise RuntimeError(f"Config yüklenirken hata oluştu: {e}")


def load_data(file_path):
    """
    CSV formatındaki dosyayı yükler.

    Args:
        file_path (str): CSV dosyasının yolu.

    Returns:
        pd.DataFrame: Yüklenen veri seti.
    """
    logger.info(f"Loading data from {file_path}")
    try:
        data = pd.read_csv(file_path)
        logger.info("Data loaded successfully")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def save_plot(plot, filename):
    """
    Grafiği belirtilen dosyaya kaydeder.

    Args:
        plot (matplotlib.figure.Figure): Kaydedilecek grafik.
        filename (str): Kaydedilecek dosyanın adı.
    """
    filepath = os.path.join(PLOTS_DIR, filename)
    plot.savefig(filepath)
    logger.info(f"Plot saved to {filepath}")
    plt.close(plot)


def basic_info(df):
    """
    Veri setinin temel bilgilerini görüntüler.

    Args:
        df (pd.DataFrame): Veri seti.
    """
    logger.info("Generating basic info of the dataset")
    info = {
        "Shape": df.shape,
        "Columns": df.columns.tolist(),
        "Data Types": df.dtypes.to_dict(),
        "Missing Values": df.isnull().sum().to_dict()
    }
    logger.info(f"Dataset Info: {info}")
    return info

def visualize_missing_values(df, save=True):

    logger.info("Visualizing missing values.")
    try:
        plt.figure(figsize=(10, 6))
        msno.matrix(df)
        if save:
            plot_path = os.path.join(PLOTS_DIR, "missing_values_matrix.png")
            plt.savefig(plot_path)
            logger.info(f"Missing values matrix plot saved to {plot_path}")
        plt.show()
    except Exception as e:
        logger.error(f"Error in missing values visualization: {e}")
        raise

def fill_missing_values(df, method="mean"):

    logger.info(f"Filling missing values using method: {method}")
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if method == "mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif method == "median":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif method == "mode":
            df = df.fillna(df.mode().iloc[0])
        else:
            raise ValueError("Invalid method for missing value handling")
        logger.info("Missing values filled successfully.")
        return df
    except Exception as e:
        logger.error(f"Error filling missing values: {e}")
        raise


def remove_null_values(df, subset_cols):
    """
    Eksik değerlere sahip satırları çıkarır.

    Args:
        df (pd.DataFrame): Veri seti.
        subset_cols (list): Eksik değer kontrolü yapılacak sütunlar.

    Returns:
        pd.DataFrame: Eksik değerlerin çıkarıldığı veri seti.
    """
    logger.info("Removing rows with null values")
    try:
        df_clean = df.dropna(subset=subset_cols)
        logger.info("Null values removed successfully")
        return df_clean
    except Exception as e:
        logger.error(f"Error in removing null values: {e}")
        raise

def distribution_analysis(df, numeric_cols):
    """
    Sayısal sütunların dağılımını analiz eder.

    Args:
        df (pd.DataFrame): Veri seti.
        numeric_cols (list): Sayısal sütun isimleri.
    """
    logger.info("Performing distribution analysis for numeric columns")
    try:
        for col in numeric_cols:
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col], kde=True, bins=30, color="blue")
            plt.title(f"Distribution of {col}")
            plt.show()
    except Exception as e:
        logger.error(f"Error in distribution analysis: {e}")



def scale_features(df, numeric_cols, method="standard"):
    """Sayısal özellikleri ölçeklendirir."""
    logger.info(f"Scaling features: {numeric_cols} using method: {method}")
    try:
        scaler = StandardScaler() if method == "standard" else MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        logger.info("Features scaled successfully.")
        return df
    except Exception as e:
        logger.error(f"Error scaling features: {e}")
        raise


def correlation_analysis(df, numeric_cols):

    logger.info("Performing correlation analysis.")
    try:
        corr_matrix = df[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plot_path = os.path.join(PLOTS_DIR, "correlation_matrix.png")
        plt.title("Correlation Matrix")
        plt.savefig(plot_path)
        logger.info(f"Correlation matrix plot saved to {plot_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Error in correlation analysis: {e}")


def detect_outliers(df, numeric_cols, save=True):
    """Aykırı değerleri tespit eder."""
    logger.info("Detecting outliers in numeric columns.")
    try:
        for col in numeric_cols:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=df[col])
            plt.title(f"Outliers in {col}")
            if save:
                save_plot(plt, f"outliers_{col}.png")
            plt.show()
    except Exception as e:
        logger.error(f"Error in outlier detection: {e}")



if __name__ == "__main__":
    # Config dosyasını yükle
    config = load_config("../config/settings.yml")
    RAW_DIR = config["paths"]["raw_dir"]
    PROCESSED_DIR = config["paths"]["processed_dir"]

    file_path = os.path.join(PROCESSED_DIR, "epa_long_preprocessed.csv")

    try:
        file_path = os.path.join(config["paths"]["processed_dir"], "epa_long_preprocessed.csv")
        df = load_data(file_path)
        logger.info("Data loaded successfully")
    except Exception as e:
        logger.critical(f"Failed to load data: {e}")
        raise

    file_path = "data/processed/epa_long_preprocessed.csv"

    # Görselleştirme ve İşleme
    visualize_missing_values(df)
    df = fill_missing_values(df, method=config["parameters"].get("missing_value_method", "mean"))
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = scale_features(df, numeric_cols, method=config["parameters"].get("scaling_method", "standard"))
    correlation_analysis(df, numeric_cols)
    detect_outliers(df, numeric_cols)

    # Kaydetme (isteğe bağlı)
    df.to_csv(os.path.join(PROCESSED_DIR, "epa_preprocessed.csv"), index=False)
    logger.info("Preprocessed data saved successfully.")
    basic_info(df)

    # Preprocessed veri kaydetme
    try:
        processed_file_path = os.path.join(config["paths"]["processed_dir"], "epa_preprocessed.csv")
        df.to_csv(processed_file_path, index=False)
        logger.info(f"Preprocessed data saved to {processed_file_path}")
    except Exception as e:
        logger.critical(f"Data preprocessing failed: {e}")


    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    distribution_analysis(df, numeric_cols)
    correlation_analysis(df, numeric_cols)
    detect_outliers(df, numeric_cols)
