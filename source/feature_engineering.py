import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.logger import setup_logger

logger = setup_logger(name="feature_engineering", log_file="../logs/feature_engineering.log", log_level="INFO")


def scale_features(df, cols, method="standard"):
    """
    Sayısal özellikleri ölçeklendirir.

    Args:
        df (pd.DataFrame): Veri seti.
        cols (list): Ölçeklendirilecek sütunlar.
        method (str): "standard" veya "minmax".

    Returns:
        pd.DataFrame: Ölçeklendirilmiş veri seti.
    """
    logger.info(f"Özellik ölçeklendirme başlıyor: {cols}, Yöntem: {method}")
    try:
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError("method must be 'standard' or 'minmax'")
        df[cols] = scaler.fit_transform(df[cols])
        logger.info("Özellik ölçeklendirme tamamlandı")
        return df
    except Exception as e:
        logger.error(f"Hata: {e}")
        raise

def create_interaction_terms(df, col1, col2, operation="multiply"):
    """
    İki özellik arasında etkileşim terimi oluşturur.

    Args:
        df (pd.DataFrame): Veri seti.
        col1 (str): Birinci sütun.
        col2 (str): İkinci sütun.
        operation (str): "multiply" veya "add".

    Returns:
        pd.Series: Oluşturulan etkileşim terimi.
    """
    logger.info(f"Etkileşim terimi oluşturuluyor: {col1}, {col2}, İşlem: {operation}")
    try:
        if operation == "multiply":
            return df[col1] * df[col2]
        elif operation == "add":
            return df[col1] + df[col2]
        else:
            raise ValueError("operation must be 'multiply' or 'add'")
    except Exception as e:
        logger.error(f"Etkileşim terimi oluşturulurken hata: {e}")
        raise
