"""
advanced_cleanup.py

Bu modül, EPA verilerinden (ör. SO2, NO2, O3, CO, PM2.5 gibi kirleticiler) elde edilen
ham DataFrame’lerdeki verilerin ileri temizleme (advanced cleanup) aşamasını gerçekleştirir.
İşlevselliğe örnek olarak:
    - Nadiren kullanılan sütunların (POC, CBSA Code/Name, FIPS kodları) otomatik olarak atılması,
    - Opsiyonel sütunların isteğe bağlı bırakılması veya çıkarılması,
    - Yüksek korelasyonlu sütunların tespit edilip otomatik silinmesi,
    - Aykırı değer (outlier) tespiti ve giderilmesi (IQR veya z‐skor yöntemi ile),
kapsamlı veri hazırlama adımlarını içerir.

Yazar: Your Name
Tarih: 2025-01-12
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal

logger = logging.getLogger("advanced_cleanup")

# Belirlediğimiz nadiren kullanılan sütunları ve opsiyonel sütunları listeleyelim:
RARELY_USED_COLUMNS: List[str] = [
    "POC",
    "CBSA Code", "CBSA Name",
    "State FIPS Code", "County FIPS Code"
]

OPTIONAL_COLUMNS: List[str] = [
    "Method Code",
    "Local Site Name",
    "Percent Complete",
    "Daily Obs Count"
]


def advanced_cleanup(
        df: pd.DataFrame,
        pollutant: str,
        remove_rarely_used: bool = True,
        drop_optional: bool = False,
        correlation_threshold: float = 1.0,
        outlier_method: Literal["none", "iqr", "zscore"] = "none",
        save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Uygulanan ileri temizlik işlemleri:
      1) Nadiren kullanılan sütunların silinmesi,
      2) Opsiyonel sütunların isteğe bağlı olarak çıkarılması,
      3) Yüksek korelasyonlu sütunların otomatik olarak atılması,
      4) Aykırı değerlerin (outlier) tespiti ve giderilmesi (IQR veya z‑skor yöntemi),
      5) Sonuç olarak temizlenmiş DataFrame’in kaydedilmesi (opsiyonel).

    Args:
        df (pd.DataFrame): EPA veri seti (örneğin, SO2, NO2, O3, CO, PM2.5).
        pollutant (str): Verinin ait olduğu kirletici (ör. "so2", "no2", "co", "o3", "pm25").
        remove_rarely_used (bool, optional): RARELY_USED_COLUMNS’da belirtilen sütunların atılıp atılmayacağı.
                                             Defaults to True.
        drop_optional (bool, optional): OPTIONAL_COLUMNS’da belirtilen sütunların çıkarılıp çıkarılmayacağı.
                                        Defaults to False.
        correlation_threshold (float, optional): (0, 1) aralığında bir eşik; bu değerin altında olan
                                                   sütun çiftleri yüksek korelasyonlu kabul edilip, fazlası atılır.
                                                   Defaults to 1.0 (yani sıfır atma).
        outlier_method (Literal["none", "iqr", "zscore"], optional): Uygulanacak outlier (aykırı değer) yöntemi.
                                                                     Defaults to "none".
        save_path (Optional[Path], optional): Son temizlenmiş DataFrame CSV'sinin kaydedileceği yol.

    Returns:
        pd.DataFrame: Temizlenmiş DataFrame.
    """
    logger.info(f"=== Advanced Cleanup for pollutant='{pollutant}', original shape={df.shape} ===")
    df_clean = df.copy()
    init_shape = df_clean.shape

    # 1) Nadiren kullanılan sütunların kaldırılması
    if remove_rarely_used:
        cols_to_drop = [col for col in RARELY_USED_COLUMNS if col in df_clean.columns]
        if cols_to_drop:
            logger.info(f"Removing rarely used columns: {cols_to_drop}")
            df_clean.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # 2) Opsiyonel sütunların kaldırılması (isteğe bağlı)
    if drop_optional:
        opt_to_drop = [col for col in OPTIONAL_COLUMNS if col in df_clean.columns]
        if opt_to_drop:
            logger.info(f"Removing optional columns: {opt_to_drop}")
            df_clean.drop(columns=opt_to_drop, inplace=True, errors="ignore")

    # 3) Yüksek korelasyonlu sütunların kaldırılması
    # Eğer correlation_threshold < 1.0 ise, yüksek korelasyonlu sütunları çıkartalım.
    if correlation_threshold < 1.0:
        df_clean = drop_highly_correlated(df_clean, threshold=correlation_threshold)

    # 4) Aykırı değer (outlier) tespiti ve giderilmesi
    if outlier_method != "none":
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"Applying outlier removal using method '{outlier_method}' on columns: {numeric_cols}")
        if outlier_method == "iqr":
            df_clean = remove_outliers_iqr(df_clean, numeric_cols)
        elif outlier_method == "zscore":
            df_clean = remove_outliers_zscore(df_clean, numeric_cols)

    final_shape = df_clean.shape
    logger.info(
        f"Advanced cleanup completed. Final shape: {final_shape}, removed {init_shape[1] - final_shape[1]} columns.")

    # 5) Eğer save_path belirtilmişse, sonucu kaydet
    if save_path:
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df_clean.to_csv(save_path, index=False)
            logger.info(f"Advanced cleaned data saved at: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save advanced cleaned data: {e}")

    return df_clean


def drop_highly_correlated(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """
    Sayısal sütunlar arasında korelasyon matrisini hesaplar ve threshold değerinin üstünde korelasyona sahip sütunları kaldırır.
    İki sütun yüksek korelasyonda ise yalnızca birini tutar.

    Args:
        df (pd.DataFrame): Temizlenecek DataFrame.
        threshold (float, optional): Kaldırma eşik değeri. Defaults to 0.95.

    Returns:
        pd.DataFrame: Korelasyon bazlı sütun kaldırmasından geçmiş DataFrame.
    """
    logger.info(f"Computing correlation matrix and dropping columns with correlation > {threshold}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        logger.debug("Insufficient numeric columns for correlation drop.")
        return df

    corr_matrix = df[numeric_cols].corr().abs()
    # Üst üçgen matrisini al
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
    if to_drop:
        logger.info(f"Dropping columns due to high correlation: {to_drop}")
        df = df.drop(columns=to_drop, errors="ignore")
    return df


def remove_outliers_iqr(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """
    IQR yöntemi ile aykırı değer tespit edip, aykırı değer bulunan satırları kaldırır.

    Args:
        df (pd.DataFrame): DataFrame.
        numeric_cols (List[str]): İncelenecek sayısal sütunlar.

    Returns:
        pd.DataFrame: Aykırı değerleri kaldırılmış DataFrame.
    """
    logger.debug("Removing outliers using IQR method.")
    df_clean = df.copy()
    for col in numeric_cols:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean


def remove_outliers_zscore(df: pd.DataFrame, numeric_cols: List[str], z_thresh: float = 3.0) -> pd.DataFrame:
    """
    Z-score yöntemi ile aykırı değerleri tespit edip, aykırı değer bulunan satırları kaldırır.

    Args:
        df (pd.DataFrame): DataFrame.
        numeric_cols (List[str]): İncelenecek sayısal sütunlar.
        z_thresh (float, optional): Z-score eşiği. Defaults to 3.0.

    Returns:
        pd.DataFrame: Aykırı değerler çıkarılmış DataFrame.
    """
    logger.debug("Removing outliers using Z-score method.")
    from scipy.stats import zscore
    df_clean = df.copy()
    z_scores = np.abs(zscore(df_clean[numeric_cols], nan_policy="omit"))
    # Her satırda tüm numeric sütunların zscore'u eşik altında olanları tut
    mask = (z_scores < z_thresh).all(axis=1)
    return df_clean[mask]
