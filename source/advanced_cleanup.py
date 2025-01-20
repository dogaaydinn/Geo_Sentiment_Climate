import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal

logger = logging.getLogger("advanced_cleanup")

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

    logger.debug("Removing outliers using Z-score method.")
    from scipy.stats import zscore
    df_clean = df.copy()
    z_scores = np.abs(zscore(df_clean[numeric_cols], nan_policy="omit"))
    # Her satırda tüm numeric sütunların zscore'u eşik altında olanları tut
    mask = (z_scores < z_thresh).all(axis=1)
    return df_clean[mask]
