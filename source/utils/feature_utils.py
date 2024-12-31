import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be 'standard' or 'minmax'")

    df[cols] = scaler.fit_transform(df[cols])
    return df

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
    if operation == "multiply":
        return df[col1] * df[col2]
    elif operation == "add":
        return df[col1] + df[col2]
    else:
        raise ValueError("operation must be 'multiply' or 'add'")
