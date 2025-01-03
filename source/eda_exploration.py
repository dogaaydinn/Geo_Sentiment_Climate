import logging

import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
import seaborn as sns

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path):

    logger.info(f"Loading data from {file_path}")
    try:
        data = pd.read_csv(file_path)
        logger.info("Data loaded successfully")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def basic_info(df):
    info = {
        "Shape": df.shape,
        "Columns": df.columns.tolist(),
        "Data Types": df.dtypes.to_dict(),
        "Missing Values": df.isnull().sum().to_dict()
    }
    return info

def missing_values(df):

    logger.info("Analyzing missing values in the dataset")
    print("[INFO] Missing Value Percentages:")
    print(df.isnull().mean() * 100)
    msno.matrix(df)
    plt.show()

def distribution_analysis(df, numeric_cols):

    logger.info("Performing distribution analysis for numeric columns")
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, bins=30, color="blue")
        plt.title(f"Distribution of {col}")
        plt.show()

def correlation_analysis(df, numeric_cols):

    logger.info("Creating correlation matrix for numeric columns")
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

def detect_outliers(df, numeric_cols):

    logger.info("Detecting outliers in numeric columns")
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Outliers in {col}")
        plt.show()