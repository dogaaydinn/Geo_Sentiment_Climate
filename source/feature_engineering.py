import pandas as pd
from typing import List
from pathlib import Path
from source.utils.logger import setup_logger
from source.config.config_utils import config
from source.utils.project_paths import ProjectPaths
from source.utils.path_utils import add_source_to_sys_path
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Add source to sys.path
add_source_to_sys_path()

# Setup logger
logger = setup_logger(
    name="feature_engineering",
    log_file=str(Path("../logs") / "feature_engineering.log"),
    log_level="INFO"
)

paths = ProjectPaths.from_config(config)

paths.ensure_directories()

def scale_features(
        df: pd.DataFrame,
        cols: List[str],
        method: str = "standard"
) -> pd.DataFrame:

    logger.info(f"Starting feature scaling: {cols}, Method: {method}")
    try:
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError("method must be 'standard' or 'minmax'")

        # Check if columns exist
        missing_cols = [col for col in cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"The following columns are missing in the DataFrame: {missing_cols}")

        df[cols] = scaler.fit_transform(df[cols])
        logger.info("Feature scaling completed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error in scaling features: {e}")
        raise


def create_interaction_terms(
        df: pd.DataFrame,
        col1: str,
        col2: str,
        operation: str = "multiply"
) -> pd.Series:
    """
    Creates an interaction term between two columns using the specified operation.

    Args:
        df (pd.DataFrame): The DataFrame containing the columns.
        col1 (str): Name of the first column.
        col2 (str): Name of the second column.
        operation (str, optional): Operation to create the interaction term ('multiply' or 'add'). Defaults to "multiply".

    Returns:
        pd.Series: A new Series representing the interaction term.
    """
    logger.info(f"Creating interaction term between '{col1}' and '{col2}' using '{operation}' operation.")
    try:
        if col1 not in df.columns or col2 not in df.columns:
            missing = [col for col in [col1, col2] if col not in df.columns]
            raise KeyError(f"The following columns are missing in the DataFrame: {missing}")

        if operation == "multiply":
            interaction = df[col1] * df[col2]
        elif operation == "add":
            interaction = df[col1] + df[col2]
        else:
            raise ValueError("operation must be 'multiply' or 'add'")

        logger.info(f"Interaction term '{col1}_{col2}_{operation}' created successfully.")
        return interaction
    except Exception as e:
        logger.error(f"Error creating interaction term: {e}")
        raise
