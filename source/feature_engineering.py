from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils.logger import setup_logger
from source.utils.path_utils import add_source_to_sys_path

logger = setup_logger(name="feature_engineering", log_file="../logs/feature_engineering.log", log_level="INFO")

# Add source to sys.path
add_source_to_sys_path()


def scale_features(df, cols, method="standard"):
    logger.info(f"Starting feature scaling: {cols}, Method: {method}")
    try:
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError("method must be 'standard' or 'minmax'")
        df[cols] = scaler.fit_transform(df[cols])
        logger.info("Feature scaling completed")
        return df
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


def create_interaction_terms(df, col1, col2, operation="multiply"):
    logger.info(f"Creating interaction term: {col1}, {col2}, Operation: {operation}")
    try:
        if operation == "multiply":
            return df[col1] * df[col2]
        elif operation == "add":
            return df[col1] + df[col2]
        else:
            raise ValueError("operation must be 'multiply' or 'add'")
    except Exception as e:
        logger.error(f"Error creating interaction term: {e}")
        raise
