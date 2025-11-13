"""
Production Inference Engine.

Provides high-performance inference capabilities with:
- Batch and real-time prediction
- Input validation and preprocessing
- Model caching
- Performance monitoring
- Error handling and logging
"""

import time
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

from source.utils.logger import setup_logger
from source.ml.model_registry import ModelRegistry

logger = setup_logger(
    name="inference",
    log_file="../logs/inference.log",
    log_level="INFO"
)


@dataclass
class PredictionResult:
    """Container for prediction results."""

    predictions: Union[List[float], np.ndarray]
    model_id: str
    inference_time_ms: float
    timestamp: str
    input_shape: tuple
    metadata: Optional[Dict[str, Any]] = None


class InferenceEngine:
    """Production inference engine for deployed models."""

    def __init__(
        self,
        model_registry: Optional[ModelRegistry] = None,
        cache_models: bool = True
    ):
        """
        Initialize InferenceEngine.

        Args:
            model_registry: ModelRegistry instance
            cache_models: Whether to cache loaded models
        """
        self.model_registry = model_registry or ModelRegistry()
        self.cache_models = cache_models
        self.model_cache = {}
        self.logger = logger

        self.logger.info("InferenceEngine initialized")

    def load_model(self, model_id: str, use_cache: bool = True) -> Any:
        """
        Load a model for inference.

        Args:
            model_id: Model ID
            use_cache: Whether to use cached model

        Returns:
            Loaded model
        """
        if use_cache and self.cache_models and model_id in self.model_cache:
            self.logger.debug(f"Using cached model: {model_id}")
            return self.model_cache[model_id]

        model = self.model_registry.get_model(model_id)

        if model is None:
            raise ValueError(f"Model not found: {model_id}")

        if self.cache_models:
            self.model_cache[model_id] = model

        return model

    def preprocess_input(
        self,
        data: Union[pd.DataFrame, Dict, List[Dict]],
        feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Preprocess input data for inference.

        Args:
            data: Input data
            feature_columns: Expected feature columns

        Returns:
            Preprocessed DataFrame
        """
        # Convert to DataFrame if necessary
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError(f"Unsupported input type: {type(data)}")

        # Select features if specified
        if feature_columns:
            missing_cols = set(feature_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
            df = df[feature_columns]

        # Handle missing values
        df = df.fillna(df.mean())

        return df

    def predict(
        self,
        data: Union[pd.DataFrame, Dict, List[Dict]],
        model_id: str,
        return_probabilities: bool = False
    ) -> PredictionResult:
        """
        Make predictions using a registered model.

        Args:
            data: Input data
            model_id: Model ID
            return_probabilities: Whether to return probabilities (classification only)

        Returns:
            PredictionResult object
        """
        from datetime import datetime

        start_time = time.time()

        # Load model
        model = self.load_model(model_id)

        # Get feature columns from metadata
        if model_id in self.model_registry.models:
            feature_columns = self.model_registry.models[model_id].feature_columns
        else:
            feature_columns = None

        # Preprocess input
        X = self.preprocess_input(data, feature_columns)

        # Make predictions
        if return_probabilities and hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(X)
        else:
            predictions = model.predict(X)

        # Calculate inference time
        inference_time_ms = (time.time() - start_time) * 1000

        # Create result
        result = PredictionResult(
            predictions=predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            model_id=model_id,
            inference_time_ms=inference_time_ms,
            timestamp=datetime.now().isoformat(),
            input_shape=X.shape
        )

        self.logger.info(
            f"Inference completed - Model: {model_id}, "
            f"Samples: {X.shape[0]}, Time: {inference_time_ms:.2f}ms"
        )

        return result

    def batch_predict(
        self,
        data: pd.DataFrame,
        model_id: str,
        batch_size: int = 1000
    ) -> PredictionResult:
        """
        Make batch predictions with chunking for large datasets.

        Args:
            data: Input DataFrame
            model_id: Model ID
            batch_size: Size of each batch

        Returns:
            PredictionResult object
        """
        from datetime import datetime

        start_time = time.time()

        all_predictions = []

        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i+batch_size]
            result = self.predict(batch, model_id)
            all_predictions.extend(result.predictions)

            self.logger.debug(
                f"Batch {i//batch_size + 1}/{(len(data)-1)//batch_size + 1} completed"
            )

        inference_time_ms = (time.time() - start_time) * 1000

        result = PredictionResult(
            predictions=all_predictions,
            model_id=model_id,
            inference_time_ms=inference_time_ms,
            timestamp=datetime.now().isoformat(),
            input_shape=data.shape,
            metadata={"batch_size": batch_size}
        )

        self.logger.info(
            f"Batch inference completed - Model: {model_id}, "
            f"Total samples: {len(data)}, Time: {inference_time_ms:.2f}ms"
        )

        return result


def main():
    """Main entry point for inference."""
    # Example usage
    engine = InferenceEngine()

    # Example prediction
    sample_data = {
        "feature_1": 10.5,
        "feature_2": 20.3,
        "feature_3": 15.7
    }

    # This will fail without a registered model, but demonstrates the API
    try:
        result = engine.predict(sample_data, model_id="example_model_v1_20240101_000000")
        print(f"Predictions: {result.predictions}")
        print(f"Inference time: {result.inference_time_ms:.2f}ms")
    except Exception as e:
        print(f"Example inference failed (expected): {e}")


if __name__ == "__main__":
    main()
