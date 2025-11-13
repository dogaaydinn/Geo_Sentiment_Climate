"""
Machine Learning Module for Geo_Sentiment_Climate.

Provides enterprise-level ML capabilities including:
- Model training and hyperparameter optimization
- Model evaluation and validation
- Model registry and versioning
- Production inference
- Experiment tracking
"""

from .model_training import ModelTrainer, TrainingConfig
from .model_evaluation import ModelEvaluator, EvaluationMetrics
from .model_registry import ModelRegistry
from .inference import InferenceEngine

__all__ = [
    "ModelTrainer",
    "TrainingConfig",
    "ModelEvaluator",
    "EvaluationMetrics",
    "ModelRegistry",
    "InferenceEngine",
]

__version__ = "2.0.0"
