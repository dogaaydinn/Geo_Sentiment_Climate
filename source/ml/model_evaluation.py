"""
Model Evaluation and Metrics Module.

Comprehensive evaluation framework for ML models including:
- Regression metrics (RMSE, MAE, R2, MAPE)
- Classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Cross-validation
- Residual analysis
- Learning curves
- Feature importance visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from sklearn.model_selection import learning_curve

from source.utils.logger import setup_logger
from source.config.config_utils import config

logger = setup_logger(
    name="model_evaluation",
    log_file="../logs/model_evaluation.log",
    log_level="INFO"
)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    # Regression metrics
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    mape: Optional[float] = None
    mse: Optional[float] = None

    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    roc_auc: Optional[float] = None

    # Additional info
    task_type: str = "regression"
    model_name: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class ModelEvaluator:
    """Comprehensive model evaluation framework."""

    def __init__(self, plots_dir: Optional[Path] = None):
        """
        Initialize ModelEvaluator.

        Args:
            plots_dir: Directory to save plots
        """
        self.logger = logger
        self.plots_dir = plots_dir or Path(config.get("paths", {}).get("plots_dir", "../plots"))
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        sns.set_style("whitegrid")

    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> EvaluationMetrics:
        """
        Evaluate regression model.

        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model

        Returns:
            EvaluationMetrics object
        """
        self.logger.info(f"Evaluating regression model: {model_name}")

        metrics = EvaluationMetrics(
            rmse=np.sqrt(mean_squared_error(y_true, y_pred)),
            mae=mean_absolute_error(y_true, y_pred),
            r2=r2_score(y_true, y_pred),
            mse=mean_squared_error(y_true, y_pred),
            task_type="regression",
            model_name=model_name
        )

        try:
            metrics.mape = mean_absolute_percentage_error(y_true, y_pred)
        except:
            pass

        self.logger.info(f"RMSE: {metrics.rmse:.4f}")
        self.logger.info(f"MAE: {metrics.mae:.4f}")
        self.logger.info(f"R2: {metrics.r2:.4f}")

        return metrics

    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        model_name: str = "Model"
    ) -> EvaluationMetrics:
        """
        Evaluate classification model.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            model_name: Name of the model

        Returns:
            EvaluationMetrics object
        """
        self.logger.info(f"Evaluating classification model: {model_name}")

        metrics = EvaluationMetrics(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, average='weighted', zero_division=0),
            recall=recall_score(y_true, y_pred, average='weighted', zero_division=0),
            f1=f1_score(y_true, y_pred, average='weighted', zero_division=0),
            task_type="classification",
            model_name=model_name
        )

        if y_prob is not None:
            try:
                metrics.roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            except:
                pass

        self.logger.info(f"Accuracy: {metrics.accuracy:.4f}")
        self.logger.info(f"F1 Score: {metrics.f1:.4f}")

        return metrics

    def plot_regression_results(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save: bool = True
    ) -> None:
        """
        Plot regression results.

        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Predicted vs Actual
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=10)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title(f'{model_name} - Predicted vs Actual')
        axes[0, 0].grid(True, alpha=0.3)

        # Residuals
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title(f'{model_name} - Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)

        # Residual histogram
        axes[1, 0].hist(residuals, bins=50, edgecolor='black')
        axes[1, 0].set_xlabel('Residual Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'{model_name} - Residual Distribution')
        axes[1, 0].grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title(f'{model_name} - Q-Q Plot')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            save_path = self.plots_dir / f"{model_name}_regression_evaluation.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Regression plot saved to {save_path}")

        plt.close()

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        model_name: str = "Model",
        save: bool = True
    ) -> None:
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            model_name: Name of the model
            save: Whether to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f'{model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save:
            save_path = self.plots_dir / f"{model_name}_confusion_matrix.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Confusion matrix saved to {save_path}")

        plt.close()

    def plot_feature_importance(
        self,
        feature_importance: pd.DataFrame,
        model_name: str = "Model",
        top_n: int = 20,
        save: bool = True
    ) -> None:
        """
        Plot feature importance.

        Args:
            feature_importance: DataFrame with 'feature' and 'importance' columns
            model_name: Name of the model
            top_n: Number of top features to show
            save: Whether to save the plot
        """
        top_features = feature_importance.head(top_n)

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'{model_name} - Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save:
            save_path = self.plots_dir / f"{model_name}_feature_importance.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Feature importance plot saved to {save_path}")

        plt.close()
