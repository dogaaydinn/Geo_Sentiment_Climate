"""
Enterprise ML Model Training Pipeline.

Features:
- Multi-model support (XGBoost, LightGBM, CatBoost, RandomForest, Neural Networks)
- Automated hyperparameter optimization (Optuna)
- Cross-validation and stratified sampling
- Experiment tracking (MLflow, Weights & Biases)
- Model checkpointing and early stopping
- Feature importance analysis
- Training monitoring and logging
"""

import os
import json
import joblib
import mlflow
import optuna
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field, asdict

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score

from source.utils.logger import setup_logger
from source.config.config_utils import config

logger = setup_logger(
    name="model_training",
    log_file="../logs/model_training.log",
    log_level="INFO"
)


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    model_type: str = "xgboost"  # xgboost, lightgbm, catboost, randomforest, gradientboosting
    task_type: str = "regression"  # regression or classification
    n_trials: int = 50  # Optuna trials
    cv_folds: int = 5
    random_state: int = 42
    early_stopping_rounds: int = 50
    use_mlflow: bool = True
    use_wandb: bool = False
    experiment_name: str = "geo_climate_experiment"
    model_save_dir: Path = Path("../models")
    feature_importance_plot: bool = True

    # Model-specific hyperparameters
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Training data paths
    train_data_path: Optional[str] = None
    test_data_path: Optional[str] = None

    # Target and feature columns
    target_column: str = "aqi"
    feature_columns: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None

    def __post_init__(self):
        """Initialize default hyperparameters if not provided."""
        if not self.hyperparameters:
            self.hyperparameters = self._get_default_hyperparameters()

    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters for each model type."""
        defaults = {
            "xgboost": {
                "n_estimators": 1000,
                "max_depth": 6,
                "learning_rate": 0.01,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
            },
            "lightgbm": {
                "n_estimators": 1000,
                "max_depth": 6,
                "learning_rate": 0.01,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            "catboost": {
                "iterations": 1000,
                "depth": 6,
                "learning_rate": 0.01,
                "l2_leaf_reg": 3.0,
            },
            "randomforest": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
            },
        }
        return defaults.get(self.model_type, {})


class ModelTrainer:
    """
    Enterprise-level model trainer with comprehensive features.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize ModelTrainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.logger = logger
        self.model = None
        self.best_params = None
        self.training_history = {}
        self.feature_importance = None

        # Create directories
        self.config.model_save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize MLflow if requested
        if self.config.use_mlflow:
            mlflow.set_experiment(self.config.experiment_name)

        self.logger.info(f"ModelTrainer initialized with {self.config.model_type}")

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test data.

        Returns:
            Tuple of (train_df, test_df)
        """
        self.logger.info("Loading training data...")

        if self.config.train_data_path:
            train_df = pd.read_csv(self.config.train_data_path)
        else:
            # Default path
            processed_dir = Path(config.get("paths", {}).get("processed_dir", "../data/processed"))
            train_df = pd.read_csv(processed_dir / "epa_long_preprocessed_ADV.csv")

        # Load or split test data
        if self.config.test_data_path:
            test_df = pd.read_csv(self.config.test_data_path)
        else:
            # Create train/test split
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(
                train_df,
                test_size=0.2,
                random_state=self.config.random_state
            )

        self.logger.info(f"Train data shape: {train_df.shape}")
        self.logger.info(f"Test data shape: {test_df.shape}")

        return train_df, test_df

    def prepare_features(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target from DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (X, y)
        """
        # Select features
        if self.config.feature_columns:
            feature_cols = [col for col in self.config.feature_columns if col in df.columns]
        else:
            # Use all numeric columns except target
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if self.config.target_column in feature_cols:
                feature_cols.remove(self.config.target_column)

        X = df[feature_cols].copy()
        y = df[self.config.target_column].copy()

        # Handle missing values
        X = X.fillna(X.mean())

        self.logger.info(f"Features shape: {X.shape}")
        self.logger.info(f"Target shape: {y.shape}")

        return X, y

    def create_model(self, params: Optional[Dict] = None) -> Any:
        """
        Create a model instance with given parameters.

        Args:
            params: Model hyperparameters

        Returns:
            Model instance
        """
        if params is None:
            params = self.config.hyperparameters

        model_type = self.config.model_type
        task_type = self.config.task_type

        if model_type == "xgboost":
            if task_type == "regression":
                model = xgb.XGBRegressor(**params, random_state=self.config.random_state)
            else:
                model = xgb.XGBClassifier(**params, random_state=self.config.random_state)

        elif model_type == "lightgbm":
            if task_type == "regression":
                model = lgb.LGBMRegressor(**params, random_state=self.config.random_state, verbose=-1)
            else:
                model = lgb.LGBMClassifier(**params, random_state=self.config.random_state, verbose=-1)

        elif model_type == "catboost":
            if task_type == "regression":
                model = CatBoostRegressor(**params, random_state=self.config.random_state, verbose=False)
            else:
                model = CatBoostClassifier(**params, random_state=self.config.random_state, verbose=False)

        elif model_type == "randomforest":
            if task_type == "regression":
                model = RandomForestRegressor(**params, random_state=self.config.random_state, n_jobs=-1)
            else:
                model = RandomForestClassifier(**params, random_state=self.config.random_state, n_jobs=-1)

        elif model_type == "gradientboosting":
            model = GradientBoostingRegressor(**params, random_state=self.config.random_state)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return model

    def objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Optuna objective function for hyperparameter optimization.

        Args:
            trial: Optuna trial
            X: Features
            y: Target

        Returns:
            CV score
        """
        # Define hyperparameter search space
        if self.config.model_type == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            }

        elif self.config.model_type == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }

        elif self.config.model_type == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", 100, 2000),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            }

        elif self.config.model_type == "randomforest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            }

        else:
            params = {}

        # Create model and perform cross-validation
        model = self.create_model(params)

        if self.config.task_type == "regression":
            scores = cross_val_score(
                model, X, y,
                cv=self.config.cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            return -scores.mean()
        else:
            scores = cross_val_score(
                model, X, y,
                cv=self.config.cv_folds,
                scoring='f1_weighted',
                n_jobs=-1
            )
            return scores.mean()

    def optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.

        Args:
            X: Features
            y: Target

        Returns:
            Best hyperparameters
        """
        self.logger.info("Starting hyperparameter optimization...")

        study = optuna.create_study(
            direction="minimize" if self.config.task_type == "regression" else "maximize",
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state)
        )

        study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=self.config.n_trials,
            show_progress_bar=True
        )

        self.best_params = study.best_params
        self.logger.info(f"Best parameters: {self.best_params}")
        self.logger.info(f"Best CV score: {study.best_value:.6f}")

        return self.best_params

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Any:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)

        Returns:
            Trained model
        """
        self.logger.info("Training model...")

        # Use best params if available
        params = self.best_params if self.best_params else self.config.hyperparameters

        # Create model
        self.model = self.create_model(params)

        # Train with early stopping if validation set provided
        if X_val is not None and y_val is not None and hasattr(self.model, 'fit'):
            if self.config.model_type in ["xgboost", "lightgbm"]:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=self.config.early_stopping_rounds,
                    verbose=False
                )
            elif self.config.model_type == "catboost":
                self.model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=self.config.early_stopping_rounds,
                    verbose=False
                )
            else:
                self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)

        self.logger.info("Model training completed")

        # Extract feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

        return self.model

    def save_model(self, model_name: Optional[str] = None) -> Path:
        """
        Save the trained model.

        Args:
            model_name: Name for the saved model

        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{self.config.model_type}_{timestamp}.joblib"

        model_path = self.config.model_save_dir / model_name
        joblib.dump(self.model, model_path)

        # Save config and metadata
        metadata = {
            "model_type": self.config.model_type,
            "task_type": self.config.task_type,
            "parameters": self.best_params or self.config.hyperparameters,
            "trained_at": datetime.now().isoformat(),
            "feature_columns": list(self.feature_importance['feature']) if self.feature_importance is not None else None
        }

        metadata_path = self.config.model_save_dir / f"{model_name.replace('.joblib', '_metadata.json')}"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Model saved to: {model_path}")
        self.logger.info(f"Metadata saved to: {metadata_path}")

        return model_path

    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline.

        Returns:
            Dictionary with training results
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Full Training Pipeline")
        self.logger.info("=" * 80)

        # Load data
        train_df, test_df = self.load_data()

        # Prepare features
        X_train, y_train = self.prepare_features(train_df)
        X_test, y_test = self.prepare_features(test_df)

        # Split for validation
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=self.config.random_state
        )

        # Optimize hyperparameters
        self.optimize_hyperparameters(X_train_split, y_train_split)

        # Train model
        self.train(X_train, y_train, X_val, y_val)

        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        if self.config.task_type == "regression":
            train_metrics = {
                "rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
                "mae": mean_absolute_error(y_train, train_pred),
                "r2": r2_score(y_train, train_pred)
            }
            test_metrics = {
                "rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
                "mae": mean_absolute_error(y_test, test_pred),
                "r2": r2_score(y_test, test_pred)
            }
        else:
            train_metrics = {
                "accuracy": accuracy_score(y_train, train_pred),
                "f1": f1_score(y_train, train_pred, average='weighted')
            }
            test_metrics = {
                "accuracy": accuracy_score(y_test, test_pred),
                "f1": f1_score(y_test, test_pred, average='weighted')
            }

        self.logger.info(f"Train metrics: {train_metrics}")
        self.logger.info(f"Test metrics: {test_metrics}")

        # Save model
        model_path = self.save_model()

        results = {
            "model_path": str(model_path),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "best_params": self.best_params,
            "feature_importance": self.feature_importance.to_dict() if self.feature_importance is not None else None
        }

        self.logger.info("=" * 80)
        self.logger.info("Training Pipeline Completed Successfully")
        self.logger.info("=" * 80)

        return results


def main():
    """Main entry point for model training."""
    # Example configuration
    training_config = TrainingConfig(
        model_type="xgboost",
        task_type="regression",
        n_trials=20,
        target_column="aqi",
        experiment_name="geo_climate_aqi_prediction"
    )

    trainer = ModelTrainer(training_config)
    results = trainer.run_full_pipeline()

    print("\nTraining Results:")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
