"""
Model Registry Module.

Manages model versioning, storage, and retrieval with features like:
- Model versioning and tagging
- Model metadata storage
- Performance tracking
- Model promotion (dev/staging/production)
- Model comparison
"""

import json
import joblib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from source.utils.logger import setup_logger

logger = setup_logger(
    name="model_registry",
    log_file="../logs/model_registry.log",
    log_level="INFO"
)


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""

    model_id: str
    model_name: str
    version: str
    model_type: str
    task_type: str
    created_at: str
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    feature_columns: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    stage: str = "dev"  # dev, staging, production
    description: Optional[str] = None


class ModelRegistry:
    """Enterprise model registry for version control and deployment."""

    def __init__(self, registry_dir: Path = Path("../models/registry")):
        """
        Initialize ModelRegistry.

        Args:
            registry_dir: Directory to store registered models
        """
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.registry_dir / "registry_metadata.json"
        self.models = self._load_registry()

        self.logger = logger
        self.logger.info(f"Model Registry initialized at {self.registry_dir}")

    def _load_registry(self) -> Dict[str, ModelMetadata]:
        """Load registry metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                return {k: ModelMetadata(**v) for k, v in data.items()}
        return {}

    def _save_registry(self) -> None:
        """Save registry metadata to file."""
        data = {k: asdict(v) for k, v in self.models.items()}
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)

    def register_model(
        self,
        model_path: Path,
        model_name: str,
        model_type: str,
        task_type: str,
        metrics: Dict[str, float],
        hyperparameters: Dict[str, Any],
        feature_columns: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        stage: str = "dev"
    ) -> str:
        """
        Register a new model.

        Args:
            model_path: Path to the model file
            model_name: Name of the model
            model_type: Type of model (xgboost, lightgbm, etc.)
            task_type: Type of task (regression, classification)
            metrics: Performance metrics
            hyperparameters: Model hyperparameters
            feature_columns: List of feature columns
            tags: Model tags
            description: Model description
            stage: Model stage (dev/staging/production)

        Returns:
            Model ID
        """
        # Generate model ID and version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v{len([m for m in self.models.values() if m.model_name == model_name]) + 1}"
        model_id = f"{model_name}_{version}_{timestamp}"

        # Copy model to registry
        model_registry_path = self.registry_dir / f"{model_id}.joblib"
        shutil.copy(model_path, model_registry_path)

        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            version=version,
            model_type=model_type,
            task_type=task_type,
            created_at=datetime.now().isoformat(),
            metrics=metrics,
            hyperparameters=hyperparameters,
            feature_columns=feature_columns,
            tags=tags or [],
            stage=stage,
            description=description
        )

        # Register model
        self.models[model_id] = metadata
        self._save_registry()

        self.logger.info(f"Model registered: {model_id}")
        self.logger.info(f"Metrics: {metrics}")

        return model_id

    def get_model(self, model_id: str) -> Optional[Any]:
        """
        Load a registered model.

        Args:
            model_id: Model ID

        Returns:
            Loaded model or None
        """
        if model_id not in self.models:
            self.logger.error(f"Model not found: {model_id}")
            return None

        model_path = self.registry_dir / f"{model_id}.joblib"
        if not model_path.exists():
            self.logger.error(f"Model file not found: {model_path}")
            return None

        model = joblib.load(model_path)
        self.logger.info(f"Model loaded: {model_id}")
        return model

    def get_latest_model(self, model_name: str, stage: Optional[str] = None) -> Optional[Any]:
        """
        Get the latest version of a model.

        Args:
            model_name: Model name
            stage: Filter by stage (optional)

        Returns:
            Latest model or None
        """
        matching_models = [
            m for m in self.models.values()
            if m.model_name == model_name and (stage is None or m.stage == stage)
        ]

        if not matching_models:
            self.logger.warning(f"No models found for {model_name}")
            return None

        latest_model = max(matching_models, key=lambda m: m.created_at)
        return self.get_model(latest_model.model_id)

    def promote_model(self, model_id: str, new_stage: str) -> bool:
        """
        Promote a model to a new stage.

        Args:
            model_id: Model ID
            new_stage: New stage (staging/production)

        Returns:
            Success status
        """
        if model_id not in self.models:
            self.logger.error(f"Model not found: {model_id}")
            return False

        self.models[model_id].stage = new_stage
        self._save_registry()

        self.logger.info(f"Model {model_id} promoted to {new_stage}")
        return True

    def list_models(self, model_name: Optional[str] = None, stage: Optional[str] = None) -> List[ModelMetadata]:
        """
        List registered models.

        Args:
            model_name: Filter by model name
            stage: Filter by stage

        Returns:
            List of model metadata
        """
        models = list(self.models.values())

        if model_name:
            models = [m for m in models if m.model_name == model_name]

        if stage:
            models = [m for m in models if m.stage == stage]

        return sorted(models, key=lambda m: m.created_at, reverse=True)

    def compare_models(self, model_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple models.

        Args:
            model_ids: List of model IDs

        Returns:
            Comparison DataFrame
        """
        import pandas as pd

        comparison_data = []
        for model_id in model_ids:
            if model_id in self.models:
                metadata = self.models[model_id]
                row = {
                    'model_id': model_id,
                    'model_name': metadata.model_name,
                    'version': metadata.version,
                    'stage': metadata.stage,
                    'created_at': metadata.created_at,
                    **metadata.metrics
                }
                comparison_data.append(row)

        return pd.DataFrame(comparison_data)
