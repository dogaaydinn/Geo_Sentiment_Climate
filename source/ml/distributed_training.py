"""
Distributed Training Infrastructure.

Provides distributed training capabilities for large-scale model training.
Part of Phase 3: Scaling & Optimization - Distributed Systems.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd

try:
    import ray
    from ray import train
    from ray.train import ScalingConfig
    from ray.air.config import RunConfig
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Ray not available. Distributed training disabled.")

try:
    import horovod.torch as hvd
    HOROVOD_AVAILABLE = True
except ImportError:
    HOROVOD_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    num_workers: int = 4
    use_gpu: bool = False
    gpu_per_worker: float = 0.5
    cpu_per_worker: int = 4
    memory_per_worker_gb: int = 8
    backend: str = "ray"  # ray, horovod
    checkpoint_frequency: int = 5


class DistributedTrainer:
    """
    Distributed model training using Ray or Horovod.

    Features:
    - Multi-worker training
    - GPU support
    - Automatic checkpointing
    - Fault tolerance
    - Hyperparameter tuning at scale
    """

    def __init__(self, config: Optional[DistributedConfig] = None):
        """
        Initialize distributed trainer.

        Args:
            config: Distributed training configuration
        """
        self.config = config or DistributedConfig()

        if self.config.backend == "ray" and RAY_AVAILABLE:
            self._init_ray()
        elif self.config.backend == "horovod" and HOROVOD_AVAILABLE:
            self._init_horovod()
        else:
            logger.warning(
                f"Backend {self.config.backend} not available. "
                "Using single-node training."
            )

    def _init_ray(self):
        """Initialize Ray cluster."""
        if not ray.is_initialized():
            try:
                # Connect to existing cluster or start local
                ray.init(
                    address=os.getenv("RAY_ADDRESS", "auto"),
                    ignore_reinit_error=True,
                    logging_level=logging.INFO
                )
                logger.info(f"Ray initialized: {ray.cluster_resources()}")
            except Exception as e:
                logger.error(f"Ray initialization failed: {e}")
                raise

    def _init_horovod(self):
        """Initialize Horovod."""
        try:
            hvd.init()
            logger.info(
                f"Horovod initialized: rank={hvd.rank()}, "
                f"size={hvd.size()}"
            )
        except Exception as e:
            logger.error(f"Horovod initialization failed: {e}")
            raise

    def train_xgboost_distributed(
        self,
        train_data: pd.DataFrame,
        target_col: str,
        params: Dict[str, Any]
    ) -> Any:
        """
        Train XGBoost model using distributed Ray.

        Args:
            train_data: Training data
            target_col: Target column name
            params: XGBoost parameters

        Returns:
            Trained model
        """
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray not available for distributed training")

        from ray.train.xgboost import XGBoostTrainer
        from ray.air.config import ScalingConfig

        # Prepare data
        X = train_data.drop(columns=[target_col])
        y = train_data[target_col]

        # Create Ray datasets
        train_dataset = ray.data.from_pandas(
            pd.concat([X, y], axis=1)
        )

        # Configure scaling
        scaling_config = ScalingConfig(
            num_workers=self.config.num_workers,
            use_gpu=self.config.use_gpu,
            resources_per_worker={
                "CPU": self.config.cpu_per_worker,
                "GPU": self.config.gpu_per_worker if self.config.use_gpu else 0
            }
        )

        # Create trainer
        trainer = XGBoostTrainer(
            scaling_config=scaling_config,
            label_column=target_col,
            params=params,
            datasets={"train": train_dataset}
        )

        # Train
        logger.info(
            f"Starting distributed XGBoost training with "
            f"{self.config.num_workers} workers"
        )

        result = trainer.fit()

        logger.info("Distributed training completed")

        return result.checkpoint

    def train_sklearn_distributed(
        self,
        train_data: pd.DataFrame,
        target_col: str,
        model_class: type,
        model_params: Dict[str, Any]
    ) -> Any:
        """
        Train scikit-learn model using Ray.

        Args:
            train_data: Training data
            target_col: Target column name
            model_class: Model class (e.g., RandomForestClassifier)
            model_params: Model parameters

        Returns:
            Trained model
        """
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray not available for distributed training")

        from ray.train.sklearn import SklearnTrainer

        # Prepare data
        X = train_data.drop(columns=[target_col])
        y = train_data[target_col]

        def train_func():
            """Training function to run on each worker."""
            model = model_class(**model_params)
            model.fit(X, y)
            return model

        # Create trainer
        trainer = SklearnTrainer(
            train_func=train_func,
            scaling_config=ScalingConfig(
                num_workers=self.config.num_workers
            )
        )

        result = trainer.fit()

        return result.checkpoint

    def hyperparameter_tuning_distributed(
        self,
        train_data: pd.DataFrame,
        target_col: str,
        model_class: type,
        param_space: Dict[str, Any],
        num_samples: int = 20
    ) -> Dict[str, Any]:
        """
        Distributed hyperparameter tuning using Ray Tune.

        Args:
            train_data: Training data
            target_col: Target column
            model_class: Model class
            param_space: Hyperparameter search space
            num_samples: Number of trials

        Returns:
            Best parameters
        """
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray not available for distributed tuning")

        from ray import tune
        from ray.tune.schedulers import ASHAScheduler

        X = train_data.drop(columns=[target_col])
        y = train_data[target_col]

        def objective(config):
            """Objective function for tuning."""
            from sklearn.model_selection import cross_val_score

            model = model_class(**config)
            scores = cross_val_score(model, X, y, cv=3)
            mean_score = scores.mean()

            return {"score": mean_score}

        # ASHA scheduler for early stopping
        scheduler = ASHAScheduler(
            max_t=100,
            grace_period=10,
            reduction_factor=3
        )

        # Run tuning
        logger.info(
            f"Starting distributed hyperparameter tuning: "
            f"{num_samples} trials"
        )

        analysis = tune.run(
            objective,
            config=param_space,
            num_samples=num_samples,
            scheduler=scheduler,
            resources_per_trial={
                "cpu": self.config.cpu_per_worker,
                "gpu": self.config.gpu_per_worker if self.config.use_gpu else 0
            },
            verbose=1
        )

        best_config = analysis.get_best_config(metric="score", mode="max")

        logger.info(f"Best parameters found: {best_config}")

        return best_config

    def data_parallel_training(
        self,
        train_func: Callable,
        data_shards: List[pd.DataFrame],
        **kwargs
    ) -> List[Any]:
        """
        Data-parallel training across multiple workers.

        Args:
            train_func: Training function
            data_shards: Data split across workers
            **kwargs: Additional arguments for train_func

        Returns:
            List of results from each worker
        """
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray not available for parallel training")

        @ray.remote
        def remote_train(shard: pd.DataFrame):
            """Remote training function."""
            return train_func(shard, **kwargs)

        # Distribute training
        logger.info(
            f"Starting data-parallel training on "
            f"{len(data_shards)} shards"
        )

        futures = [
            remote_train.remote(shard)
            for shard in data_shards
        ]

        results = ray.get(futures)

        logger.info("Data-parallel training completed")

        return results

    def get_cluster_resources(self) -> Dict[str, Any]:
        """Get available cluster resources."""
        if RAY_AVAILABLE and ray.is_initialized():
            resources = ray.cluster_resources()
            return {
                "total_cpus": resources.get("CPU", 0),
                "total_gpus": resources.get("GPU", 0),
                "total_memory_gb": resources.get("memory", 0) / (1024**3),
                "num_nodes": len(ray.nodes())
            }
        else:
            return {
                "total_cpus": os.cpu_count(),
                "total_gpus": 0,
                "total_memory_gb": 0,
                "num_nodes": 1
            }

    def shutdown(self):
        """Shutdown distributed training cluster."""
        if RAY_AVAILABLE and ray.is_initialized():
            ray.shutdown()
            logger.info("Ray cluster shutdown")


class DistributedDataLoader:
    """
    Distributed data loading and preprocessing.

    Efficiently loads and preprocesses large datasets across workers.
    """

    def __init__(self, num_workers: int = 4):
        """
        Initialize distributed data loader.

        Args:
            num_workers: Number of worker processes
        """
        self.num_workers = num_workers

        if RAY_AVAILABLE:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)

    def load_and_preprocess(
        self,
        file_paths: List[str],
        preprocess_func: Callable
    ) -> pd.DataFrame:
        """
        Load and preprocess files in parallel.

        Args:
            file_paths: List of file paths
            preprocess_func: Preprocessing function

        Returns:
            Combined preprocessed DataFrame
        """
        if not RAY_AVAILABLE:
            # Fallback to sequential processing
            dfs = [preprocess_func(path) for path in file_paths]
            return pd.concat(dfs, ignore_index=True)

        @ray.remote
        def process_file(path: str):
            """Process single file."""
            return preprocess_func(path)

        logger.info(f"Processing {len(file_paths)} files in parallel")

        # Process files in parallel
        futures = [process_file.remote(path) for path in file_paths]
        dfs = ray.get(futures)

        # Combine results
        combined_df = pd.concat(dfs, ignore_index=True)

        logger.info(f"Loaded {len(combined_df)} rows from {len(file_paths)} files")

        return combined_df

    def create_data_shards(
        self,
        data: pd.DataFrame,
        num_shards: int
    ) -> List[pd.DataFrame]:
        """
        Split data into shards for distributed processing.

        Args:
            data: DataFrame to split
            num_shards: Number of shards

        Returns:
            List of DataFrame shards
        """
        shard_size = len(data) // num_shards
        shards = []

        for i in range(num_shards):
            start_idx = i * shard_size
            end_idx = start_idx + shard_size if i < num_shards - 1 else len(data)
            shards.append(data.iloc[start_idx:end_idx].copy())

        logger.info(f"Created {num_shards} data shards")

        return shards


# Global distributed trainer
distributed_trainer = DistributedTrainer()
distributed_loader = DistributedDataLoader()
