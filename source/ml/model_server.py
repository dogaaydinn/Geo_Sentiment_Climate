"""
Model Serving Infrastructure.

Provides high-performance model serving with load balancing and optimization.
Part of Phase 3: Scaling & Optimization - Distributed Systems.
"""

import numpy as np
import asyncio
from typing import Dict, Any, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import logging

from source.ml.model_registry import ModelRegistry
from source.ml.inference import InferenceEngine
from source.utils.cache_manager import prediction_cache, cache_manager
from source.utils.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


@dataclass
class ServingConfig:
    """Configuration for model serving."""
    max_batch_size: int = 32
    max_wait_time_ms: int = 50
    num_workers: int = 4
    enable_caching: bool = True
    cache_ttl: int = 300
    enable_batching: bool = True


class ModelServer:
    """
    High-performance model server with batching and caching.

    Features:
    - Dynamic batching for throughput
    - Prediction caching
    - Multi-threading for parallel inference
    - Circuit breaker protection
    - Model warming
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        config: Optional[ServingConfig] = None
    ):
        """
        Initialize model server.

        Args:
            model_registry: Model registry instance
            config: Serving configuration
        """
        self.model_registry = model_registry
        self.config = config or ServingConfig()
        self.inference_engine = InferenceEngine(model_registry)

        # Thread pool for parallel inference
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.num_workers
        )

        # Batching queue
        self.batch_queue: List[Dict] = []
        self.batch_lock = asyncio.Lock()

        # Circuit breakers per model
        self.model_breakers: Dict[str, CircuitBreaker] = {}

        logger.info(
            f"Model server initialized with {self.config.num_workers} workers"
        )

    def _get_model_breaker(self, model_id: str) -> CircuitBreaker:
        """Get or create circuit breaker for model."""
        if model_id not in self.model_breakers:
            self.model_breakers[model_id] = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=30,
                name=f"Model-{model_id}"
            )
        return self.model_breakers[model_id]

    async def predict(
        self,
        model_id: str,
        input_data: Dict[str, Any],
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Make prediction with caching and batching.

        Args:
            model_id: Model ID
            input_data: Input features
            use_cache: Whether to use cache

        Returns:
            Prediction result
        """
        # Check cache first
        if use_cache and self.config.enable_caching:
            cached = prediction_cache.get_cached_prediction(
                model_id, input_data
            )
            if cached:
                logger.debug(f"Cache hit for model {model_id}")
                return {
                    "prediction": cached,
                    "cached": True,
                    "model_id": model_id
                }

        # Get circuit breaker
        breaker = self._get_model_breaker(model_id)

        # Make prediction with circuit breaker
        try:
            prediction = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: breaker.call(
                    self.inference_engine.predict,
                    input_data,
                    model_id
                )
            )

            # Cache result
            if use_cache and self.config.enable_caching:
                prediction_cache.cache_prediction(
                    model_id,
                    input_data,
                    prediction,
                    ttl=self.config.cache_ttl
                )

            return {
                "prediction": prediction.predictions,
                "cached": False,
                "model_id": model_id,
                "inference_time_ms": prediction.inference_time_ms
            }

        except Exception as e:
            logger.error(f"Prediction error for model {model_id}: {e}")
            raise

    async def batch_predict(
        self,
        model_id: str,
        batch_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Batch prediction for multiple inputs.

        Args:
            model_id: Model ID
            batch_data: List of input data

        Returns:
            List of predictions
        """
        # Split into chunks if batch too large
        if len(batch_data) > self.config.max_batch_size:
            results = []
            for i in range(0, len(batch_data), self.config.max_batch_size):
                chunk = batch_data[i:i + self.config.max_batch_size]
                chunk_results = await self._predict_batch(model_id, chunk)
                results.extend(chunk_results)
            return results
        else:
            return await self._predict_batch(model_id, batch_data)

    async def _predict_batch(
        self,
        model_id: str,
        batch_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Internal batch prediction."""
        import pandas as pd

        breaker = self._get_model_breaker(model_id)

        try:
            # Convert to DataFrame for batch inference
            df = pd.DataFrame(batch_data)

            # Make batch prediction
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: breaker.call(
                    self.inference_engine.batch_predict,
                    df,
                    model_id,
                    self.config.max_batch_size
                )
            )

            # Format results
            predictions = [
                {
                    "prediction": pred,
                    "model_id": model_id,
                    "cached": False
                }
                for pred in result.predictions
            ]

            return predictions

        except Exception as e:
            logger.error(f"Batch prediction error for model {model_id}: {e}")
            raise

    def warm_models(self, model_ids: Optional[List[str]] = None):
        """
        Warm up models by loading into memory.

        Args:
            model_ids: List of model IDs to warm (None = all production)
        """
        if model_ids is None:
            # Warm all production models
            models = self.model_registry.list_models(stage="production")
            model_ids = [m.model_id for m in models]

        logger.info(f"Warming {len(model_ids)} models...")

        for model_id in model_ids:
            try:
                # Load model
                self.model_registry.load_model(model_id)
                logger.info(f"Warmed model: {model_id}")
            except Exception as e:
                logger.error(f"Error warming model {model_id}: {e}")

    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        cache_stats = cache_manager.get_stats()

        # Circuit breaker states
        breaker_states = {
            model_id: breaker.state.value
            for model_id, breaker in self.model_breakers.items()
        }

        return {
            "config": {
                "max_batch_size": self.config.max_batch_size,
                "num_workers": self.config.num_workers,
                "caching_enabled": self.config.enable_caching,
                "batching_enabled": self.config.enable_batching
            },
            "cache": cache_stats,
            "circuit_breakers": breaker_states,
            "loaded_models": len(self.model_registry.models)
        }

    async def health_check(self) -> Dict[str, Any]:
        """Health check for model server."""
        try:
            # Check if we can list models
            models = self.model_registry.list_models()

            # Check cache
            cache_healthy = cache_manager.redis_client.ping()

            return {
                "status": "healthy",
                "models_available": len(models),
                "cache_connected": cache_healthy,
                "workers": self.config.num_workers
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def shutdown(self):
        """Shutdown model server."""
        self.executor.shutdown(wait=True)
        logger.info("Model server shutdown complete")


class LoadBalancer:
    """
    Load balancer for distributing requests across model replicas.

    Implements round-robin and least-connections strategies.
    """

    def __init__(self, servers: List[ModelServer], strategy: str = "round_robin"):
        """
        Initialize load balancer.

        Args:
            servers: List of model servers
            strategy: Load balancing strategy (round_robin, least_connections)
        """
        self.servers = servers
        self.strategy = strategy
        self.current_index = 0
        self.request_counts = {i: 0 for i in range(len(servers))}

    async def predict(
        self,
        model_id: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route prediction to available server."""
        server = self._select_server()

        # Track request
        server_idx = self.servers.index(server)
        self.request_counts[server_idx] += 1

        try:
            result = await server.predict(model_id, input_data)
            return result
        finally:
            self.request_counts[server_idx] -= 1

    def _select_server(self) -> ModelServer:
        """Select server based on strategy."""
        if self.strategy == "round_robin":
            server = self.servers[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.servers)
            return server

        elif self.strategy == "least_connections":
            # Select server with least active requests
            min_idx = min(self.request_counts.items(), key=lambda x: x[1])[0]
            return self.servers[min_idx]

        else:
            # Default to round-robin
            return self.servers[self.current_index]

    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        return {
            "strategy": self.strategy,
            "num_servers": len(self.servers),
            "request_distribution": self.request_counts
        }
