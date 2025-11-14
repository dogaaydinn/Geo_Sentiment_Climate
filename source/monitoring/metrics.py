"""Prometheus metrics collection."""

import time
from functools import wraps
from typing import Callable
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    generate_latest, REGISTRY, CollectorRegistry
)
from fastapi import Request
from source.utils.logger import setup_logger

logger = setup_logger(name="metrics", log_file="../logs/metrics.log")


class PrometheusMetrics:
    """
    Prometheus metrics for API and ML monitoring.

    Tracks:
    - Request counts
    - Response times
    - Error rates
    - Model inference metrics
    - System health
    """

    def __init__(self, app_name: str = "geo_climate"):
        """Initialize metrics."""
        self.app_name = app_name

        # Request metrics
        self.request_count = Counter(
            f"{app_name}_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"]
        )

        self.request_duration = Histogram(
            f"{app_name}_request_duration_seconds",
            "HTTP request duration",
            ["method", "endpoint"]
        )

        self.request_size = Summary(
            f"{app_name}_request_size_bytes",
            "HTTP request size"
        )

        self.response_size = Summary(
            f"{app_name}_response_size_bytes",
            "HTTP response size"
        )

        # ML metrics
        self.prediction_count = Counter(
            f"{app_name}_predictions_total",
            "Total predictions made",
            ["model_id", "model_type"]
        )

        self.prediction_duration = Histogram(
            f"{app_name}_prediction_duration_seconds",
            "Prediction inference time",
            ["model_id"]
        )

        self.prediction_error = Counter(
            f"{app_name}_prediction_errors_total",
            "Prediction errors",
            ["model_id", "error_type"]
        )

        self.model_cache_hit = Counter(
            f"{app_name}_model_cache_hits_total",
            "Model cache hits",
            ["model_id"]
        )

        self.model_cache_miss = Counter(
            f"{app_name}_model_cache_misses_total",
            "Model cache misses",
            ["model_id"]
        )

        # System metrics
        self.active_connections = Gauge(
            f"{app_name}_active_connections",
            "Number of active connections"
        )

        self.memory_usage = Gauge(
            f"{app_name}_memory_usage_bytes",
            "Memory usage in bytes"
        )

        self.cpu_usage = Gauge(
            f"{app_name}_cpu_usage_percent",
            "CPU usage percentage"
        )

        # Database metrics
        self.db_connection_pool_size = Gauge(
            f"{app_name}_db_connection_pool_size",
            "Database connection pool size"
        )

        self.db_query_duration = Histogram(
            f"{app_name}_db_query_duration_seconds",
            "Database query duration",
            ["query_type"]
        )

        # Cache metrics
        self.cache_hit = Counter(
            f"{app_name}_cache_hits_total",
            "Cache hits",
            ["cache_type"]
        )

        self.cache_miss = Counter(
            f"{app_name}_cache_misses_total",
            "Cache misses",
            ["cache_type"]
        )

        logger.info("Prometheus metrics initialized")

    def track_request(self, request: Request, response_time: float, status_code: int):
        """Track HTTP request."""
        self.request_count.labels(
            method=request.method,
            endpoint=request.url.path,
            status=status_code
        ).inc()

        self.request_duration.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(response_time)

    def track_prediction(
        self,
        model_id: str,
        model_type: str,
        duration: float,
        error: bool = False,
        error_type: str = None
    ):
        """Track ML prediction."""
        if not error:
            self.prediction_count.labels(
                model_id=model_id,
                model_type=model_type
            ).inc()

            self.prediction_duration.labels(
                model_id=model_id
            ).observe(duration)
        else:
            self.prediction_error.labels(
                model_id=model_id,
                error_type=error_type or "unknown"
            ).inc()

    def track_cache(self, cache_type: str, hit: bool):
        """Track cache hit/miss."""
        if hit:
            self.cache_hit.labels(cache_type=cache_type).inc()
        else:
            self.cache_miss.labels(cache_type=cache_type).inc()


# Global metrics instance
metrics = PrometheusMetrics()


def track_time(metric_name: str = None):
    """Decorator to track function execution time."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                logger.debug(f"{func.__name__} took {duration:.3f}s")

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                logger.debug(f"{func.__name__} took {duration:.3f}s")

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def count_requests(endpoint: str):
    """Decorator to count endpoint requests."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                metrics.request_count.labels(
                    method="GET",
                    endpoint=endpoint,
                    status=200
                ).inc()
                return result
            except Exception as e:
                metrics.request_count.labels(
                    method="GET",
                    endpoint=endpoint,
                    status=500
                ).inc()
                raise

        return wrapper

    return decorator


def observe_latency(histogram: Histogram):
    """Decorator to observe function latency."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                histogram.observe(time.time() - start_time)

        return wrapper

    return decorator


def register_metrics():
    """Register all metrics with Prometheus."""
    return generate_latest(REGISTRY)


import asyncio
