"""
Enterprise-Grade Monitoring and Metrics Module.

Comprehensive Prometheus metrics following Google SRE best practices:
- Golden Signals (Latency, Traffic, Errors, Saturation)
- Application metrics (API, ML models, business)
- Infrastructure metrics (CPU, memory, disk, network)
- Custom business metrics

Part of Week 5: Prometheus & Metrics Implementation
"""

from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    generate_latest, REGISTRY, CollectorRegistry
)
from fastapi import Response
from typing import Dict, Optional, Callable
from functools import wraps
from datetime import datetime
import time
import psutil
import os
import structlog

# Configure logger
logger = structlog.get_logger()

# Create custom registry
registry = CollectorRegistry(auto_describe=True)


# ============================================================================
# GOLDEN SIGNALS - Google SRE Best Practices
# ============================================================================

# 1. LATENCY - Request Duration
# ============================================================================

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['method', 'endpoint', 'status'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=registry
)

http_request_size_bytes = Histogram(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint'],
    buckets=[100, 1000, 10000, 100000, 1000000, 10000000],
    registry=registry
)

http_response_size_bytes = Histogram(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint', 'status'],
    buckets=[100, 1000, 10000, 100000, 1000000, 10000000],
    registry=registry
)


# 2. TRAFFIC - Request Rate
# ============================================================================

http_requests_total = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

http_requests_in_progress = Gauge(
    'http_requests_in_progress',
    'Number of HTTP requests currently being processed',
    ['method', 'endpoint'],
    registry=registry
)

api_requests_per_second = Gauge(
    'api_requests_per_second',
    'Current API requests per second',
    registry=registry
)


# 3. ERRORS - Error Rate
# ============================================================================

http_request_exceptions_total = Counter(
    'http_request_exceptions_total',
    'Total number of exceptions during request processing',
    ['method', 'endpoint', 'exception_type'],
    registry=registry
)

http_client_errors_total = Counter(
    'http_client_errors_total',
    'Total number of 4xx client errors',
    ['method', 'endpoint', 'status'],
    registry=registry
)

http_server_errors_total = Counter(
    'http_server_errors_total',
    'Total number of 5xx server errors',
    ['method', 'endpoint', 'status'],
    registry=registry
)

error_rate_percentage = Gauge(
    'error_rate_percentage',
    'Current error rate as percentage',
    ['time_window'],
    registry=registry
)


# 4. SATURATION - Resource Usage
# ============================================================================

system_cpu_usage_percent = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage',
    registry=registry
)

system_cpu_count = Gauge(
    'system_cpu_count',
    'Number of CPU cores',
    registry=registry
)

system_memory_usage_bytes = Gauge(
    'system_memory_usage_bytes',
    'System memory usage in bytes',
    registry=registry
)

system_memory_total_bytes = Gauge(
    'system_memory_total_bytes',
    'Total system memory in bytes',
    registry=registry
)

system_disk_usage_bytes = Gauge(
    'system_disk_usage_bytes',
    'Disk usage in bytes',
    ['mountpoint'],
    registry=registry
)

system_disk_total_bytes = Gauge(
    'system_disk_total_bytes',
    'Total disk space in bytes',
    ['mountpoint'],
    registry=registry
)

system_network_io_bytes_total = Counter(
    'system_network_io_bytes_total',
    'Total network I/O in bytes',
    ['direction'],  # sent, received
    registry=registry
)

database_connections_active = Gauge(
    'database_connections_active',
    'Number of active database connections',
    registry=registry
)

database_connections_total = Gauge(
    'database_connections_total',
    'Total number of database connections',
    registry=registry
)

redis_connections_active = Gauge(
    'redis_connections_active',
    'Number of active Redis connections',
    registry=registry
)

# Prediction Metrics
predictions_total = Counter(
    'predictions_total',
    'Total number of predictions made',
    ['model_id', 'model_type'],
    registry=registry
)

prediction_duration_seconds = Histogram(
    'prediction_duration_seconds',
    'Prediction duration in seconds',
    ['model_id'],
    registry=registry
)

# Model Metrics
active_models = Gauge(
    'active_models',
    'Number of active models',
    ['stage'],
    registry=registry
)

model_accuracy = Gauge(
    'model_accuracy',
    'Model accuracy score',
    ['model_id', 'model_name'],
    registry=registry
)

# System Metrics
system_cpu_usage = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage',
    registry=registry
)

system_memory_usage = Gauge(
    'system_memory_usage_percent',
    'System memory usage percentage',
    registry=registry
)

system_disk_usage = Gauge(
    'system_disk_usage_percent',
    'System disk usage percentage',
    registry=registry
)

# Cache Metrics
cache_hits = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type'],
    registry=registry
)

cache_misses = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type'],
    registry=registry
)


class MetricsCollector:
    """Collects and manages application metrics."""

    def __init__(self):
        self.start_time = time.time()

    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics."""
        http_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
        http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)

    def record_prediction(
        self,
        model_id: str,
        model_type: str,
        duration: float
    ):
        """Record prediction metrics."""
        predictions_total.labels(model_id=model_id, model_type=model_type).inc()
        prediction_duration_seconds.labels(model_id=model_id).observe(duration)

    def update_model_metrics(self, models: list):
        """Update model-related metrics."""
        # Count models by stage
        stage_counts = {}
        for model in models:
            stage = getattr(model, 'stage', 'unknown')
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

        # Update gauges
        for stage, count in stage_counts.items():
            active_models.labels(stage=stage).set(count)

        # Update model accuracy
        for model in models:
            if hasattr(model, 'metrics') and 'accuracy' in model.metrics:
                model_accuracy.labels(
                    model_id=model.model_id,
                    model_name=model.model_name
                ).set(model.metrics['accuracy'])

    def update_system_metrics(self):
        """Update system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            system_cpu_usage.set(cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            system_memory_usage.set(memory.percent)

            # Disk usage
            disk = psutil.disk_usage('/')
            system_disk_usage.set(disk.percent)

        except Exception as e:
            # Log error but don't fail
            print(f"Error updating system metrics: {e}")

    def record_cache_hit(self, cache_type: str = "default"):
        """Record cache hit."""
        cache_hits.labels(cache_type=cache_type).inc()

    def record_cache_miss(self, cache_type: str = "default"):
        """Record cache miss."""
        cache_misses.labels(cache_type=cache_type).inc()

    def get_uptime_seconds(self) -> float:
        """Get application uptime in seconds."""
        return time.time() - self.start_time


# Global metrics collector
metrics_collector = MetricsCollector()


def get_prometheus_metrics() -> Response:
    """
    Generate Prometheus metrics endpoint response.

    Returns:
        Response with Prometheus metrics in text format
    """
    # Update system metrics before generating output
    metrics_collector.update_system_metrics()

    # Generate metrics
    metrics_output = generate_latest(registry)

    return Response(
        content=metrics_output,
        media_type="text/plain; charset=utf-8"
    )


def get_health_metrics() -> Dict:
    """
    Get health metrics in JSON format.

    Returns:
        Dictionary with health metrics
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        return {
            "uptime_seconds": metrics_collector.get_uptime_seconds(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            },
            "process": {
                "pid": os.getpid(),
                "cpu_percent": psutil.Process().cpu_percent(),
                "memory_mb": psutil.Process().memory_info().rss / (1024**2)
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "uptime_seconds": metrics_collector.get_uptime_seconds()
        }
