"""Distributed tracing support."""

import time
import uuid
from functools import wraps
from typing import Callable
from fastapi import Request
from source.utils.logger import setup_logger

logger = setup_logger(name="tracing", log_file="../logs/tracing.log")


class TracingMiddleware:
    """Middleware for distributed tracing."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Generate trace ID
        trace_id = str(uuid.uuid4())
        scope["trace_id"] = trace_id

        # Add trace ID to headers
        async def send_with_trace(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-trace-id", trace_id.encode()))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_trace)


def trace_function(func: Callable) -> Callable:
    """Decorator to trace function execution."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        trace_id = str(uuid.uuid4())
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            logger.info(f"TRACE {trace_id}: {func.__name__} completed in {time.time() - start:.3f}s")
            return result
        except Exception as e:
            logger.error(f"TRACE {trace_id}: {func.__name__} failed: {e}")
            raise

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        trace_id = str(uuid.uuid4())
        start = time.time()
        try:
            result = func(*args, **kwargs)
            logger.info(f"TRACE {trace_id}: {func.__name__} completed in {time.time() - start:.3f}s")
            return result
        except Exception as e:
            logger.error(f"TRACE {trace_id}: {func.__name__} failed: {e}")
            raise

    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper
