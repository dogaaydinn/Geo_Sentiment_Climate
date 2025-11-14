"""
Geo Climate Python SDK.

Official Python client for the Geo Sentiment Climate API.
Part of Phase 4: Advanced Features - API Marketplace.

Example usage:
    >>> from geo_climate_sdk import GeoClimateClient
    >>> client = GeoClimateClient(api_key="your-api-key")
    >>> prediction = client.predict(data={"pm25": 35.5, "temp": 72, ...})
    >>> print(prediction)
"""

__version__ = "1.0.0"
__author__ = "Doğa Aydın"
__email__ = "dogaa882@gmail.com"

from .client import GeoClimateClient
from .exceptions import (
    GeoClimateError,
    AuthenticationError,
    RateLimitError,
    ValidationError
)

__all__ = [
    "GeoClimateClient",
    "GeoClimateError",
    "AuthenticationError",
    "RateLimitError",
    "ValidationError"
]
