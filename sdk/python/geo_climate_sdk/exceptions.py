"""
Custom exceptions for Geo Climate SDK.
"""


class GeoClimateError(Exception):
    """Base exception for Geo Climate SDK."""
    pass


class AuthenticationError(GeoClimateError):
    """Raised when authentication fails."""
    pass


class RateLimitError(GeoClimateError):
    """Raised when rate limit is exceeded."""
    pass


class ValidationError(GeoClimateError):
    """Raised when request validation fails."""
    pass


class ModelNotFoundError(GeoClimateError):
    """Raised when model is not found."""
    pass


class PredictionError(GeoClimateError):
    """Raised when prediction fails."""
    pass
