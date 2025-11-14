"""Input validation and sanitization."""

import re
import html
from typing import Any, Dict
import bleach


class InputValidator:
    """Validate and sanitize user inputs."""

    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            value = str(value)

        # Remove HTML tags and escape special characters
        value = bleach.clean(value, tags=[], strip=True)
        value = html.escape(value)

        # Limit length
        return value[:max_length]

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    @staticmethod
    def sanitize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize dictionary."""
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = InputValidator.sanitize_string(value)
            elif isinstance(value, dict):
                sanitized[key] = InputValidator.sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    InputValidator.sanitize_string(v) if isinstance(v, str) else v
                    for v in value
                ]
            else:
                sanitized[key] = value
        return sanitized


def sanitize_input(data: Any) -> Any:
    """Sanitize input data."""
    if isinstance(data, str):
        return InputValidator.sanitize_string(data)
    elif isinstance(data, dict):
        return InputValidator.sanitize_dict(data)
    return data
