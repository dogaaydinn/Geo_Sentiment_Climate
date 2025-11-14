"""Security tests for input validation."""

import pytest
from source.security.validation import InputValidator, sanitize_input


class TestInputValidation:
    """Test input validation and sanitization."""

    def test_sanitize_string_removes_html(self):
        """Test HTML removal from strings."""
        malicious_input = "<script>alert('XSS')</script>Hello"
        sanitized = InputValidator.sanitize_string(malicious_input)

        assert "<script>" not in sanitized
        assert "alert" not in sanitized

    def test_sanitize_string_limits_length(self):
        """Test string length limiting."""
        long_string = "a" * 2000
        sanitized = InputValidator.sanitize_string(long_string, max_length=100)

        assert len(sanitized) == 100

    def test_validate_email_correct_format(self):
        """Test email validation with correct format."""
        assert InputValidator.validate_email("test@example.com")
        assert InputValidator.validate_email("user.name@domain.co.uk")

    def test_validate_email_incorrect_format(self):
        """Test email validation with incorrect format."""
        assert not InputValidator.validate_email("invalid-email")
        assert not InputValidator.validate_email("@example.com")
        assert not InputValidator.validate_email("test@")

    def test_sanitize_dict(self):
        """Test dictionary sanitization."""
        malicious_dict = {
            "name": "<script>alert('XSS')</script>",
            "email": "test@example.com",
            "nested": {
                "value": "<b>bold</b>"
            }
        }

        sanitized = InputValidator.sanitize_dict(malicious_dict)

        assert "<script>" not in sanitized["name"]
        assert "<b>" not in sanitized["nested"]["value"]
        assert sanitized["email"] == "test@example.com"

    def test_sql_injection_prevention(self):
        """Test SQL injection pattern detection."""
        sql_injection = "'; DROP TABLE users; --"
        sanitized = InputValidator.sanitize_string(sql_injection)

        # Should escape special characters
        assert sanitized != sql_injection
