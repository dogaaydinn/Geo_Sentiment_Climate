"""Security tests for rate limiting."""

import pytest
import time
from source.security.rate_limiter import RateLimiter, RateLimitExceeded


class TestRateLimiter:
    """Test rate limiting functionality."""

    def test_rate_limiter_allows_requests_within_limit(self):
        """Test that requests within limit are allowed."""
        limiter = RateLimiter(default_limit=5, default_window=60)

        class FakeRequest:
            def __init__(self):
                self.url = type('obj', (object,), {'path': '/test'})()
                self.client = type('obj', (object,), {'host': '127.0.0.1'})()
                self.headers = {}

        request = FakeRequest()

        # Should allow first 5 requests
        for i in range(5):
            allowed, info = limiter.check_rate_limit(request, limit=5, window=60)
            assert allowed
            assert info['remaining'] >= 0

    def test_rate_limiter_blocks_excess_requests(self):
        """Test that requests exceeding limit are blocked."""
        limiter = RateLimiter(default_limit=3, default_window=60)

        class FakeRequest:
            def __init__(self):
                self.url = type('obj', (object,), {'path': '/test'})()
                self.client = type('obj', (object,), {'host': '127.0.0.1'})()
                self.headers = {}

        request = FakeRequest()

        # First 3 requests should pass
        for i in range(3):
            allowed, _ = limiter.check_rate_limit(request, limit=3, window=60)
            assert allowed

        # 4th request should fail
        allowed, info = limiter.check_rate_limit(request, limit=3, window=60)
        assert not allowed
        assert info['retry_after'] > 0

    def test_sliding_window(self):
        """Test sliding window algorithm."""
        limiter = RateLimiter(default_limit=2, default_window=2)

        class FakeRequest:
            def __init__(self):
                self.url = type('obj', (object,), {'path': '/test'})()
                self.client = type('obj', (object,), {'host': '127.0.0.1'})()
                self.headers = {}

        request = FakeRequest()

        # Use 2 requests
        limiter.check_rate_limit(request, limit=2, window=2)
        limiter.check_rate_limit(request, limit=2, window=2)

        # Wait for window to partially expire
        time.sleep(2.1)

        # Should be able to make requests again
        allowed, _ = limiter.check_rate_limit(request, limit=2, window=2)
        assert allowed
