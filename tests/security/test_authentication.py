"""Security tests for authentication system."""

import pytest
from fastapi.testclient import TestClient
from source.security.auth import JWTHandler, OAuth2Handler, User


class TestJWTAuthentication:
    """Test JWT authentication."""

    def test_create_access_token(self):
        """Test access token creation."""
        handler = JWTHandler()
        token_data = {"user_id": "123", "username": "testuser"}
        token = handler.create_access_token(token_data)

        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_valid_token(self):
        """Test token verification with valid token."""
        handler = JWTHandler()
        token_data = {
            "user_id": "123",
            "username": "testuser",
            "email": "test@example.com",
            "roles": ["user"],
            "permissions": ["read"]
        }
        token = handler.create_access_token(token_data)
        decoded = handler.verify_token(token)

        assert decoded.user_id == "123"
        assert decoded.username == "testuser"
        assert "user" in decoded.roles

    def test_verify_expired_token(self):
        """Test token verification with expired token."""
        from datetime import timedelta
        handler = JWTHandler()
        token_data = {"user_id": "123", "username": "testuser"}
        token = handler.create_access_token(
            token_data,
            expires_delta=timedelta(seconds=-1)
        )

        with pytest.raises(Exception):
            handler.verify_token(token)

    def test_refresh_token(self):
        """Test token refresh."""
        handler = JWTHandler()
        token_data = {
            "user_id": "123",
            "username": "testuser",
            "email": "test@example.com",
            "roles": ["user"]
        }
        refresh_token = handler.create_refresh_token(token_data)
        new_tokens = handler.refresh_access_token(refresh_token)

        assert new_tokens.access_token
        assert new_tokens.refresh_token
        assert new_tokens.token_type == "bearer"


class TestPasswordHashing:
    """Test password hashing."""

    def test_hash_password(self):
        """Test password hashing."""
        oauth_handler = OAuth2Handler()
        password = "testpassword123"
        hashed = oauth_handler.get_password_hash(password)

        assert hashed != password
        assert len(hashed) > 0

    def test_verify_correct_password(self):
        """Test password verification with correct password."""
        oauth_handler = OAuth2Handler()
        password = "testpassword123"
        hashed = oauth_handler.get_password_hash(password)

        assert oauth_handler.verify_password(password, hashed)

    def test_verify_incorrect_password(self):
        """Test password verification with incorrect password."""
        oauth_handler = OAuth2Handler()
        password = "testpassword123"
        hashed = oauth_handler.get_password_hash(password)

        assert not oauth_handler.verify_password("wrongpassword", hashed)
