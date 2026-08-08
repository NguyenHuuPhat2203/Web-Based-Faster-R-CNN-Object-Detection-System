"""Unit tests for authentication service logic."""

from backend.core.exceptions import ConflictError, UnauthorizedError
from backend.services.auth_service import AuthService


class TestAuthService:
    """Auth service unit tests."""

    def test_register_duplicate_email(self, db_session) -> None:
        """Raise when registering a duplicate email."""
        service = AuthService(db_session)
        service.register("dup@example.com", "UserA", "secret123")
        try:
            service.register("dup@example.com", "UserB", "secret123")
            raise AssertionError("Expected ConflictError")
        except ConflictError:
            pass

    def test_authenticate_wrong_password(self, db_session) -> None:
        """Raise when the password is invalid."""
        service = AuthService(db_session)
        service.register("auth@example.com", "User", "secret123")
        try:
            service.authenticate("auth@example.com", "wrong")
            raise AssertionError("Expected UnauthorizedError")
        except UnauthorizedError:
            pass
