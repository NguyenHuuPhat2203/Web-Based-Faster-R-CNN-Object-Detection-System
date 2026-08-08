"""Domain-level exceptions mapped to HTTP responses in the API layer."""

from __future__ import annotations


class DomainError(Exception):
    """Base class for domain errors raised by services."""

    def __init__(self, message: str) -> None:
        """Store the human-readable error message."""
        self.message = message
        super().__init__(message)


class UnauthorizedError(DomainError):
    """Raised when authentication or authorization fails."""


class NotFoundError(DomainError):
    """Raised when a requested resource does not exist."""


class ForbiddenError(DomainError):
    """Raised when a user lacks permission for a resource."""


class ConflictError(DomainError):
    """Raised when a resource conflicts with existing state."""


class ValidationError(DomainError):
    """Raised when input validation fails at the service layer."""


class PayloadTooLargeError(ValidationError):
    """Raised when an uploaded payload exceeds configured limits."""


class TokenError(UnauthorizedError):
    """Raised when a JWT is invalid or expired."""


class OAuthError(UnauthorizedError):
    """Raised when Google OAuth exchange fails."""
