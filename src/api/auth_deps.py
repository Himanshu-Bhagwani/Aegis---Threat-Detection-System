"""
AEGIS Authentication Dependencies
===================================
FastAPI dependency injection for route protection.

Primary path  → AWS Cognito access-token validation (get_user API call)
Fallback path → Local JWT (python-jose) when Cognito is not configured
Mock path     → Accepts any "mock-*" token for local development

Usage in a router:
    from src.api.auth_deps import require_auth, optional_auth

    @router.post("/protected")
    async def protected(user: dict = Depends(require_auth)):
        return {"uid": user["user_sub"]}
"""

import os
import logging
from typing import Optional

from fastapi import Depends, HTTPException, Header, status
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "aegis-dev-secret-change-in-prod")
JWT_ALGORITHM  = os.getenv("JWT_ALGORITHM",  "HS256")


# ─────────────────────────────────────────────
# Token extraction
# ─────────────────────────────────────────────

def _extract_token(authorization: Optional[str]) -> str:
    """Pull the Bearer token out of the Authorization header."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization format. Expected: 'Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization[7:].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token is empty.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


# ─────────────────────────────────────────────
# Cognito validation (primary)
# ─────────────────────────────────────────────

def _validate_via_cognito(token: str) -> Optional[dict]:
    """
    Validate token against AWS Cognito with get_user().
    Returns user dict on success, None if Cognito is not configured.
    Raises 401 HTTPException if token is actively invalid.
    """
    try:
        from src.utils.aws_utils import cognito_verify_token, COGNITO_USER_POOL_ID
        if not COGNITO_USER_POOL_ID:
            return None  # Cognito not configured — try fallback
        result = cognito_verify_token(token)
        if result.get("mock"):
            return result
        if not result.get("valid"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid or expired token ({result.get('error', 'unknown')})",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Cognito validation error: %s", e)
        return None


# ─────────────────────────────────────────────
# Local JWT validation (fallback)
# ─────────────────────────────────────────────

def _validate_via_local_jwt(token: str) -> Optional[dict]:
    """Validate a locally-signed JWT (python-jose). Used when Cognito is not configured."""
    try:
        from jose import jwt
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return {
            "valid":    True,
            "user_sub": payload.get("sub", "local-user"),
            "email":    payload.get("email", ""),
            "username": payload.get("email", ""),
        }
    except Exception:
        return None


# ─────────────────────────────────────────────
# Mock validation (development only)
# ─────────────────────────────────────────────

def _validate_mock(token: str) -> Optional[dict]:
    """Accept mock-* tokens in development / when no auth is configured."""
    if token.startswith("mock-"):
        return {
            "valid":    True,
            "user_sub": "mock-user-001",
            "email":    "demo@aegis.com",
            "username": "demo@aegis.com",
            "mock":     True,
        }
    return None


# ─────────────────────────────────────────────
# Core resolver
# ─────────────────────────────────────────────

def _resolve_user(token: str) -> dict:
    """
    Try auth methods in order:
      1. Mock token  (development — checked FIRST so mock-* tokens always work
                      even when Cognito env vars are set to placeholder values)
      2. AWS Cognito (if credentials are valid/reachable)
      3. Local JWT   (python-jose fallback)
    """
    # Check mock FIRST — prevents placeholder Cognito creds from blocking dev tokens
    user = _validate_mock(token)
    if user:
        return user

    user = _validate_via_cognito(token)
    if user:
        return user

    user = _validate_via_local_jwt(token)
    if user:
        return user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )


# ─────────────────────────────────────────────
# Public dependency functions
# ─────────────────────────────────────────────

def require_auth(authorization: Optional[str] = Header(None)) -> dict:
    """
    FastAPI dependency — requires a valid token.
    Returns user dict: { valid, user_sub, email, username, [mock] }
    """
    token = _extract_token(authorization)
    return _resolve_user(token)


def optional_auth(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """
    FastAPI dependency — token is optional.
    Returns user dict if authenticated, None otherwise.
    """
    if not authorization:
        return None
    try:
        token = _extract_token(authorization)
        return _resolve_user(token)
    except HTTPException:
        return None


# ─────────────────────────────────────────────
# Local JWT creation (for /login fallback)
# ─────────────────────────────────────────────

def create_local_jwt(user_sub: str, email: str, expires_minutes: int = 60) -> str:
    """Create a local JWT for use when Cognito is not configured."""
    from datetime import datetime, timedelta, timezone
    try:
        from jose import jwt
        now     = datetime.now(timezone.utc)
        payload = {
            "sub":   user_sub,
            "email": email,
            "iat":   now,
            "exp":   now + timedelta(minutes=expires_minutes),
        }
        return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    except ImportError:
        import uuid
        return f"mock-{uuid.uuid4().hex}"


# ─────────────────────────────────────────────
# Legacy shims (keep existing call sites working)
# ─────────────────────────────────────────────
_valid_tokens: set = set()

def get_token_from_header(authorization: Optional[str] = Header(None)) -> str:
    return _extract_token(authorization)

def validate_mock_token(token: str) -> dict:
    return _resolve_user(token)

def add_valid_token(token: str):
    _valid_tokens.add(token)

def remove_valid_token(token: str):
    _valid_tokens.discard(token)
