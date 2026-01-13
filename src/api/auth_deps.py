"""
Authentication dependencies for protecting API routes.

This module provides FastAPI dependencies for route protection.
Currently uses mock token validation (for demo purposes).
For production, replace with proper JWT/Firebase token validation.
"""

from fastapi import Depends, HTTPException, Header
from typing import Optional

# Store valid tokens (in production, use proper token validation)
_valid_tokens: set[str] = set()


def get_token_from_header(authorization: Optional[str] = Header(None)) -> str:
    """
    Extract token from Authorization header.
    
    Expected format: "Bearer <token>"
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing. Please login first."
        )
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization format. Expected 'Bearer <token>'"
        )
    
    token = authorization.replace("Bearer ", "").strip()
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Token is empty"
        )
    
    return token


def validate_mock_token(token: str) -> dict:
    """
    Validate mock token (for demo purposes).
    
    In production, replace this with proper JWT validation
    or Firebase token verification.
    """
    # For mock auth, accept tokens that start with "mock-token-"
    if token.startswith("mock-token-"):
        # Extract user info from token or return default
        return {
            "uid": "demo-user",
            "email": "demo@aegis.com",
            "token": token
        }
    
    # If you want stricter validation, check against _valid_tokens
    # if token not in _valid_tokens:
    #     raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    raise HTTPException(
        status_code=401,
        detail="Invalid authentication token. Please login again."
    )


def require_auth(authorization: Optional[str] = Header(None)) -> dict:
    """
    FastAPI dependency for protecting routes.
    
    Usage:
        @router.post("/protected")
        async def protected_route(user: dict = Depends(require_auth)):
            return {"message": f"Hello {user['email']}"}
    """
    token = get_token_from_header(authorization)
    user = validate_mock_token(token)
    return user


def optional_auth(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """
    FastAPI dependency for optional authentication.
    
    Returns user dict if authenticated, None otherwise.
    Useful for endpoints that have different behavior for authenticated vs anonymous users.
    
    Usage:
        @router.get("/public")
        async def public_route(user: Optional[dict] = Depends(optional_auth)):
            if user:
                return {"message": f"Hello {user['email']}"}
            return {"message": "Hello anonymous user"}
    """
    try:
        token = get_token_from_header(authorization)
        return validate_mock_token(token)
    except HTTPException:
        return None


def add_valid_token(token: str):
    """Add a token to the valid tokens set (for testing/strict validation)"""
    _valid_tokens.add(token)


def remove_valid_token(token: str):
    """Remove a token from valid tokens (logout)"""
    _valid_tokens.discard(token)

