"""
AEGIS Breach Detection Router
==============================
Endpoints:
  POST /breach/check/password  - Check if password was exposed in data breaches
  POST /breach/check/email     - Check if email was exposed in data breaches
  GET  /breach/health          - Module health check
"""

import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.breach.hibp_checker import score_password_breach, check_email_breach, check_password_hash
from src.utils.aws_utils import dynamo_put_event, cw_put_detection_metrics

router = APIRouter()


class PasswordBreachRequest(BaseModel):
    password:   str = Field(..., min_length=1, description="Password to check", example="hunter2")
    user_id:    Optional[str] = Field(None, description="User ID for event logging")

    class Config:
        json_schema_extra = {
            "example": {
                "password": "hunter2",
                "user_id":  "user_12345",
            }
        }


class EmailBreachRequest(BaseModel):
    email:   str = Field(..., description="Email address to check", example="user@example.com")
    user_id: Optional[str] = Field(None, description="User ID for event logging")


@router.post("/check/password", summary="Check password against known data breaches")
async def check_password_breach_endpoint(body: PasswordBreachRequest, request: Request):
    """
    Uses HIBP k-anonymity API to check if a password has appeared in data breaches.
    Only the first 5 characters of the SHA-1 hash are sent to HIBP — the full
    password never leaves your server.

    Returns breach probability, strength score, entropy, and recommendations.
    """
    start  = time.perf_counter()
    result = score_password_breach(body.password)
    elapsed_ms = (time.perf_counter() - start) * 1000

    cw_put_detection_metrics(
        module     = "breach",
        risk_score = result["breach_probability"],
        latency_ms = elapsed_ms,
        confidence = 0.9 if result["api_available"] else 0.6,
    )

    # NOTE: persistence happens centrally in broadcast_detection_event so the
    # event is tenant-scoped and folded into the user's profile. Writing here
    # too would create a duplicate, un-namespaced row.
    try:
        broadcast = getattr(request.app.state, "broadcast", None)
        if broadcast:
            uid = body.user_id or "anonymous"
            # Send the full assessment so the profile can retain the detail
            # (strength, length, character classes, advice) for the dashboard.
            await broadcast("breach_score", uid, {
                "breach_probability": result["breach_probability"],
                "risk_level":        result["risk_level"],
                "is_pwned":          result["is_pwned"],
                "pwned_count":       result.get("pwned_count", 0),
                "strength_score":    result.get("strength_score", 0.0),
                "entropy_bits":      result.get("entropy_bits", 0.0),
                "length":            result.get("length", 0),
                "char_types":        result.get("char_types", 0),
                "recommendations":   result.get("recommendations", []),
                "user_id":           body.user_id,
            })
    except Exception:
        pass

    return {
        **result,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "latency_ms": round(elapsed_ms, 2),
    }


@router.post("/check/email", summary="Check email against known data breaches")
async def check_email_breach_endpoint(body: EmailBreachRequest):
    """
    Uses HIBP API to check if an email has been exposed in known data breaches.
    Requires HIBP_API_KEY to be set in .env.
    """
    result = check_email_breach(body.email)

    if body.user_id:
        dynamo_put_event(
            user_id    = body.user_id,
            event_type = "email_breach_check",
            scores     = {
                "breach_count": result["breach_count"],
                "email":        body.email,
            },
        )

    return {**result, "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/health", summary="Breach module health check")
async def breach_health():
    """Check if the HIBP API is reachable."""
    api_reachable = check_password_hash("AAAAA")
    return {
        "module":        "breach",
        "status":        "healthy" if api_reachable else "degraded",
        "hibp_api":      "reachable" if api_reachable else "unreachable",
        "api_key_set":   bool(__import__("os").getenv("HIBP_API_KEY")),
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }
