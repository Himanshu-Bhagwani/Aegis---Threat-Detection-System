"""
AEGIS Device Fingerprint Router
=================================
Endpoints:
  POST /device/score    - Score device risk (fingerprint + behavioral signals)
  POST /device/register - Register a known-good device for a user
  GET  /device/health   - Module health check
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.device.device_risk_model import score_device_risk, generate_device_id
from src.utils.aws_utils import (
    dynamo_put_event,
    dynamo_get_user_profile,
    dynamo_update_user_profile,
    cw_put_detection_metrics,
)

router = APIRouter()


class DeviceFingerprintInput(BaseModel):
    """Device fingerprint fields collected client-side."""
    user_agent:           Optional[str] = Field(None, example="Mozilla/5.0 ...")
    platform:             Optional[str] = Field(None, example="Win32")
    screen_resolution:    Optional[str] = Field(None, example="1920x1080")
    timezone:             Optional[str] = Field(None, example="America/New_York")
    language:             Optional[str] = Field(None, example="en-US")
    hardware_concurrency: Optional[int] = Field(None, example=8)
    color_depth:          Optional[int] = Field(None, example=24)
    touch_support:        Optional[bool] = Field(None, example=False)
    webgl_renderer:       Optional[str] = Field(None, example="NVIDIA GeForce RTX 3080")

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


class DeviceScoreRequest(BaseModel):
    user_id:                 str = Field(..., description="User identifier")
    fingerprint:             DeviceFingerprintInput = Field(...)
    failed_biometric:        bool = Field(False)
    failed_biometric_count:  int  = Field(0, ge=0)
    app_unlock_attempted:    bool = Field(False)
    app_bundle:              Optional[str] = Field(None, example="com.bank.app")
    unusual_time:            bool = Field(False)
    location_mismatch:       bool = Field(False)
    rooted_jailbroken:       bool = Field(False)
    emulator_detected:       bool = Field(False)
    vpn_detected:            bool = Field(False)
    multiple_failures_today: int  = Field(0, ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_12345",
                "fingerprint": {
                    "user_agent":        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                    "platform":          "Win32",
                    "screen_resolution": "1920x1080",
                    "timezone":          "America/New_York",
                },
                "failed_biometric":       False,
                "failed_biometric_count": 0,
                "app_unlock_attempted":   False,
            }
        }


class DeviceRegisterRequest(BaseModel):
    user_id:     str = Field(..., description="User identifier")
    fingerprint: DeviceFingerprintInput = Field(...)
    label:       Optional[str] = Field(None, example="Work laptop")


@router.post("/score", summary="Score device fingerprint risk")
async def score_device(body: DeviceScoreRequest, request: Request):
    """
    Score the risk of a device interaction.

    Checks:
    - Whether the device fingerprint matches known devices for this user
    - Biometric failure patterns
    - App-lock access attempts
    - Contextual signals (unusual time, location, rooted device, emulator)

    Integrates with DynamoDB to fetch the user's known device list.
    Persists the event back to DynamoDB for history/baseline building.
    """
    start = time.perf_counter()

    # Fetch known devices from user profile
    profile           = dynamo_get_user_profile(body.user_id) or {}
    known_fingerprints = profile.get("known_devices", [])

    # Score
    fp_dict = body.fingerprint.to_dict()
    result  = score_device_risk(
        fingerprint             = fp_dict,
        known_fingerprints      = known_fingerprints,
        failed_biometric        = body.failed_biometric,
        failed_biometric_count  = body.failed_biometric_count,
        app_unlock_attempted    = body.app_unlock_attempted,
        app_bundle              = body.app_bundle,
        unusual_time            = body.unusual_time,
        location_mismatch       = body.location_mismatch,
        rooted_jailbroken       = body.rooted_jailbroken,
        emulator_detected       = body.emulator_detected,
        vpn_detected            = body.vpn_detected,
        multiple_failures_today = body.multiple_failures_today,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    # CloudWatch metrics
    cw_put_detection_metrics(
        module     = "device",
        risk_score = result["device_risk_score"],
        latency_ms = elapsed_ms,
        confidence = 0.85,
    )

    # Persist event
    dynamo_put_event(
        user_id    = body.user_id,
        event_type = "device_score",
        scores     = {
            "device_risk_score": result["device_risk_score"],
            "is_new_device":     result["is_new_device"],
            "risk_level":        result["risk_level"],
            "signals":           result["signals"],
        },
        raw_input  = {
            "device_id":              result["device_id"],
            "app_bundle":             body.app_bundle,
            "failed_biometric_count": body.failed_biometric_count,
            "app_unlock_attempted":   body.app_unlock_attempted,
        },
    )

    try:
        broadcast = getattr(request.app.state, "broadcast", None)
        if broadcast:
            await broadcast("device_score", body.user_id, {
                "device_risk_score": result["device_risk_score"],
                "risk_level":        result["risk_level"],
                "is_new_device":     result["is_new_device"],
                "user_id":           body.user_id,
            })
    except Exception:
        pass

    return {
        **result,
        "user_id":    body.user_id,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "latency_ms": round(elapsed_ms, 2),
    }


@router.post("/register", summary="Register a known-good device for a user")
async def register_device(body: DeviceRegisterRequest):
    """
    Register a new trusted device for a user.
    Adds the fingerprint to the user's known_devices list in DynamoDB.
    """
    fp_dict   = body.fingerprint.to_dict()
    device_id = generate_device_id(fp_dict)

    # Fetch existing profile
    profile = dynamo_get_user_profile(body.user_id) or {"user_id": body.user_id}
    known   = profile.get("known_devices", [])

    # Check if already registered (by device_id)
    existing_ids = [generate_device_id(kfp) for kfp in known]
    if device_id in existing_ids:
        return {
            "success":   False,
            "message":   "Device already registered",
            "device_id": device_id,
        }

    # Add new device entry
    entry = {**fp_dict, "_device_id": device_id, "_label": body.label or ""}
    known.append(entry)
    profile["known_devices"] = known[-20:]  # keep last 20 devices

    ok = dynamo_update_user_profile(body.user_id, profile)

    return {
        "success":        ok,
        "device_id":      device_id,
        "label":          body.label,
        "devices_stored": len(known),
        "timestamp":      datetime.now(timezone.utc).isoformat(),
    }


@router.get("/health", summary="Device module health check")
async def device_health():
    return {
        "module":    "device",
        "status":    "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
