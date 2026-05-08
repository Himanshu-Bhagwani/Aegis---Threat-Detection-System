"""
GPS Spoofing Detection API Router

Endpoints:
  POST /gps/score        - Score a full GPS trajectory
  POST /gps/score/point  - Score a single GPS point (limited accuracy)
  GET  /gps/health       - Module health check
"""

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.utils.aws_utils import dynamo_put_event, cw_put_detection_metrics

router = APIRouter()


# ── Pydantic models ───────────────────────────────────────────────────────────

class GPSPoint(BaseModel):
    latitude:      float           = Field(..., ge=-90, le=90)
    longitude:     float           = Field(..., ge=-180, le=180)
    speed:         Optional[float] = Field(0.0, ge=0)
    acceleration:  Optional[float] = Field(0.0)
    heading:       Optional[float] = Field(0.0, ge=0, le=360)
    heading_change: Optional[float] = Field(0.0)
    timestamp:     Optional[float] = Field(None, description="Unix timestamp (seconds)")
    time_delta:    Optional[float] = Field(0.0, ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "latitude": 37.7749, "longitude": -122.4194,
                "speed": 15.5, "heading": 45.0,
                "timestamp": 1_700_000_000.0,
            }
        }


class GPSTrajectoryRequest(BaseModel):
    trajectory: List[GPSPoint] = Field(..., min_length=1)
    user_id:    Optional[str]  = Field(None)
    device_id:  Optional[str]  = Field(None)
    ensemble:   bool           = Field(True)

    class Config:
        json_schema_extra = {
            "example": {
                "trajectory": [
                    {"latitude": 37.7749, "longitude": -122.4194,
                     "timestamp": 1_700_000_000.0, "speed": 0},
                    {"latitude": 37.7750, "longitude": -122.4195,
                     "timestamp": 1_700_000_010.0, "speed": 1.2},
                    {"latitude": 37.7751, "longitude": -122.4196,
                     "timestamp": 1_700_000_020.0, "speed": 1.8},
                ],
                "user_id": "user_12345",
                "ensemble": True,
            }
        }


class SinglePointRequest(BaseModel):
    latitude:      float          = Field(..., ge=-90,  le=90)
    longitude:     float          = Field(..., ge=-180, le=180)
    speed:         Optional[float] = Field(0.0, ge=0)
    heading:       Optional[float] = Field(0.0, ge=0, le=360)
    prev_latitude: Optional[float] = Field(None, ge=-90,  le=90)
    prev_longitude: Optional[float] = Field(None, ge=-180, le=180)


class GPSSpoofResponse(BaseModel):
    spoof_probability: float            = Field(..., ge=0, le=1)
    risk_score:        float            = Field(..., ge=0, le=1,
                                              description="Alias for spoof_probability")
    is_spoofed:        bool
    confidence:        float            = Field(..., ge=0, le=1)
    model_scores:      Dict[str, float]
    models_used:       List[str]
    risk_factors:      List[str]        = Field(default_factory=list,
                                              description="Triggered rule descriptions")
    user_id:           Optional[str]    = None
    device_id:         Optional[str]    = None
    timestamp:         Optional[str]    = None
    latency_ms:        Optional[float]  = None


# ── Lazy imports ──────────────────────────────────────────────────────────────

_scorer = None

def _get_scorer():
    global _scorer
    if _scorer is None:
        from src.gps.score_gps import score_gps_trajectory, score_single_point
        _scorer = {"trajectory": score_gps_trajectory, "single": score_single_point}
    return _scorer


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/score", response_model=GPSSpoofResponse, summary="Score GPS trajectory for spoofing")
async def score_trajectory(body: GPSTrajectoryRequest, request: Request):
    """
    Analyse a GPS trajectory for spoofing using an ensemble of:
    - Rule-based physics checks (always active)
    - Isolation Forest
    - Gradient Boosting (GBM)
    - Autoencoder reconstruction error
    - 1D-CNN + BiLSTM

    Returns a spoof probability [0-1] and which rules / models triggered.
    """
    start = time.perf_counter()
    try:
        scorer = _get_scorer()
        trajectory_dicts = [p.model_dump() for p in body.trajectory]
        result = scorer["trajectory"](trajectory_dicts, ensemble=body.ensemble)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {e}")

    elapsed_ms = (time.perf_counter() - start) * 1_000
    uid = body.user_id or "anonymous"

    # ── Persist + metrics ─────────────────────────────────
    try:
        cw_put_detection_metrics(
            module     = "gps",
            risk_score = result["spoof_probability"],
            latency_ms = elapsed_ms,
            confidence = result["confidence"],
        )
    except Exception:
        pass

    if body.user_id:
        try:
            dynamo_put_event(
                user_id    = body.user_id,
                event_type = "gps_score",
                scores     = {
                    "spoof_probability": result["spoof_probability"],
                    "is_spoofed":        result["is_spoofed"],
                    "models_used":       result["models_used"],
                    "risk_factors":      result["risk_factors"],
                },
            )
        except Exception:
            pass

    # ── WebSocket broadcast ───────────────────────────────
    try:
        broadcast = getattr(request.app.state, "broadcast", None)
        if broadcast and body.user_id:
            await broadcast("gps_score", uid, result)
    except Exception:
        pass
    # ─────────────────────────────────────────────────────

    return GPSSpoofResponse(
        spoof_probability = result["spoof_probability"],
        risk_score        = result["risk_score"],
        is_spoofed        = result["is_spoofed"],
        confidence        = result["confidence"],
        model_scores      = result["model_scores"],
        models_used       = result["models_used"],
        risk_factors      = result.get("risk_factors", []),
        user_id           = body.user_id,
        device_id         = body.device_id,
        timestamp         = datetime.now(timezone.utc).isoformat(),
        latency_ms        = round(elapsed_ms, 2),
    )


@router.post("/score/point", summary="Score a single GPS point (limited accuracy)")
async def score_single(body: SinglePointRequest):
    """
    Quick scoring for one GPS point.
    Trajectory-based scoring is significantly more accurate.
    """
    try:
        scorer = _get_scorer()
        result = scorer["single"](
            lat      = body.latitude,
            lng      = body.longitude,
            speed    = body.speed    or 0.0,
            heading  = body.heading  or 0.0,
            prev_lat = body.prev_latitude,
            prev_lng = body.prev_longitude,
        )
        return {
            **result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "warning": "Single-point scoring has limited accuracy. Use trajectory scoring.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", summary="GPS module health check")
async def health_check():
    """Check which GPS scoring models are available."""
    try:
        scorer = _get_scorer()
        test_traj = [
            {"latitude": 37.7749, "longitude": -122.4194, "timestamp": 1_700_000_000.0},
            {"latitude": 37.7750, "longitude": -122.4195, "timestamp": 1_700_000_010.0},
            {"latitude": 37.7751, "longitude": -122.4196, "timestamp": 1_700_000_020.0},
        ]
        result = scorer["trajectory"](test_traj)
        return {
            "status":           "healthy",
            "models_available": result.get("models_used", []),
            "test_score":       round(result["spoof_probability"], 4),
            "timestamp":        datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {
            "status":    "unhealthy",
            "error":     str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
