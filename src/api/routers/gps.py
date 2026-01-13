"""
GPS Spoofing Detection API Router

Endpoints for detecting GPS spoofing from trajectory data.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

router = APIRouter()


class GPSPoint(BaseModel):
    """Single GPS data point."""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in degrees")
    speed: Optional[float] = Field(0.0, ge=0, description="Speed in m/s")
    acceleration: Optional[float] = Field(0.0, description="Acceleration in m/s²")
    heading: Optional[float] = Field(0.0, ge=0, le=360, description="Heading in degrees")
    heading_change: Optional[float] = Field(0.0, description="Change in heading")
    timestamp: Optional[float] = Field(None, description="Unix timestamp")
    time_delta: Optional[float] = Field(0.0, ge=0, description="Time since last point in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "latitude": 37.7749,
                "longitude": -122.4194,
                "speed": 15.5,
                "heading": 45.0,
                "timestamp": 1701234567.0
            }
        }


class GPSTrajectoryRequest(BaseModel):
    """Request body for GPS trajectory scoring."""
    trajectory: List[GPSPoint] = Field(..., min_length=1, description="List of GPS points forming a trajectory")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    device_id: Optional[str] = Field(None, description="Optional device identifier")
    ensemble: bool = Field(True, description="Whether to use ensemble of all models")
    
    class Config:
        json_schema_extra = {
            "example": {
                "trajectory": [
                    {"latitude": 37.7749, "longitude": -122.4194, "speed": 0, "heading": 0},
                    {"latitude": 37.7750, "longitude": -122.4195, "speed": 5, "heading": 45},
                    {"latitude": 37.7751, "longitude": -122.4196, "speed": 10, "heading": 45}
                ],
                "user_id": "user_12345",
                "ensemble": True
            }
        }


class SinglePointRequest(BaseModel):
    """Request body for single GPS point scoring."""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    speed: Optional[float] = Field(0.0, ge=0)
    heading: Optional[float] = Field(0.0, ge=0, le=360)
    prev_latitude: Optional[float] = Field(None, ge=-90, le=90)
    prev_longitude: Optional[float] = Field(None, ge=-180, le=180)


class GPSSpoofResponse(BaseModel):
    """Response from GPS spoofing detection."""
    spoof_probability: float = Field(..., ge=0, le=1, description="Probability that the trajectory is spoofed")
    is_spoofed: bool = Field(..., description="Binary classification result")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in the prediction")
    model_scores: Dict[str, float] = Field(..., description="Individual model scores")
    models_used: List[str] = Field(..., description="List of models used in scoring")
    user_id: Optional[str] = None
    device_id: Optional[str] = None


# Lazy import to avoid circular imports and slow startup
_score_gps = None

def get_score_function():
    global _score_gps
    if _score_gps is None:
        from src.gps.score_gps import score_gps_trajectory, score_single_point
        _score_gps = {
            "trajectory": score_gps_trajectory,
            "single": score_single_point
        }
    return _score_gps


@router.post("/score", response_model=GPSSpoofResponse, summary="Score GPS trajectory for spoofing")
async def score_trajectory(request: GPSTrajectoryRequest):
    """
    Analyze a GPS trajectory for potential spoofing.
    
    The endpoint uses multiple ML models to detect anomalies:
    - Isolation Forest (unsupervised anomaly detection)
    - Gradient Boosting (supervised classification)
    - Autoencoder (reconstruction error)
    - CNN-RNN (deep learning on sequences)
    
    Returns a probability score [0-1] where higher values indicate 
    higher likelihood of GPS spoofing.
    """
    try:
        score_fn = get_score_function()
        
        # Convert Pydantic models to dicts
        trajectory_dicts = [point.model_dump() for point in request.trajectory]
        
        # Score the trajectory
        result = score_fn["trajectory"](trajectory_dicts, ensemble=request.ensemble)
        
        return GPSSpoofResponse(
            spoof_probability=result["spoof_probability"],
            is_spoofed=result["is_spoofed"],
            confidence=result["confidence"],
            model_scores=result["model_scores"],
            models_used=result.get("models_used", []),
            user_id=request.user_id,
            device_id=request.device_id
        )
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503, 
            detail=f"GPS spoofing models not available: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/score/point", summary="Score single GPS point (limited accuracy)")
async def score_single(request: SinglePointRequest):
    """
    Quick scoring for a single GPS point.
    
    Note: Trajectory-based scoring is more accurate. Use this endpoint 
    only when full trajectory data is not available.
    """
    try:
        score_fn = get_score_function()
        
        result = score_fn["single"](
            lat=request.latitude,
            lng=request.longitude,
            speed=request.speed or 0.0,
            heading=request.heading or 0.0,
            prev_lat=request.prev_latitude,
            prev_lng=request.prev_longitude
        )
        
        return {
            "spoof_probability": result["spoof_probability"],
            "is_spoofed": result["is_spoofed"],
            "confidence": result["confidence"],
            "warning": "Single-point scoring has limited accuracy. Use trajectory scoring for better results."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scoring point: {str(e)}")


@router.get("/health", summary="Check GPS scoring service health")
async def health_check():
    """Check if GPS scoring models are loaded and ready."""
    try:
        score_fn = get_score_function()
        
        # Try loading models
        models_available = []
        models_missing = []
        
        # Test with minimal trajectory
        test_trajectory = [
            {"latitude": 0, "longitude": 0, "speed": 0, "heading": 0}
        ]
        
        try:
            result = score_fn["trajectory"](test_trajectory)
            models_available = result.get("models_used", [])
        except Exception as e:
            models_missing.append(f"Error loading models: {str(e)}")
        
        return {
            "status": "healthy" if models_available else "degraded",
            "models_available": models_available,
            "models_missing": models_missing
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


