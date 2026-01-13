"""
Login Anomaly Detection API Router

Endpoints for detecting anomalous login patterns.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

router = APIRouter()


class LoginEventRequest(BaseModel):
    """Request body for login event scoring."""
    # Core features (matching LANL dataset training)
    user_deg: Optional[int] = Field(1, ge=0, description="User's connection degree (unique computers logged into)")
    comp_deg: Optional[int] = Field(1, ge=0, description="Computer's connection degree (unique users)")
    time_since_user_last: Optional[float] = Field(3600, description="Seconds since user's last login")
    time_since_comp_last: Optional[float] = Field(3600, description="Seconds since last login on this computer")
    hour_of_day: Optional[int] = Field(12, ge=0, le=23, description="Hour of day (0-23)")
    is_new_user: Optional[int] = Field(0, ge=0, le=1, description="1 if first time seeing this user")
    is_new_comp: Optional[int] = Field(0, ge=0, le=1, description="1 if first time seeing this computer")
    
    # Additional features (rule-based scoring)
    failed_10min: Optional[int] = Field(0, ge=0, description="Failed login attempts in last 10 minutes")
    impossible_travel: Optional[int] = Field(0, ge=0, le=1, description="1 if impossible travel detected")
    device_changed: Optional[int] = Field(0, ge=0, le=1, description="1 if device changed from usual")
    
    # Identifiers
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    source_ip: Optional[str] = Field(None, description="Source IP address")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_deg": 5,
                "comp_deg": 150,
                "time_since_user_last": 3600,
                "time_since_comp_last": 300,
                "hour_of_day": 3,
                "is_new_user": 0,
                "is_new_comp": 1,
                "failed_10min": 2,
                "impossible_travel": 0,
                "user_id": "user_12345"
            }
        }


class BatchLoginRequest(BaseModel):
    """Request body for batch login event scoring."""
    events: List[LoginEventRequest] = Field(..., min_length=1, max_length=1000)


class LoginAnomalyResponse(BaseModel):
    """Response from login anomaly detection."""
    anomaly_probability: float = Field(..., ge=0, le=1, description="Probability of anomalous login")
    is_anomalous: bool = Field(..., description="Binary classification result")
    risk_level: str = Field(..., description="Risk level: minimal, low, medium, high, critical")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in the prediction")
    model_scores: Dict[str, float] = Field(..., description="Individual model scores")
    rule_scores: Dict[str, float] = Field(default_factory=dict, description="Rule-based scores")
    models_used: List[str] = Field(..., description="List of models used")
    user_id: Optional[str] = None
    session_id: Optional[str] = None


# Lazy import
_score_login = None

def get_score_function():
    global _score_login
    if _score_login is None:
        from src.login.score_login import score_login_event, score_login_batch
        _score_login = {
            "single": score_login_event,
            "batch": score_login_batch
        }
    return _score_login


@router.post("/score", response_model=LoginAnomalyResponse, summary="Score login event for anomalies")
async def score_login(request: LoginEventRequest):
    """
    Analyze a login event for potential anomalies.
    
    The endpoint uses multiple ML models trained on LANL dataset:
    - Isolation Forest (unsupervised anomaly detection)
    - Gradient Boosting (trained on pseudo-labels)
    - Autoencoder (reconstruction error)
    - Rule-based heuristics (brute force, impossible travel, etc.)
    
    Returns a probability score [0-1] where higher values indicate 
    higher likelihood of anomalous/malicious login attempt.
    """
    try:
        score_fn = get_score_function()
        
        # Convert to dict for scoring function
        event_dict = request.model_dump(exclude_none=True)
        
        # Score the event
        result = score_fn["single"](event_dict)
        
        return LoginAnomalyResponse(
            anomaly_probability=result["anomaly_probability"],
            is_anomalous=result["is_anomalous"],
            risk_level=result["risk_level"],
            confidence=result["confidence"],
            model_scores=result["model_scores"],
            rule_scores=result.get("rule_scores", {}),
            models_used=result.get("models_used", []),
            user_id=request.user_id,
            session_id=request.session_id
        )
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Login anomaly models not available: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/score/batch", summary="Score multiple login events")
async def score_login_batch(request: BatchLoginRequest):
    """
    Score multiple login events in a single request.
    
    Useful for bulk processing historical data or analyzing patterns
    across multiple events.
    """
    try:
        score_fn = get_score_function()
        
        # Convert to list of dicts
        events = [event.model_dump(exclude_none=True) for event in request.events]
        
        # Score batch
        results = score_fn["batch"](events)
        
        return {
            "results": results,
            "total_events": len(results),
            "high_risk_count": sum(1 for r in results if r["risk_level"] in ["high", "critical"]),
            "anomalous_count": sum(1 for r in results if r["is_anomalous"])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch scoring error: {str(e)}")


@router.get("/health", summary="Check login scoring service health")
async def health_check():
    """Check if login scoring models are loaded and ready."""
    try:
        score_fn = get_score_function()
        
        # Test with sample event
        test_event = {
            "user_deg": 1,
            "comp_deg": 1,
            "time_since_user_last": 3600,
            "time_since_comp_last": 3600,
            "hour_of_day": 12,
            "is_new_user": 0,
            "is_new_comp": 0
        }
        
        result = score_fn["single"](test_event)
        
        return {
            "status": "healthy",
            "models_available": result.get("models_used", []),
            "rules_active": list(result.get("rule_scores", {}).keys())
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


