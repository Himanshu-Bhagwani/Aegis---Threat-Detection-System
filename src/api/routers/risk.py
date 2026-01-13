"""
Unified Risk Assessment API Router

Endpoints for computing combined risk scores from all detection layers.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

router = APIRouter()


class GPSData(BaseModel):
    """GPS trajectory data for risk assessment."""
    trajectory: List[Dict[str, float]] = Field(
        default=None,
        description="List of GPS points with lat, lng, speed, etc."
    )
    spoof_probability: Optional[float] = Field(
        None, ge=0, le=1,
        description="Pre-computed GPS spoof probability (if already scored)"
    )
    confidence: Optional[float] = Field(None, ge=0, le=1)


class LoginData(BaseModel):
    """Login event data for risk assessment."""
    user_deg: Optional[int] = Field(1, ge=0)
    comp_deg: Optional[int] = Field(1, ge=0)
    time_since_user_last: Optional[float] = Field(3600)
    time_since_comp_last: Optional[float] = Field(3600)
    hour_of_day: Optional[int] = Field(12, ge=0, le=23)
    is_new_user: Optional[int] = Field(0, ge=0, le=1)
    is_new_comp: Optional[int] = Field(0, ge=0, le=1)
    failed_10min: Optional[int] = Field(0, ge=0)
    impossible_travel: Optional[int] = Field(0, ge=0, le=1)
    # Pre-computed score
    anomaly_probability: Optional[float] = Field(None, ge=0, le=1)
    confidence: Optional[float] = Field(None, ge=0, le=1)


class TransactionData(BaseModel):
    """Transaction data for fraud assessment."""
    amount: Optional[float] = Field(None, ge=0)
    merchant_category: Optional[str] = None
    is_international: Optional[bool] = Field(False)
    distance_from_home: Optional[float] = Field(None, ge=0)
    time_since_last_txn: Optional[float] = Field(None, ge=0)
    # Pre-computed or raw payload for fraud model
    fraud_probability: Optional[float] = Field(None, ge=0, le=1)
    payload: Optional[Dict[str, Any]] = Field(None, description="Raw transaction payload for fraud model")


class UnifiedRiskRequest(BaseModel):
    """Request body for unified risk assessment."""
    # User/Event identifiers
    user_id: Optional[str] = Field(None, description="User identifier")
    event_id: Optional[str] = Field(None, description="Event/session identifier")
    
    # Input data from each detection layer
    gps_data: Optional[GPSData] = Field(None, description="GPS trajectory or pre-computed score")
    login_data: Optional[LoginData] = Field(None, description="Login event features or pre-computed score")
    password: Optional[str] = Field(None, description="Password to evaluate (raw)")
    password_score: Optional[float] = Field(None, ge=0, le=1, description="Pre-computed password risk score")
    transaction_data: Optional[TransactionData] = Field(None, description="Transaction data or pre-computed fraud score")
    
    # Fusion options
    fusion_strategy: str = Field(
        "weighted_average",
        description="Fusion strategy: weighted_average, max_threat, or bayesian"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_12345",
                "event_id": "evt_67890",
                "gps_data": {
                    "spoof_probability": 0.3,
                    "confidence": 0.8
                },
                "login_data": {
                    "user_deg": 5,
                    "comp_deg": 150,
                    "hour_of_day": 3,
                    "failed_10min": 2
                },
                "password_score": 0.4,
                "transaction_data": {
                    "fraud_probability": 0.7
                },
                "fusion_strategy": "weighted_average"
            }
        }


class UnifiedRiskResponse(BaseModel):
    """Response from unified risk assessment."""
    # Overall assessment
    unified_score: float = Field(..., ge=0, le=1, description="Combined risk score")
    risk_level: str = Field(..., description="Risk level: minimal, low, medium, high, critical")
    confidence: float = Field(..., ge=0, le=1, description="Overall confidence")
    
    # Component scores
    gps_risk: float = Field(..., ge=0, le=1)
    login_risk: float = Field(..., ge=0, le=1)
    password_risk: float = Field(..., ge=0, le=1)
    fraud_risk: float = Field(..., ge=0, le=1)
    
    # Threat analysis
    primary_threats: List[str] = Field(..., description="Top threat categories identified")
    threat_signals: List[Dict] = Field(..., description="Detailed threat signals")
    recommended_actions: List[str] = Field(..., description="Recommended response actions")
    
    # Metadata
    user_id: Optional[str] = None
    event_id: Optional[str] = None
    timestamp: str = Field(..., description="Assessment timestamp")
    models_used: List[str] = Field(..., description="Models used in assessment")


# Lazy imports
_fusion_engine = None
_gps_scorer = None
_login_scorer = None
_password_scorer = None
_fraud_scorer = None


def get_fusion_engine():
    global _fusion_engine
    if _fusion_engine is None:
        from src.fusion.risk_scoring import compute_unified_risk
        _fusion_engine = compute_unified_risk
    return _fusion_engine


def get_gps_scorer():
    global _gps_scorer
    if _gps_scorer is None:
        from src.gps.score_gps import score_gps_trajectory
        _gps_scorer = score_gps_trajectory
    return _gps_scorer


def get_login_scorer():
    global _login_scorer
    if _login_scorer is None:
        from src.login.score_login import score_login_event
        _login_scorer = score_login_event
    return _login_scorer


def get_password_scorer():
    global _password_scorer
    if _password_scorer is None:
        from src.passwords.score_password import score_password
        _password_scorer = score_password
    return _password_scorer


def get_fraud_scorer():
    global _fraud_scorer
    if _fraud_scorer is None:
        try:
            from src.api.routers.fraud import score_fraud_rule_based, load_artifacts, prepare_row
            
            def score_fraud(payload):
                """Score fraud with graceful fallback."""
                model, scaler, features = load_artifacts()
                
                # If ML model available, use it
                if model is not None and scaler is not None and features is not None:
                    try:
                        import xgboost as xgb
                        X = prepare_row(payload, features)
                        Xs = scaler.transform(X)
                        dmat = xgb.DMatrix(Xs)
                        prob = model.predict(dmat)[0]
                        return float(prob)
                    except Exception:
                        # Fall through to rule-based
                        pass
                
                # Use rule-based fallback
                return score_fraud_rule_based(payload)
            
            _fraud_scorer = score_fraud
        except Exception as e:
            # Final fallback
            _fraud_scorer = lambda x: 0.5  # Neutral score
    return _fraud_scorer


@router.post("/unified", response_model=UnifiedRiskResponse, summary="Compute unified risk score")
async def compute_unified_risk(request: UnifiedRiskRequest):
    """
    Compute a unified risk score combining all threat detection layers.
    
    This endpoint accepts:
    - **Pre-computed scores**: If you've already scored GPS, login, password, or fraud
      individually, pass those scores directly.
    - **Raw data**: If you pass raw data (GPS trajectory, login features, password, transaction),
      the system will score them automatically.
    
    The fusion engine combines scores using configurable strategies:
    - **weighted_average**: Default. Weights each threat category by importance.
    - **max_threat**: Uses the maximum threat probability (any high threat triggers alert).
    - **bayesian**: Bayesian fusion assuming independent threat sources.
    
    Returns a comprehensive risk assessment with:
    - Overall unified risk score [0-1]
    - Risk level classification
    - Component breakdown
    - Primary threats identified
    - Recommended actions
    """
    try:
        fusion_fn = get_fusion_engine()
        
        # Prepare GPS score
        gps_score = None
        if request.gps_data:
            if request.gps_data.spoof_probability is not None:
                # Use pre-computed score
                gps_score = {
                    "spoof_probability": request.gps_data.spoof_probability,
                    "confidence": request.gps_data.confidence or 0.7,
                    "models_used": ["pre_computed"]
                }
            elif request.gps_data.trajectory:
                # Score the trajectory
                try:
                    gps_scorer = get_gps_scorer()
                    gps_score = gps_scorer(request.gps_data.trajectory)
                except Exception as e:
                    gps_score = {"spoof_probability": 0.0, "confidence": 0.0, "error": str(e)}
        
        # Prepare login score
        login_score = None
        if request.login_data:
            if request.login_data.anomaly_probability is not None:
                # Use pre-computed score
                login_score = {
                    "anomaly_probability": request.login_data.anomaly_probability,
                    "confidence": request.login_data.confidence or 0.7,
                    "models_used": ["pre_computed"]
                }
            else:
                # Score the login event
                try:
                    login_scorer = get_login_scorer()
                    login_dict = request.login_data.model_dump(
                        exclude={"anomaly_probability", "confidence"}
                    )
                    login_score = login_scorer(login_dict)
                except Exception as e:
                    login_score = {"anomaly_probability": 0.0, "confidence": 0.0, "error": str(e)}
        
        # Prepare password score
        password_risk = None
        if request.password_score is not None:
            password_risk = request.password_score
        elif request.password:
            try:
                password_scorer = get_password_scorer()
                password_risk = password_scorer(request.password)
            except Exception:
                password_risk = None
        
        # Prepare fraud score
        fraud_risk = None
        if request.transaction_data:
            if request.transaction_data.fraud_probability is not None:
                fraud_risk = request.transaction_data.fraud_probability
            elif request.transaction_data.payload:
                try:
                    fraud_scorer = get_fraud_scorer()
                    fraud_risk = fraud_scorer(request.transaction_data.payload)
                except Exception:
                    fraud_risk = None
        
        # Compute unified risk
        result = fusion_fn(
            gps_score=gps_score,
            login_score=login_score,
            password_score=password_risk,
            fraud_score=fraud_risk,
            user_id=request.user_id,
            event_id=request.event_id,
            fusion_strategy=request.fusion_strategy
        )
        
        return UnifiedRiskResponse(
            unified_score=result["unified_score"],
            risk_level=result["risk_level"],
            confidence=result["confidence"],
            gps_risk=result["gps_risk"],
            login_risk=result["login_risk"],
            password_risk=result["password_risk"],
            fraud_risk=result["fraud_risk"],
            primary_threats=result["primary_threats"],
            threat_signals=result["threat_signals"],
            recommended_actions=result["recommended_actions"],
            user_id=result["user_id"],
            event_id=result["event_id"],
            timestamp=result["timestamp"],
            models_used=result["models_used"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk computation error: {str(e)}")


@router.post("/unified/quick", summary="Quick unified risk from pre-computed scores")
async def quick_unified_risk(
    gps_risk: Optional[float] = None,
    login_risk: Optional[float] = None,
    password_risk: Optional[float] = None,
    fraud_risk: Optional[float] = None,
    user_id: Optional[str] = None,
    fusion_strategy: str = "weighted_average"
):
    """
    Quick unified risk assessment from pre-computed component scores.
    
    Use this endpoint when you've already computed individual risk scores
    and just need the fusion result.
    """
    try:
        fusion_fn = get_fusion_engine()
        
        # Create score dicts from raw probabilities
        gps_score = {"spoof_probability": gps_risk, "confidence": 0.8} if gps_risk is not None else None
        login_score = {"anomaly_probability": login_risk, "confidence": 0.8} if login_risk is not None else None
        
        result = fusion_fn(
            gps_score=gps_score,
            login_score=login_score,
            password_score=password_risk,
            fraud_score=fraud_risk,
            user_id=user_id,
            fusion_strategy=fusion_strategy
        )
        
        return {
            "unified_score": result["unified_score"],
            "risk_level": result["risk_level"],
            "primary_threats": result["primary_threats"],
            "recommended_actions": result["recommended_actions"][:3]  # Top 3 actions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/thresholds", summary="Get risk level thresholds")
async def get_thresholds():
    """
    Get the current risk level threshold configuration.
    
    Returns the score thresholds used to classify risk levels.
    """
    return {
        "thresholds": {
            "minimal": {"min": 0.0, "max": 0.1},
            "low": {"min": 0.1, "max": 0.25},
            "medium": {"min": 0.25, "max": 0.5},
            "high": {"min": 0.5, "max": 0.75},
            "critical": {"min": 0.75, "max": 1.0}
        },
        "component_weights": {
            "gps_spoofing": 1.5,
            "login_anomaly": 2.0,
            "password_weakness": 1.0,
            "transaction_fraud": 2.5
        },
        "fusion_strategies_available": ["weighted_average", "max_threat", "bayesian"]
    }


@router.get("/health", summary="Check unified risk service health")
async def health_check():
    """Check if all risk scoring components are available."""
    status = {
        "fusion_engine": False,
        "gps_scorer": False,
        "login_scorer": False,
        "password_scorer": False,
        "fraud_scorer": False
    }
    
    try:
        get_fusion_engine()
        status["fusion_engine"] = True
    except Exception:
        pass
    
    try:
        get_gps_scorer()
        status["gps_scorer"] = True
    except Exception:
        pass
    
    try:
        get_login_scorer()
        status["login_scorer"] = True
    except Exception:
        pass
    
    try:
        get_password_scorer()
        status["password_scorer"] = True
    except Exception:
        pass
    
    try:
        get_fraud_scorer()
        status["fraud_scorer"] = True
    except Exception:
        pass
    
    all_healthy = all(status.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "components": status,
        "available_count": sum(status.values()),
        "total_components": len(status)
    }


