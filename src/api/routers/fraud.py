from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json, joblib, os, numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

router = APIRouter()

# Paths (relative to project root)
MODEL_PATH = "data/processed/fraud/xgb_model.bst"
SCALER_PATH = "data/processed/fraud/scaler.pkl"
FEATURES_PATH = "data/processed/fraud/features.json"

class Transaction(BaseModel):
    # Accept arbitrary JSON fields (we'll pick the numeric features needed)
    payload: dict

# Lazy-loaded artifacts
_model = None
_scaler = None
_features = None
_artifacts_loaded = False
_artifacts_available = False


def load_artifacts():
    """Load fraud detection artifacts. Returns None if not available."""
    global _model, _scaler, _features, _artifacts_loaded, _artifacts_available
    
    if _artifacts_loaded:
        if not _artifacts_available:
            return None, None, None
        return _model, _scaler, _features
    
    _artifacts_loaded = True
    
    # Try to load model artifacts
    try:
        # Check if all files exist
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(FEATURES_PATH):
            print(f"Warning: Fraud detection model artifacts not found. Using rule-based fallback.")
            _artifacts_available = False
            return None, None, None
        
        # Load XGBoost model
        import xgboost as xgb
        _model = xgb.Booster()
        _model.load_model(MODEL_PATH)
        
        # Load scaler
        _scaler = joblib.load(SCALER_PATH)
        
        # Load features
        with open(FEATURES_PATH) as f:
            _features = json.load(f)
        
        _artifacts_available = True
        print(f"✅ Fraud detection model loaded successfully with {len(_features)} features")
        return _model, _scaler, _features
        
    except Exception as e:
        print(f"Warning: Could not load fraud detection model: {e}. Using rule-based fallback.")
        _artifacts_available = False
        return None, None, None


def prepare_row(payload: Dict[str, Any], features: Optional[list]) -> np.ndarray:
    """Prepare feature row for model prediction."""
    if features is None:
        # Extract common numeric features from payload
        numeric_keys = [k for k, v in payload.items() if isinstance(v, (int, float))]
        features = numeric_keys[:20]  # Limit to first 20 numeric features
    
    row = [float(payload.get(f, 0.0)) for f in features]
    return np.array(row).reshape(1, -1)


def score_fraud_rule_based(payload: Dict[str, Any]) -> float:
    """Rule-based fraud detection when ML model is not available."""
    risk_score = 0.0
    
    # Rule 1: High transaction amount
    amount = float(payload.get("amount", payload.get("transaction_amount", 0)))
    if amount > 10000:
        risk_score += 0.4
    elif amount > 5000:
        risk_score += 0.25
    elif amount > 1000:
        risk_score += 0.1
    
    # Rule 2: International transaction flag
    if payload.get("is_international", payload.get("international", False)):
        risk_score += 0.3
    
    # Rule 3: Time-based anomalies (late night transactions)
    hour = payload.get("hour", payload.get("hour_of_day", 12))
    if hour >= 22 or hour <= 5:  # 10 PM - 5 AM
        risk_score += 0.2
    
    # Rule 4: Rapid transactions (many transactions in short time)
    tx_count_1h = payload.get("tx_count_1h", payload.get("transactions_last_hour", 0))
    if tx_count_1h > 10:
        risk_score += 0.4
    elif tx_count_1h > 5:
        risk_score += 0.2
    
    # Rule 5: Time since last transaction (very short = suspicious)
    time_since_last = payload.get("time_since_last_tx", payload.get("time_since_last_transaction", 3600))
    if time_since_last < 60:  # Less than 1 minute
        risk_score += 0.3
    elif time_since_last < 300:  # Less than 5 minutes
        risk_score += 0.15
    
    # Rule 6: Amount ratio (transaction much larger than user average)
    amount_ratio = payload.get("amount_ratio", 1.0)
    if amount_ratio > 5.0:
        risk_score += 0.4
    elif amount_ratio > 3.0:
        risk_score += 0.25
    elif amount_ratio > 2.0:
        risk_score += 0.1
    
    # Rule 7: New merchant (user hasn't used this merchant before)
    merchant_freq = payload.get("merchant_freq_user", payload.get("merchant_frequency", 1))
    if merchant_freq == 1:
        risk_score += 0.15
    
    # Rule 8: Device change
    if payload.get("device_changed", payload.get("new_device", False)):
        risk_score += 0.2
    
    # Rule 9: Distance from home (if available)
    distance = payload.get("distance_from_home", payload.get("distance", None))
    if distance is not None:
        if distance > 1000:  # More than 1km (or 1000 units)
            risk_score += 0.25
        elif distance > 500:
            risk_score += 0.15
    
    return min(1.0, risk_score)


@router.post("/score")
def score_txn(body: Transaction):
    """
    Score a transaction for fraud probability.
    
    Uses ML model if available, otherwise falls back to rule-based scoring.
    """
    try:
        model, scaler, features = load_artifacts()
        
        # If ML model available, use it
        if model is not None and scaler is not None and features is not None:
            try:
                import xgboost as xgb
                X = prepare_row(body.payload, features)
                Xs = scaler.transform(X)
                dmat = xgb.DMatrix(Xs)
                prob = model.predict(dmat)[0]
                return {
                    "fraud_probability": float(prob),
                    "method": "ml_model",
                    "confidence": 0.85
                }
            except Exception as e:
                print(f"Warning: ML model prediction failed: {e}. Falling back to rule-based.")
                # Fall through to rule-based
        
        # Use rule-based fallback
        prob = score_fraud_rule_based(body.payload)
        return {
            "fraud_probability": prob,
            "method": "rule_based",
            "confidence": 0.6
        }
        
    except Exception as e:
        # Final fallback if everything fails
        print(f"Error in fraud scoring: {e}")
        return {
            "fraud_probability": 0.5,  # Neutral score
            "method": "fallback",
            "confidence": 0.0,
            "error": str(e)
        }
