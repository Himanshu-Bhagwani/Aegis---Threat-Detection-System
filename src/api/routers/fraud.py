from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict
import json, joblib, math, os, numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

router = APIRouter()

# Paths (relative to project root)
MODEL_PATH = "data/processed/fraud/xgb_model.bst"
SCALER_PATH = "data/processed/fraud/scaler.pkl"
FEATURES_PATH = "data/processed/fraud/features.json"

class Transaction(BaseModel):
    """A transaction to score.

    Accepts either the wrapped form used by the dashboard:
        {"payload": {"amount": 5000, ...}, "user_id": "alice@corp.com"}
    or a flat form (what the SDK / integrating apps send):
        {"amount": 5000, "hour": 21, "user_id": "alice@corp.com"}
    """
    payload: Optional[dict] = None
    user_id: Optional[str] = None

    model_config = ConfigDict(extra="allow")

    def resolved_payload(self) -> Dict[str, Any]:
        """Return the transaction fields regardless of which form was sent."""
        if self.payload:
            return self.payload
        data = self.model_dump(exclude_none=True)
        data.pop("payload", None)
        data.pop("user_id", None)
        return data

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
    
    # Rule 1: Absolute size — only a fallback prior.
    # Fixed currency thresholds are meaningless across currencies (₹10,000 is
    # routine in INR but the old rule gave it the same +0.4 as ₹50Cr, which is
    # why very different amounts all scored 55%). Once we know what's normal
    # for this user, Rule 6 does the work and this stays out of the way.
    amount = float(payload.get("amount", payload.get("transaction_amount", 0)))
    baseline_count = int(payload.get("baseline_count", 0) or 0)
    if baseline_count < 2 and amount > 1000:
        # No usable history yet — fall back to a mild, smoothly-scaling prior so
        # even here a larger amount always outranks a smaller one.
        risk_score += max(0.0, min(0.35, 0.10 * math.log10(amount / 1000.0)))
    
    # Rule 2: International transaction flag
    if payload.get("is_international", payload.get("international", False)):
        risk_score += 0.3
    
    # Rule 3: Time-based anomalies (late night transactions)
    hour = payload.get("hour", payload.get("hour_of_day", 12))
    if hour >= 22 or hour <= 5:  # 10 PM - 5 AM
        risk_score += 0.2
    
    # Rule 4: Velocity — many transactions in a short time.
    tx_count_1h = payload.get("tx_count_1h", payload.get("transactions_last_hour", 0))
    if tx_count_1h > 10:
        risk_score += 0.4
    elif tx_count_1h > 5:
        risk_score += 0.2

    # Rule 4b: Velocity relative to THIS user's usual pace. A burst of ordinary
    # amounts (e.g. six ₹30k payments when they normally make one an hour) is a
    # classic drain pattern that per-transaction amount checks miss entirely.
    typical_rate = float(payload.get("typical_per_hour", 0) or 0)
    if typical_rate > 0 and tx_count_1h > 0:
        velocity_ratio = tx_count_1h / typical_rate
        if velocity_ratio >= 2.0:
            risk_score += min(0.35, 0.18 * math.log2(velocity_ratio))
    
    # Rule 5: Time since last transaction (very short = suspicious)
    time_since_last = payload.get("time_since_last_tx", payload.get("time_since_last_transaction", 3600))
    if time_since_last < 60:  # Less than 1 minute
        risk_score += 0.3
    elif time_since_last < 300:  # Less than 5 minutes
        risk_score += 0.15
    
    # Rule 6: How far above the user's normal spend this is.
    # Continuous (log-scaled) rather than stepped: banded thresholds created a
    # cliff where a 4.9x payment scored far below a 5.1x one, and could even
    # rank a bigger transaction as safer than a smaller one.
    #   2x -> 0.19   5x -> 0.43   10x -> 0.62   30x -> 0.92   100x -> 0.95(cap)
    amount_ratio = float(payload.get("amount_ratio", 1.0) or 1.0)
    if amount_ratio > 1.2:
        risk_score += max(0.0, min(0.95, 0.62 * math.log10(amount_ratio)))
    
    # Rule 7: New merchant — only meaningful when merchant identity was actually
    # supplied. It previously fired on every transaction (the default is 1),
    # adding a constant +0.15 of noise to every score.
    if payload.get("merchant") or payload.get("merchant_id"):
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
async def score_txn(body: Transaction, request: Request):
    """
    Score a transaction for fraud probability.

    Uses ML model if available, otherwise falls back to rule-based scoring.
    """
    payload = body.resolved_payload()
    uid_raw = body.user_id or payload.get("user_id")

    # Derive how far this transaction deviates from the user's OWN spending
    # history. Without this every large payment scores the same regardless of
    # whether it's normal for them (a fixed currency threshold can't tell a
    # ₹50Cr corporate transfer from a ₹50Cr anomaly).
    # Treat a missing ratio — or the neutral 1.0 that older clients hard-coded —
    # as "no deviation information supplied", and derive it from history instead.
    try:
        _supplied_ratio = float(payload.get("amount_ratio", 1.0) or 1.0)
    except (TypeError, ValueError):
        _supplied_ratio = 1.0
    _needs_ratio = "amount_ratio" not in payload or abs(_supplied_ratio - 1.0) < 1e-9

    computed_ratio = None
    baseline = {}
    if uid_raw and "amount" in payload:
        try:
            from src.api.tenant_context import scope_user_id
            from src.utils.profiles import transaction_baseline
            baseline = transaction_baseline(scope_user_id(uid_raw))
            amount = float(payload.get("amount") or 0)

            # Compare against the MEDIAN so a single outlier can't reset what
            # counts as normal for this user.
            if _needs_ratio and baseline.get("median", 0) > 0 and amount > 0:
                computed_ratio = round(amount / baseline["median"], 4)
                payload["amount_ratio"] = computed_ratio

            # Give the velocity rule this user's usual pace to compare against,
            # and tell the amount rule whether we have a usable baseline yet.
            if baseline.get("typical_per_hour"):
                payload.setdefault("typical_per_hour", baseline["typical_per_hour"])
            payload["baseline_count"] = baseline.get("count", 0)
        except Exception as e:
            print(f"Warning: could not derive spending baseline: {e}")

    try:
        model, scaler, features = load_artifacts()

        if model is not None and scaler is not None and features is not None:
            try:
                import xgboost as xgb
                X = prepare_row(payload, features)
                Xs = scaler.transform(X)
                dmat = xgb.DMatrix(Xs)
                prob = model.predict(dmat)[0]
                result = {"fraud_probability": float(prob), "method": "ml_model", "confidence": 0.85}
            except Exception as e:
                print(f"Warning: ML model prediction failed: {e}. Falling back to rule-based.")
                prob = score_fraud_rule_based(payload)
                result = {"fraud_probability": prob, "method": "rule_based", "confidence": 0.6}
        else:
            prob = score_fraud_rule_based(payload)
            result = {"fraud_probability": prob, "method": "rule_based", "confidence": 0.6}

    except Exception as e:
        print(f"Error in fraud scoring: {e}")
        result = {"fraud_probability": 0.5, "method": "fallback", "confidence": 0.0, "error": str(e)}

    # Attribute the event to the real user so it lands on their profile
    # (previously hard-coded to "anonymous", so fraud never reached a profile).
    uid = uid_raw or "anonymous"
    amount = float(payload.get("amount") or 0)

    # ── Damp the score for spending the user has already vouched for ──
    # Once a large payment is confirmed as theirs, anything at or below that
    # level is normal for them and shouldn't keep getting flagged.
    max_confirmed = float(baseline.get("max_confirmed") or 0)
    if max_confirmed > 0 and amount > 0 and amount <= max_confirmed:
        result["fraud_probability"] = min(result["fraud_probability"], 0.30)
        result["damped_by_confirmation"] = True

    result["user_id"] = uid
    if computed_ratio is not None:
        # Surfaced so the UI can explain *why* a transaction was flagged.
        result["amount_ratio_vs_history"] = computed_ratio
    if baseline:
        result["baseline_median"] = baseline.get("median", 0)
        result["baseline_count"]  = baseline.get("count", 0)

    # ── Should we challenge the user about this transaction? ──
    # Policy lives here so every client behaves consistently.
    prob      = result["fraud_probability"]
    seen      = int(baseline.get("count", 0))
    median    = float(baseline.get("median") or 0)
    ratio     = computed_ratio or 0
    challenge = False
    tone      = "confirm"          # softer wording while we're still learning

    if seen < 3:
        # Barely any history — only bother them about genuinely large amounts.
        challenge = amount > 10000
    elif seen < 5:
        challenge = amount > 10000 and (ratio == 0 or ratio >= 1.5)
    else:
        # Enough history to judge: challenge clear outliers. The absolute floor
        # stops tiny-but-proportionally-large amounts nagging the user.
        big_multiple = ratio >= 3.0 if median >= 5000 else ratio >= 5.0
        challenge = (prob >= 0.60 or big_multiple) and amount > 1000
        tone = "verify"

    if result.get("damped_by_confirmation"):
        challenge = False

    result["challenge"]       = bool(challenge)
    result["challenge_tone"]  = tone
    result["transactions_seen"] = seen
    # Keep the transaction context on the event so the dashboard can chart
    # real spend history (amount, timing, velocity) rather than a static curve.
    for k in ("amount", "hour", "is_international", "tx_count_1h", "amount_ratio",
              "time_since_last_tx", "currency", "merchant"):
        if k in payload and payload[k] is not None:
            result[k] = payload[k]
    result["risk_level"] = (
        "critical" if result["fraud_probability"] >= 0.75 else
        "high"     if result["fraud_probability"] >= 0.50 else
        "medium"   if result["fraud_probability"] >= 0.25 else
        "low"      if result["fraud_probability"] >= 0.10 else "minimal"
    )

    try:
        broadcast = getattr(request.app.state, "broadcast", None)
        if broadcast:
            await broadcast("fraud_score", uid, result)
    except Exception:
        pass

    return result
