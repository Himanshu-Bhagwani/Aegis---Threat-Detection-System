#!/usr/bin/env python3
"""
Login Anomaly Detection - Scoring Module

Score login events using trained models from LANL dataset:
- Isolation Forest (unsupervised anomaly detection)
- Gradient Boosting (trained on pseudo-labels)
- Autoencoder (reconstruction error)

Each model outputs an anomaly probability [0, 1].
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import joblib

# Lazy imports for TensorFlow
_tf_loaded = False
_tf = None
_keras = None

def _load_tensorflow():
    global _tf_loaded, _tf, _keras
    if not _tf_loaded:
        import tensorflow as tf
        _tf = tf
        _keras = tf.keras
        _tf_loaded = True
    return _tf, _keras

# Model paths
MODEL_DIR = Path("models/login")

# Cached model instances
_models_cache = {}
_meta_cache = {}


def get_model_dir() -> Path:
    """Get model directory, handling both relative and absolute paths."""
    if MODEL_DIR.exists():
        return MODEL_DIR
    # Try from project root
    project_root = Path(__file__).parent.parent.parent
    return project_root / "models" / "login"


def load_metadata() -> Dict:
    """Load training metadata including feature columns and scaler path."""
    if "meta" not in _meta_cache:
        meta_path = get_model_dir() / "lanl_meta.json"
        if not meta_path.exists():
            # Default feature columns based on LANL training
            _meta_cache["meta"] = {
                "feature_cols": [
                    "user_deg",
                    "comp_deg", 
                    "time_since_user_last",
                    "time_since_comp_last",
                    "hour_of_day",
                    "is_new_user",
                    "is_new_comp"
                ]
            }
        else:
            with open(meta_path) as f:
                _meta_cache["meta"] = json.load(f)
    return _meta_cache["meta"]


def load_thresholds() -> Dict:
    """Load anomaly detection thresholds from training."""
    if "thresholds" not in _meta_cache:
        thresh_path = get_model_dir() / "lanl_thresholds.json"
        if thresh_path.exists():
            with open(thresh_path) as f:
                _meta_cache["thresholds"] = json.load(f)
        else:
            # Default thresholds
            _meta_cache["thresholds"] = {
                "iso_score_p95": 0.1,
                "ae_mse_p95": 0.05
            }
    return _meta_cache["thresholds"]


def load_scaler():
    """Load the fitted StandardScaler."""
    if "scaler" not in _models_cache:
        scaler_path = get_model_dir() / "lanl_scaler.joblib"
        if not scaler_path.exists():
            _models_cache["scaler"] = None
            return None
        try:
            _models_cache["scaler"] = joblib.load(scaler_path)
        except Exception as e:
            print(f"Warning: Could not load Login scaler: {e}")
            _models_cache["scaler"] = None
            return None
    return _models_cache.get("scaler")


def load_isolation_forest():
    """Load the trained Isolation Forest model."""
    if "isolation_forest" not in _models_cache:
        model_path = get_model_dir() / "lanl_isolation_forest.joblib"
        if not model_path.exists():
            _models_cache["isolation_forest"] = None
            return None
        try:
            _models_cache["isolation_forest"] = joblib.load(model_path)
        except Exception as e:
            print(f"Warning: Could not load Login Isolation Forest model: {e}")
            _models_cache["isolation_forest"] = None
            return None
    return _models_cache.get("isolation_forest")


def load_gbm():
    """Load the trained Gradient Boosting model."""
    if "gbm" not in _models_cache:
        model_path = get_model_dir() / "lanl_gbm_pseudo.joblib"
        if not model_path.exists():
            _models_cache["gbm"] = None
            return None
        try:
            _models_cache["gbm"] = joblib.load(model_path)
        except Exception as e:
            print(f"Warning: Could not load Login GBM model: {e}")
            _models_cache["gbm"] = None
            return None
    return _models_cache.get("gbm")


def load_autoencoder():
    """Load the trained Autoencoder model."""
    if "autoencoder" not in _models_cache:
        model_path = get_model_dir() / "lanl_autoencoder.h5"
        if not model_path.exists():
            # Mark as unavailable rather than raising
            _models_cache["autoencoder"] = None
            return None
        
        try:
            tf, keras = _load_tensorflow()
            # Try loading with compile=False to avoid metric deserialization issues
            _models_cache["autoencoder"] = keras.models.load_model(
                str(model_path), 
                compile=False
            )
        except Exception as e:
            # If that fails, try legacy format
            try:
                import h5py
                _models_cache["autoencoder"] = keras.models.load_model(
                    str(model_path),
                    compile=False,
                    safe_mode=False
                )
            except Exception as e2:
                # Keras compatibility issue - mark as unavailable
                print(f"Warning: Could not load Login Autoencoder model (Keras compatibility?): {e2}. Skipping.")
                _models_cache["autoencoder"] = None
                return None
    
    return _models_cache.get("autoencoder")


def preprocess_login_event(event: Dict) -> np.ndarray:
    """
    Convert a login event dictionary to feature vector.
    
    Args:
        event: Dictionary with login event features:
            - user_deg: User's connection degree (how many computers they've logged into)
            - comp_deg: Computer's connection degree (how many users have logged in)
            - time_since_user_last: Seconds since user's last login
            - time_since_comp_last: Seconds since last login on this computer
            - hour_of_day: Hour (0-23)
            - is_new_user: 1 if first time seeing this user, 0 otherwise
            - is_new_comp: 1 if first time seeing this computer, 0 otherwise
            
        Alternative keys accepted:
            - failed_attempts / failed_10min: Recent failed login attempts
            - device_changed: Whether device changed from usual
            - impossible_travel: Flag for impossible travel detection
            - source_ip, user_agent, etc. (converted to derived features)
            
    Returns:
        Scaled feature vector ready for model input
    """
    meta = load_metadata()
    feature_cols = meta.get("feature_cols", [])
    
    # Map alternative field names
    field_mapping = {
        "failed_attempts": "failed_10min",
        "device_changed": "is_new_comp",
        "hour": "hour_of_day",
        "user_degree": "user_deg",
        "computer_degree": "comp_deg",
    }
    
    # Normalize field names
    normalized_event = {}
    for key, value in event.items():
        mapped_key = field_mapping.get(key, key)
        normalized_event[mapped_key] = value
    
    # Build feature vector
    features = []
    for col in feature_cols:
        val = normalized_event.get(col, 0.0)
        
        # Handle special cases
        if col in ["time_since_user_last", "time_since_comp_last"]:
            if val < 0:
                val = 3600.0  # Default to 1 hour for first events
        
        features.append(float(val))
    
    X = np.array(features, dtype=np.float32).reshape(1, -1)
    
    # Scale features
    scaler = load_scaler()
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        # If no scaler, use basic normalization
        X_scaled = X
    
    return X_scaled


def score_isolation_forest(X_scaled: np.ndarray) -> float:
    """Score using Isolation Forest. Returns anomaly probability [0, 1]."""
    try:
        model = load_isolation_forest()
        if model is None:
            return -1.0  # Model not available
        
        # decision_function: negative = anomaly, positive = normal
        score = -model.decision_function(X_scaled)[0]  # Invert so higher = more anomalous
        
        # Get threshold from training
        thresholds = load_thresholds()
        threshold = thresholds.get("iso_score_p95", 0.1)
        
        # Convert to probability using sigmoid scaling around threshold
        prob = 1.0 / (1.0 + np.exp(-5 * (score - threshold)))
        
        return float(np.clip(prob, 0.0, 1.0))
    except Exception:
        return -1.0


def score_gbm(X_scaled: np.ndarray) -> float:
    """Score using Gradient Boosting. Returns anomaly probability [0, 1]."""
    try:
        model = load_gbm()
        if model is None:
            return -1.0  # Model not available
        
        prob = model.predict_proba(X_scaled)[0, 1]
        return float(prob)
    except Exception:
        return -1.0


def score_autoencoder(X_scaled: np.ndarray) -> float:
    """Score using Autoencoder reconstruction error. Returns anomaly probability [0, 1]."""
    try:
        model = load_autoencoder()
        if model is None:
            return -1.0  # Model not available
        
        # Get reconstruction
        X_rec = model.predict(X_scaled, verbose=0)
        
        # Compute MSE
        mse = np.mean((X_scaled - X_rec) ** 2)
        
        # Get threshold from training
        thresholds = load_thresholds()
        threshold = thresholds.get("ae_mse_p95", 0.05)
        
        # Convert MSE to probability
        prob = 1.0 / (1.0 + np.exp(-10 * (mse - threshold)))
        
        return float(np.clip(prob, 0.0, 1.0))
    except Exception as e:
        # Autoencoder not available or Keras compatibility issue
        return -1.0


def apply_rule_based_scoring(event: Dict) -> Dict[str, float]:
    """
    Apply rule-based heuristics for login anomaly detection.
    
    Returns individual rule scores and flags.
    """
    rules = {}
    
    # Rule 1: Failed attempts in last 10 minutes
    failed_attempts = event.get("failed_10min", event.get("failed_attempts", 0))
    if failed_attempts >= 5:
        rules["brute_force_risk"] = min(failed_attempts / 10, 1.0)
    else:
        rules["brute_force_risk"] = failed_attempts / 10
    
    # Rule 2: Impossible travel (if provided)
    if event.get("impossible_travel", 0) == 1:
        rules["impossible_travel_risk"] = 0.9
    else:
        rules["impossible_travel_risk"] = 0.0
    
    # Rule 3: New user accessing high-degree computer
    is_new_user = event.get("is_new_user", 0)
    comp_deg = event.get("comp_deg", event.get("computer_degree", 0))
    if is_new_user == 1 and comp_deg > 100:
        rules["new_user_high_target"] = 0.7
    else:
        rules["new_user_high_target"] = 0.0
    
    # Rule 4: Off-hours login (2am - 5am)
    hour = event.get("hour_of_day", event.get("hour", 12))
    if 2 <= hour <= 5:
        rules["off_hours_risk"] = 0.4
    else:
        rules["off_hours_risk"] = 0.0
    
    # Rule 5: Device/location change
    if event.get("device_changed", event.get("is_new_comp", 0)) == 1:
        rules["device_change_risk"] = 0.3
    else:
        rules["device_change_risk"] = 0.0
    
    # Rule 6: Very long time since last login
    time_since = event.get("time_since_user_last", 0)
    if time_since > 86400 * 30:  # > 30 days
        rules["dormant_account_risk"] = 0.5
    elif time_since > 86400 * 7:  # > 7 days
        rules["dormant_account_risk"] = 0.2
    else:
        rules["dormant_account_risk"] = 0.0
    
    # Aggregate rule score
    rules["combined_rule_score"] = min(sum(rules.values()) / 3, 1.0)
    
    return rules


def score_login_event(event: Dict, include_rules: bool = True) -> Dict:
    """
    Score a login event for anomaly detection.
    
    Args:
        event: Dictionary with login event features
        include_rules: Whether to include rule-based scoring
        
    Returns:
        Dictionary with:
            - anomaly_probability: Combined score [0, 1]
            - model_scores: Individual scores from each model
            - rule_scores: Scores from rule-based heuristics (if include_rules)
            - is_anomalous: Boolean classification
            - risk_level: "low", "medium", "high", or "critical"
            - confidence: Confidence level of the prediction
    """
    # Preprocess event
    X_scaled = preprocess_login_event(event)
    
    # Get ML model scores
    ml_scores = {}
    ml_scores["isolation_forest"] = score_isolation_forest(X_scaled)
    ml_scores["gbm"] = score_gbm(X_scaled)
    ml_scores["autoencoder"] = score_autoencoder(X_scaled)
    
    # Filter available models
    available_ml = {k: v for k, v in ml_scores.items() if v >= 0}
    
    # Get rule-based scores
    rule_scores = apply_rule_based_scoring(event) if include_rules else {}
    
    # Combine scores
    if available_ml:
        # Weighted ML ensemble
        weights = {
            "isolation_forest": 1.0,
            "gbm": 1.5,
            "autoencoder": 1.0
        }
        ml_combined = sum(ml_scores[k] * weights.get(k, 1.0) 
                        for k in available_ml) / sum(weights.get(k, 1.0) 
                        for k in available_ml)
    else:
        ml_combined = 0.5  # Neutral if no models available
    
    # Combine ML with rules
    if include_rules and rule_scores:
        rule_combined = rule_scores.get("combined_rule_score", 0.0)
        # ML models get 70% weight, rules get 30%
        final_score = 0.7 * ml_combined + 0.3 * rule_combined
    else:
        final_score = ml_combined
    
    # Calculate confidence
    if len(available_ml) > 1:
        variance = np.var(list(available_ml.values()))
        confidence = 1.0 - min(variance * 4, 1.0)
    else:
        confidence = 0.6
    
    # Determine risk level
    if final_score >= 0.8:
        risk_level = "critical"
    elif final_score >= 0.6:
        risk_level = "high"
    elif final_score >= 0.3:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    return {
        "anomaly_probability": float(final_score),
        "model_scores": ml_scores,
        "rule_scores": rule_scores if include_rules else {},
        "is_anomalous": final_score >= 0.5,
        "risk_level": risk_level,
        "confidence": float(confidence),
        "models_used": list(available_ml.keys())
    }


def score_login_batch(events: List[Dict]) -> List[Dict]:
    """Score multiple login events efficiently."""
    return [score_login_event(event) for event in events]


if __name__ == "__main__":
    # Example usage
    sample_event = {
        "user_deg": 5,
        "comp_deg": 150,
        "time_since_user_last": 3600,
        "time_since_comp_last": 300,
        "hour_of_day": 3,
        "is_new_user": 0,
        "is_new_comp": 1,
        "failed_10min": 3,
        "impossible_travel": 0
    }
    
    result = score_login_event(sample_event)
    print("Login Anomaly Detection Result:")
    print(f"  Anomaly Probability: {result['anomaly_probability']:.4f}")
    print(f"  Is Anomalous: {result['is_anomalous']}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Model Scores: {result['model_scores']}")
    print(f"  Rule Scores: {result['rule_scores']}")
