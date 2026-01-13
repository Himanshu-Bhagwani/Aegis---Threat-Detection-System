#!/usr/bin/env python3
"""
GPS Spoofing Detection - Scoring Module

Score GPS trajectory windows using trained models:
- Isolation Forest (unsupervised anomaly detection)
- Gradient Boosting (supervised)
- Autoencoder (reconstruction error)
- CNN-RNN (deep learning)

Each model outputs a spoofing probability [0, 1].
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib

# Lazy imports for TensorFlow (only load when needed)
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
MODEL_DIR = Path("models/gps")
FEATURE_NAMES_PATH = Path("data/windows/gps/feature_names.json")

# Cached model instances
_models_cache = {}


def get_model_dir() -> Path:
    """Get model directory, handling both relative and absolute paths."""
    if MODEL_DIR.exists():
        return MODEL_DIR
    # Try from project root
    project_root = Path(__file__).parent.parent.parent
    return project_root / "models" / "gps"


def load_feature_names() -> List[str]:
    """Load feature names used during training."""
    if FEATURE_NAMES_PATH.exists():
        with open(FEATURE_NAMES_PATH) as f:
            return json.load(f)
    # Try from project root
    project_root = Path(__file__).parent.parent.parent
    alt_path = project_root / "data" / "windows" / "gps" / "feature_names.json"
    if alt_path.exists():
        with open(alt_path) as f:
            return json.load(f)
    # Default feature names based on typical GPS trajectory features
    return [
        "latitude", "longitude", "speed", "acceleration",
        "heading", "heading_change", "time_delta",
        "distance_delta", "sudden_jump", "impossible_speed_flag"
    ]


def load_isolation_forest():
    """Load the trained Isolation Forest model."""
    if "isolation_forest" not in _models_cache:
        model_path = get_model_dir() / "gps_isolation_forest.joblib"
        if not model_path.exists():
            _models_cache["isolation_forest"] = None
            return None
        try:
            _models_cache["isolation_forest"] = joblib.load(model_path)
        except Exception as e:
            print(f"Warning: Could not load GPS Isolation Forest model: {e}")
            _models_cache["isolation_forest"] = None
            return None
    return _models_cache.get("isolation_forest")


def load_gbm():
    """Load the trained Gradient Boosting model."""
    if "gbm" not in _models_cache:
        model_path = get_model_dir() / "gps_gbm.joblib"
        if not model_path.exists():
            _models_cache["gbm"] = None
            return None
        try:
            _models_cache["gbm"] = joblib.load(model_path)
        except Exception as e:
            print(f"Warning: Could not load GPS GBM model: {e}")
            _models_cache["gbm"] = None
            return None
    return _models_cache.get("gbm")


def load_autoencoder():
    """Load the trained Autoencoder model."""
    if "autoencoder" not in _models_cache:
        model_path = get_model_dir() / "gps_autoencoder.h5"
        if not model_path.exists():
            # Try best model
            model_path = get_model_dir() / "gps_ae_best.h5"
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
            try:
                _models_cache["autoencoder"] = keras.models.load_model(
                    str(model_path),
                    compile=False,
                    safe_mode=False
                )
            except Exception as e2:
                # Keras compatibility issue - mark as unavailable
                print(f"Warning: Could not load GPS Autoencoder model (Keras compatibility?): {e2}. Skipping.")
                _models_cache["autoencoder"] = None
                return None
    
    return _models_cache.get("autoencoder")


def load_cnn_rnn():
    """Load the trained CNN-RNN model."""
    if "cnn_rnn" not in _models_cache:
        model_path = get_model_dir() / "gps_cnn_rnn.h5"
        if not model_path.exists():
            model_path = get_model_dir() / "gps_cnn_rnn_best.h5"
        if not model_path.exists():
            # Mark as unavailable rather than raising
            _models_cache["cnn_rnn"] = None
            return None
        
        try:
            tf, keras = _load_tensorflow()
            # Try loading with compile=False to avoid metric deserialization issues
            _models_cache["cnn_rnn"] = keras.models.load_model(
                str(model_path),
                compile=False
            )
        except Exception as e:
            try:
                _models_cache["cnn_rnn"] = keras.models.load_model(
                    str(model_path),
                    compile=False,
                    safe_mode=False
                )
            except Exception as e2:
                # Keras compatibility issue - mark as unavailable
                print(f"Warning: Could not load GPS CNN-RNN model (Keras compatibility?): {e2}. Skipping.")
                _models_cache["cnn_rnn"] = None
                return None
    
    return _models_cache.get("cnn_rnn")


def make_window_level_features(X: np.ndarray) -> np.ndarray:
    """
    Aggregate over time dimension to get fixed-length features per window.
    X: (n_windows, T, F) or (T, F) -> aggregated features
    
    Computes: mean, std, max for each feature.
    """
    if X.ndim == 2:
        X = X[np.newaxis, ...]  # Add batch dimension
    
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    maxv = X.max(axis=1)
    return np.concatenate([mean, std, maxv], axis=1)


def preprocess_gps_data(trajectory: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert raw GPS trajectory to feature arrays.
    
    Args:
        trajectory: List of GPS points with lat, lng, timestamp, etc.
        
    Returns:
        X_window: Shape (T, F) for sequence models
        X_flat: Shape (F*3,) for traditional ML models
    """
    feature_names = load_feature_names()
    n_features = len(feature_names)
    T = len(trajectory)
    
    if T == 0:
        raise ValueError("Empty trajectory provided")
    
    # Build feature matrix
    X = np.zeros((T, n_features), dtype=np.float32)
    
    for t, point in enumerate(trajectory):
        for i, feat in enumerate(feature_names):
            X[t, i] = float(point.get(feat, 0.0))
    
    # Compute derived features if raw lat/lng provided
    if T > 1:
        for t in range(1, T):
            # Distance delta (simple Euclidean approximation)
            if "distance_delta" in feature_names:
                idx = feature_names.index("distance_delta")
                lat_idx = feature_names.index("latitude") if "latitude" in feature_names else -1
                lng_idx = feature_names.index("longitude") if "longitude" in feature_names else -1
                if lat_idx >= 0 and lng_idx >= 0:
                    dlat = X[t, lat_idx] - X[t-1, lat_idx]
                    dlng = X[t, lng_idx] - X[t-1, lng_idx]
                    X[t, idx] = np.sqrt(dlat**2 + dlng**2) * 111000  # Approx meters
            
            # Speed-based features
            if "sudden_jump" in feature_names:
                idx = feature_names.index("sudden_jump")
                speed_idx = feature_names.index("speed") if "speed" in feature_names else -1
                if speed_idx >= 0:
                    # Flag if speed jumps unrealistically
                    X[t, idx] = 1.0 if X[t, speed_idx] > 500 else 0.0  # 500 m/s threshold
    
    # Create aggregated features for traditional ML
    X_window = X  # (T, F)
    X_flat = make_window_level_features(X)  # (1, F*3)
    
    return X_window, X_flat.squeeze()


def score_isolation_forest(X_flat: np.ndarray) -> float:
    """Score using Isolation Forest. Returns anomaly probability [0, 1]."""
    try:
        model = load_isolation_forest()
        if model is None:
            return -1.0  # Model not available
        
        X = X_flat.reshape(1, -1)
        
        # decision_function: negative = anomaly, positive = normal
        score = model.decision_function(X)[0]
        
        # Convert to probability [0, 1] where 1 = highly anomalous
        # Typical range is [-0.5, 0.5], map to [0, 1]
        prob = 1.0 - (score + 0.5)
        return float(np.clip(prob, 0.0, 1.0))
    except Exception:
        return -1.0  # Model not available


def score_gbm(X_flat: np.ndarray) -> float:
    """Score using Gradient Boosting. Returns spoofing probability [0, 1]."""
    try:
        model = load_gbm()
        if model is None:
            return -1.0  # Model not available
        
        X = X_flat.reshape(1, -1)
        
        prob = model.predict_proba(X)[0, 1]
        return float(prob)
    except Exception:
        return -1.0


def score_autoencoder(X_window: np.ndarray) -> float:
    """Score using Autoencoder reconstruction error. Returns anomaly probability [0, 1]."""
    try:
        model = load_autoencoder()
        if model is None:
            return -1.0  # Model not available
        
        # Flatten for dense autoencoder
        T, F = X_window.shape
        X_flat = X_window.reshape(1, T * F)
        
        # Get reconstruction
        X_rec = model.predict(X_flat, verbose=0)
        
        # Compute MSE
        mse = np.mean((X_flat - X_rec) ** 2)
        
        # Convert MSE to probability using sigmoid-like scaling
        # Threshold calibrated from training (adjust based on your data)
        threshold = 0.1  # Typical threshold from training
        prob = 1.0 / (1.0 + np.exp(-10 * (mse - threshold)))
        
        return float(np.clip(prob, 0.0, 1.0))
    except Exception as e:
        # Autoencoder not available or Keras compatibility issue
        return -1.0


def score_cnn_rnn(X_window: np.ndarray) -> float:
    """Score using CNN-RNN model. Returns spoofing probability [0, 1]."""
    try:
        model = load_cnn_rnn()
        if model is None:
            return -1.0  # Model not available
        
        # Add batch dimension: (T, F) -> (1, T, F)
        X = X_window[np.newaxis, ...]
        
        prob = model.predict(X, verbose=0)[0, 0]
        return float(prob)
    except Exception as e:
        # CNN-RNN not available or Keras compatibility issue
        return -1.0


def score_gps_trajectory(trajectory: List[Dict], ensemble: bool = True) -> Dict:
    """
    Score a GPS trajectory for spoofing detection.
    
    Args:
        trajectory: List of GPS points, each with keys like:
            - latitude, longitude
            - speed, acceleration
            - heading, heading_change
            - timestamp or time_delta
        ensemble: If True, combine all model scores; else return individual scores
        
    Returns:
        Dictionary with:
            - spoof_probability: Combined or primary score [0, 1]
            - model_scores: Individual scores from each model
            - is_spoofed: Boolean classification
            - confidence: Confidence level of the prediction
    """
    # Preprocess trajectory
    X_window, X_flat = preprocess_gps_data(trajectory)
    
    # Get scores from each model
    scores = {}
    
    scores["isolation_forest"] = score_isolation_forest(X_flat)
    scores["gbm"] = score_gbm(X_flat)
    scores["autoencoder"] = score_autoencoder(X_window)
    scores["cnn_rnn"] = score_cnn_rnn(X_window)
    
    # Filter out unavailable models (-1)
    available_scores = {k: v for k, v in scores.items() if v >= 0}
    
    if not available_scores:
        return {
            "spoof_probability": 0.5,
            "model_scores": scores,
            "is_spoofed": False,
            "confidence": 0.0,
            "error": "No trained models available"
        }
    
    if ensemble:
        # Weighted ensemble: supervised models get higher weight
        weights = {
            "isolation_forest": 1.0,
            "gbm": 2.0,
            "autoencoder": 1.0,
            "cnn_rnn": 2.5  # Deep learning model typically best
        }
        
        weighted_sum = sum(scores[k] * weights.get(k, 1.0) 
                         for k in available_scores)
        total_weight = sum(weights.get(k, 1.0) for k in available_scores)
        
        combined_prob = weighted_sum / total_weight
    else:
        # Use best available model (prefer CNN-RNN > GBM > others)
        priority = ["cnn_rnn", "gbm", "autoencoder", "isolation_forest"]
        for model in priority:
            if model in available_scores:
                combined_prob = available_scores[model]
                break
    
    # Calculate confidence based on model agreement
    if len(available_scores) > 1:
        score_values = list(available_scores.values())
        variance = np.var(score_values)
        confidence = 1.0 - min(variance * 4, 1.0)  # High variance = low confidence
    else:
        confidence = 0.7  # Single model confidence
    
    return {
        "spoof_probability": float(combined_prob),
        "model_scores": scores,
        "is_spoofed": combined_prob >= 0.5,
        "confidence": float(confidence),
        "models_used": list(available_scores.keys())
    }


# Convenience function for single-point scoring (limited accuracy)
def score_single_point(lat: float, lng: float, speed: float = 0.0, 
                       heading: float = 0.0, prev_lat: float = None,
                       prev_lng: float = None) -> Dict:
    """
    Quick scoring for a single GPS point.
    Note: Trajectory-based scoring is more accurate.
    """
    point = {
        "latitude": lat,
        "longitude": lng,
        "speed": speed,
        "heading": heading,
    }
    
    # Create minimal trajectory
    trajectory = [point]
    
    if prev_lat is not None and prev_lng is not None:
        prev_point = {
            "latitude": prev_lat,
            "longitude": prev_lng,
            "speed": 0.0,
            "heading": 0.0,
        }
        trajectory = [prev_point, point]
    
    return score_gps_trajectory(trajectory)


if __name__ == "__main__":
    # Example usage
    sample_trajectory = [
        {"latitude": 37.7749, "longitude": -122.4194, "speed": 0, "heading": 0},
        {"latitude": 37.7750, "longitude": -122.4195, "speed": 5, "heading": 45},
        {"latitude": 37.7751, "longitude": -122.4196, "speed": 10, "heading": 45},
        {"latitude": 37.7752, "longitude": -122.4197, "speed": 15, "heading": 45},
        {"latitude": 37.7753, "longitude": -122.4198, "speed": 20, "heading": 45},
    ]
    
    result = score_gps_trajectory(sample_trajectory)
    print("GPS Spoofing Detection Result:")
    print(f"  Spoof Probability: {result['spoof_probability']:.4f}")
    print(f"  Is Spoofed: {result['is_spoofed']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Model Scores: {result['model_scores']}")


