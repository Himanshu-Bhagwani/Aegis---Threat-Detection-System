#!/usr/bin/env python3
"""
Score a single password using the trained password risk model.

Usage (from project root, after training):

  python3 src/passwords/score_password.py "MyP@ssw0rd!"
"""

import sys
import json
import math
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

RANDOM_STATE = 42

# Same feature logic as in password_pipeline.py
SEQUENTIAL_PATTERNS = [
    "1234", "2345", "3456", "4567", "5678", "6789",
    "abcd", "bcde", "cdef", "qwerty", "asdf", "zxcv"
]

def estimate_entropy(pw: str) -> float:
    if not pw:
        return 0.0
    charset = set(pw)
    N = len(charset)
    if N <= 1:
        return 0.0
    return len(pw) * math.log2(N)

def has_seq_pattern(pw: str) -> int:
    pw_lower = pw.lower()
    for pat in SEQUENTIAL_PATTERNS:
        if pat in pw_lower:
            return 1
    return 0

def password_to_features(pw: str) -> dict:
    length = len(pw)
    n_lower = sum(c.islower() for c in pw)
    n_upper = sum(c.isupper() for c in pw)
    n_digit = sum(c.isdigit() for c in pw)
    n_symbol = sum(not c.isalnum() for c in pw)

    has_lower = 1 if n_lower > 0 else 0
    has_upper = 1 if n_upper > 0 else 0
    has_digit = 1 if n_digit > 0 else 0
    has_symbol = 1 if n_symbol > 0 else 0

    unique_chars = len(set(pw))
    entropy_est = estimate_entropy(pw)
    seq = has_seq_pattern(pw)
    repeat_ratio = 0.0
    if length > 0:
        most_freq = max([pw.count(c) for c in set(pw)])
        repeat_ratio = most_freq / length

    return {
        "length": length,
        "n_lower": n_lower,
        "n_upper": n_upper,
        "n_digit": n_digit,
        "n_symbol": n_symbol,
        "has_lower": has_lower,
        "has_upper": has_upper,
        "has_digit": has_digit,
        "has_symbol": has_symbol,
        "unique_chars": unique_chars,
        "entropy_est": entropy_est,
        "has_seq_pattern": seq,
        "repeat_ratio": repeat_ratio,
    }

_model_cache = {}

def load_artifacts(base_dir="data/processed/passwords"):
    """Load model artifacts. Returns None if not available."""
    if "loaded" in _model_cache:
        return _model_cache.get("model"), _model_cache.get("scaler"), _model_cache.get("feature_names")
    
    model_path = Path(base_dir) / "password_model_rf.pkl"
    scaler_path = Path(base_dir) / "password_scaler.pkl"
    features_path = Path(base_dir) / "password_features.json"

    if not model_path.exists() or not scaler_path.exists() or not features_path.exists():
        _model_cache["loaded"] = True
        _model_cache["model"] = None
        return None, None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(features_path) as f:
        feature_names = json.load(f)
    
    _model_cache["loaded"] = True
    _model_cache["model"] = model
    _model_cache["scaler"] = scaler
    _model_cache["feature_names"] = feature_names
    return model, scaler, feature_names


def score_password_rule_based(pw: str) -> float:
    """Rule-based password strength scoring when ML model not available."""
    feat = password_to_features(pw)
    
    score = 0.0
    
    # Length scoring (longer = stronger = lower risk)
    if feat["length"] < 6:
        score += 0.4
    elif feat["length"] < 8:
        score += 0.2
    elif feat["length"] < 12:
        score += 0.05
    # 12+ chars gets no penalty
    
    # Complexity
    complexity = feat["has_lower"] + feat["has_upper"] + feat["has_digit"] + feat["has_symbol"]
    if complexity < 2:
        score += 0.3
    elif complexity < 3:
        score += 0.1
    
    # Sequential patterns (very bad)
    if feat["has_seq_pattern"]:
        score += 0.25
    
    # Low entropy
    if feat["entropy_est"] < 20:
        score += 0.2
    elif feat["entropy_est"] < 35:
        score += 0.1
    
    # High repeat ratio (like "aaaaaa")
    if feat["repeat_ratio"] > 0.5:
        score += 0.2
    
    # Common password patterns
    pw_lower = pw.lower()
    common_words = ["password", "123456", "qwerty", "admin", "letmein", "welcome", "monkey", "dragon"]
    for word in common_words:
        if word in pw_lower:
            score += 0.35
            break
    
    return min(1.0, score)


def score_password(pw: str) -> float:
    """Score password using ML model or fall back to rule-based."""
    model, scaler, feature_names = load_artifacts()
    
    if model is None:
        # Fall back to rule-based scoring
        return score_password_rule_based(pw)
    
    feat_dict = password_to_features(pw)
    row = [feat_dict.get(col, 0.0) for col in feature_names]
    X = np.array(row, dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prob_breached = model.predict_proba(X_scaled)[0, 1]
    return float(prob_breached)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 src/passwords/score_password.py \"MyP@ssw0rd!\"")
        sys.exit(1)
    pw = sys.argv[1]
    prob = score_password(pw)
    print(f"Password: {pw}")
    print(f"Estimated breach/weak probability: {prob:.4f}")
