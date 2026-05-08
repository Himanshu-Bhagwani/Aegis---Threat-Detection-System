#!/usr/bin/env python3
"""
GPS Spoofing Detection - Scoring Module

Feature engineering matches the training pipeline exactly:
  speed_m_s, accel, bearing_diff, dist_m, dt,
  hour, dayofweek, sudden_jump, impossible_speed_flag  (9 features)

Model input expectations:
  GBM / Isolation Forest : flat (27,)  = mean + std + max of each feature
  CNN-RNN                : (32, 9)     padded / truncated window
  Autoencoder            : (288,)      = 32 × 9 flat
"""

import math
import json
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import joblib

# ── Constants ─────────────────────────────────────────────────────────────────

FEATURE_NAMES: List[str] = [
    "speed_m_s", "accel", "bearing_diff", "dist_m", "dt",
    "hour", "dayofweek", "sudden_jump", "impossible_speed_flag",
]
N_FEATURES  = len(FEATURE_NAMES)   # 9
WINDOW_SIZE = 32                    # must match training  (CNN-RNN: (None,32,9))
N_FLAT      = N_FEATURES * 3       # 27  (mean‖std‖max)

MODEL_DIR = Path("models/gps")

_models_cache: Dict[str, object] = {}

# ── Geo helpers ───────────────────────────────────────────────────────────────

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(max(0.0, a)))


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Forward azimuth in degrees [0, 360)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x  = math.sin(dl) * math.cos(p2)
    y  = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _ang_diff(a: float, b: float) -> float:
    """Absolute angular difference [0, 180]."""
    d = abs(a - b) % 360
    return min(d, 360 - d)


def _parse_ts(point: Dict) -> Optional[float]:
    """Return Unix timestamp as float, or None."""
    ts = point.get("timestamp")
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


# ── Feature engineering ───────────────────────────────────────────────────────

def compute_features(trajectory: List[Dict]) -> np.ndarray:
    """
    Compute the 9 training features from raw GPS points.

    Each point must have 'latitude' and 'longitude'.
    'timestamp' (unix float or ISO string) is strongly recommended;
    if absent, a 1-second interval is assumed.

    Returns array of shape (T, 9).
    """
    T = len(trajectory)
    X = np.zeros((T, N_FEATURES), dtype=np.float32)

    if T == 0:
        return X

    times = [_parse_ts(p) for p in trajectory]

    # If timestamps are missing, synthesise them at 1-second intervals
    if all(t is None for t in times):
        times = [float(i) for i in range(T)]
    else:
        # Fill gaps with linear interpolation
        for i, t in enumerate(times):
            if t is None:
                # find nearest non-None neighbours
                prev = next((times[j] for j in range(i - 1, -1, -1) if times[j] is not None), None)
                nxt  = next((times[j] for j in range(i + 1, T)       if times[j] is not None), None)
                if prev is not None and nxt is not None:
                    times[i] = (prev + nxt) / 2
                elif prev is not None:
                    times[i] = prev + 1.0
                elif nxt is not None:
                    times[i] = nxt - 1.0
                else:
                    times[i] = float(i)

    prev_lat = prev_lon = prev_brng = None
    prev_speed = 0.0
    prev_ts    = None

    for t, point in enumerate(trajectory):
        lat = float(point.get("latitude", 0.0))
        lon = float(point.get("longitude", 0.0))
        ts  = times[t]

        hour_val = dayofweek_val = 0
        if ts is not None:
            try:
                dt_obj       = datetime.fromtimestamp(ts)
                hour_val     = dt_obj.hour
                dayofweek_val = dt_obj.weekday()
            except Exception:
                pass

        if prev_lat is None:
            X[t, FEATURE_NAMES.index("hour")]      = hour_val
            X[t, FEATURE_NAMES.index("dayofweek")] = dayofweek_val
            prev_lat, prev_lon = lat, lon
            prev_ts   = ts
            prev_brng = 0.0
            continue

        dt      = max(0.0, ts - prev_ts) if ts is not None and prev_ts is not None else 1.0
        dist_m  = _haversine(prev_lat, prev_lon, lat, lon)
        spd     = dist_m / dt if dt > 0 else 0.0
        brng    = _bearing(prev_lat, prev_lon, lat, lon) if dist_m > 0 else prev_brng
        brng_diff = _ang_diff(brng, prev_brng)
        accel   = (spd - prev_speed) / dt if dt > 0 else 0.0

        # Clamp to plausible ranges before writing
        X[t, FEATURE_NAMES.index("speed_m_s")]             = float(np.clip(spd,   0, 1_000))
        X[t, FEATURE_NAMES.index("accel")]                  = float(np.clip(accel, -500, 500))
        X[t, FEATURE_NAMES.index("bearing_diff")]           = float(brng_diff)
        X[t, FEATURE_NAMES.index("dist_m")]                 = float(np.clip(dist_m, 0, 1e6))
        X[t, FEATURE_NAMES.index("dt")]                     = float(np.clip(dt, 0, 3_600))
        X[t, FEATURE_NAMES.index("hour")]                   = float(hour_val)
        X[t, FEATURE_NAMES.index("dayofweek")]              = float(dayofweek_val)
        X[t, FEATURE_NAMES.index("sudden_jump")]            = 1.0 if (dist_m > 1_000 and dt < 60) else 0.0
        X[t, FEATURE_NAMES.index("impossible_speed_flag")]  = 1.0 if spd > 100 else 0.0

        prev_lat, prev_lon = lat, lon
        prev_ts    = ts
        prev_speed = spd
        prev_brng  = brng

    return X


def _make_flat(X: np.ndarray) -> np.ndarray:
    """Aggregate (T, 9) → (27,) via [mean ‖ std ‖ max]."""
    if X.shape[0] == 0:
        return np.zeros(N_FLAT, dtype=np.float32)
    return np.concatenate([X.mean(0), X.std(0), X.max(0)]).astype(np.float32)


def _pad_window(X: np.ndarray) -> np.ndarray:
    """Resize (T, 9) → (WINDOW_SIZE, 9), truncating tail or zero-padding head."""
    T = X.shape[0]
    if T >= WINDOW_SIZE:
        return X[-WINDOW_SIZE:].astype(np.float32)
    pad = np.zeros((WINDOW_SIZE - T, N_FEATURES), dtype=np.float32)
    return np.concatenate([pad, X]).astype(np.float32)


# ── Rule-based scorer (always available) ─────────────────────────────────────

def score_rule_based(X: np.ndarray) -> Tuple[float, List[str]]:
    """
    Physics-based GPS spoof probability from the 9 computed features.
    Returns (probability [0,1], list of triggered rules).
    """
    T = X.shape[0]
    if T == 0:
        return 0.5, ["empty_trajectory"]

    si = FEATURE_NAMES.index
    score     = 0.0
    triggered: List[str] = []

    # 1. Impossible speed (> 100 m/s ≈ 360 km/h)
    impos_frac = float(X[:, si("impossible_speed_flag")].mean())
    if impos_frac > 0:
        score += 0.55 * min(1.0, impos_frac * 10)
        triggered.append(f"impossible_speed ({impos_frac:.1%} of points)")

    # 2. Sudden location jumps (> 1 km in < 60 s)
    jump_frac = float(X[:, si("sudden_jump")].mean())
    if jump_frac > 0:
        score += 0.45 * min(1.0, jump_frac * 5)
        triggered.append(f"sudden_jump ({jump_frac:.1%} of points)")

    # 3. Physically unreachable acceleration (> 50 m/s² ≈ 5 g)
    max_accel = float(np.abs(X[:, si("accel")]).max())
    if max_accel > 50:
        score += 0.30
        triggered.append(f"extreme_acceleration ({max_accel:.0f} m/s²)")

    # 4. Extreme top speed
    max_speed = float(X[:, si("speed_m_s")].max())
    if max_speed > 250:   # > 900 km/h
        score += 0.40
        triggered.append(f"extreme_speed ({max_speed:.0f} m/s)")
    elif max_speed > 150: # > 540 km/h
        score += 0.20
        triggered.append(f"high_speed ({max_speed:.0f} m/s)")

    # 5. Static spoof: device moves < 1 m total over > 60 s  (lock on fake position)
    if T > 2:
        total_dist = float(X[:, si("dist_m")].sum())
        total_time = float(X[:, si("dt")].sum())
        if total_time > 60 and total_dist < 1.0:
            score += 0.50
            triggered.append(f"static_spoof ({total_dist:.2f} m in {total_time:.0f} s)")

    # 6. Pathological bearing flips (> 150° turn fraction > 20%)
    flip_frac = float((X[:, si("bearing_diff")] > 150).mean())
    if flip_frac > 0.20:
        score += 0.25
        triggered.append(f"bearing_flips ({flip_frac:.1%} of transitions)")

    # 7. Null-island proximity (all points near 0°N 0°E — typical mock GPS default)
    # Approximate via very small distances with any non-trivial time
    if T > 1:
        total_dist = float(X[:, si("dist_m")].sum())
        total_time = float(X[:, si("dt")].sum())
        # Real devices rarely stay < 5 m total over > 30 s unless indoors
        if total_dist < 5.0 and total_time > 30 and max_speed < 1.0:
            score += 0.35
            triggered.append("near_zero_movement (possible mock GPS)")

    return float(np.clip(score, 0.0, 1.0)), triggered


# ── Model loaders ─────────────────────────────────────────────────────────────

def _model_dir() -> Path:
    if MODEL_DIR.exists():
        return MODEL_DIR
    return Path(__file__).parent.parent.parent / "models" / "gps"


def _load_joblib(name: str, filename: str):
    if name not in _models_cache:
        path = _model_dir() / filename
        if not path.exists():
            _models_cache[name] = None
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _models_cache[name] = joblib.load(path)
        except Exception as e:
            print(f"Warning: could not load {filename}: {e}")
            _models_cache[name] = None
    return _models_cache[name]


def _load_keras(name: str, *filenames: str):
    if name not in _models_cache:
        for fn in filenames:
            path = _model_dir() / fn
            if not path.exists():
                continue
            try:
                import tensorflow as tf
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _models_cache[name] = tf.keras.models.load_model(
                        str(path), compile=False
                    )
                break
            except Exception as e:
                print(f"Warning: could not load {fn}: {e}")
        if name not in _models_cache:
            _models_cache[name] = None
    return _models_cache[name]


# ── Individual model scorers ──────────────────────────────────────────────────

def _score_isolation_forest(X_flat: np.ndarray) -> float:
    model = _load_joblib("isolation_forest", "gps_isolation_forest.joblib")
    if model is None:
        return -1.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = model.decision_function(X_flat.reshape(1, -1))[0]
        # decision_function: negative = anomaly; map to [0, 1]
        return float(np.clip(1.0 - (score + 0.5), 0.0, 1.0))
    except Exception:
        return -1.0


def _score_gbm(X_flat: np.ndarray) -> float:
    model = _load_joblib("gbm", "gps_gbm.joblib")
    if model is None:
        return -1.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(model.predict_proba(X_flat.reshape(1, -1))[0, 1])
    except Exception:
        return -1.0


def _score_autoencoder(X_win: np.ndarray) -> float:
    model = _load_keras("autoencoder", "gps_autoencoder.h5", "gps_ae_best.h5")
    if model is None:
        return -1.0
    try:
        flat = _pad_window(X_win).reshape(1, WINDOW_SIZE * N_FEATURES)  # (1, 288)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rec  = model.predict(flat, verbose=0)
        mse = float(np.mean((flat - rec) ** 2))
        # Calibrated threshold from training (~90th-percentile normal MSE)
        threshold = 0.05
        return float(np.clip(1.0 / (1.0 + math.exp(-10 * (mse - threshold))), 0.0, 1.0))
    except Exception:
        return -1.0


def _score_cnn_rnn(X_win: np.ndarray) -> float:
    model = _load_keras("cnn_rnn", "gps_cnn_rnn.h5", "gps_cnn_rnn_best.h5")
    if model is None:
        return -1.0
    try:
        win = _pad_window(X_win)[np.newaxis, ...]  # (1, 32, 9)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(model.predict(win, verbose=0)[0, 0])
    except Exception:
        return -1.0


# ── Public API ────────────────────────────────────────────────────────────────

def score_gps_trajectory(trajectory: List[Dict], ensemble: bool = True) -> Dict:
    """
    Score a GPS trajectory for spoofing.

    Each point should have:
        latitude, longitude           (required)
        timestamp                     (unix float or ISO string; recommended)
        speed, heading, time_delta    (optional, ignored by feature engine)

    Returns:
        spoof_probability  float [0, 1]
        risk_score         alias for spoof_probability (frontend compat)
        is_spoofed         bool
        confidence         float [0, 1]
        model_scores       dict  individual model scores
        models_used        list
        risk_factors       list  triggered rule descriptions
    """
    if not trajectory:
        return {
            "spoof_probability": 0.0, "risk_score": 0.0,
            "is_spoofed": False, "confidence": 0.0,
            "model_scores": {}, "models_used": [],
            "risk_factors": ["empty_trajectory"],
        }

    # 1. Feature engineering
    X = compute_features(trajectory)           # (T, 9)
    X_flat = _make_flat(X)                     # (27,)

    # 2. Rule-based score (always computed)
    rule_prob, triggered = score_rule_based(X)

    # 3. ML model scores
    scores: Dict[str, float] = {"rule_based": rule_prob}
    scores["isolation_forest"] = _score_isolation_forest(X_flat)
    scores["gbm"]              = _score_gbm(X_flat)
    scores["autoencoder"]      = _score_autoencoder(X)
    scores["cnn_rnn"]          = _score_cnn_rnn(X)

    available = {k: v for k, v in scores.items() if v >= 0.0}

    if ensemble:
        # Weighted ensemble; unsupported models fall back to rule_based weight
        weights = {
            "rule_based":       1.0,
            "isolation_forest": 1.5,
            "gbm":              2.5,
            "autoencoder":      1.0,
            "cnn_rnn":          3.0,
        }
        w_sum = sum(weights[k] for k in available)
        combined = sum(available[k] * weights[k] for k in available) / w_sum if w_sum else rule_prob
    else:
        priority = ["cnn_rnn", "gbm", "isolation_forest", "autoencoder", "rule_based"]
        combined = next(available[m] for m in priority if m in available)

    # 4. Confidence: inversely proportional to score variance across models
    vals = list(available.values())
    confidence = float(np.clip(1.0 - min(np.var(vals) * 4, 1.0), 0.3, 1.0)) if len(vals) > 1 else 0.6

    combined = float(np.clip(combined, 0.0, 1.0))

    return {
        "spoof_probability": combined,
        "risk_score":        combined,   # alias for frontend
        "is_spoofed":        combined >= 0.5,
        "confidence":        confidence,
        "model_scores":      scores,
        "models_used":       list(available.keys()),
        "risk_factors":      triggered,
    }


def score_single_point(
    lat: float, lng: float,
    speed: float = 0.0,
    heading: float = 0.0,
    prev_lat: Optional[float] = None,
    prev_lng: Optional[float] = None,
) -> Dict:
    """Quick scoring for a single GPS point (limited accuracy)."""
    import time as _time
    now = _time.time()
    trajectory = []
    if prev_lat is not None and prev_lng is not None:
        trajectory.append({"latitude": prev_lat, "longitude": prev_lng,
                            "timestamp": now - 10.0})
    trajectory.append({"latitude": lat, "longitude": lng,
                        "timestamp": now, "speed": speed, "heading": heading})
    return score_gps_trajectory(trajectory)


if __name__ == "__main__":
    # Quick smoke-test
    normal = [
        {"latitude": 37.7749, "longitude": -122.4194, "timestamp": 1_700_000_000.0},
        {"latitude": 37.7750, "longitude": -122.4195, "timestamp": 1_700_000_010.0},
        {"latitude": 37.7751, "longitude": -122.4196, "timestamp": 1_700_000_020.0},
    ]
    spoofed = [
        {"latitude": 37.7749, "longitude": -122.4194, "timestamp": 1_700_000_000.0},
        {"latitude": 51.5074, "longitude":   -0.1278, "timestamp": 1_700_000_005.0},  # NYC → London
    ]

    for label, traj in [("normal", normal), ("spoofed", spoofed)]:
        r = score_gps_trajectory(traj)
        print(f"\n[{label}]")
        print(f"  spoof_probability : {r['spoof_probability']:.4f}")
        print(f"  is_spoofed        : {r['is_spoofed']}")
        print(f"  risk_factors      : {r['risk_factors']}")
        print(f"  models_used       : {r['models_used']}")
