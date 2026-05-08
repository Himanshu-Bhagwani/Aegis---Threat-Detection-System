"""
AEGIS Device Fingerprint Risk Scoring
=======================================
Scores the risk of a device based on:
  - Fingerprint match against known devices for the user
  - Behavioral signals (failed biometric, new device, unusual time)
  - App-lock access attempts
  - Device anomaly indicators

Works entirely with rule-based logic (no trained model required).
Integrates with DynamoDB user profiles for known-device registry.
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Fingerprint utilities
# ─────────────────────────────────────────────

def generate_device_id(fingerprint: Dict) -> str:
    """
    Generate a stable device ID from fingerprint components.
    Uses SHA-256 of sorted key-value pairs.
    """
    stable_keys = [
        "user_agent", "platform", "screen_resolution",
        "timezone", "language", "hardware_concurrency",
    ]
    parts = []
    for k in stable_keys:
        if k in fingerprint:
            parts.append(f"{k}={str(fingerprint[k]).lower()}")
    raw = "|".join(sorted(parts)) or str(fingerprint)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def fingerprint_match_score(fp_a: Dict, fp_b: Dict) -> float:
    """
    Compare two device fingerprints and return a similarity score [0-1].
    1.0 = identical, 0.0 = completely different.
    """
    if not fp_a or not fp_b:
        return 0.0

    fields = [
        ("user_agent",          1.5),
        ("platform",            1.0),
        ("screen_resolution",   0.8),
        ("timezone",            0.8),
        ("language",            0.5),
        ("hardware_concurrency",0.4),
        ("color_depth",         0.3),
        ("touch_support",       0.4),
        ("webgl_renderer",      0.8),
    ]

    total_weight = sum(w for _, w in fields)
    matched      = 0.0

    for field, weight in fields:
        a = fp_a.get(field)
        b = fp_b.get(field)
        if a is not None and b is not None:
            if str(a).lower() == str(b).lower():
                matched += weight

    return round(matched / total_weight, 4)


# ─────────────────────────────────────────────
# Main scoring function
# ─────────────────────────────────────────────

def score_device_risk(
    fingerprint:              Dict,
    known_fingerprints:       Optional[List[Dict]] = None,
    failed_biometric:         bool = False,
    failed_biometric_count:   int  = 0,
    app_unlock_attempted:     bool = False,
    app_bundle:               Optional[str] = None,
    unusual_time:             bool = False,
    location_mismatch:        bool = False,
    rooted_jailbroken:        bool = False,
    emulator_detected:        bool = False,
    vpn_detected:             bool = False,
    multiple_failures_today:  int  = 0,
) -> Dict:
    """
    Score the risk level of a device interaction.

    Args:
      fingerprint            : Current device fingerprint dict
      known_fingerprints     : List of known-good fingerprints for this user
      failed_biometric       : True if biometric auth failed this attempt
      failed_biometric_count : How many times in a row biometric has failed
      app_unlock_attempted   : True if a protected app unlock was attempted
      app_bundle             : Bundle ID of the app being unlocked
      unusual_time           : True if access is outside normal hours for this user
      location_mismatch      : True if device location differs from usual area
      rooted_jailbroken      : True if device appears rooted/jailbroken
      emulator_detected      : True if running on emulator/VM
      vpn_detected           : True if VPN is active
      multiple_failures_today: Total failed attempts today

    Returns:
      {
        device_risk_score: float [0-1],
        is_new_device:     bool,
        best_match_score:  float,
        device_id:         str,
        signals:           list[str],
        risk_level:        str,
        recommended_actions: list[str],
      }
    """
    device_id  = generate_device_id(fingerprint)
    signals:   List[str] = []
    risk_score = 0.0

    # ── New device check ──────────────────────────────
    is_new_device   = True
    best_match      = 0.0

    if known_fingerprints:
        scores = [fingerprint_match_score(fingerprint, kfp) for kfp in known_fingerprints]
        best_match = max(scores) if scores else 0.0
        is_new_device = best_match < 0.65  # below 65% match = treat as new device

    if is_new_device:
        risk_score += 0.30
        signals.append("new_device_or_fingerprint_mismatch")

    # ── Biometric failures ────────────────────────────
    if failed_biometric:
        risk_score += 0.15
        signals.append("biometric_failure")
    if failed_biometric_count >= 3:
        risk_score += 0.20
        signals.append(f"repeated_biometric_failures_{failed_biometric_count}x")
    if failed_biometric_count >= 5:
        risk_score += 0.10
        signals.append("brute_force_biometric_suspected")

    # ── App-lock access attempt ───────────────────────
    if app_unlock_attempted:
        base = 0.10
        if failed_biometric:
            base += 0.15  # failed unlock attempt on a locked app is high risk
        risk_score += base
        label = f"app_unlock_attempt:{app_bundle}" if app_bundle else "app_unlock_attempt"
        signals.append(label)

    # ── Contextual anomalies ──────────────────────────
    if unusual_time:
        risk_score += 0.10
        signals.append("unusual_access_time")

    if location_mismatch:
        risk_score += 0.15
        signals.append("location_mismatch")

    # ── Device integrity ──────────────────────────────
    if rooted_jailbroken:
        risk_score += 0.25
        signals.append("rooted_or_jailbroken_device")

    if emulator_detected:
        risk_score += 0.30
        signals.append("emulator_or_vm_detected")

    if vpn_detected:
        risk_score += 0.05
        signals.append("vpn_active")

    # ── Repeated failures ─────────────────────────────
    if multiple_failures_today >= 5:
        risk_score += 0.15
        signals.append(f"high_failure_rate_today:{multiple_failures_today}")
    elif multiple_failures_today >= 2:
        risk_score += 0.08
        signals.append(f"multiple_failures_today:{multiple_failures_today}")

    # ── Normalise ─────────────────────────────────────
    risk_score = round(min(risk_score, 1.0), 4)

    # ── Risk level ────────────────────────────────────
    if risk_score >= 0.75:
        risk_level = "critical"
    elif risk_score >= 0.50:
        risk_level = "high"
    elif risk_score >= 0.25:
        risk_level = "medium"
    elif risk_score >= 0.10:
        risk_level = "low"
    else:
        risk_level = "minimal"

    # ── Recommended actions ───────────────────────────
    actions: List[str] = []
    if is_new_device:
        actions.append("Prompt user to verify new device via email/SMS.")
    if failed_biometric_count >= 3:
        actions.append("Temporarily lock account — repeated biometric failures.")
    if rooted_jailbroken:
        actions.append("Block sensitive operations on compromised device.")
    if emulator_detected:
        actions.append("Reject request — emulator/automation detected.")
    if risk_score >= 0.75:
        actions.append("Require step-up authentication (OTP or admin approval).")

    return {
        "device_risk_score":   risk_score,
        "is_new_device":       is_new_device,
        "best_match_score":    round(best_match, 4),
        "device_id":           device_id,
        "signals":             signals,
        "risk_level":          risk_level,
        "recommended_actions": actions,
        "fingerprint_received": bool(fingerprint),
        "known_devices_checked": len(known_fingerprints) if known_fingerprints else 0,
    }


# Legacy shim
def score(fp: Dict) -> float:
    result = score_device_risk(fp)
    return result["device_risk_score"]
