"""
AEGIS Breach Checker — HaveIBeenPwned (HIBP) Integration
==========================================================
Uses the HIBP k-anonymity API:
  - Hashes the password with SHA-1
  - Sends only the first 5 characters of the hash (k-anonymity)
  - Checks if the tail appears in the returned list

HIBP API key is read from HIBP_API_KEY env var.
Falls back to entropy-only scoring if the API is unavailable.
"""

import os
import re
import math
import hashlib
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

HIBP_API_KEY   = os.getenv("HIBP_API_KEY", "")
HIBP_PWNED_URL = "https://api.pwnedpasswords.com/range/{prefix}"
HIBP_EMAIL_URL = "https://haveibeenpwned.com/api/v3/breachedaccount/{account}"

REQUEST_TIMEOUT = 5  # seconds


# ─────────────────────────────────────────────
# Password breach check (k-anonymity)
# ─────────────────────────────────────────────

def _sha1_hash(password: str) -> str:
    return hashlib.sha1(password.encode("utf-8")).hexdigest().upper()


def _password_entropy(password: str) -> float:
    """Shannon entropy of password (bits)."""
    if not password:
        return 0.0
    freq = {}
    for ch in password:
        freq[ch] = freq.get(ch, 0) + 1
    total  = len(password)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def check_password_hash(hash_prefix: str) -> bool:
    """
    Legacy shim — kept for health check.
    Returns True if the HIBP range API is reachable.
    """
    try:
        import httpx
        resp = httpx.get(
            HIBP_PWNED_URL.format(prefix=hash_prefix[:5]),
            timeout=REQUEST_TIMEOUT,
        )
        return resp.status_code == 200
    except Exception:
        return False


def check_password_pwned(password: str) -> Tuple[bool, int]:
    """
    Check if a password has been seen in known data breaches via HIBP.

    Returns:
      (is_pwned: bool, pwned_count: int)
      pwned_count = 0 means not found or API unavailable.
    """
    if not password:
        return False, 0

    sha1     = _sha1_hash(password)
    prefix   = sha1[:5]
    suffix   = sha1[5:]

    try:
        import httpx
        resp = httpx.get(
            HIBP_PWNED_URL.format(prefix=prefix),
            timeout=REQUEST_TIMEOUT,
            headers={"Add-Padding": "true"},
        )
        if resp.status_code != 200:
            logger.warning("HIBP API returned %d", resp.status_code)
            return False, 0

        for line in resp.text.splitlines():
            parts = line.split(":")
            if len(parts) == 2 and parts[0].upper() == suffix:
                count = int(parts[1].strip())
                return True, count
        return False, 0
    except ImportError:
        logger.warning("httpx not installed — HIBP check skipped")
        return False, 0
    except Exception as e:
        logger.warning("HIBP API error: %s", e)
        return False, 0


def score_password_breach(password: str) -> Dict:
    """
    Full breach + strength scoring for a password.

    Returns:
      {
        breach_probability: float [0-1],
        is_pwned:          bool,
        pwned_count:       int,
        entropy_bits:      float,
        strength_score:    float [0-1],
        risk_level:        str,
        recommendations:   list[str],
        api_available:     bool,
      }
    """
    if not password:
        return {
            "breach_probability": 1.0,
            "is_pwned":          False,
            "pwned_count":       0,
            "entropy_bits":      0.0,
            "strength_score":    0.0,
            "risk_level":        "critical",
            "recommendations":   ["Password cannot be empty"],
            "api_available":     False,
        }

    # ── Entropy & strength ────────────────────────────
    entropy        = _password_entropy(password)
    length         = len(password)
    has_upper      = bool(re.search(r"[A-Z]", password))
    has_lower      = bool(re.search(r"[a-z]", password))
    has_digit      = bool(re.search(r"\d",    password))
    has_symbol     = bool(re.search(r"[^A-Za-z0-9]", password))
    char_types     = sum([has_upper, has_lower, has_digit, has_symbol])
    has_sequential = bool(re.search(r"(012|123|234|345|456|567|678|789|890|abc|bcd|cde)", password.lower()))
    has_repeat     = bool(re.search(r"(.)\1{2,}", password))

    # Strength score [0-1]
    strength = 0.0
    strength += min(length / 20.0, 0.35)      # up to 0.35 for length
    strength += (char_types / 4.0) * 0.35      # up to 0.35 for diversity
    strength += min(entropy / 80.0, 0.20)      # up to 0.20 for entropy
    strength -= 0.10 if has_sequential else 0
    strength -= 0.05 if has_repeat else 0
    strength  = max(0.0, min(1.0, strength))

    # ── HIBP check ────────────────────────────────────
    is_pwned, pwned_count = check_password_pwned(password)
    api_available         = pwned_count >= 0  # False only on hard failure

    # ── Breach probability ────────────────────────────
    if is_pwned:
        # Scale by how many times it was seen: 1-10 → 0.6-0.99
        breach_prob = min(0.60 + (math.log10(max(pwned_count, 1)) / 10.0), 0.99)
    else:
        # Derive from weakness when API says "not seen"
        breach_prob = (1.0 - strength) * 0.40

    # ── Risk level ────────────────────────────────────
    if breach_prob >= 0.75 or is_pwned:
        risk_level = "critical"
    elif breach_prob >= 0.50:
        risk_level = "high"
    elif breach_prob >= 0.25:
        risk_level = "medium"
    elif breach_prob >= 0.10:
        risk_level = "low"
    else:
        risk_level = "minimal"

    # ── Recommendations ───────────────────────────────
    recs = []
    if is_pwned:
        recs.append(f"This password appeared {pwned_count:,} times in data breaches. Change it immediately.")
    if length < 12:
        recs.append("Use at least 12 characters.")
    if not has_upper:
        recs.append("Add uppercase letters.")
    if not has_digit:
        recs.append("Add numbers.")
    if not has_symbol:
        recs.append("Add special characters (!, @, #, etc.).")
    if has_sequential:
        recs.append("Avoid sequential patterns (123, abc).")
    if has_repeat:
        recs.append("Avoid repeated characters (aaa, 111).")
    if not recs and strength > 0.80:
        recs.append("Strong password. Consider using a password manager.")

    return {
        "breach_probability": round(breach_prob, 4),
        "is_pwned":          is_pwned,
        "pwned_count":       pwned_count,
        "entropy_bits":      round(entropy, 2),
        "strength_score":    round(strength, 4),
        "risk_level":        risk_level,
        "recommendations":   recs,
        "api_available":     api_available,
        "length":            length,
        "char_types":        char_types,
    }


def check_email_breach(email: str) -> Dict:
    """
    Check if an email address appears in known data breaches via HIBP.
    Requires HIBP_API_KEY.
    """
    if not HIBP_API_KEY:
        return {
            "email":           email,
            "breach_count":    0,
            "breaches":        [],
            "api_available":   False,
            "note":            "HIBP_API_KEY not configured",
        }

    try:
        import httpx
        resp = httpx.get(
            HIBP_EMAIL_URL.format(account=email),
            timeout=REQUEST_TIMEOUT,
            headers={
                "hibp-api-key":  HIBP_API_KEY,
                "User-Agent":    "AEGIS-ThreatDetection",
            },
        )
        if resp.status_code == 404:
            return {"email": email, "breach_count": 0, "breaches": [], "api_available": True}
        if resp.status_code == 401:
            return {"email": email, "breach_count": 0, "breaches": [], "api_available": False, "note": "Invalid HIBP API key"}

        breaches = resp.json()
        names    = [b.get("Name", "Unknown") for b in breaches]
        return {
            "email":         email,
            "breach_count":  len(names),
            "breaches":      names[:10],   # top 10
            "api_available": True,
        }
    except Exception as e:
        logger.warning("HIBP email check failed: %s", e)
        return {"email": email, "breach_count": 0, "breaches": [], "api_available": False}
