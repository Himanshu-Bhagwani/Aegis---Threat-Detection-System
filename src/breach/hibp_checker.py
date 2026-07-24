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
from typing import Dict, List, Optional, Tuple

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


# Throwaway / disposable mail providers — accounts here are high-churn and are
# a common signal for fraud and abuse.
_DISPOSABLE_DOMAINS = {
    "mailinator.com", "guerrillamail.com", "10minutemail.com", "tempmail.com",
    "temp-mail.org", "throwawaymail.com", "yopmail.com", "trashmail.com",
    "sharklasers.com", "getnada.com", "dispostable.com", "fakeinbox.com",
    "maildrop.cc", "mintemail.com", "spamgourmet.com", "mailnesia.com",
}
# Free consumer providers — fine, but far more exposed in credential-stuffing
# lists than a corporate domain.
_FREEMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "icloud.com", "mail.com", "gmx.com", "yandex.com", "protonmail.com",
    "live.com", "msn.com", "rediffmail.com",
}
# Shared/role mailboxes — usually multiple readers, weak accountability.
_ROLE_PREFIXES = {
    "admin", "administrator", "info", "support", "sales", "contact", "help",
    "billing", "office", "team", "hello", "noreply", "no-reply", "root",
    "webmaster", "postmaster", "security", "abuse", "test", "demo",
}

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def assess_email_risk(email: str) -> Dict:
    """Local, always-available exposure assessment for an email address.

    HIBP's breach-by-email endpoint needs a paid key. When it isn't available
    we still return a real, explainable assessment rather than a misleading
    "Clean" — derived from the address structure and its domain.
    """
    raw    = (email or "").strip()
    lower  = raw.lower()
    valid  = bool(_EMAIL_RE.match(lower))
    local, _, domain = lower.partition("@")

    signals: List[str] = []
    risk = 0.10  # a valid address is never zero-risk — it's a public identifier

    if not valid:
        return {
            "valid": False, "risk_score": 0.0, "risk_level": "unknown",
            "signals": ["Not a valid email address format."],
            "domain": domain, "is_disposable": False, "is_freemail": False,
            "is_role_account": False,
        }

    is_disposable = domain in _DISPOSABLE_DOMAINS
    is_freemail   = domain in _FREEMAIL_DOMAINS
    is_role       = local in _ROLE_PREFIXES or any(local.startswith(p) for p in _ROLE_PREFIXES)

    if is_disposable:
        risk += 0.55
        signals.append(f"'{domain}' is a disposable/temporary mail provider — high abuse and fraud signal.")
    elif is_freemail:
        risk += 0.22
        signals.append(f"'{domain}' is a free consumer provider — heavily represented in credential-stuffing lists.")
    else:
        signals.append(f"'{domain}' is a custom/corporate domain — lower generic exposure.")

    if is_role:
        risk += 0.20
        signals.append(f"'{local}@' looks like a shared role mailbox — usually multiple readers and weaker accountability.")

    if len(local) <= 3:
        risk += 0.12
        signals.append(f"Very short local part ('{local}') — short addresses are guessed and scraped far more often.")

    if local.isalpha() and len(local) <= 6:
        risk += 0.08
        signals.append("Simple dictionary-style address — trivially enumerable.")

    if domain.count(".") == 1 and len(domain.split(".")[0]) <= 3:
        risk += 0.05
        signals.append("Very short domain — commonly used for throwaway addresses.")

    risk = max(0.0, min(1.0, risk))
    level = ("critical" if risk >= 0.75 else "high" if risk >= 0.50
             else "medium" if risk >= 0.25 else "low" if risk >= 0.10 else "minimal")

    return {
        "valid": True, "risk_score": round(risk, 4), "risk_level": level,
        "signals": signals, "domain": domain,
        "is_disposable": is_disposable, "is_freemail": is_freemail,
        "is_role_account": is_role,
    }


def check_email_breach(email: str) -> Dict:
    """
    Check if an email address appears in known data breaches.

    Uses HIBP when an API key is configured; otherwise falls back to a local
    structural/domain assessment so the caller always gets a real answer
    instead of a misleading "0 breaches / Clean".
    """
    assessment = assess_email_risk(email)

    if not HIBP_API_KEY:
        return {
            "email":          email,
            "breach_count":   0,
            "breaches":       [],
            "api_available":  False,
            "verified":       False,
            "note":           "Not verified against breach corpora (no HIBP API key) — showing local exposure assessment.",
            "assessment":     assessment,
            "risk_score":     assessment["risk_score"],
            "risk_level":     assessment["risk_level"],
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
            # Genuinely not in any known breach — but structural risk still applies.
            return {
                "email": email, "breach_count": 0, "breaches": [],
                "api_available": True, "verified": True,
                "assessment": assessment,
                "risk_score": assessment["risk_score"],
                "risk_level": assessment["risk_level"],
                "note": "Not found in any known breach.",
            }
        if resp.status_code == 401:
            return {
                "email": email, "breach_count": 0, "breaches": [],
                "api_available": False, "verified": False,
                "note": "Invalid HIBP API key — showing local exposure assessment.",
                "assessment": assessment,
                "risk_score": assessment["risk_score"],
                "risk_level": assessment["risk_level"],
            }

        breaches = resp.json()
        names    = [b.get("Name", "Unknown") for b in breaches]
        # Confirmed breaches dominate the score.
        breach_risk = min(1.0, 0.55 + 0.09 * len(names))
        combined    = max(breach_risk, assessment["risk_score"])
        level = ("critical" if combined >= 0.75 else "high" if combined >= 0.50
                 else "medium" if combined >= 0.25 else "low")
        return {
            "email":         email,
            "breach_count":  len(names),
            "breaches":      names[:10],   # top 10
            "api_available": True,
            "verified":      True,
            "assessment":    assessment,
            "risk_score":    round(combined, 4),
            "risk_level":    level,
            "note":          f"Found in {len(names)} known breach{'es' if len(names) != 1 else ''}.",
        }
    except Exception as e:
        logger.warning("HIBP email check failed: %s", e)
        return {
            "email": email, "breach_count": 0, "breaches": [],
            "api_available": False, "verified": False,
            "note": "Breach lookup unavailable — showing local exposure assessment.",
            "assessment": assessment,
            "risk_score": assessment["risk_score"],
            "risk_level": assessment["risk_level"],
        }
