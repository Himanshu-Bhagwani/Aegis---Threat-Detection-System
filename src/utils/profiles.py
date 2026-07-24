"""
Per-user behavioral profile aggregation
========================================
Turns the stream of scored detection events into a durable, per-user profile
that the dashboard can read — the "real data" behind each user card.

Each profile row lives in the profiles table (PK = tenant-scoped user_id) and
holds the latest score for every detection module plus login bookkeeping:

  user_id (scoped)  tenant_id  name  email
  gps_spoof  login_anomaly  password_leak  fraud_risk  breach_risk
  unified_score  risk_level  login_count  last_login  created_at  last_updated

register_profile()      → called when a user signs in (sets name/email)
update_from_event()     → called after every scored event (updates module scores)
get_tenant_profiles()   → lists a tenant's profiles for the dashboard
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.utils.aws_utils import (
    get_resource,
    dynamo_get_user_profile,
    dynamo_update_user_profile,
    DYNAMODB_PROFILES_TABLE,
)

logger = logging.getLogger(__name__)

# Fusion weights — kept in sync with src/fusion/risk_scoring.py and the frontend.
# Login anomaly and fraud dominate — they're the signals of an account actually
# under attack. GPS / password / breach contribute, but far less. Tuned so two
# strong signals (login + fraud both ~90%) land the unified score around 75%.
_W = {"gps_spoof": 0.6, "login_anomaly": 3.5, "password_leak": 0.4, "fraud_risk": 3.5, "breach_risk": 0.7}
_W_TOTAL = sum(_W.values())  # 8.7

_MODULE_FIELDS = ["gps_spoof", "login_anomaly", "password_leak", "fraud_risk", "breach_risk"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def score_to_level(s: float) -> str:
    if s >= 0.75: return "critical"
    if s >= 0.50: return "high"
    if s >= 0.25: return "medium"
    if s >= 0.10: return "low"
    return "minimal"


def _compute_unified(m: Dict[str, float]) -> float:
    """Combine module scores into one risk figure.

    Two parts, take the higher:
      • a login/fraud-dominant weighted average across all modules (so two strong
        signals like login+fraud at ~90% land near 75%, not dragged to ~90% or
        diluted to ~45%); and
      • a floor of 0.6 × the single strongest module, so one critical signal on
        its own (e.g. a lone fraudulent transaction) still surfaces as high
        rather than being washed out by the clean modules.
    """
    vals = {k: max(0.0, min(1.0, float(m.get(k, 0.0) or 0.0))) for k in _W}
    weighted  = sum(vals[k] * _W[k] for k in _W) / _W_TOTAL
    strongest = max(vals.values()) if vals else 0.0
    return round(max(weighted, 0.6 * strongest), 4)


def _to_profile_number(obj: Any) -> Any:
    """DynamoDB rejects raw floats — mirror aws_utils' Decimal conversion."""
    from decimal import Decimal
    if isinstance(obj, float):
        return Decimal(str(round(obj, 6)))
    if isinstance(obj, dict):
        return {k: _to_profile_number(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_profile_number(v) for v in obj]
    return obj


def _from_profile_number(obj: Any) -> Any:
    from decimal import Decimal
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _from_profile_number(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_profile_number(v) for v in obj]
    return obj


def _blank_profile(user_id: str, tenant_id: str) -> Dict[str, Any]:
    now = _now()
    return {
        "user_id":       user_id,
        "tenant_id":     tenant_id,
        "name":          "",
        "email":         "",
        "gps_spoof":     0.0,
        "login_anomaly": 0.0,
        "password_leak": 0.0,
        "fraud_risk":    0.0,
        "breach_risk":   0.0,
        "unified_score": 0.0,
        "risk_level":    "minimal",
        "login_count":   0,
        "last_login":    None,
        "created_at":    now,
        "last_updated":  now,
    }


def register_profile(
    tenant_id: str,
    user_id_scoped: str,
    name: str = "",
    email: str = "",
) -> Optional[Dict[str, Any]]:
    """Create the profile if new, or refresh its name/email. Called on sign-in."""
    existing = dynamo_get_user_profile(user_id_scoped)
    profile  = existing or _blank_profile(user_id_scoped, tenant_id)

    profile["tenant_id"] = tenant_id
    if name:
        profile["name"] = name
    if email:
        profile["email"] = email
    profile.setdefault("created_at", _now())

    dynamo_update_user_profile(user_id_scoped, profile)
    return profile


# event_type → the module field(s) it updates, and where to read the score from.
def _scores_from_event(event_type: str, result: Dict[str, Any]) -> Dict[str, float]:
    """Return {module_field: score} contributed by this event."""
    def g(*keys, default=None):
        for k in keys:
            v = result.get(k)
            if v is not None:
                return float(v)
        return default

    if event_type == "unified_risk":
        # The unified endpoint reports every sub-score, but returns 0.0 for the
        # modules that weren't part of *this* request. Only take non-zero
        # values, otherwise a routine login would wipe out a user's previously
        # recorded fraud and breach scores.
        out = {}
        for field, key in [
            ("gps_spoof", "gps_risk"), ("login_anomaly", "login_risk"),
            ("password_leak", "password_risk"), ("fraud_risk", "fraud_risk"),
            ("breach_risk", "breach_risk"),
        ]:
            v = result.get(key)
            if v is not None and float(v) > 0:
                out[field] = float(v)
        return out
    if event_type == "login_score":
        v = g("anomaly_probability", "login_risk")
        return {"login_anomaly": v} if v is not None else {}
    if event_type == "gps_score":
        v = g("spoof_probability", "risk_score", "gps_risk")
        return {"gps_spoof": v} if v is not None else {}
    if event_type == "fraud_score":
        v = g("fraud_probability", "fraud_risk")
        return {"fraud_risk": v} if v is not None else {}
    if event_type == "breach_score":
        v = g("breach_probability", "breach_risk")
        return {"breach_risk": v} if v is not None else {}
    if event_type == "password_score":
        v = g("password_risk", "risk_score")
        return {"password_leak": v} if v is not None else {}
    return {}


def update_from_event(
    tenant_id: str,
    user_id_scoped: str,
    event_type: str,
    result: Dict[str, Any],
    name: str = "",
    email: str = "",
) -> Optional[Dict[str, Any]]:
    """Merge a scored event into the user's profile and recompute the unified score."""
    updates = _scores_from_event(event_type, result)

    existing = dynamo_get_user_profile(user_id_scoped)
    profile  = existing or _blank_profile(user_id_scoped, tenant_id)
    profile["tenant_id"] = tenant_id
    if name:  profile["name"]  = name
    if email: profile["email"] = email

    for field, value in updates.items():
        profile[field] = max(0.0, min(1.0, value))

    # Keep the detail from the last password breach check so the dashboard can
    # show "weak password / no special character / pwned" without re-asking for it.
    if event_type == "breach_score" and "strength_score" in result:
        profile["password_meta"] = _to_profile_number({
            "is_pwned":       bool(result.get("is_pwned")),
            "pwned_count":    int(result.get("pwned_count") or 0),
            "strength_score": float(result.get("strength_score") or 0.0),
            "entropy_bits":   float(result.get("entropy_bits") or 0.0),
            "length":         int(result.get("length") or 0),
            "char_types":     int(result.get("char_types") or 0),
            "risk_level":     result.get("risk_level", "unknown"),
            "recommendations": list(result.get("recommendations") or [])[:6],
            "checked_at":     _now(),
        })

    # Login bookkeeping — count real sign-ins.
    if event_type in ("login_score", "unified_risk"):
        profile["login_count"] = int(profile.get("login_count", 0)) + 1
        profile["last_login"]  = _now()

    # Always recompute from the profile's accumulated module scores. The
    # endpoint's own unified_score only reflects the modules in that single
    # request, whereas the profile should express the user's overall risk
    # across every signal seen so far (login + fraud + breach + GPS).
    profile["unified_score"] = _compute_unified({k: profile.get(k, 0.0) for k in _MODULE_FIELDS})

    profile["risk_level"]   = score_to_level(profile["unified_score"])
    profile["last_updated"] = _now()

    dynamo_update_user_profile(user_id_scoped, profile)
    return profile


def to_dashboard_profile(row: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
    """Shape a stored profile row into what the Apeilo dashboard expects."""
    scoped = row.get("user_id", "")
    prefix = f"{tenant_id}#"
    raw_id = scoped[len(prefix):] if scoped.startswith(prefix) else scoped
    return {
        "id":         raw_id,
        "name":       row.get("name") or raw_id,
        "email":      row.get("email", ""),
        "created_at": row.get("created_at", ""),
        "is_demo":    False,
        "notes":      f"Live profile · {int(row.get('login_count', 0))} logins",
        "login_count": int(row.get("login_count", 0)),
        "last_login":  row.get("last_login"),
        # Detail from this user's most recent password breach check, if any.
        "password_meta": _from_profile_number(row.get("password_meta")) if row.get("password_meta") else None,
        "metrics": {
            "gps_spoof":     float(row.get("gps_spoof", 0.0)),
            "login_anomaly": float(row.get("login_anomaly", 0.0)),
            "password_leak": float(row.get("password_leak", 0.0)),
            "fraud_risk":    float(row.get("fraud_risk", 0.0)),
            "breach_risk":   float(row.get("breach_risk", 0.0)),
            "unified_score": float(row.get("unified_score", 0.0)),
            "risk_level":    row.get("risk_level", "minimal"),
            "last_updated":  row.get("last_updated", ""),
        },
    }


# ═══════════════════════════════════════════════════════
# Personal login-hour histogram (adaptive baseline)
# ═══════════════════════════════════════════════════════

# Generic risk-by-hour prior, used before we know anything about a user.
# Mirrors the frontend curve: dangerous overnight, safe during business hours.
HOUR_BASELINE = [
    0.72, 0.81, 0.90, 0.95, 0.88, 0.60,   # 00-05 night
    0.28, 0.18, 0.14, 0.12, 0.13, 0.15,   # 06-11 morning
    0.16, 0.17, 0.18, 0.19, 0.21, 0.24,   # 12-17 business hours
    0.30, 0.35, 0.42, 0.52, 0.63, 0.70,   # 18-23 evening
]

_LOGIN_EVENT_TYPES = ("unified_risk", "login_score")

# How much a user's own habit can suppress the generic prior.
_MAX_FAMILIARITY_DISCOUNT = 0.85
# Logins at one hour before that hour counts as fully "normal for this user".
_FAMILIARITY_SATURATION = 5


def _hour_label(h: int) -> str:
    suffix = "AM" if h < 12 else "PM"
    hour12 = h % 12 or 12
    return f"{hour12} {suffix}"


def login_stats(tenant_id: str, user_id_scoped: str, limit: int = 400) -> Dict[str, Any]:
    """Build a per-user login-hour histogram with an adapted risk curve.

    The generic prior says "3 AM is dangerous" — but if *this* user signs in at
    3 AM every night, it isn't dangerous for them. Each hour's risk is
    discounted by how habitually the user logs in at that hour, so the chart
    adapts to real behaviour instead of showing a fixed curve.
    """
    from src.utils.aws_utils import dynamo_get_user_events

    events = dynamo_get_user_events(user_id_scoped, limit=limit) or []
    logins = [e for e in events if e.get("event_type") in _LOGIN_EVENT_TYPES]

    counts = [0] * 24          # successful sign-ins per hour (drives familiarity)
    failed_counts = [0] * 24   # failed attempts per hour (never lowers risk)
    failed_total = 0
    success_total = 0
    recent: List[Dict[str, Any]] = []

    for e in logins:
        scores = e.get("scores") or {}

        # An explicit hour means a caller deliberately reported a sign-in
        # attempt (internal score recomputes no longer record one).
        hour = scores.get("hour_of_day")
        if hour is None:
            continue
        hour = max(0, min(23, int(hour)))

        failed  = int(scores.get("failed_10min") or 0)
        outcome = scores.get("login_success")

        if outcome is None:
            # ── Legacy event (recorded before the success flag existed) ──
            # 12 used to be the field default, so legacy events at exactly
            # 12:00 are indistinguishable from internal recomputes — skip them
            # rather than invent logins the user never made.
            if hour == 12:
                continue
            # Otherwise infer: a clean attempt is a sign-in, one carrying
            # failures in the same window is a failed attempt.
            succeeded = failed == 0
        else:
            succeeded = bool(outcome)

        if succeeded:
            # Only a SUCCESSFUL sign-in makes an hour "normal for you".
            counts[hour] += 1
            success_total += 1
        else:
            failed_counts[hour] += 1
            failed_total += 1

        recent.append({
            "timestamp":       e.get("timestamp"),
            "hour":            hour,
            "hour_label":      _hour_label(hour),
            "succeeded":       succeeded,
            "failed_attempts": failed,
            "risk": float(
                scores.get("unified_score")
                or scores.get("anomaly_probability")
                or 0.0
            ),
            "event_type": e.get("event_type"),
        })

    recent.sort(key=lambda r: r.get("timestamp") or "", reverse=True)

    total = success_total
    peak  = max(counts) if total else 0

    hours = []
    for h in range(24):
        baseline = HOUR_BASELINE[h]
        # 0..1 — how habitual this hour is, from SUCCESSFUL sign-ins only.
        familiarity = min(1.0, counts[h] / _FAMILIARITY_SATURATION) if counts[h] else 0.0
        adapted = baseline * (1.0 - _MAX_FAMILIARITY_DISCOUNT * familiarity)
        hours.append({
            "hour":         h,
            "label":        f"{h:02d}:00",
            "count":        counts[h],          # successful sign-ins
            "failed_count": failed_counts[h],   # shown, but never lowers risk
            "baseline":     round(baseline, 4),
            "risk":         round(adapted, 4),
            "familiarity":  round(familiarity, 3),
            "is_usual":     bool(peak and counts[h] == peak and counts[h] > 0),
        })

    usual_hour = counts.index(peak) if total and peak > 0 else None

    return {
        "user_id":       user_id_scoped.split("#", 1)[-1],
        "tenant_id":     tenant_id,
        "total_logins":  total,
        "failed_total":  failed_total,
        "usual_hour":    usual_hour,
        "usual_label":   _hour_label(usual_hour) if usual_hour is not None else None,
        "usual_count":   peak if usual_hour is not None else 0,
        "hours":         hours,
        "recent":        recent[:20],
        # Below this many logins the curve is still mostly the generic prior.
        "is_personalised": total >= 3,
    }


# ═══════════════════════════════════════════════════════
# Transaction / fraud history
# ═══════════════════════════════════════════════════════

_RISKY_THRESHOLD = 0.60


# ═══════════════════════════════════════════════════════
# Location history / impossible-travel
# ═══════════════════════════════════════════════════════

# Coordinates are stored rounded to 3 decimals (~110 m). That is far more
# precision than impossible-travel needs, while avoiding retaining a precise
# movement history for every user.
GEO_PRECISION = 3

# Civil aviation tops out around 900 km/h; beyond this no legitimate journey
# explains the gap between two sign-ins.
IMPOSSIBLE_KMH = 1000.0
# Faster than road or rail — only a flight could explain it. Worth noting.
FLIGHT_KMH = 400.0
# Below this gap between sign-ins, speed is meaningless (division by ~zero).
MIN_GAP_HOURS = 1.0 / 60.0        # 1 minute
# …so within that window only a jump this large counts as impossible. Smaller
# hops are almost always a network change or a coarse location fix.
INSTANT_JUMP_KM = 50.0


def round_coord(v: float) -> float:
    return round(float(v), GEO_PRECISION)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    import math
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def last_login_location(user_id_scoped: str, limit: int = 60) -> Optional[Dict[str, Any]]:
    """The most recent sign-in location recorded for this user, if any."""
    from src.utils.aws_utils import dynamo_get_user_events

    for e in (dynamo_get_user_events(user_id_scoped, limit=limit) or []):
        if e.get("event_type") not in _LOGIN_EVENT_TYPES:
            continue
        s = e.get("scores") or {}
        if s.get("geo_lat") is None or s.get("geo_lng") is None:
            continue
        try:
            return {
                "lat":       float(s["geo_lat"]),
                "lng":       float(s["geo_lng"]),
                "timestamp": e.get("timestamp"),
            }
        except Exception:
            continue
    return None


def assess_travel(prev: Dict[str, Any], lat: float, lng: float, now_iso: str) -> Dict[str, Any]:
    """Compare this sign-in against the previous one's location and time."""
    out: Dict[str, Any] = {"impossible_travel": False}
    try:
        t_prev = datetime.fromisoformat(str(prev["timestamp"]).replace("Z", "+00:00"))
        t_now  = datetime.fromisoformat(str(now_iso).replace("Z", "+00:00"))
        hours  = max(0.0, (t_now - t_prev).total_seconds() / 3600.0)
    except Exception:
        return out

    km = haversine_km(prev["lat"], prev["lng"], lat, lng)
    out.update({
        "distance_km":   round(km, 1),
        "hours_elapsed": round(hours, 2),
        "from_lat": prev["lat"], "from_lng": prev["lng"],
    })

    # Same place (within rounding) — nothing to say.
    if km < 1.0:
        out["implied_kmh"] = 0.0
        out["travel_verdict"] = "plausible"
        return out

    # Back-to-back sign-ins can't yield a meaningful speed (dividing by ~zero
    # makes everything look impossible). A few kilometres of drift here is far
    # more likely a different network or a coarse fix than teleportation, so
    # only a genuinely large jump counts.
    if hours <= MIN_GAP_HOURS:
        out["implied_kmh"] = None
        if km >= INSTANT_JUMP_KM:
            out["impossible_travel"] = True
            out["travel_verdict"] = "impossible"
        else:
            out["travel_verdict"] = "plausible"
        return out

    speed = km / hours
    out["implied_kmh"] = round(speed, 1)

    if speed >= IMPOSSIBLE_KMH:
        out["impossible_travel"] = True
        out["travel_verdict"] = "impossible"
    elif speed >= FLIGHT_KMH:
        out["travel_verdict"] = "flight_speed"
    else:
        out["travel_verdict"] = "plausible"
    return out


def transaction_baseline(user_id_scoped: str, limit: int = 100) -> Dict[str, Any]:
    """A robust picture of what 'normal spending' looks like for this user.

    Uses the MEDIAN rather than the mean: one huge transaction would otherwise
    drag the average up so far that the *next* huge transaction looks ordinary
    (a real defect — a ₹5L outlier made a following ₹6L score lower).

    Transactions the user has explicitly confirmed as theirs are folded into
    the baseline, so an approved large payment stops being treated as an
    anomaly and raises what counts as normal.
    """
    from src.utils.aws_utils import dynamo_get_user_events

    events = dynamo_get_user_events(user_id_scoped, limit=limit) or []

    amounts: List[float] = []
    confirmed: List[float] = []
    hourly_counts: Dict[str, int] = {}

    # Amounts the user vouched for — these become part of "normal".
    for e in events:
        if e.get("event_type") != "activity_verification":
            continue
        s = e.get("scores") or {}
        if s.get("confirmed") and s.get("amount"):
            try:
                confirmed.append(float(s["amount"]))
            except Exception:
                pass

    for e in events:
        if e.get("event_type") != "fraud_score":
            continue
        s = e.get("scores") or {}
        if "amount" not in s:
            continue
        try:
            amt = float(s["amount"])
        except Exception:
            continue
        if amt <= 0:
            continue
        amounts.append(amt)
        ts = (e.get("timestamp") or "")[:13]   # hour bucket, e.g. 2026-07-21T09
        if ts:
            hourly_counts[ts] = hourly_counts.get(ts, 0) + 1

    # Median of everything seen so far (outlier-resistant).
    median = 0.0
    if amounts:
        ordered = sorted(amounts)
        mid = len(ordered) // 2
        median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2

    # A confirmed large payment lifts the bar for what looks normal.
    if confirmed:
        median = max(median, sum(confirmed) / len(confirmed) * 0.5)

    # Typical transactions-per-hour (median across active hours).
    rates = sorted(hourly_counts.values())
    typical_per_hour = 0.0
    if rates:
        m = len(rates) // 2
        typical_per_hour = rates[m] if len(rates) % 2 else (rates[m - 1] + rates[m]) / 2

    return {
        "median":           round(median, 2),
        "count":            len(amounts),
        "typical_per_hour": typical_per_hour,
        "confirmed_count":  len(confirmed),
        "max_confirmed":    max(confirmed) if confirmed else 0.0,
    }


def prior_transaction_amounts(user_id_scoped: str, limit: int = 100) -> List[float]:
    """Amounts of this user's previously scored transactions (newest first)."""
    from src.utils.aws_utils import dynamo_get_user_events
    events = dynamo_get_user_events(user_id_scoped, limit=limit) or []
    out: List[float] = []
    for e in events:
        if e.get("event_type") != "fraud_score":
            continue
        s = e.get("scores") or {}
        if "amount" in s:
            try:
                amt = float(s["amount"])
                if amt > 0:
                    out.append(amt)
            except Exception:
                continue
    return out


def transaction_stats(tenant_id: str, user_id_scoped: str, limit: int = 200) -> Dict[str, Any]:
    """Real transaction history for the fraud view.

    Returns the recent transactions actually scored for this user, plus the
    aggregates the dashboard shows: average/peak amount, transactions per hour,
    and how risky transactions compare with this user's normal spend.
    """
    from src.utils.aws_utils import dynamo_get_user_events

    events = dynamo_get_user_events(user_id_scoped, limit=limit) or []
    txns   = [e for e in events if e.get("event_type") == "fraud_score"]

    recent: List[Dict[str, Any]] = []
    for e in txns:
        s = e.get("scores") or {}
        if "amount" not in s:
            continue   # scored without a transaction body — nothing to chart
        risk = float(s.get("fraud_probability") or 0.0)
        recent.append({
            "timestamp":        e.get("timestamp"),
            "amount":           float(s.get("amount") or 0),
            "hour":             int(s.get("hour")) if s.get("hour") is not None else None,
            "risk":             round(risk, 4),
            "risk_level":       score_to_level(risk),
            "is_international": bool(s.get("is_international")),
            "is_risky":         risk >= _RISKY_THRESHOLD,
        })

    recent.sort(key=lambda t: t.get("timestamp") or "", reverse=True)

    amounts = [t["amount"] for t in recent]
    total   = len(recent)
    risky   = [t for t in recent if t["is_risky"]]
    normal  = [t for t in recent if not t["is_risky"]]

    avg_amount        = (sum(amounts) / total) if total else 0.0
    avg_normal_amount = (sum(t["amount"] for t in normal) / len(normal)) if normal else 0.0
    avg_risky_amount  = (sum(t["amount"] for t in risky) / len(risky)) if risky else 0.0

    # How many times larger a risky transaction is than this user's normal one.
    amount_ratio = (avg_risky_amount / avg_normal_amount) if avg_normal_amount > 0 and risky else 0.0

    # How far the single biggest transaction stands out from everything else —
    # the clearest "this one doesn't belong" signal.
    largest_txn = max(recent, key=lambda t: t["amount"]) if recent else None
    rest = sorted(amounts, reverse=True)[1:]
    avg_rest = (sum(rest) / len(rest)) if rest else 0.0
    largest_vs_rest = (max(amounts) / avg_rest) if rest and avg_rest > 0 else 0.0

    # Velocity: transactions in the last hour, by event time.
    per_hour = 0
    if recent:
        try:
            newest = datetime.fromisoformat((recent[0]["timestamp"] or "").replace("Z", "+00:00"))
            for t in recent:
                ts = datetime.fromisoformat((t["timestamp"] or "").replace("Z", "+00:00"))
                if (newest - ts).total_seconds() <= 3600:
                    per_hour += 1
        except Exception:
            per_hour = 0

    by_hour = [0.0] * 24
    for t in recent:
        if t["hour"] is not None:
            by_hour[max(0, min(23, t["hour"]))] += t["amount"]

    return {
        "user_id":            user_id_scoped.split("#", 1)[-1],
        "tenant_id":          tenant_id,
        "total_transactions": total,
        "risky_count":        len(risky),
        "risky_ratio":        round(len(risky) / total, 4) if total else 0.0,
        "avg_amount":         round(avg_amount, 2),
        "avg_normal_amount":  round(avg_normal_amount, 2),
        "avg_risky_amount":   round(avg_risky_amount, 2),
        "max_amount":         round(max(amounts), 2) if amounts else 0.0,
        "total_amount":       round(sum(amounts), 2),
        "amount_ratio":       round(amount_ratio, 2),
        # Biggest transaction vs the average of all the others.
        "largest_vs_rest":    round(largest_vs_rest, 2),
        "avg_excluding_largest": round(avg_rest, 2),
        "largest_txn":        largest_txn,
        "per_hour":           per_hour,
        "amount_by_hour":     [round(v, 2) for v in by_hour],
        "recent":             recent[:25],
    }


def delete_profile(tenant_id: str, user_id_scoped: str) -> bool:
    """Delete a user's profile row. Returns True if the delete was issued."""
    resource = get_resource("dynamodb")
    if resource is None:
        return False
    try:
        table = resource.Table(DYNAMODB_PROFILES_TABLE)
        table.delete_item(Key={"user_id": user_id_scoped})
        return True
    except Exception as e:
        logger.warning("delete_profile failed for %s: %s", user_id_scoped, e)
        return False


def get_tenant_profiles(tenant_id: str) -> List[Dict[str, Any]]:
    """List all of a tenant's profiles, shaped for the dashboard (newest first)."""
    resource = get_resource("dynamodb")
    if resource is None:
        return []
    from boto3.dynamodb.conditions import Attr
    try:
        table = resource.Table(DYNAMODB_PROFILES_TABLE)
        resp  = table.scan(FilterExpression=Attr("tenant_id").eq(tenant_id))
        rows  = resp.get("Items", [])
    except Exception as e:
        logger.warning("get_tenant_profiles scan failed: %s", e)
        return []

    profiles = [to_dashboard_profile(r, tenant_id) for r in rows]
    profiles.sort(key=lambda p: p["metrics"]["unified_score"], reverse=True)
    return profiles
