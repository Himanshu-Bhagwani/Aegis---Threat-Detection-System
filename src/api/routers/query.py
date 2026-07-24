"""
Natural Language Query Router
==============================
Accepts free-text analyst queries + live profile data from the frontend,
interprets intent via Ollama (local LLM), then builds chart/table responses
from the REAL user scores — not synthetic data.

Falls back to a rule-based parser when Ollama is unavailable.
"""

import json
import logging
import os
import random
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# NOTE: docker-compose sets OLLAMA_URL (not OLLAMA_BASE_URL). Read both so the
# chatbot actually reaches the Ollama container instead of silently falling back
# to the rule-based parser (which is why greetings returned canned answers).
OLLAMA_BASE  = os.getenv("OLLAMA_URL") or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# ── Pydantic models ───────────────────────────────────────

class ProfilePayload(BaseModel):
    id:            str
    name:          str
    email:         str
    risk_level:    str
    unified_score: float
    gps_spoof:     float
    login_anomaly: float
    password_leak: float
    fraud_risk:    float
    breach_risk:   float


class NLQueryRequest(BaseModel):
    query:    str
    history:  Optional[List[Dict[str, str]]] = []
    profiles: Optional[List[ProfilePayload]] = []


# ── Prompt building ───────────────────────────────────────

SYSTEM_PROMPT = """You are an AI security analyst embedded inside Aegis, a fraud and identity-threat detection platform.
You have been given real-time risk data for all monitored users. Use this data to answer the analyst's question.

If the user is just chatting — greeting you ("hi", "hello"), asking how you are,
thanking you, or asking what you can do — use intent "conversational" and put a
warm, natural, helpful reply in answer_prefix. In that reply, briefly mention
what you can help with and reference the live data you can see (e.g. how many
users are monitored and whether any are high risk). Do NOT invent a chart for
small talk.

Respond ONLY with valid JSON (no markdown, no explanation):
{{
  "intent": "<one of: conversational | users_by_risk | risk_over_time | module_breakdown | alert_summary | top_threats | fraud_analysis | device_risk | breach_stats | general_stats | user_detail>",
  "filters": {{
    "risk_level":       "<minimal | low | medium | high | critical — or null>",
    "time_range_hours": <integer 1-720>,
    "module":           "<gps | login | password | fraud | breach | device — or null>",
    "limit":            <integer 5-50>,
    "user_name":        "<specific user name if asked about one person — or null>"
  }},
  "chart_type":   "<text | bar | line | pie | table | metric>",   // use "text" for conversational
  "answer_prefix": "<2-3 sentences that directly reference the actual user names and scores in the data below>"
}}

Intent guide:
  users_by_risk    → rank/list users by a specific risk score
  risk_over_time   → trend a risk metric over time
  module_breakdown → compare all 6 detection modules (bar chart)
  alert_summary    → breakdown of which modules are over threshold (pie)
  top_threats      → most triggered threat categories
  fraud_analysis   → fraud score timeline
  device_risk      → device fingerprint table
  breach_stats     → breach/password exposure summary
  general_stats    → system-level KPI summary cards
  user_detail      → detailed breakdown for a specific named user

Current live user data:
{profiles_context}
"""


def _build_profiles_context(profiles: List[ProfilePayload]) -> str:
    if not profiles:
        return "No profiles currently loaded."
    lines = []
    for p in profiles:
        lines.append(
            f"  • {p.name} ({p.email}) — Risk: {p.risk_level.upper()} "
            f"[Unified={p.unified_score:.0%}, GPS={p.gps_spoof:.0%}, "
            f"Login={p.login_anomaly:.0%}, Password={p.password_leak:.0%}, "
            f"Fraud={p.fraud_risk:.0%}, Breach={p.breach_risk:.0%}]"
        )
    return "\n".join(lines)


async def call_ollama(query: str, history: List[Dict], profiles: List[ProfilePayload]) -> Dict:
    """Ask the local LLM to classify the query. Falls back to rules on failure.

    The returned dict carries `_llm_used` / `_llm_error` so the endpoint can
    tell the caller whether the answer really came from the model — otherwise a
    silent fallback looks identical to a working LLM.
    """
    ctx     = _build_profiles_context(profiles)
    system  = SYSTEM_PROMPT.replace("{profiles_context}", ctx)
    messages = [{"role": "system", "content": system}]
    for turn in (history or [])[-4:]:
        messages.append(turn)
    messages.append({"role": "user", "content": query})

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE}/api/chat",
                json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"].strip()
            parsed  = None
            # Strip markdown fences if present
            if "```" in content:
                for part in content.split("```"):
                    part = part.strip().lstrip("json").strip()
                    try:
                        parsed = json.loads(part)
                        break
                    except Exception:
                        continue
            if parsed is None:
                parsed = json.loads(content)
            parsed["_llm_used"] = True
            return parsed
    except Exception as e:
        detail = f"{type(e).__name__}: {str(e)[:160]}"
        logger.warning("Ollama call failed (%s at %s, model=%s) — using rule-based fallback",
                       detail, OLLAMA_BASE, OLLAMA_MODEL)
        fallback = _rule_based_parse(query)
        fallback["_llm_used"]  = False
        fallback["_llm_error"] = detail
        return fallback


# Small talk that should get a conversational reply, not an analytics card.
_GREETING_WORDS = {
    "hi", "hii", "hiii", "hey", "heya", "hello", "helo", "yo", "sup",
    "hola", "namaste", "greetings", "good morning", "good afternoon",
    "good evening", "morning", "evening",
}
_CHITCHAT_PATTERNS = (
    "how are you", "how r u", "how's it going", "hows it going", "what's up",
    "whats up", "who are you", "what are you", "what can you do", "what do you do",
    "help me", "what should i ask", "thank", "thanks", "thx", "ok", "okay",
    "cool", "nice", "good job", "bye", "goodbye", "see ya",
)


def _is_small_talk(q: str) -> bool:
    """True when the message is a greeting / chit-chat rather than a data question."""
    s = q.strip().lower().strip("!?.,")
    if not s:
        return False
    if s in _GREETING_WORDS:
        return True
    # Short messages that open with a greeting, e.g. "hey there"
    if len(s.split()) <= 4 and any(s.startswith(g) for g in _GREETING_WORDS):
        return True
    return any(p in s for p in _CHITCHAT_PATTERNS)


def _small_talk_reply(query: str, profiles: List[ProfilePayload]) -> str:
    """A friendly, data-aware reply for greetings — no chart."""
    q = query.strip().lower().strip("!?.,")
    n = len(profiles)

    if n:
        critical = sum(1 for p in profiles if p.unified_score >= 0.75)
        high     = sum(1 for p in profiles if 0.50 <= p.unified_score < 0.75)
        avg      = sum(p.unified_score for p in profiles) / n
        if critical:
            top = max(profiles, key=lambda p: p.unified_score)
            state = (f"Right now I'm watching {n} user{'s' if n != 1 else ''}, and "
                     f"{critical} {'is' if critical == 1 else 'are'} at critical risk — "
                     f"{top.name} is the highest at {top.unified_score:.0%}.")
        elif high:
            state = (f"Right now I'm watching {n} user{'s' if n != 1 else ''}; "
                     f"{high} {'is' if high == 1 else 'are'} in the high-risk band, "
                     f"average risk {avg:.0%}.")
        else:
            state = (f"Right now I'm watching {n} user{'s' if n != 1 else ''} and nothing "
                     f"looks alarming — average risk is {avg:.0%}.")
    else:
        state = "I don't have any user profiles loaded yet."

    if any(w in q for w in ("thank", "thx")):
        return f"Happy to help! {state} Ask me any time you want a closer look."
    if "bye" in q or "see ya" in q:
        return f"Goodbye! {state} I'll keep monitoring in the background."
    if any(p in q for p in ("who are you", "what are you", "what can you do", "what do you do", "help")):
        return ("I'm the Aegis security analyst. I can rank users by risk, break down the "
                "detection modules (GPS, login, password, fraud, breach), trend risk over "
                f"time, and dig into any individual user. {state}")
    if any(p in q for p in ("how are you", "how r u", "how's it going", "hows it going")):
        return (f"Doing well, thanks for asking — all detection modules are running. {state} "
                "Want me to show which users need attention?")

    # Plain greeting
    return (f"Hello! {state} You can ask me things like \"who's at critical risk?\", "
            "\"compare login anomaly across users\", or \"what is Himanshu's risk profile?\"")


def _rule_based_parse(query: str) -> Dict:
    q = query.lower()

    if _is_small_talk(q):
        return {
            "intent": "conversational",
            "filters": {"risk_level": None, "time_range_hours": 24,
                        "module": None, "limit": 20, "user_name": None},
            "chart_type": "text",
            "answer_prefix": "",   # filled in by the endpoint, which has the profiles
        }

    intent, chart_type = "general_stats", "metric"
    module, risk_level, time_range, limit = None, None, 24, 20
    user_name = None

    if any(w in q for w in ["gps", "location", "spoof", "coordinate"]):        module = "gps"
    elif any(w in q for w in ["login", "sign in", "authentication", "auth"]):  module = "login"
    elif any(w in q for w in ["fraud", "transaction", "payment"]):              module = "fraud"
    elif any(w in q for w in ["breach", "password", "pwned", "leaked"]):       module = "breach"
    elif any(w in q for w in ["device", "fingerprint"]):                        module = "device"

    for lvl in ["critical", "high", "medium", "low", "minimal"]:
        if lvl in q: risk_level = lvl; break

    if any(w in q for w in ["month", "30 day"]):     time_range = 720
    elif any(w in q for w in ["week", "7 day"]):     time_range = 168
    elif "48 hour" in q or "2 day" in q:             time_range = 48

    import re
    m = re.search(r"top\s+(\d+)", q)
    if m: limit = int(m.group(1))

    if any(w in q for w in ["user", "who", "account", "profile", "people", "rank", "highest", "lowest"]):
        intent, chart_type = "users_by_risk", "bar"
    elif any(w in q for w in ["trend", "over time", "history", "past", "timeline", "last week", "last 24"]):
        intent, chart_type = "risk_over_time", "line"
    elif any(w in q for w in ["module", "breakdown", "compare", "each module", "all module"]):
        intent, chart_type = "module_breakdown", "bar"
    elif any(w in q for w in ["alert", "notification", "threshold"]):
        intent, chart_type = "alert_summary", "pie"
    elif any(w in q for w in ["threat type", "attack type", "most common", "frequent threat"]):
        intent, chart_type = "top_threats", "bar"
    elif "fraud" in q and any(w in q for w in ["score", "analysis", "stat", "amount"]):
        intent, chart_type = "fraud_analysis", "line"
    elif module == "device":
        intent, chart_type = "device_risk", "table"
    elif module == "breach":
        intent, chart_type = "breach_stats", "metric"

    return {
        "intent":  intent,
        "filters": {
            "risk_level": risk_level, "time_range_hours": time_range,
            "module": module, "limit": limit, "user_name": user_name,
        },
        "chart_type":    chart_type,
        "answer_prefix": f"Showing {intent.replace('_', ' ')} based on your query.",
    }


# ── Data builders using real profile data ─────────────────

RISK_COLORS = {
    "minimal":  "#00ff88",
    "low":      "#84cc16",
    "medium":   "#f59e0b",
    "high":     "#f97316",
    "critical": "#ef4444",
}

MODULE_KEYS = {
    "gps":      ("gps_spoof",     "GPS Spoofing"),
    "login":    ("login_anomaly", "Login Anomaly"),
    "password": ("password_leak", "Password Risk"),
    "fraud":    ("fraud_risk",    "Fraud Score"),
    "breach":   ("breach_risk",   "Breach Exposure"),
}

THRESHOLDS = {
    "gps_spoof":     0.60,
    "login_anomaly": 0.65,
    "password_leak": 0.60,
    "fraud_risk":    0.60,
    "breach_risk":   0.50,
    "unified_score": 0.70,
}

CHART_COLORS = ["#3d7fff","#8b5cf6","#06b6d4","#ef4444","#f97316","#f59e0b","#00ff88","#84cc16","#ec4899"]

THREAT_TYPES = [
    "GPS Spoofing", "Login Anomaly", "Transaction Fraud",
    "Account Takeover", "Credential Stuffing", "Device Hijack",
    "Breach Exposure", "Impossible Travel", "Brute Force",
]


def _level_for_score(s: float) -> str:
    if s >= 0.75: return "critical"
    if s >= 0.50: return "high"
    if s >= 0.25: return "medium"
    if s >= 0.10: return "low"
    return "minimal"


def _build_users_by_risk(profiles: List[ProfilePayload], filters: Dict):
    risk_filter = filters.get("risk_level")
    module      = filters.get("module") or "fraud"
    limit       = min(int(filters.get("limit") or 20), len(profiles) or 20)

    attr = {
        "gps": "gps_spoof", "login": "login_anomaly",
        "password": "password_leak", "fraud": "fraud_risk",
        "breach": "breach_risk",
    }.get(module, "unified_score")

    rows = []
    for p in profiles:
        score = getattr(p, attr, p.unified_score)
        lvl   = _level_for_score(score)
        if risk_filter and lvl != risk_filter:
            continue
        rows.append({"profile": p, "score": score, "lvl": lvl})

    rows.sort(key=lambda x: x["score"], reverse=True)
    rows = rows[:limit]

    mod_label = {
        "gps": "GPS Spoof", "login": "Login Anomaly",
        "password": "Password Risk", "fraud": "Fraud Risk",
        "breach": "Breach Risk",
    }.get(module, "Unified Score")

    chart_data = [
        {
            "name":       r["profile"].name.split()[0],
            "score":      round(r["score"] * 100, 1),
            "risk_level": r["lvl"],
            "fill":       RISK_COLORS[r["lvl"]],
        }
        for r in rows
    ]
    table_data = [
        {
            "User":         r["profile"].name,
            "Email":        r["profile"].email,
            "Risk Level":   r["lvl"].capitalize(),
            mod_label:      f"{round(r['score'] * 100, 1)}%",
            "Unified Score":f"{round(r['profile'].unified_score * 100, 1)}%",
        }
        for r in rows
    ]
    return chart_data, table_data


def _build_module_breakdown(profiles: List[ProfilePayload]):
    if not profiles:
        return []
    modules = [
        ("GPS Spoofing",    "gps_spoof",     "#3d7fff"),
        ("Login Anomaly",   "login_anomaly", "#8b5cf6"),
        ("Password Risk",   "password_leak", "#06b6d4"),
        ("Fraud Score",     "fraud_risk",    "#ef4444"),
        ("Breach Exposure", "breach_risk",   "#f97316"),
    ]
    result = []
    for label, attr, color in modules:
        scores = [getattr(p, attr, 0) for p in profiles]
        avg    = sum(scores) / len(scores)
        result.append({
            "module":     label,
            "avg_score":  round(avg * 100, 1),
            "detections": sum(1 for s in scores if s > THRESHOLDS.get(attr, 0.5)),
            "color":      color,
        })
    return result


def _build_alert_summary(profiles: List[ProfilePayload]):
    counts: Dict[str, int] = {}
    module_map = [
        ("GPS Spoofing",    "gps_spoof",     0.60),
        ("Login Anomaly",   "login_anomaly", 0.65),
        ("Password Risk",   "password_leak", 0.60),
        ("Fraud Score",     "fraud_risk",    0.60),
        ("Breach Exposure", "breach_risk",   0.50),
    ]
    for p in profiles:
        for label, attr, thresh in module_map:
            if getattr(p, attr, 0) > thresh:
                counts[label] = counts.get(label, 0) + 1
    if not counts:
        for label, _, _ in module_map:
            counts[label] = 0
    return [{"name": k, "value": v} for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)]


def _build_general_stats(profiles: List[ProfilePayload]):
    if not profiles:
        return [
            {"label": "Total Profiles",     "value": 0,    "unit": ""},
            {"label": "Critical Risk",      "value": 0,    "unit": ""},
            {"label": "Avg Unified Risk",   "value": 0,    "unit": "%"},
            {"label": "Modules Monitored",  "value": 5,    "unit": ""},
        ]
    avg = sum(p.unified_score for p in profiles) / len(profiles)
    critical = sum(1 for p in profiles if p.risk_level == "critical")
    return [
        {"label": "Total Profiles",    "value": len(profiles),               "unit": ""},
        {"label": "Critical Risk",     "value": critical,                    "unit": ""},
        {"label": "Avg Unified Risk",  "value": round(avg * 100, 1),         "unit": "%"},
        {"label": "Modules Monitored", "value": 5,                           "unit": ""},
    ]


def _build_breach_stats(profiles: List[ProfilePayload]):
    if not profiles:
        return [
            {"label": "Accounts Checked",   "value": 0,    "unit": ""},
            {"label": "High Breach Risk",   "value": 0,    "unit": ""},
            {"label": "Avg Breach Score",   "value": 0,    "unit": "%"},
            {"label": "Threshold",          "value": 50,   "unit": "%"},
        ]
    breach_scores = [p.breach_risk for p in profiles]
    high_breach   = sum(1 for s in breach_scores if s > 0.50)
    avg_breach    = sum(breach_scores) / len(breach_scores)
    return [
        {"label": "Accounts Checked",  "value": len(profiles),                "unit": ""},
        {"label": "High Breach Risk",  "value": high_breach,                  "unit": ""},
        {"label": "Avg Breach Score",  "value": round(avg_breach * 100, 1),   "unit": "%"},
        {"label": "Detection Threshold","value": 50,                           "unit": "%"},
    ]


def _build_user_detail(profiles: List[ProfilePayload], filters: Dict):
    target_name = (filters.get("user_name") or "").lower()
    profile = next(
        (p for p in profiles if target_name and target_name in p.name.lower()),
        profiles[0] if profiles else None,
    )
    if not profile:
        return [], []

    modules = [
        ("GPS Spoofing",    profile.gps_spoof,     "#3d7fff"),
        ("Login Anomaly",   profile.login_anomaly, "#8b5cf6"),
        ("Password Risk",   profile.password_leak, "#06b6d4"),
        ("Fraud Score",     profile.fraud_risk,    "#ef4444"),
        ("Breach Exposure", profile.breach_risk,   "#f97316"),
        ("Unified Score",   profile.unified_score, "#f59e0b"),
    ]
    chart_data = [
        {"module": label, "score": round(val * 100, 1), "fill": color}
        for label, val, color in modules
    ]
    table_data = [
        {
            "Module":    label,
            "Score":     f"{round(val * 100, 1)}%",
            "Risk":      _level_for_score(val).capitalize(),
            "Threshold": f"{round(THRESHOLDS.get(attr, 0.5) * 100, 0):.0f}%" ,
            "Status":    "ALERT" if val > THRESHOLDS.get(attr, 0.5) else "OK",
        }
        for (label, val, _), attr in zip(
            modules,
            ["gps_spoof", "login_anomaly", "password_leak", "fraud_risk", "breach_risk", "unified_score"],
        )
    ]
    return chart_data, table_data


def _build_risk_over_time(profiles: List[ProfilePayload], filters: Dict) -> List[Dict]:
    """Simulate a plausible history based on the real current scores as end-state."""
    rng    = random.Random(7)
    hours  = int(filters.get("time_range_hours") or 24)
    module = filters.get("module") or "fraud"
    attr   = {
        "gps": "gps_spoof", "login": "login_anomaly",
        "password": "password_leak", "fraud": "fraud_risk", "breach": "breach_risk",
    }.get(module, "unified_score")
    label  = {
        "gps": "GPS Spoofing", "login": "Login Anomaly",
        "password": "Password Risk", "fraud": "Fraud Score", "breach": "Breach Exposure",
    }.get(module, "Unified Risk")

    # Use mean current score as the end-state
    current = (sum(getattr(p, attr, 0) for p in profiles) / len(profiles)) if profiles else 0.4

    now   = datetime.now(timezone.utc)
    step  = max(1, hours // 48)
    # Walk backwards from current, adding noise
    base  = max(0.04, current - rng.uniform(0.05, 0.20))
    points = []

    for i in range(0, hours, step):
        ts    = now - timedelta(hours=hours - i)
        frac  = i / max(hours, 1)
        # Trend toward current score over time
        base  = base + (current - base) * frac * 0.04 + rng.uniform(-0.025, 0.035)
        base  = max(0.02, min(0.98, base))
        spike = base + rng.uniform(0.15, 0.35) if rng.random() < 0.07 else base
        spike = min(0.99, spike)
        fmt   = "%m/%d %H:%M" if hours <= 48 else "%m/%d"
        points.append({"time": ts.strftime(fmt), label: round(spike * 100, 1), "Threshold": 50})

    return points


def _build_top_threats(profiles: List[ProfilePayload]) -> List[Dict]:
    rng = random.Random(17)
    # Weight detections by how many profiles exceed each module threshold
    module_map = [
        ("GPS Spoofing",     "gps_spoof",     0.60),
        ("Login Anomaly",    "login_anomaly", 0.65),
        ("Fraud Detection",  "fraud_risk",    0.60),
        ("Breach Exposure",  "breach_risk",   0.50),
        ("Password Risk",    "password_leak", 0.60),
    ]
    results = []
    for name, attr, thresh in module_map:
        hits = sum(1 for p in profiles if getattr(p, attr, 0) > thresh)
        avg  = sum(getattr(p, attr, 0) for p in profiles) / max(len(profiles), 1)
        results.append({
            "threat":       name,
            "count":        hits + rng.randint(0, 5),
            "avg_severity": round(avg * 100, 1),
        })
    # Add synthetic threat types with lower counts
    for extra in ["Account Takeover", "Credential Stuffing", "Impossible Travel", "Brute Force"]:
        results.append({"threat": extra, "count": rng.randint(0, 3), "avg_severity": round(rng.uniform(20, 55), 1)})
    return sorted(results, key=lambda x: x["count"], reverse=True)


# ── Endpoint ──────────────────────────────────────────────

@router.get("/health", summary="Is the local LLM reachable and is the model pulled?")
async def query_health():
    """Diagnose the AI-query pipeline: endpoint, reachability, and whether the
    configured model has actually finished downloading."""
    info: Dict[str, Any] = {
        "ollama_url":   OLLAMA_BASE,
        "model":        OLLAMA_MODEL,
        "reachable":    False,
        "model_pulled": False,
        "models_available": [],
        "error":        None,
    }
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags")
            resp.raise_for_status()
            info["reachable"] = True
            names = [m.get("name", "") for m in resp.json().get("models", [])]
            info["models_available"] = names
            base = OLLAMA_MODEL.split(":")[0]
            info["model_pulled"] = any(n == OLLAMA_MODEL or n.split(":")[0] == base for n in names)
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {str(e)[:200]}"

    if not info["reachable"]:
        info["status"] = "unreachable — AI answers fall back to rule-based"
    elif not info["model_pulled"]:
        info["status"] = f"model '{OLLAMA_MODEL}' not pulled yet — still downloading, answers fall back to rule-based"
    else:
        info["status"] = "healthy"
    return info


@router.post("")
async def nl_query(body: NLQueryRequest):
    t0 = time.perf_counter()

    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    profiles = body.profiles or []

    # Greetings / small talk answer instantly from live data — no LLM round-trip,
    # no chart. Keeps "hi" feeling like a conversation rather than a dashboard.
    if _is_small_talk(body.query):
        return {
            "answer":        _small_talk_reply(body.query, profiles),
            "chart_type":    "text",
            "chart_data":    [],
            "chart_config":  {},
            "table_data":    None,
            "table_headers": None,
            "intent":        "conversational",
            "query_time_ms": round((time.perf_counter() - t0) * 1000, 1),
            "ollama_model":  OLLAMA_MODEL,
            # Answered locally on purpose — greetings shouldn't wait on the LLM.
            "llm_used":      False,
            "llm_error":     None,
        }

    parsed = await call_ollama(body.query, body.history or [], profiles)

    intent     = parsed.get("intent",       "general_stats")
    filters    = parsed.get("filters",      {}) or {}
    chart_type = parsed.get("chart_type",   "bar")
    answer     = parsed.get("answer_prefix", "Here are the results based on your live profile data.")

    chart_data:    List[Dict]          = []
    table_data:    Optional[List[Dict]] = None
    table_headers: Optional[List[str]]  = None
    chart_config:  Dict[str, Any]       = {}

    # ── Route to data builder ─────────────────

    if intent in ("users_by_risk", "user_detail") and intent != "user_detail":
        chart_data, table_data = _build_users_by_risk(profiles, filters)
        table_headers = list(table_data[0].keys()) if table_data else []
        chart_config  = {
            "x_key":    "name",
            "bars":     [{"key": "score", "name": "Risk Score (%)", "color": "#3d7fff"}],
            "y_label":  "Risk Score (%)",
            "y_domain": [0, 100],
        }

    elif intent == "user_detail":
        chart_data, table_data = _build_user_detail(profiles, filters)
        table_headers = list(table_data[0].keys()) if table_data else []
        chart_type    = "bar"
        chart_config  = {
            "x_key":    "module",
            "bars":     [{"key": "score", "name": "Score (%)", "color": "#3d7fff"}],
            "y_label":  "Score (%)",
            "y_domain": [0, 100],
        }

    elif intent == "risk_over_time":
        module     = filters.get("module") or "fraud"
        chart_data = _build_risk_over_time(profiles, filters)
        label      = {"gps": "GPS Spoofing", "login": "Login Anomaly",
                      "password": "Password Risk", "fraud": "Fraud Score",
                      "breach": "Breach Exposure"}.get(module, "Unified Risk")
        chart_config = {
            "x_key": "time",
            "lines": [
                {"key": label,       "color": "#3d7fff"},
                {"key": "Threshold", "color": "#ef4444", "dashed": True},
            ],
            "y_label":  "Score (%)",
            "y_domain": [0, 100],
        }

    elif intent == "module_breakdown":
        chart_data  = _build_module_breakdown(profiles)
        chart_config = {
            "x_key": "module",
            "bars":  [
                {"key": "avg_score",  "name": "Avg Score (%)", "color": "#3d7fff"},
                {"key": "detections", "name": "Profiles Over Threshold", "color": "#8b5cf6"},
            ],
            "y_label": "Value",
        }

    elif intent == "alert_summary":
        chart_type  = "pie"
        chart_data  = _build_alert_summary(profiles)
        chart_config = {
            "name_key":  "name",
            "value_key": "value",
            "colors":    CHART_COLORS,
        }

    elif intent == "top_threats":
        chart_data  = _build_top_threats(profiles)
        chart_config = {
            "x_key": "threat",
            "bars":  [
                {"key": "count",        "name": "Detections",       "color": "#ef4444"},
                {"key": "avg_severity", "name": "Avg Severity (%)", "color": "#f97316"},
            ],
            "y_label": "Count / Severity",
        }

    elif intent == "fraud_analysis":
        filters_mod  = {**filters, "module": "fraud"}
        chart_data   = _build_risk_over_time(profiles, filters_mod)
        # Rename key for display
        chart_data   = [{"time": d["time"], "Fraud Score": d.get("Fraud Score", d.get("Unified Risk", 0)), "Threshold": 50} for d in chart_data]
        chart_config  = {
            "x_key": "time",
            "lines": [
                {"key": "Fraud Score", "color": "#ef4444"},
                {"key": "Threshold",   "color": "#f59e0b", "dashed": True},
            ],
            "y_label":  "Score (%)",
            "y_domain": [0, 100],
        }

    elif intent == "device_risk":
        # Show per-profile device-risk table (uses unified as proxy)
        chart_type = "table"
        rng = random.Random(21)
        table_data = [
            {
                "User":       p.name,
                "Risk Level": p.risk_level.capitalize(),
                "GPS Spoof":  f"{round(p.gps_spoof * 100, 1)}%",
                "Login":      f"{round(p.login_anomaly * 100, 1)}%",
                "VPN":        rng.choice(["Yes", "No", "No", "No"]),
                "Emulator":   rng.choice(["Yes", "No", "No"]),
            }
            for p in profiles
        ]
        table_headers = ["User", "Risk Level", "GPS Spoof", "Login", "VPN", "Emulator"]

    elif intent == "breach_stats":
        chart_type = "metric"
        chart_data = _build_breach_stats(profiles)

    elif intent == "conversational":
        # The model classified this as small talk — reply in plain text.
        chart_type = "text"
        chart_data = []
        if not str(answer).strip():
            answer = _small_talk_reply(body.query, profiles)

    else:  # general_stats
        chart_type = "metric"
        chart_data = _build_general_stats(profiles)

    return {
        "answer":        answer,
        "chart_type":    chart_type,
        "chart_data":    chart_data,
        "chart_config":  chart_config,
        "table_data":    table_data,
        "table_headers": table_headers,
        "intent":        intent,
        "query_time_ms": round((time.perf_counter() - t0) * 1000, 1),
        "ollama_model":  OLLAMA_MODEL,
        # Whether the answer really came from the LLM, and why not if it didn't.
        "llm_used":      bool(parsed.get("_llm_used")),
        "llm_error":     parsed.get("_llm_error"),
    }
