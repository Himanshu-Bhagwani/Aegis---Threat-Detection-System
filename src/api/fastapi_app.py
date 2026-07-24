"""
AEGIS Threat Detection API v2.0
=================================
Multi-layer threat detection and identity management system.

Architecture:
  - FastAPI + Uvicorn  (async, high-performance)
  - AWS Cognito        : Real user authentication & JWT
  - AWS DynamoDB       : Event persistence & user profiles
  - AWS CloudWatch     : Metrics and structured logging
  - AWS SNS            : Critical-risk alerts
  - WebSocket          : Real-time event streaming to dashboard

Endpoints:
  /auth/*     - Cognito-backed sign-up, sign-in, sign-out
  /gps/*      - GPS spoofing detection
  /login/*    - Login anomaly detection
  /password/* - Password risk assessment
  /fraud/*    - Transaction fraud detection
  /breach/*   - Data breach exposure check
  /device/*   - Device fingerprint risk scoring
  /risk/*     - Unified risk fusion (all layers)
  /identity/* - Per-user behavioral profiles & history
  /alerts/*   - Security alert management
  /ws         - WebSocket real-time feed
  /health     - System health (all components + AWS)
"""

import os
import time
import uuid
import json
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set

from fastapi import (
    FastAPI, Request, WebSocket, WebSocketDisconnect,
    HTTPException, Depends, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Import detection routers ──────────────────────────
from src.api.routers import fraud    as fraud_router
from src.api.routers import gps      as gps_router
from src.api.routers import login    as login_router
from src.api.routers import password as password_router
from src.api.routers import risk     as risk_router
from src.api.routers import breach   as breach_router
from src.api.routers import device   as device_router
from src.api.routers import query    as query_router

# ── Auth dependency ───────────────────────────────────
from src.api.auth_deps import require_auth, optional_auth, create_local_jwt

# ── AWS helpers ───────────────────────────────────────
from src.utils.aws_utils import (
    cognito_sign_up,
    cognito_sign_in,
    cognito_sign_out,
    dynamo_put_event,
    dynamo_put_alert,
    dynamo_upsert_alert,
    dynamo_find_open_alert,
    dynamo_resolve_alert,
    dynamo_get_user_events,
    dynamo_get_user_profile,
    dynamo_get_recent_alerts,
    dynamo_dismiss_alert,
    cw_put_metric,
    sns_alert_critical_risk,
    check_aws_health,
    aws_available,
    COGNITO_USER_POOL_ID,
)

# ── Multi-tenant (API-key) + webhook helpers ──────────
from src.utils.api_keys import get_tenant_for_key
from src.utils.webhooks import send_threat_webhook
from src.utils.profiles import (
    register_profile,
    update_from_event,
    get_tenant_profiles,
    to_dashboard_profile,
    login_stats,
    transaction_stats,
    delete_profile,
)
from src.api.tenant_context import (
    set_current_tenant,
    get_current_tenant,
    current_tenant_id,
    scope_user_id,
    unscope_user_id,
    DEFAULT_TENANT,
)

CRITICAL_THRESHOLD = float(os.getenv("CRITICAL_RISK_THRESHOLD", "0.75"))
HIGH_THRESHOLD     = float(os.getenv("HIGH_RISK_THRESHOLD",     "0.50"))

# When true, the detection endpoints require a valid X-Api-Key header.
REQUIRE_API_KEY = os.getenv("APEILO_REQUIRE_API_KEY", "false").lower() in ("1", "true", "yes")
# Path prefixes that represent scored detection traffic (subject to API-key auth).
_API_KEY_PROTECTED_PREFIXES = (
    "/gps", "/login", "/password", "/fraud", "/risk", "/breach", "/device", "/query",
    "/identity", "/alerts", "/profiles",
)


# ══════════════════════════════════════════════════════
# WebSocket Connection Manager
# ══════════════════════════════════════════════════════

class ConnectionManager:
    """
    Manages active WebSocket connections.
    Broadcasts real-time detection events to all connected dashboard clients.
    """

    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)
        logger.info("WS client connected. Total: %d", len(self.active))

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)
        logger.info("WS client disconnected. Total: %d", len(self.active))

    async def broadcast(self, message: dict):
        """Send a JSON message to all connected clients."""
        if not self.active:
            return
        payload = json.dumps(message, default=str)
        dead    = set()
        for ws in self.active:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.active.discard(ws)


ws_manager = ConnectionManager()


# ══════════════════════════════════════════════════════
# Pydantic Models (Auth)
# ══════════════════════════════════════════════════════

class SignUpRequest(BaseModel):
    email:       str = Field(..., example="user@example.com")
    password:    str = Field(..., min_length=8, example="SecurePass123")
    given_name:  str = Field("",  example="Ada")
    family_name: str = Field("",  example="Lovelace")


class SignInRequest(BaseModel):
    email:    str = Field(..., example="user@example.com")
    password: str = Field(..., example="SecurePass123")


class SignOutRequest(BaseModel):
    access_token: str = Field(..., description="The access token to invalidate")


class RegisterIdentityRequest(BaseModel):
    user_id: str = Field(..., description="Stable id for the user in the calling app")
    name:    str = Field("", description="Display name")
    email:   str = Field("", description="Email address")


class LockdownRequest(BaseModel):
    """Ask the connected app to block login access for a period of time."""
    user_id: str = Field(..., description="Account to lock")
    lock_minutes: int = Field(
        15, ge=1, le=1440,
        description="How long to block new logins (minutes). 10/15/30/60 are the presets.",
    )


class VerifyActivityRequest(BaseModel):
    """The user's answer to a 'was this you?' challenge."""
    user_id:    str  = Field(..., description="Who was challenged")
    activity:   str  = Field("transaction", description="'transaction' or 'login'")
    confirmed:  bool = Field(..., description="True = 'yes, that was me'")
    risk_score: float = Field(0.0, ge=0, le=1)
    detail:     Optional[Dict[str, Any]] = Field(None, description="Amount, hour, etc.")
    alert_id:   Optional[str] = Field(None, description="Challenge being answered (marks it resolved)")


# ══════════════════════════════════════════════════════
# FastAPI App
# ══════════════════════════════════════════════════════

app = FastAPI(
    title="AEGIS Threat Detection API",
    description="""
## AEGIS — Multi-Layer Threat Detection & Identity Management

### Detection Layers
| Layer | Endpoint | Technology |
|-------|----------|------------|
| GPS Spoofing | `/gps/*` | Isolation Forest + GBM + Autoencoder + CNN-RNN |
| Login Anomaly | `/login/*` | LANL-trained IF + GBM + Autoencoder |
| Password Risk | `/password/*` | XGBoost + entropy analysis |
| Transaction Fraud | `/fraud/*` | XGBoost + 9 rule heuristics |
| Breach Exposure | `/breach/*` | HIBP k-anonymity API |
| Device Fingerprint | `/device/*` | Behavioral fingerprint scoring |

### Fusion
All detection layers feed the **Unified Risk Engine** (`/risk/unified`).

### Real-time Feed
Connect to `/ws` for a WebSocket stream of detection events.
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────
# Default local origins plus any extra ones from APEILO_CORS_ORIGINS
# (comma-separated) so an integrating app on another port/host can connect.
_default_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://localhost:5173",   # SODA (Vite dev / container)
    "http://127.0.0.1:5173",
]
_extra_origins = [o.strip() for o in os.getenv("APEILO_CORS_ORIGINS", "").split(",") if o.strip()]
_cors_origins = _default_origins + _extra_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in _extra_origins else _cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing + CloudWatch ───────────────────────
@app.middleware("http")
async def telemetry_middleware(request: Request, call_next):
    start    = time.perf_counter()
    response = await call_next(request)
    elapsed  = (time.perf_counter() - start) * 1000

    response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"

    try:
        path = request.url.path.strip("/").split("/")[0] or "root"
        cw_put_metric(
            "APILatencyMs",
            elapsed,
            unit="Milliseconds",
            dimensions=[{"Name": "Endpoint", "Value": path}],
        )
    except Exception:
        pass

    return response


# ── API-key → tenant resolution ───────────────────────
def _path_is_protected(path: str) -> bool:
    return any(path.startswith(p) for p in _API_KEY_PROTECTED_PREFIXES)


@app.middleware("http")
async def tenant_middleware(request: Request, call_next):
    """Resolve the caller's tenant from the X-Api-Key header.

    - A valid key  → that tenant (isolated namespace + its webhook).
    - No / bad key → the default 'apeilo' tenant (keeps the bundled dashboard
      working). If APEILO_REQUIRE_API_KEY=true, protected paths 401 instead.
    """
    api_key = request.headers.get("x-api-key", "")
    tenant  = None
    if api_key:
        try:
            tenant = get_tenant_for_key(api_key)
        except Exception as e:
            logger.warning("API key lookup failed: %s", e)
            tenant = None

    set_current_tenant(tenant)             # None resolves to the default tenant
    request.state.tenant = get_current_tenant()

    if REQUIRE_API_KEY and _path_is_protected(request.url.path) and not tenant:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error":   "invalid_api_key",
                "message": "A valid X-Api-Key header is required for this endpoint.",
            },
        )

    return await call_next(request)


# ── Include detection routers ─────────────────────────
app.include_router(gps_router.router,      prefix="/gps",      tags=["GPS Spoofing"])
app.include_router(login_router.router,    prefix="/login",    tags=["Login Anomaly"])
app.include_router(password_router.router, prefix="/password", tags=["Password Risk"])
app.include_router(fraud_router.router,    prefix="/fraud",    tags=["Fraud Detection"])
app.include_router(risk_router.router,     prefix="/risk",     tags=["Unified Risk"])
app.include_router(breach_router.router,   prefix="/breach",   tags=["Breach Detection"])
app.include_router(device_router.router,   prefix="/device",   tags=["Device Fingerprint"])
app.include_router(query_router.router,    prefix="/query/nl", tags=["AI Query"])


# ══════════════════════════════════════════════════════
# AUTH ENDPOINTS (Cognito-backed)
# ══════════════════════════════════════════════════════

@app.post("/auth/signup", tags=["Auth"], summary="Register a new user")
async def signup(body: SignUpRequest):
    result = cognito_sign_up(
        email=body.email,
        password=body.password,
        given_name=body.given_name,
        family_name=body.family_name,
    )
    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("message", "Registration failed"),
        )
    cw_put_metric("UserSignups", 1.0, "Count")
    return {
        "success":   True,
        "user_sub":  result.get("user_sub"),
        "confirmed": result.get("confirmed", False),
        "message":   result.get("message"),
        "mock_mode": result.get("mock", False),
    }


@app.post("/auth/signin", tags=["Auth"], summary="Sign in and get tokens")
async def signin(body: SignInRequest):
    result = cognito_sign_in(email=body.email, password=body.password)
    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=result.get("message", "Authentication failed"),
        )
    # Fallback to local JWT if Cognito not configured
    if result.get("mock") and not COGNITO_USER_POOL_ID:
        token = create_local_jwt(user_sub=body.email, email=body.email)
        result["access_token"] = token
    cw_put_metric("UserSignIns", 1.0, "Count")
    return result


@app.post("/auth/signout", tags=["Auth"], summary="Sign out")
async def signout(body: SignOutRequest):
    ok = cognito_sign_out(body.access_token)
    return {"success": ok, "message": "Signed out successfully" if ok else "Sign-out failed"}


# Legacy endpoints kept for frontend compatibility
@app.post("/signup", tags=["Auth (Legacy)"], include_in_schema=False)
async def signup_legacy(body: SignUpRequest):
    return await signup(body)

@app.post("/login", tags=["Auth (Legacy)"], include_in_schema=False)
async def login_legacy(body: SignInRequest):
    return await signin(body)


# ══════════════════════════════════════════════════════
# IDENTITY & EVENT HISTORY
# ══════════════════════════════════════════════════════

@app.post("/identity/register", tags=["Identity"], summary="Register/refresh a user profile on sign-in")
async def register_identity(body: RegisterIdentityRequest):
    """Called by an integrating app (e.g. SODA) when a user signs in.

    Creates the user's profile if new, or refreshes name/email, so the very
    first login shows up on the dashboard even before any scores exist.
    """
    tid    = current_tenant_id()
    scoped = scope_user_id(body.user_id)
    row    = register_profile(tid, scoped, body.name, body.email)
    return {
        "success":   True,
        "tenant_id": tid,
        "profile":   to_dashboard_profile(row, tid) if row else None,
    }


@app.get("/profiles", tags=["Identity"], summary="List the current tenant's live user profiles")
async def list_profiles():
    """Real per-user profiles aggregated from scored events — what the dashboard shows."""
    tid   = current_tenant_id()
    items = get_tenant_profiles(tid)
    return {"profiles": items, "count": len(items), "tenant_id": tid}


@app.post("/identity/lockdown", tags=["Identity"],
          summary="Block login access to a compromised account for a period")
async def lockdown_account(body: LockdownRequest):
    """Sends an `account.lockdown` webhook telling the connected app to block
    new logins for `lock_minutes` and end existing sessions. Apeilo can't touch
    another app's sessions itself — the app enforces the lock and shows the
    countdown to the user."""
    tenant = get_current_tenant()
    tid    = tenant.get("tenant_id", "apeilo")
    scoped = scope_user_id(body.user_id)

    now        = datetime.now(timezone.utc)
    lock_until = now + timedelta(minutes=body.lock_minutes)
    payload = {
        "risk_level":     "critical",
        "lock_minutes":   body.lock_minutes,
        "lock_until":     lock_until.isoformat(),
        "primary_threats": ["account_compromise_confirmed"],
        "recommended_actions": [
            f"Block new sign-ins for {body.lock_minutes} minutes.",
            "End active sessions for this account.",
        ],
    }
    try:
        dynamo_put_event(
            user_id=scoped, event_type="account_lockdown",
            scores={"lock_minutes": body.lock_minutes, "lock_until": lock_until.isoformat()},
            tenant_id=tid,
        )
    except Exception as e:
        logger.debug("lockdown persist skipped: %s", e)

    delivered = False
    try:
        delivered = send_threat_webhook(
            tenant=tenant, event_type="account.lockdown",
            user_id=body.user_id, risk_score=1.0,
            risk_level="critical", result=payload,
        )
    except Exception as e:
        logger.warning("lockdown webhook failed: %s", e)

    return {
        "success": True,
        "user_id": body.user_id,
        "lock_minutes": body.lock_minutes,
        "lock_until": lock_until.isoformat(),
        "webhook_delivered": delivered,
    }


@app.get("/identity/{user_id}/stepup-status", tags=["Identity"],
         summary="Step-up verification status for a suspicious success (polled by the connected app)")
async def stepup_status(user_id: str):
    """After a login succeeds on the heels of failed attempts, the connected app
    holds access and polls this until the account owner answers on the Apeilo
    dashboard: 'pending' → keep waiting, 'confirmed' → grant access, 'denied' →
    block, 'none' → no step-up is outstanding."""
    scoped = scope_user_id(user_id)
    # Newest step-up challenge for this user, whatever its status.
    from src.utils.aws_utils import dynamo_get_recent_alerts as _gra
    tid = current_tenant_id()
    alerts = _gra(limit=100, tenant_id=tid)
    mine = [
        a for a in alerts
        if a.get("user_id") == scoped and a.get("alert_type") == "challenge_login_stepup"
    ]
    if not mine:
        return {"status": "none", "user_id": user_id}
    latest = max(mine, key=lambda a: a.get("timestamp", ""))
    st = latest.get("status", "open")
    mapped = {"open": "pending", "confirmed": "confirmed", "denied": "denied"}.get(st, "pending")
    return {"status": mapped, "user_id": user_id, "alert_id": latest.get("alert_id")}


@app.get("/challenges", tags=["Identity"],
         summary="Pending 'was this you?' challenges for this tenant")
async def list_challenges(limit: int = 20):
    """Unanswered challenges, newest first — shown as modals in the Apeilo
    dashboard so the account owner/analyst can verify the activity."""
    tid   = current_tenant_id()
    items = dynamo_get_recent_alerts(limit=100, tenant_id=tid)
    out = []
    for a in items:
        if not str(a.get("alert_type", "")).startswith("challenge_"):
            continue
        if a.get("status") != "open":
            continue
        d = a.get("details") or {}
        out.append({
            "alert_id":   a.get("alert_id"),
            "user_id":    d.get("display_user") or unscope_user_id(a.get("user_id", "")),
            "activity":   d.get("activity", "transaction"),
            "risk_score": float(a.get("risk_score") or 0),
            "risk_level": d.get("risk_level", "unknown"),
            "tone":       d.get("tone", "verify"),
            "timestamp":  a.get("timestamp"),
            "detail":     d,
        })
    return {"challenges": out[:limit], "count": len(out[:limit]), "tenant_id": tid}


@app.post("/identity/verify-activity", tags=["Identity"],
          summary="Record the user's answer to a 'was this you?' challenge")
async def verify_activity(body: VerifyActivityRequest):
    """When the user denies an activity it becomes a confirmed incident: an
    alert is raised and the tenant's webhook fires immediately. When they
    confirm it, the event is recorded as benign so it stops being treated as
    an anomaly."""
    tenant    = get_current_tenant()
    tid       = tenant.get("tenant_id", "apeilo")
    scoped    = scope_user_id(body.user_id)
    detail    = body.detail or {}

    try:
        dynamo_put_event(
            user_id    = scoped,
            event_type = "activity_verification",
            scores     = {
                "activity":   body.activity,
                "confirmed":  body.confirmed,
                "risk_score": body.risk_score,
                # `amount` is read back by transaction_baseline() so a payment
                # the user approves raises what counts as normal for them.
                **detail,
            },
            tenant_id  = tid,
        )
    except Exception as e:
        logger.debug("verification persist skipped: %s", e)

    # Close the queued challenge, recording its outcome so a waiting connected
    # app (step-up flow) can read whether the user confirmed or denied.
    if body.alert_id:
        try:
            dynamo_resolve_alert(body.alert_id, scoped, "confirmed" if body.confirmed else "denied")
        except Exception as e:
            logger.debug("challenge resolve skipped: %s", e)

    # A confirmed step-up ("yes, the sign-in was me, despite the failures") is
    # trusted but not fully cleared — the burst really happened — so login risk
    # settles at a cautious ~45% rather than snapping back to near-zero.
    if body.confirmed and body.activity in ("login_stepup", "login", "login_failed"):
        try:
            update_from_event(tid, scoped, "login_score",
                              {"anomaly_probability": 0.45, "risk_level": "medium"})
        except Exception as e:
            logger.debug("step-up risk update skipped: %s", e)
        # Confirming the success also clears the sibling failed-login challenge
        # from the same burst, so the owner isn't left with a stale popup.
        if body.activity == "login_stepup":
            try:
                leftover = dynamo_find_open_alert(scoped, "challenge_login_failed")
                if leftover:
                    dynamo_resolve_alert(leftover["alert_id"], scoped, "confirmed")
            except Exception as e:
                logger.debug("sibling challenge cleanup skipped: %s", e)

    if not body.confirmed:
        # The user says it wasn't them — treat as a confirmed incident.
        payload = {
            "risk_level":          "critical",
            "primary_threats":     [f"user_denied_{body.activity}"],
            "recommended_actions": [
                "User denied this activity — treat as confirmed fraud.",
                "Force re-authentication and review recent activity.",
            ],
            "confirmed_by_user":   False,
            **detail,
        }
        try:
            dynamo_put_alert(
                user_id    = scoped,
                alert_type = f"denied_{body.activity}",
                risk_score = max(0.9, body.risk_score),
                details    = payload,
                tenant_id  = tid,
            )
        except Exception as e:
            logger.debug("verification alert skipped: %s", e)
        try:
            send_threat_webhook(
                tenant     = tenant,
                event_type = f"denied_{body.activity}",
                user_id    = body.user_id,
                risk_score = max(0.9, body.risk_score),
                risk_level = "critical",
                result     = payload,
            )
        except Exception as e:
            logger.warning("verification webhook failed: %s", e)

    return {
        "success":   True,
        "confirmed": body.confirmed,
        "escalated": not body.confirmed,
        "user_id":   body.user_id,
    }


@app.get("/identity/{user_id}/login-stats", tags=["Identity"],
         summary="Personal login-hour histogram + recent attempts")
async def get_login_stats(user_id: str, limit: int = 400):
    """Per-user login behaviour: how often they sign in at each hour, the risk
    curve adapted to that habit, their usual hour, and recent attempts with
    failed-attempt counts and timestamps."""
    tid = current_tenant_id()
    return login_stats(tid, scope_user_id(user_id), limit=limit)


@app.get("/identity/{user_id}/transaction-stats", tags=["Identity"],
         summary="Real transaction history + fraud aggregates")
async def get_transaction_stats(user_id: str, limit: int = 200):
    """Recent scored transactions for this user with amounts, timing, velocity
    and how risky transactions compare against their normal spend."""
    tid = current_tenant_id()
    return transaction_stats(tid, scope_user_id(user_id), limit=limit)


@app.delete("/profiles/{user_id}", tags=["Identity"], summary="Delete a user profile")
async def delete_user_profile(user_id: str):
    """Remove a profile from the current tenant. Stored events are left intact
    (they expire via TTL); only the aggregated profile is removed."""
    tid = current_tenant_id()
    ok  = delete_profile(tid, scope_user_id(user_id))
    return {"success": ok, "user_id": user_id, "tenant_id": tid}


@app.get("/identity/{user_id}", tags=["Identity"])
async def get_identity_profile(
    user_id: str,
    user: Optional[dict] = Depends(optional_auth),
):
    """Fetch a user's behavioral profile and recent event history from DynamoDB."""
    scoped  = scope_user_id(user_id)
    profile = dynamo_get_user_profile(scoped)
    events  = dynamo_get_user_events(scoped, limit=20)
    return {
        "user_id":       user_id,
        "tenant_id":     current_tenant_id(),
        "profile":       profile or {"user_id": user_id, "note": "No profile data yet"},
        "recent_events": events,
        "event_count":   len(events),
    }


@app.get("/identity/{user_id}/events", tags=["Identity"])
async def get_user_events(
    user_id:    str,
    limit:      int = 50,
    event_type: Optional[str] = None,
    user: Optional[dict] = Depends(optional_auth),
):
    """Fetch paginated detection event history for a user."""
    events = dynamo_get_user_events(scope_user_id(user_id), limit=limit, event_type=event_type)
    return {"user_id": user_id, "tenant_id": current_tenant_id(), "events": events, "count": len(events)}


# ══════════════════════════════════════════════════════
# ALERT MANAGEMENT
# ══════════════════════════════════════════════════════

@app.get("/alerts", tags=["Alerts"])
async def get_alerts(limit: int = 20, user: Optional[dict] = Depends(optional_auth)):
    """Fetch recent security alerts for the current tenant's dashboard."""
    tid    = current_tenant_id()
    alerts = dynamo_get_recent_alerts(limit=limit, tenant_id=tid)
    return {"alerts": alerts, "count": len(alerts), "tenant_id": tid}


@app.post("/alerts/{alert_id}/dismiss", tags=["Alerts"])
async def dismiss_alert(alert_id: str, user: dict = Depends(require_auth)):
    """Mark a security alert as dismissed."""
    ok = dynamo_dismiss_alert(alert_id)
    return {"success": ok, "alert_id": alert_id}


# ══════════════════════════════════════════════════════
# WEBSOCKET
# ══════════════════════════════════════════════════════

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    Real-time detection event stream.

    Connect from frontend JS:
      const socket = new WebSocket("ws://localhost:8000/ws");
      socket.onmessage = (e) => handleEvent(JSON.parse(e.data));

    Message types:
      connected       - initial handshake
      detection_event - a scored detection event
      heartbeat       - 30-second keepalive
    """
    await ws_manager.connect(ws)
    try:
        await ws.send_json({
            "type":      "connected",
            "message":   "AEGIS real-time feed active",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        while True:
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
                if data == "ping":
                    await ws.send_text("pong")
            except asyncio.TimeoutError:
                await ws.send_json({
                    "type":      "heartbeat",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)


async def broadcast_detection_event(event_type: str, user_id: str, result: dict):
    """
    Central sink for every scored event (called by all detection routers).

    Responsibilities:
      1. Persist the event to DynamoDB, namespaced by the caller's tenant.
      2. Stream it to connected dashboard WebSocket clients.
      3. On a critical score: persist an alert, POST the tenant's webhook, and
         (if still configured) fire the legacy SNS alert.

    Tenant scoping means one integrating app (e.g. SODA) never sees another's
    users, events, or alerts.
    """
    risk_score = float(
        result.get("unified_score")
        or result.get("spoof_probability")
        or result.get("anomaly_probability")
        or result.get("fraud_probability")
        or result.get("breach_probability")
        or 0.0
    )
    risk_level = result.get("risk_level", "unknown")

    tenant     = get_current_tenant()
    tenant_id  = tenant.get("tenant_id", "apeilo")
    scoped_uid = scope_user_id(user_id)

    # Scoring calls that carry no user (e.g. an ad-hoc password check from the
    # dashboard) must not spawn a phantom "anonymous" profile.
    is_real_user = bool(user_id) and user_id not in ("anonymous", "guest", "unknown")

    # ── 1. Persist the event + fold it into the user's live profile ──
    if is_real_user:
        try:
            dynamo_put_event(
                user_id    = scoped_uid,
                event_type = event_type,
                scores     = result,
                tenant_id  = tenant_id,
            )
        except Exception as e:
            logger.debug("event persist skipped: %s", e)

        try:
            update_from_event(tenant_id, scoped_uid, event_type, result)
        except Exception as e:
            logger.debug("profile update skipped: %s", e)

    # ── 1c. Queue a "was this you?" challenge for the Apeilo dashboard ──
    # Challenges are answered in Apeilo (by whoever monitors the account), not
    # in the connected app — that app only shows a passive alert toast.
    #
    # Login challenges are COALESCED: a brute-force burst produces one challenge
    # (raised on the 5th failed attempt) that keeps updating with the running
    # count, rather than a fresh popup per attempt. A success right after a burst
    # is a distinct "step-up" challenge that gates access to the connected app.
    if is_real_user:
        failed        = int(result.get("failed_10min") or 0)
        login_success = result.get("login_success")   # True / False / None
        alert_type    = None
        activity      = None
        coalesce      = True

        if event_type == "fraud_score" and result.get("challenge"):
            alert_type, activity, coalesce = "challenge_transaction", "transaction", False
        elif event_type in ("unified_risk", "login_score"):
            if login_success is False and failed >= 5:
                # Brute force in progress — ask once, coalesce further failures.
                alert_type, activity = "challenge_login_failed", "login_failed"
            elif login_success is True and failed >= 5:
                # A success on the heels of failures — step up before granting access.
                alert_type, activity = "challenge_login_stepup", "login_stepup"
            elif login_success is None and risk_score >= CRITICAL_THRESHOLD:
                # Other high-risk login (e.g. impossible travel), outcome unknown.
                alert_type, activity = "challenge_login", "login"

        if alert_type:
            details = {
                "activity":        activity,
                "risk_level":      risk_level,
                "tone":            result.get("challenge_tone", "verify"),
                "amount":          result.get("amount"),
                "hour":            result.get("hour", result.get("hour_of_day")),
                "source":          result.get("source"),
                "failed_attempts": failed,
                "amount_ratio":    result.get("amount_ratio_vs_history"),
                "display_user":    user_id,
            }
            try:
                if coalesce:
                    dynamo_upsert_alert(scoped_uid, alert_type, risk_score, details, tenant_id)
                else:
                    dynamo_put_alert(scoped_uid, alert_type, risk_score, details, tenant_id)
            except Exception as e:
                logger.debug("challenge queue skipped: %s", e)

    # ── 2. WebSocket broadcast (raw user id for display) ──
    message = {
        "type":       "detection_event",
        "event_type": event_type,
        "tenant_id":  tenant_id,
        "user_id":    user_id,
        "risk_score": round(risk_score, 4),
        "risk_level": risk_level,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "details":    result,
    }
    await ws_manager.broadcast(message)

    # ── 3. Critical: alert + webhook + (legacy) SNS ──
    if risk_score >= CRITICAL_THRESHOLD and is_real_user:
        event_id = result.get("event_id", str(uuid.uuid4()))

        try:
            dynamo_put_alert(
                user_id    = scoped_uid,
                alert_type = event_type,
                risk_score = risk_score,
                details    = {
                    "risk_level":          risk_level,
                    "primary_threats":     result.get("primary_threats", [event_type]),
                    "event_id":            event_id,
                    "recommended_actions": result.get("recommended_actions", []),
                },
                tenant_id  = tenant_id,
            )
        except Exception as e:
            logger.debug("alert persist skipped: %s", e)

        # Outbound webhook to the tenant's registered URL (replaces SNS).
        try:
            send_threat_webhook(
                tenant     = tenant,
                event_type = event_type,
                user_id    = user_id,
                risk_score = risk_score,
                risk_level = risk_level,
                result     = result,
            )
        except Exception as e:
            logger.warning("webhook delivery failed: %s", e)

        try:
            sns_alert_critical_risk(
                user_id             = user_id,
                unified_score       = risk_score,
                primary_threats     = result.get("primary_threats", [event_type]),
                event_id            = event_id,
                recommended_actions = result.get("recommended_actions", []),
            )
            cw_put_metric("CriticalAlertsFired", 1.0, "Count")
        except Exception as e:
            logger.warning("SNS alert failed: %s", e)


app.state.broadcast  = broadcast_detection_event
app.state.ws_manager = ws_manager


# ══════════════════════════════════════════════════════
# HEALTH & ROOT
# ══════════════════════════════════════════════════════

@app.get("/", tags=["System"])
async def root():
    return {
        "name":      "AEGIS Threat Detection API",
        "version":   "2.0.0",
        "status":    "operational",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aws_mode":  aws_available(),
        "auth_mode": "cognito" if COGNITO_USER_POOL_ID else "local_jwt",
    }


@app.get("/health", tags=["System"])
async def health():
    """Full system health: modules + AWS services + WebSocket connections."""
    h: Dict[str, Any] = {
        "status":    "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version":   "2.0.0",
        "ws_clients": len(ws_manager.active),
        "aws":       {},
        "modules":   {},
    }

    module_checks = {
        "gps":      ("src.gps.score_gps",          "score_gps_trajectory"),
        "login":    ("src.login.score_login",        "score_login_event"),
        "password": ("src.passwords.score_password", "score_password"),
        "fraud":    ("src.api.routers.fraud",        "load_artifacts"),
        "fusion":   ("src.fusion.risk_scoring",      "compute_unified_risk"),
    }
    all_ok = True
    for name, (module_path, func_name) in module_checks.items():
        try:
            import importlib
            mod = importlib.import_module(module_path)
            getattr(mod, func_name)
            h["modules"][name] = "healthy"
        except Exception as e:
            h["modules"][name] = f"degraded: {str(e)[:60]}"
            all_ok = False

    for name, (module_path, func_name) in {
        "breach": ("src.breach.hibp_checker",      "check_password_hash"),
        "device": ("src.device.device_risk_model", "score_device_risk"),
    }.items():
        try:
            import importlib
            mod = importlib.import_module(module_path)
            getattr(mod, func_name)
            h["modules"][name] = "healthy"
        except Exception as e:
            h["modules"][name] = f"degraded: {str(e)[:60]}"

    try:
        h["aws"] = check_aws_health()
    except Exception as e:
        h["aws"] = {"error": str(e)[:80]}

    if not all_ok:
        h["status"] = "degraded"

    return h


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc), "path": str(request.url)},
    )


@app.on_event("startup")
async def startup():
    # Create local DynamoDB tables + seed the bootstrap API key (no-op unless
    # APEILO_LOCAL_MODE is on and a local DynamoDB endpoint is configured).
    try:
        from src.utils.local_bootstrap import bootstrap
        bootstrap()
    except Exception as e:
        logger.warning("Local bootstrap skipped: %s", e)

    print("=" * 60)
    print("  APEILO THREAT DETECTION API v2.0")
    print("=" * 60)
    print(f"  Data : {'DynamoDB' if aws_available() else 'mock/local'}")
    print(f"  Auth : {'Cognito' if COGNITO_USER_POOL_ID else 'Local JWT'}")
    print(f"  Keys : {'required' if REQUIRE_API_KEY else 'optional (default tenant)'}")
    print(f"  WS   : ws://localhost:8000/ws")
    print(f"  Docs : http://localhost:8000/docs")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown():
    print("AEGIS API shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
