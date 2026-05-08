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
from datetime import datetime, timezone
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

# ── Auth dependency ───────────────────────────────────
from src.api.auth_deps import require_auth, optional_auth, create_local_jwt

# ── AWS helpers ───────────────────────────────────────
from src.utils.aws_utils import (
    cognito_sign_up,
    cognito_sign_in,
    cognito_sign_out,
    dynamo_put_event,
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

CRITICAL_THRESHOLD = float(os.getenv("CRITICAL_RISK_THRESHOLD", "0.75"))
HIGH_THRESHOLD     = float(os.getenv("HIGH_RISK_THRESHOLD",     "0.50"))


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
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
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


# ── Include detection routers ─────────────────────────
app.include_router(gps_router.router,      prefix="/gps",      tags=["GPS Spoofing"])
app.include_router(login_router.router,    prefix="/login",    tags=["Login Anomaly"])
app.include_router(password_router.router, prefix="/password", tags=["Password Risk"])
app.include_router(fraud_router.router,    prefix="/fraud",    tags=["Fraud Detection"])
app.include_router(risk_router.router,     prefix="/risk",     tags=["Unified Risk"])
app.include_router(breach_router.router,   prefix="/breach",   tags=["Breach Detection"])
app.include_router(device_router.router,   prefix="/device",   tags=["Device Fingerprint"])


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

@app.get("/identity/{user_id}", tags=["Identity"])
async def get_identity_profile(
    user_id: str,
    user: dict = Depends(require_auth),
):
    """Fetch a user's behavioral profile and recent event history from DynamoDB."""
    profile = dynamo_get_user_profile(user_id)
    events  = dynamo_get_user_events(user_id, limit=20)
    return {
        "user_id":       user_id,
        "profile":       profile or {"user_id": user_id, "note": "No profile data yet"},
        "recent_events": events,
        "event_count":   len(events),
    }


@app.get("/identity/{user_id}/events", tags=["Identity"])
async def get_user_events(
    user_id:    str,
    limit:      int = 50,
    event_type: Optional[str] = None,
    user: dict  = Depends(require_auth),
):
    """Fetch paginated detection event history for a user."""
    events = dynamo_get_user_events(user_id, limit=limit, event_type=event_type)
    return {"user_id": user_id, "events": events, "count": len(events)}


# ══════════════════════════════════════════════════════
# ALERT MANAGEMENT
# ══════════════════════════════════════════════════════

@app.get("/alerts", tags=["Alerts"])
async def get_alerts(limit: int = 20, user: dict = Depends(require_auth)):
    """Fetch recent security alerts for the dashboard."""
    alerts = dynamo_get_recent_alerts(limit=limit)
    return {"alerts": alerts, "count": len(alerts)}


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
    Called by routers after scoring.
    Broadcasts to WebSocket clients and fires SNS alert if critical.
    """
    risk_score = float(
        result.get("unified_score")
        or result.get("spoof_probability")
        or result.get("anomaly_probability")
        or result.get("fraud_probability")
        or result.get("breach_probability")
        or 0.0
    )

    message = {
        "type":       "detection_event",
        "event_type": event_type,
        "user_id":    user_id,
        "risk_score": round(risk_score, 4),
        "risk_level": result.get("risk_level", "unknown"),
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "details":    result,
    }
    await ws_manager.broadcast(message)

    if risk_score >= CRITICAL_THRESHOLD:
        event_id = result.get("event_id", str(uuid.uuid4()))
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
    print("=" * 60)
    print("  AEGIS THREAT DETECTION API v2.0")
    print("=" * 60)
    print(f"  AWS  : {'active' if aws_available() else 'mock/local'}")
    print(f"  Auth : {'Cognito' if COGNITO_USER_POOL_ID else 'Local JWT'}")
    print(f"  WS   : ws://localhost:8000/ws")
    print(f"  Docs : http://localhost:8000/docs")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown():
    print("AEGIS API shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
