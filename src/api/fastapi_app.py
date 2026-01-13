"""
Aegis Threat Detection API

A unified multi-layer threat detection system that identifies suspicious behavior
across GPS movement, login patterns, password integrity, and transactional fraud,
combining them into a single risk score.

Endpoints:
- /gps/*     - GPS spoofing detection
- /login/*   - Login anomaly detection
- /password/* - Password strength and breach risk assessment
- /fraud/*   - Transaction fraud detection
- /risk/*    - Unified risk scoring combining all layers
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import time
import uuid
from datetime import datetime

# Import routers
from src.api.routers import fraud as fraud_router
from src.api.routers import gps as gps_router
from src.api.routers import login as login_router
from src.api.routers import password as password_router
from src.api.routers import risk as risk_router


# ============================================================
# Simple Mock Authentication (for testing without Firebase)
# ============================================================
class UserCredentials(BaseModel):
    email: str
    password: str

# In-memory user store (for demo/testing only)
MOCK_USERS = {
    "demo@aegis.com": {"password": "demo123", "uid": "demo-user-001"},
    "admin@aegis.com": {"password": "admin123", "uid": "admin-user-001"},
    "test@test.com": {"password": "test123", "uid": "test-user-001"},
}

# Create FastAPI app
app = FastAPI(
    title="Aegis Threat Detection API",
    description="""
## Multi-Layer Threat Detection System

Aegis provides comprehensive threat detection across multiple vectors:

### 🛰️ GPS Spoofing Detection
Analyzes GPS trajectories to detect location spoofing using ML models (Isolation Forest, GBM, Autoencoder, CNN-RNN).

### 🔐 Login Anomaly Detection
Identifies anomalous login patterns using models trained on the LANL dataset with rule-based heuristics.

### 🔑 Password Risk Assessment
Evaluates password strength and breach risk using feature engineering and ML classification.

### 💳 Transaction Fraud Detection
Flags potentially fraudulent transactions using XGBoost and behavioral analysis.

### ⚡ Unified Risk Scoring
Combines all threat signals into a single risk score using configurable fusion strategies.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "health", "description": "Health check endpoints"},
        {"name": "gps", "description": "GPS spoofing detection"},
        {"name": "login", "description": "Login anomaly detection"},
        {"name": "password", "description": "Password risk assessment"},
        {"name": "fraud", "description": "Transaction fraud detection"},
        {"name": "risk", "description": "Unified risk scoring"},
    ]
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        # Add your production domains here
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Include routers
app.include_router(gps_router.router, prefix="/gps", tags=["gps"])
app.include_router(login_router.router, prefix="/login", tags=["login"])
app.include_router(password_router.router, prefix="/password", tags=["password"])
app.include_router(fraud_router.router, prefix="/fraud", tags=["fraud"])
app.include_router(risk_router.router, prefix="/risk", tags=["risk"])


# Root endpoint
@app.get("/", tags=["health"])
async def root():
    """Welcome endpoint with API overview."""
    return {
        "name": "Aegis Threat Detection API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "gps_scoring": "/gps/score",
            "login_scoring": "/login/score",
            "password_scoring": "/password/score",
            "fraud_scoring": "/fraud/score",
            "unified_risk": "/risk/unified"
        }
    }


# ============================================================
# Mock Auth Endpoints (for testing dashboard without Firebase)
# ============================================================

@app.post("/signup", tags=["auth"])
async def signup(user: UserCredentials):
    """
    Mock signup endpoint for testing.
    Creates a new user in the demo user store.
    """
    if user.email in MOCK_USERS:
        return {"error": "User already exists"}
    
    new_uid = f"user-{uuid.uuid4().hex[:8]}"
    MOCK_USERS[user.email] = {
        "password": user.password,
        "uid": new_uid
    }
    
    return {
        "uid": new_uid,
        "email": user.email,
        "message": "User created successfully"
    }


@app.post("/login", tags=["auth"])
async def login(user: UserCredentials):
    """
    Mock login endpoint for testing.
    
    Pre-configured test accounts:
    - demo@aegis.com / demo123
    - admin@aegis.com / admin123
    - test@test.com / test123
    
    Or signup to create your own!
    """
    if user.email not in MOCK_USERS:
        return {"error": "User not found. Try demo@aegis.com / demo123"}
    
    stored_user = MOCK_USERS[user.email]
    
    if stored_user["password"] != user.password:
        return {"error": "Invalid password"}
    
    # Generate a mock token
    mock_token = f"mock-token-{uuid.uuid4().hex}"
    
    return {
        "idToken": mock_token,
        "email": user.email,
        "uid": stored_user["uid"],
        "message": "Login successful!"
    }


# Health check endpoint
@app.get("/health", tags=["health"])
async def health():
    """
    Comprehensive health check for all system components.
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {}
    }
    
    # Check each component
    components_healthy = True
    
    # GPS health
    try:
        from src.gps.score_gps import score_gps_trajectory
        health_status["components"]["gps"] = "healthy"
    except Exception as e:
        health_status["components"]["gps"] = f"degraded: {str(e)}"
        components_healthy = False
    
    # Login health
    try:
        from src.login.score_login import score_login_event
        health_status["components"]["login"] = "healthy"
    except Exception as e:
        health_status["components"]["login"] = f"degraded: {str(e)}"
        components_healthy = False
    
    # Password health
    try:
        from src.passwords.score_password import score_password
        health_status["components"]["password"] = "healthy"
    except Exception as e:
        health_status["components"]["password"] = f"degraded: {str(e)}"
        components_healthy = False
    
    # Fraud health
    try:
        from src.api.routers.fraud import load_artifacts
        health_status["components"]["fraud"] = "healthy"
    except Exception as e:
        health_status["components"]["fraud"] = f"degraded: {str(e)}"
        components_healthy = False
    
    # Fusion health
    try:
        from src.fusion.risk_scoring import compute_unified_risk
        health_status["components"]["fusion"] = "healthy"
    except Exception as e:
        health_status["components"]["fusion"] = f"degraded: {str(e)}"
        components_healthy = False
    
    if not components_healthy:
        health_status["status"] = "degraded"
    
    return health_status


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url)
        }
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models and connections on startup."""
    print("=" * 60)
    print("🛡️  AEGIS THREAT DETECTION API STARTING")
    print("=" * 60)
    print("Loading detection models...")
    
    # Optionally pre-load models here for faster first requests
    # This is commented out to allow lazy loading for faster startup
    # try:
    #     from src.gps.score_gps import load_isolation_forest
    #     load_isolation_forest()
    #     print("  ✓ GPS models loaded")
    # except Exception as e:
    #     print(f"  ⚠ GPS models not available: {e}")
    
    print("API ready at http://localhost:8000")
    print("Documentation at http://localhost:8000/docs")
    print("=" * 60)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("Aegis API shutting down...")


# Run with: uvicorn src.api.fastapi_app:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
