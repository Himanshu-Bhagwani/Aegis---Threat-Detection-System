# main.py
from fastapi import Depends, FastAPI
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import firebase_config
from .auth import login_user, signup_user, verify_token

# Initialize Firebase first
firebase_config.initialize_firebase()

# Create security scheme
security = HTTPBearer()

app = FastAPI(
    title="Ageis API",
    description="Firebase + FastAPI Backend",
    version="1.0.0",
)

allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class User(BaseModel):
    email: str
    password: str


@app.post("/signup")
async def signup(user: User):
    return signup_user(user.email, user.password)


@app.post("/login")
async def login(user: User):
    return login_user(user.email, user.password)


@app.get("/secure-data")
async def secure_data(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Extract token from Bearer header value
    id_token = credentials.credentials

    # Re-use existing verify_token helper by mimicking a FastAPI request
    class MockRequest:
        def __init__(self, token: str):
            self.headers = {"Authorization": f"Bearer {token}"}

    mock_request = MockRequest(id_token)
    user = await verify_token(mock_request)
    return {"message": f"Hello {user['email']}, this is protected data!"}


@app.get("/")
async def root():
    return {"message": "Welcome to Firebase + FastAPI backend"}
