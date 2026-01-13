# auth.py
from fastapi import HTTPException, Request
from firebase_admin import auth as firebase_auth
import requests
import os
from dotenv import load_dotenv
load_dotenv()

# Load credentials
# cred = credentials.Certificate("./ageis-dead3-firebase-adminsdk-fbsvc-589b74d7f8.json")

def signup_user(email, password):
    try:
        user = firebase_auth.create_user(
            email=email,
            password=password
        )
        return {"uid": user.uid, "email": user.email}
    except Exception as e:
        return {"error": str(e)}

def login_user(email, password):
    try:
        # Use Firebase Auth REST API to get ID token
        api_key = os.getenv("API_KEY") 
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
        
        payload = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }
        
        response = requests.post(url, json=payload)
        data = response.json()
        
        if "idToken" in data:
            return {
                "idToken": data["idToken"],
                "email": data["email"],
                "uid": data["localId"]
            }
        else:
            return {"error": data.get("error", {}).get("message", "Login failed")}
            
    except Exception as e:
        return {"error": str(e)}


async def verify_token(request: Request):
    id_token = request.headers.get("Authorization")

    if not id_token:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    if id_token.startswith("Bearer "):
        id_token = id_token.split("Bearer ")[1]

    try:
        decoded_token = firebase_auth.verify_id_token(id_token)
        return decoded_token
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")