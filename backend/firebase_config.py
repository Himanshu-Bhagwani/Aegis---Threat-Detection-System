# firebase_config.py
import os
from pathlib import Path

import firebase_admin
from firebase_admin import credentials

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CREDENTIAL_PATH = (PACKAGE_DIR / "ageis-dead3-firebase-adminsdk-fbsvc-589b74d7f8.json").resolve()


def initialize_firebase():
    """Initialize Firebase if not already initialized"""
    if not firebase_admin._apps:
        raw_path = os.getenv("FIREBASE_KEY_PATH")
        credential_path = DEFAULT_CREDENTIAL_PATH

        if raw_path:
            candidate = Path(raw_path).expanduser()
            if not candidate.is_absolute():
                candidate = (Path.cwd() / candidate).resolve()

            if candidate.exists():
                credential_path = candidate
            else:
                package_relative = (PACKAGE_DIR / raw_path).resolve()
                if package_relative.exists():
                    credential_path = package_relative
                else:
                    print(
                        f"Warning: FIREBASE_KEY_PATH '{raw_path}' not found. "
                        f"Falling back to default credentials at {credential_path}."
                    )

        cred = credentials.Certificate(str(credential_path))
        firebase_admin.initialize_app(cred)
        print("Firebase initialized successfully")
    else:
        print("Firebase already initialized")
