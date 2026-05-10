"""
AEGIS AWS Services Layer
========================
Complete boto3 wrappers for:
  - Cognito  : User auth, JWT validation, user pool management
  - S3       : Model artifact storage and retrieval
  - DynamoDB : Event persistence, user profiles, behavioral baselines
  - CloudWatch: Metrics emission, structured logging
  - SNS      : Real-time alerting on critical risk events
  - Secrets Manager: Secure credential storage

All functions degrade gracefully — if AWS is not configured the system
falls back to local/mock behaviour without crashing.
"""

import os
import io
import json
import time
import uuid
import logging
import hashlib
import hmac
import base64
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from decimal import Decimal

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Configuration (pulled from env / .env)
# ─────────────────────────────────────────────
AWS_REGION            = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")

COGNITO_USER_POOL_ID  = os.getenv("COGNITO_USER_POOL_ID", "")
COGNITO_CLIENT_ID     = os.getenv("COGNITO_CLIENT_ID", "")
COGNITO_CLIENT_SECRET = os.getenv("COGNITO_CLIENT_SECRET", "")

S3_MODELS_BUCKET      = os.getenv("S3_MODELS_BUCKET", "aegis-ml-models")
S3_LOGS_BUCKET        = os.getenv("S3_LOGS_BUCKET", "aegis-event-logs")

DYNAMODB_EVENTS_TABLE   = os.getenv("DYNAMODB_EVENTS_TABLE", "Apeilo-events")
DYNAMODB_PROFILES_TABLE = os.getenv("DYNAMODB_PROFILES_TABLE", "apeilo-users-profile")
DYNAMODB_ALERTS_TABLE   = os.getenv("DYNAMODB_ALERTS_TABLE", "Apeilo-alerts")

CLOUDWATCH_NAMESPACE  = os.getenv("CLOUDWATCH_NAMESPACE", "AEGIS/ThreatDetection")
SNS_ALERTS_TOPIC_ARN  = os.getenv("SNS_ALERTS_TOPIC_ARN", "")

# ─────────────────────────────────────────────
# AWS Client Factory (lazy, cached)
# ─────────────────────────────────────────────
_clients: Dict[str, Any] = {}


def _boto_kwargs() -> Dict[str, str]:
    """Build common boto3 kwargs from env."""
    kwargs: Dict[str, str] = {"region_name": AWS_REGION}
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"]     = AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
    return kwargs


def get_client(service: str):
    """Return a cached boto3 client for the given service."""
    if service not in _clients:
        try:
            _clients[service] = boto3.client(service, **_boto_kwargs())
        except Exception as exc:
            logger.warning("Could not create boto3 client for %s: %s", service, exc)
            return None
    return _clients[service]


def get_resource(service: str):
    """Return a cached boto3 resource for the given service."""
    key = f"resource_{service}"
    if key not in _clients:
        try:
            _clients[key] = boto3.resource(service, **_boto_kwargs())
        except Exception as exc:
            logger.warning("Could not create boto3 resource for %s: %s", service, exc)
            return None
    return _clients[key]


def aws_available() -> bool:
    """Quick check: are AWS credentials configured?"""
    return bool(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)


# ═══════════════════════════════════════════════════════
# SECTION 1 — COGNITO AUTH
# ═══════════════════════════════════════════════════════

def _cognito_secret_hash(username: str) -> Optional[str]:
    """Compute HMAC secret hash required by Cognito when client secret is set."""
    if not COGNITO_CLIENT_SECRET:
        return None
    message = username + COGNITO_CLIENT_ID
    dig = hmac.new(
        COGNITO_CLIENT_SECRET.encode("utf-8"),
        msg=message.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    return base64.b64encode(dig).decode()


def cognito_sign_up(
    email: str,
    password: str,
    given_name: str = "",
    family_name: str = "",
) -> Dict:
    """
    Register a new user in Cognito User Pool.
    Returns: { success, user_sub, message }
    """
    client = get_client("cognito-idp")
    if not client or not COGNITO_USER_POOL_ID or not COGNITO_CLIENT_ID:
        return {
            "success":  True,
            "user_sub": f"mock-{uuid.uuid4().hex[:8]}",
            "message":  "User created (mock mode — AWS Cognito not configured)",
            "mock":     True,
        }

    kwargs: Dict[str, Any] = {
        "ClientId":       COGNITO_CLIENT_ID,
        "Username":       email,
        "Password":       password,
        "UserAttributes": [{"Name": "email", "Value": email}],
    }
    if given_name:
        kwargs["UserAttributes"].append({"Name": "given_name",  "Value": given_name})
    if family_name:
        kwargs["UserAttributes"].append({"Name": "family_name", "Value": family_name})

    secret = _cognito_secret_hash(email)
    if secret:
        kwargs["SecretHash"] = secret

    try:
        resp = client.sign_up(**kwargs)
        return {
            "success":   True,
            "user_sub":  resp["UserSub"],
            "confirmed": resp.get("UserConfirmed", False),
            "message":   "Registration successful. Check your email for the verification code.",
        }
    except (ClientError, NoCredentialsError) as e:
        err  = getattr(e, "response", {}).get("Error", {})
        code = err.get("Code", type(e).__name__)
        msg  = err.get("Message", str(e))
        logger.error("Cognito sign_up error: %s — %s", code, msg)
        return {"success": False, "error": code, "message": msg}


def _mock_signin_response(email: str) -> Dict:
    """Return a mock sign-in response for development / unconfigured Cognito."""
    token = f"mock-access-{uuid.uuid4().hex}"
    return {
        "success":       True,
        "access_token":  token,
        "id_token":      f"mock-id-{uuid.uuid4().hex}",
        "refresh_token": f"mock-refresh-{uuid.uuid4().hex}",
        "expires_in":    3600,
        "user_sub":      "mock-user-001",
        "email":         email,
        "mock":          True,
        "mock_mode":     True,
    }


def cognito_sign_in(email: str, password: str) -> Dict:
    """
    Authenticate a user and return JWT tokens.
    Returns: { success, access_token, id_token, refresh_token, expires_in }
    Falls back to mock response when Cognito is not configured or unreachable.
    """
    client = get_client("cognito-idp")
    if not client or not COGNITO_CLIENT_ID:
        return _mock_signin_response(email)

    auth_params: Dict[str, str] = {
        "USERNAME": email,
        "PASSWORD": password,
    }
    secret = _cognito_secret_hash(email)
    if secret:
        auth_params["SECRET_HASH"] = secret

    try:
        resp   = client.initiate_auth(
            AuthFlow="USER_PASSWORD_AUTH",
            AuthParameters=auth_params,
            ClientId=COGNITO_CLIENT_ID,
        )
        result = resp["AuthenticationResult"]
        return {
            "success":       True,
            "access_token":  result["AccessToken"],
            "id_token":      result["IdToken"],
            "refresh_token": result.get("RefreshToken", ""),
            "expires_in":    result.get("ExpiresIn", 3600),
            "token_type":    result.get("TokenType", "Bearer"),
        }
    except (ClientError, NoCredentialsError) as e:
        err  = getattr(e, "response", {}).get("Error", {})
        code = err.get("Code", type(e).__name__)
        msg  = err.get("Message", str(e))
        logger.error("Cognito sign_in error: %s — %s", code, msg)
        if code in ("ResourceNotFoundException", "UnrecognizedClientException",
                    "InvalidClientTokenId", "InvalidParameterException", "NoCredentialsError"):
            logger.warning("Cognito not reachable — returning mock sign-in response")
            return _mock_signin_response(email)
        return {"success": False, "error": code, "message": msg}


def cognito_verify_token(access_token: str) -> Dict:
    """
    Validate an access token and return user attributes.
    Returns: { valid, user_sub, email, username } or { valid: False, error }
    Mock tokens (mock-*) are always accepted for development without hitting AWS.
    """
    # Always accept mock tokens — checked before any network call
    if access_token.startswith("mock-"):
        return {
            "valid":    True,
            "user_sub": "mock-user-001",
            "email":    "demo@aegis.com",
            "username": "demo@aegis.com",
            "mock":     True,
        }

    client = get_client("cognito-idp")
    if not client or not COGNITO_USER_POOL_ID:
        return {"valid": False, "error": "AWS Cognito not configured"}

    try:
        resp  = client.get_user(AccessToken=access_token)
        attrs = {a["Name"]: a["Value"] for a in resp["UserAttributes"]}
        return {
            "valid":    True,
            "user_sub": attrs.get("sub", ""),
            "email":    attrs.get("email", ""),
            "username": resp["Username"],
        }
    except (ClientError, NoCredentialsError) as e:
        code = getattr(e, "response", {}).get("Error", {}).get("Code", str(e))
        logger.warning("Cognito token validation failed: %s", code)
        return {"valid": False, "error": code}


def cognito_sign_out(access_token: str) -> bool:
    """Global sign-out — invalidates all tokens for the user."""
    client = get_client("cognito-idp")
    if not client:
        return True
    try:
        client.global_sign_out(AccessToken=access_token)
        return True
    except (ClientError, NoCredentialsError) as e:
        logger.warning("Cognito sign_out error: %s", str(e))
        return False


def cognito_get_user_profile(username: str) -> Dict:
    """Admin fetch of full user profile from Cognito."""
    client = get_client("cognito-idp")
    if not client or not COGNITO_USER_POOL_ID:
        return {"username": username, "mock": True}
    try:
        resp  = client.admin_get_user(UserPoolId=COGNITO_USER_POOL_ID, Username=username)
        attrs = {a["Name"]: a["Value"] for a in resp.get("UserAttributes", [])}
        return {
            "username":   resp["Username"],
            "status":     resp["UserStatus"],
            "enabled":    resp["Enabled"],
            "created":    resp["UserCreateDate"].isoformat(),
            "modified":   resp["UserLastModifiedDate"].isoformat(),
            "email":      attrs.get("email", ""),
            "given_name": attrs.get("given_name", ""),
            "sub":        attrs.get("sub", ""),
        }
    except (ClientError, NoCredentialsError) as e:
        logger.warning("cognito_get_user_profile: %s", str(e))
        return {}


# ═══════════════════════════════════════════════════════
# SECTION 2 — S3 MODEL STORAGE
# ═══════════════════════════════════════════════════════

def s3_upload_model(local_path: str, s3_key: str, bucket: str = S3_MODELS_BUCKET) -> bool:
    """Upload a model file to S3. Returns True on success."""
    client = get_client("s3")
    if not client:
        logger.warning("S3 not available, skipping upload of %s", local_path)
        return False
    try:
        client.upload_file(local_path, bucket, s3_key)
        logger.info("Uploaded %s → s3://%s/%s", local_path, bucket, s3_key)
        return True
    except (ClientError, FileNotFoundError) as e:
        logger.error("S3 upload failed: %s", e)
        return False


def s3_download_model(s3_key: str, local_path: str, bucket: str = S3_MODELS_BUCKET) -> bool:
    """Download a model file from S3 to local_path. Returns True on success."""
    client = get_client("s3")
    if not client:
        return False
    try:
        client.download_file(bucket, s3_key, local_path)
        logger.info("Downloaded s3://%s/%s → %s", bucket, s3_key, local_path)
        return True
    except (ClientError, NoCredentialsError) as e:
        code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
        if code == "404":
            logger.warning("Model not found in S3: %s", s3_key)
        else:
            logger.error("S3 download failed: %s", e)
        return False


def s3_model_exists(s3_key: str, bucket: str = S3_MODELS_BUCKET) -> bool:
    """Check if a model artifact exists in S3."""
    client = get_client("s3")
    if not client:
        return False
    try:
        client.head_object(Bucket=bucket, Key=s3_key)
        return True
    except (ClientError, NoCredentialsError):
        return False


def s3_upload_bytes(
    data: bytes,
    s3_key: str,
    bucket: str = S3_LOGS_BUCKET,
    content_type: str = "application/json",
) -> bool:
    """Upload raw bytes to S3."""
    client = get_client("s3")
    if not client:
        return False
    try:
        client.put_object(Bucket=bucket, Key=s3_key, Body=data, ContentType=content_type)
        return True
    except (ClientError, NoCredentialsError) as e:
        logger.error("S3 put_object failed: %s", e)
        return False


def s3_list_models(prefix: str = "", bucket: str = S3_MODELS_BUCKET) -> List[str]:
    """List model artifacts in S3."""
    client = get_client("s3")
    if not client:
        return []
    try:
        resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        return [obj["Key"] for obj in resp.get("Contents", [])]
    except (ClientError, NoCredentialsError) as e:
        logger.error("S3 list_objects failed: %s", e)
        return []


def ensure_model_local(
    s3_key: str,
    local_path: str,
    bucket: str = S3_MODELS_BUCKET,
) -> bool:
    """
    Ensure a model file is available locally.
    If the file is missing locally, downloads from S3.
    Returns True if file is ready to use.
    """
    if os.path.exists(local_path):
        return True
    logger.info("Model not found locally, fetching from S3: %s", s3_key)
    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    return s3_download_model(s3_key, local_path, bucket)


# ═══════════════════════════════════════════════════════
# SECTION 3 — DYNAMODB EVENT STORE
# ═══════════════════════════════════════════════════════

def _to_decimal(obj: Any) -> Any:
    """Recursively convert floats to Decimal for DynamoDB."""
    if isinstance(obj, float):
        return Decimal(str(round(obj, 6)))
    if isinstance(obj, dict):
        return {k: _to_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_decimal(i) for i in obj]
    return obj


def _from_decimal(obj: Any) -> Any:
    """Recursively convert Decimal back to float for JSON responses."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _from_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_decimal(i) for i in obj]
    return obj


def dynamo_put_event(
    user_id: str,
    event_type: str,
    scores: Dict,
    raw_input: Optional[Dict] = None,
    table_name: str = DYNAMODB_EVENTS_TABLE,
) -> Optional[str]:
    """
    Persist a detection event to DynamoDB.

    Schema:
      PK (user_id): partition key
      SK (sk):      ISO timestamp + UUID, sort key
      event_type, scores, raw_input, ttl (90-day auto-expire)

    Returns event_id on success, None on failure / AWS not configured.
    """
    resource = get_resource("dynamodb")
    if not resource:
        logger.debug("DynamoDB not available, event not persisted")
        return None

    event_id  = str(uuid.uuid4())
    now       = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    ttl       = int(now.timestamp()) + (90 * 86400)

    # Sort key in DynamoDB table is `event_id`. Prefix with timestamp so the
    # natural string ordering of event_id reflects chronological order.
    sortable_event_id = f"{timestamp}#{event_id}"
    item: Dict[str, Any] = {
        "user_id":    user_id,
        "event_id":   sortable_event_id,
        "event_uuid": event_id,
        "event_type": event_type,
        "timestamp":  timestamp,
        "scores":     _to_decimal(scores),
        "ttl":        ttl,
    }
    if raw_input:
        item["raw_input"] = _to_decimal(raw_input)

    try:
        table = resource.Table(table_name)
        table.put_item(Item=item)
        logger.info("Event persisted: %s / %s", user_id, event_id)
        return event_id
    except (ClientError, NoCredentialsError) as e:
        logger.error("DynamoDB put_event failed: %s", str(e))
        return None


def dynamo_get_user_events(
    user_id: str,
    limit: int = 50,
    event_type: Optional[str] = None,
    table_name: str = DYNAMODB_EVENTS_TABLE,
) -> List[Dict]:
    """Fetch recent events for a user (newest first). Optionally filter by event_type."""
    resource = get_resource("dynamodb")
    if not resource:
        return []

    from boto3.dynamodb.conditions import Key, Attr

    try:
        table  = resource.Table(table_name)
        kwargs: Dict[str, Any] = {
            "KeyConditionExpression": Key("user_id").eq(user_id),
            "ScanIndexForward": False,
            "Limit": limit,
        }
        if event_type:
            kwargs["FilterExpression"] = Attr("event_type").eq(event_type)
        resp = table.query(**kwargs)
        return [_from_decimal(item) for item in resp.get("Items", [])]
    except (ClientError, NoCredentialsError) as e:
        logger.error("DynamoDB query failed: %s", str(e))
        return []


def dynamo_update_user_profile(
    user_id: str,
    profile_data: Dict,
    table_name: str = DYNAMODB_PROFILES_TABLE,
) -> bool:
    """
    Upsert a user behavioral profile.
    Fields: typical_hour_range, usual_location_bbox, avg_transaction_amount,
            known_device_ids, risk_baseline, last_login_ip, etc.
    """
    resource = get_resource("dynamodb")
    if not resource:
        return False

    profile_data               = _to_decimal(profile_data)
    profile_data["user_id"]    = user_id
    profile_data["last_updated"] = datetime.now(timezone.utc).isoformat()

    try:
        table = resource.Table(table_name)
        table.put_item(Item=profile_data)
        return True
    except (ClientError, NoCredentialsError) as e:
        logger.error("DynamoDB profile update failed: %s", str(e))
        return False


def dynamo_get_user_profile(
    user_id: str,
    table_name: str = DYNAMODB_PROFILES_TABLE,
) -> Optional[Dict]:
    """Fetch a user's behavioral profile."""
    resource = get_resource("dynamodb")
    if not resource:
        return None
    try:
        table = resource.Table(table_name)
        resp  = table.get_item(Key={"user_id": user_id})
        item  = resp.get("Item")
        return _from_decimal(item) if item else None
    except (ClientError, NoCredentialsError) as e:
        logger.error("DynamoDB get_profile failed: %s", str(e))
        return None


def dynamo_put_alert(
    user_id: str,
    alert_type: str,
    risk_score: float,
    details: Dict,
    table_name: str = DYNAMODB_ALERTS_TABLE,
) -> Optional[str]:
    """Persist a security alert to DynamoDB."""
    resource = get_resource("dynamodb")
    if not resource:
        return None

    alert_id = str(uuid.uuid4())
    now      = datetime.now(timezone.utc)
    ttl      = int(now.timestamp()) + (30 * 86400)

    item = {
        "alert_id":   alert_id,
        "user_id":    user_id,
        "alert_type": alert_type,
        "risk_score": _to_decimal(risk_score),
        "details":    _to_decimal(details),
        "timestamp":  now.isoformat(),
        "status":     "open",
        "ttl":        ttl,
    }
    try:
        table = resource.Table(table_name)
        table.put_item(Item=item)
        return alert_id
    except (ClientError, NoCredentialsError) as e:
        logger.error("DynamoDB put_alert failed: %s", str(e))
        return None


def dynamo_get_recent_alerts(
    limit: int = 20,
    table_name: str = DYNAMODB_ALERTS_TABLE,
) -> List[Dict]:
    """Scan for recent alerts across all users (for the dashboard alert panel)."""
    resource = get_resource("dynamodb")
    if not resource:
        return []
    try:
        table = resource.Table(table_name)
        resp  = table.scan(Limit=limit)
        items = sorted(
            [_from_decimal(i) for i in resp.get("Items", [])],
            key=lambda x: x.get("timestamp", ""),
            reverse=True,
        )
        return items[:limit]
    except (ClientError, NoCredentialsError) as e:
        logger.error("DynamoDB scan alerts failed: %s", str(e))
        return []


def dynamo_dismiss_alert(
    alert_id: str,
    user_id: Optional[str] = None,
    table_name: str = DYNAMODB_ALERTS_TABLE,
) -> bool:
    """Mark an alert as dismissed. Requires user_id since it's the partition key."""
    resource = get_resource("dynamodb")
    if not resource:
        return False
    try:
        table = resource.Table(table_name)
        if not user_id:
            # Caller didn't pass user_id — locate the alert by scan to get it.
            from boto3.dynamodb.conditions import Attr
            resp = table.scan(
                FilterExpression=Attr("alert_id").eq(alert_id),
                Limit=1,
            )
            items = resp.get("Items", [])
            if not items:
                logger.warning("dynamo_dismiss_alert: alert_id %s not found", alert_id)
                return False
            user_id = items[0]["user_id"]
        table.update_item(
            Key={"user_id": user_id, "alert_id": alert_id},
            UpdateExpression="SET #s = :s, dismissed_at = :t",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":s": "dismissed",
                ":t": datetime.now(timezone.utc).isoformat(),
            },
        )
        return True
    except (ClientError, NoCredentialsError) as e:
        logger.error("DynamoDB dismiss_alert failed: %s", str(e))
        return False


# ═══════════════════════════════════════════════════════
# SECTION 4 — CLOUDWATCH METRICS
# ═══════════════════════════════════════════════════════

def cw_put_metric(
    metric_name: str,
    value: float,
    unit: str = "None",
    dimensions: Optional[List[Dict[str, str]]] = None,
    namespace: str = CLOUDWATCH_NAMESPACE,
) -> bool:
    """
    Emit a single custom metric to CloudWatch.

    Examples:
      cw_put_metric("RiskScore", 0.87, dimensions=[{"Name":"Module","Value":"GPS"}])
      cw_put_metric("InferenceLatencyMs", 42.3, unit="Milliseconds")
    """
    client = get_client("cloudwatch")
    if not client:
        logger.debug("CloudWatch not available: %s = %s", metric_name, value)
        return False

    metric: Dict[str, Any] = {
        "MetricName": metric_name,
        "Value":      value,
        "Unit":       unit,
        "Timestamp":  datetime.now(timezone.utc),
    }
    if dimensions:
        metric["Dimensions"] = dimensions

    try:
        client.put_metric_data(Namespace=namespace, MetricData=[metric])
        return True
    except NoCredentialsError:
        logger.debug("CloudWatch put_metric skipped: no AWS credentials")
        return False
    except (ClientError, NoCredentialsError) as e:
        logger.warning("CloudWatch put_metric failed: %s", str(e))
        return False


def cw_put_detection_metrics(
    module: str,
    risk_score: float,
    latency_ms: float,
    confidence: float,
    model_used: str = "ensemble",
) -> None:
    """Emit the standard set of detection metrics for a given module."""
    dims = [{"Name": "Module", "Value": module}]
    cw_put_metric("RiskScore",          risk_score,  "None",         dims)
    cw_put_metric("InferenceLatencyMs", latency_ms,  "Milliseconds", dims)
    cw_put_metric("Confidence",         confidence,  "None",         dims)
    cw_put_metric("RequestCount",       1.0,         "Count",        dims)


def cw_put_alert_metric(risk_level: str) -> None:
    """Emit an alert-count metric broken down by risk level."""
    cw_put_metric(
        "AlertsFired", 1.0, "Count",
        dimensions=[{"Name": "RiskLevel", "Value": risk_level}],
    )


def cw_log_event(log_group: str, log_stream: str, message: str) -> bool:
    """Write a single structured message to CloudWatch Logs."""
    client = get_client("logs")
    if not client:
        return False
    try:
        try:
            client.create_log_stream(logGroupName=log_group, logStreamName=log_stream)
        except (ClientError, NoCredentialsError):
            pass  # Already exists — ignore
        client.put_log_events(
            logGroupName=log_group,
            logStreamName=log_stream,
            logEvents=[{"timestamp": int(time.time() * 1000), "message": message}],
        )
        return True
    except (ClientError, NoCredentialsError) as e:
        logger.warning("CloudWatch log failed: %s", str(e))
        return False


# ═══════════════════════════════════════════════════════
# SECTION 5 — SNS ALERTING
# ═══════════════════════════════════════════════════════

def sns_publish_alert(
    subject: str,
    message: str,
    risk_level: str = "high",
    topic_arn: str = SNS_ALERTS_TOPIC_ARN,
    attributes: Optional[Dict] = None,
) -> Optional[str]:
    """
    Publish a security alert to SNS.
    Returns MessageId on success, None on failure / not configured.
    """
    client = get_client("sns")
    if not client or not topic_arn:
        logger.warning("SNS not configured — alert not sent: %s", subject)
        return None

    msg_attrs: Dict[str, Any] = {
        "risk_level": {"DataType": "String", "StringValue": risk_level},
        "source":     {"DataType": "String", "StringValue": "AEGIS"},
    }
    if attributes:
        for k, v in attributes.items():
            msg_attrs[k] = {"DataType": "String", "StringValue": str(v)}

    try:
        resp = client.publish(
            TopicArn=topic_arn,
            Subject=subject[:100],
            Message=message,
            MessageAttributes=msg_attrs,
        )
        logger.info("SNS alert sent: %s / MessageId=%s", subject, resp["MessageId"])
        return resp["MessageId"]
    except (ClientError, NoCredentialsError) as e:
        logger.error("SNS publish failed: %s", str(e))
        return None


def sns_alert_critical_risk(
    user_id: str,
    unified_score: float,
    primary_threats: List[str],
    event_id: str,
    recommended_actions: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Fire a structured critical-risk alert via SNS.
    Call this when unified_score > 0.75.
    """
    subject = f"[AEGIS CRITICAL] User {user_id} — Risk Score {unified_score:.2f}"
    body = {
        "alert_type":        "critical_risk",
        "user_id":           user_id,
        "event_id":          event_id,
        "unified_score":     round(unified_score, 4),
        "primary_threats":   primary_threats,
        "recommended_actions": recommended_actions or [],
        "timestamp":         datetime.now(timezone.utc).isoformat(),
    }
    return sns_publish_alert(
        subject=subject,
        message=json.dumps(body, indent=2),
        risk_level="critical",
        attributes={"user_id": user_id, "event_id": event_id},
    )


def sns_alert_account_takeover(
    user_id: str,
    signals: List[str],
    event_id: str,
) -> Optional[str]:
    """Fire an account-takeover alert."""
    subject = f"[AEGIS] Possible Account Takeover — User {user_id}"
    body = {
        "alert_type": "account_takeover",
        "user_id":    user_id,
        "event_id":   event_id,
        "signals":    signals,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }
    return sns_publish_alert(
        subject=subject,
        message=json.dumps(body, indent=2),
        risk_level="high",
    )


# ═══════════════════════════════════════════════════════
# SECTION 6 — SECRETS MANAGER
# ═══════════════════════════════════════════════════════

def get_secret(secret_name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Retrieve a secret string from AWS Secrets Manager.
    Falls back to `default` if not available.
    """
    client = get_client("secretsmanager")
    if not client:
        return default
    try:
        resp = client.get_secret_value(SecretId=secret_name)
        return resp.get("SecretString", default)
    except (ClientError, NoCredentialsError) as e:
        code = getattr(e, "response", {}).get("Error", {}).get("Code", str(e))
        logger.warning("Secrets Manager get_secret (%s): %s", secret_name, code)
        return default


def get_secret_json(secret_name: str) -> Dict:
    """Retrieve and parse a JSON secret."""
    raw = get_secret(secret_name)
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return {}


# ═══════════════════════════════════════════════════════
# SECTION 7 — INFRASTRUCTURE HEALTH CHECK
# ═══════════════════════════════════════════════════════

def check_aws_health() -> Dict[str, Any]:
    """
    Check connectivity to all AWS services used by AEGIS.
    Returns a dict with per-service status. Safe to call in health endpoints.
    """
    status: Dict[str, Any] = {
        "aws_configured": aws_available(),
        "region":         AWS_REGION,
        "services":       {},
    }

    # Cognito
    try:
        c = get_client("cognito-idp")
        if c and COGNITO_USER_POOL_ID:
            c.describe_user_pool(UserPoolId=COGNITO_USER_POOL_ID)
            status["services"]["cognito"] = "healthy"
        else:
            status["services"]["cognito"] = "not_configured"
    except Exception as e:
        status["services"]["cognito"] = f"error: {str(e)[:80]}"

    # S3
    try:
        c = get_client("s3")
        if c:
            c.head_bucket(Bucket=S3_MODELS_BUCKET)
            status["services"]["s3"] = "healthy"
        else:
            status["services"]["s3"] = "not_configured"
    except Exception as e:
        status["services"]["s3"] = f"error: {str(e)[:80]}"

    # DynamoDB
    try:
        r = get_resource("dynamodb")
        if r:
            r.Table(DYNAMODB_EVENTS_TABLE).load()
            status["services"]["dynamodb"] = "healthy"
        else:
            status["services"]["dynamodb"] = "not_configured"
    except Exception as e:
        status["services"]["dynamodb"] = f"error: {str(e)[:80]}"

    # CloudWatch
    try:
        c = get_client("cloudwatch")
        if c:
            c.list_metrics(Namespace=CLOUDWATCH_NAMESPACE, Limit=1)
            status["services"]["cloudwatch"] = "healthy"
        else:
            status["services"]["cloudwatch"] = "not_configured"
    except Exception as e:
        status["services"]["cloudwatch"] = f"error: {str(e)[:80]}"

    # SNS
    try:
        c = get_client("sns")
        if c and SNS_ALERTS_TOPIC_ARN:
            c.get_topic_attributes(TopicArn=SNS_ALERTS_TOPIC_ARN)
            status["services"]["sns"] = "healthy"
        else:
            status["services"]["sns"] = "not_configured"
    except Exception as e:
        status["services"]["sns"] = f"error: {str(e)[:80]}"

    return status


# ─────────────────────────────────────────────
# Legacy shim — keep existing call sites working
# ─────────────────────────────────────────────
def upload_file(local_path: str, s3_key: str, bucket: str = S3_MODELS_BUCKET) -> bool:
    """Legacy wrapper — use s3_upload_model() for new code."""
    return s3_upload_model(local_path, s3_key, bucket)
