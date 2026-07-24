"""
Outbound webhook notifications
==============================
Replaces AWS SNS for the self-hosted / multi-tenant deployment. When a threat
crosses the alert threshold, Apeilo POSTs a JSON payload to the tenant's
registered webhook URL. The integrating app (SODA, etc.) receives it and can
notify the user, force MFA, lock the account, and so on.

Every request is signed so the receiver can verify it really came from Apeilo:
  X-Apeilo-Signature: sha256=<hex hmac of the raw body using the tenant secret>
  X-Apeilo-Event:     threat.detected
"""

import hmac
import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 5.0


def sign_payload(body: bytes, secret: str) -> str:
    """Compute the `sha256=<hex>` signature for a raw request body."""
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def send_threat_webhook(
    tenant: Dict[str, Any],
    event_type: str,
    user_id: str,
    risk_score: float,
    risk_level: str,
    result: Dict[str, Any],
) -> bool:
    """POST a threat notification to a tenant's webhook URL.

    `tenant` is an api-key record (see api_keys.py). No-op when the tenant has
    no webhook configured. Failures are swallowed (logged) so scoring latency
    is never affected by a slow/broken receiver.
    """
    webhook_url = (tenant or {}).get("webhook_url", "")
    if not webhook_url:
        return False

    secret  = (tenant or {}).get("webhook_secret", "")
    payload = {
        "type":        "threat.detected",
        "event_type":  event_type,
        "tenant_id":   tenant.get("tenant_id", ""),
        "user_id":     user_id,
        "risk_score":  round(float(risk_score), 4),
        "risk_level":  risk_level,
        "primary_threats":     result.get("primary_threats", [event_type]),
        "recommended_actions": result.get("recommended_actions", []),
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "details":     result,
    }
    body = json.dumps(payload, default=str).encode("utf-8")

    headers = {
        "Content-Type":      "application/json",
        "X-Apeilo-Event":    "threat.detected",
        "User-Agent":        "Apeilo-Webhook/1.0",
    }
    if secret:
        headers["X-Apeilo-Signature"] = sign_payload(body, secret)

    try:
        import httpx
        with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
            resp = client.post(webhook_url, content=body, headers=headers)
        if resp.status_code >= 300:
            logger.warning("Webhook %s returned %s", webhook_url, resp.status_code)
            return False
        logger.info("Webhook delivered to %s (%s)", webhook_url, resp.status_code)
        return True
    except Exception as e:
        logger.warning("Webhook delivery to %s failed: %s", webhook_url, e)
        return False
