"""
API-key management (multi-tenant)
=================================
Integrating apps (SODA, or any other project) authenticate to Apeilo with an
API key sent as the `X-Api-Key` header. Each key maps to a *tenant* — an
isolated namespace of users, events and alerts — and optionally a webhook URL
that Apeilo POSTs to when a threat is detected.

Storage: the `Apeilo-apikeys` DynamoDB table.
  PK key_hash (S)  — SHA-256 of the raw key (raw key is never stored)
  attributes: tenant_id, name, webhook_url, webhook_secret, active, created_at

Keys look like:  apeilo_sk_<40 hex chars>
"""

import hashlib
import logging
import secrets
from datetime import datetime, timezone
from typing import Optional, Dict, List

from src.utils.aws_utils import get_resource, DYNAMODB_APIKEYS_TABLE

logger = logging.getLogger(__name__)

KEY_PREFIX = "apeilo_sk_"

# Small in-process cache so we don't hit DynamoDB on every scored request.
_cache: Dict[str, dict] = {}


def generate_key() -> str:
    """Return a new random API key (raw — show once, store only the hash)."""
    return KEY_PREFIX + secrets.token_hex(20)


def hash_key(raw_key: str) -> str:
    """Deterministic SHA-256 hash used as the DynamoDB partition key."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_webhook_secret() -> str:
    """Return a new webhook signing secret."""
    return "whsec_" + secrets.token_hex(24)


def _table():
    resource = get_resource("dynamodb")
    if resource is None:
        return None
    return resource.Table(DYNAMODB_APIKEYS_TABLE)


def upsert_api_key(
    raw_key: str,
    tenant_id: str,
    name: str = "",
    webhook_url: Optional[str] = None,
    webhook_secret: Optional[str] = None,
    active: bool = True,
) -> Optional[dict]:
    """Create or update an API key record. Returns the stored record (no raw key)."""
    table = _table()
    if table is None:
        logger.warning("apikeys table unavailable; cannot store key")
        return None

    key_hash = hash_key(raw_key)
    record = {
        "key_hash":       key_hash,
        "tenant_id":      tenant_id,
        "name":           name,
        "webhook_url":    webhook_url or "",
        "webhook_secret": webhook_secret or generate_webhook_secret(),
        "active":         active,
        "created_at":     datetime.now(timezone.utc).isoformat(),
    }
    table.put_item(Item=record)
    _cache.pop(key_hash, None)
    return record


def get_tenant_for_key(raw_key: str) -> Optional[dict]:
    """Resolve an API key to its tenant record, or None if invalid/revoked."""
    if not raw_key:
        return None
    key_hash = hash_key(raw_key)

    if key_hash in _cache:
        rec = _cache[key_hash]
        return rec if rec.get("active", True) else None

    table = _table()
    if table is None:
        return None
    try:
        resp = table.get_item(Key={"key_hash": key_hash})
        rec  = resp.get("Item")
    except Exception as e:
        logger.warning("get_tenant_for_key lookup failed: %s", e)
        return None

    if not rec:
        return None
    _cache[key_hash] = rec
    return rec if rec.get("active", True) else None


def list_api_keys() -> List[dict]:
    """List all key records (without raw keys — those are never stored)."""
    table = _table()
    if table is None:
        return []
    try:
        return table.scan().get("Items", [])
    except Exception as e:
        logger.warning("list_api_keys failed: %s", e)
        return []


def revoke_api_key(raw_key: str) -> bool:
    """Deactivate a key by raw value."""
    return _set_active_by_hash(hash_key(raw_key), False)


def revoke_api_key_by_hash(key_hash: str) -> bool:
    return _set_active_by_hash(key_hash, False)


def _set_active_by_hash(key_hash: str, active: bool) -> bool:
    table = _table()
    if table is None:
        return False
    try:
        table.update_item(
            Key={"key_hash": key_hash},
            UpdateExpression="SET active = :a",
            ExpressionAttributeValues={":a": active},
        )
        _cache.pop(key_hash, None)
        return True
    except Exception as e:
        logger.warning("revoke failed: %s", e)
        return False
