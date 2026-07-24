"""
Per-request tenant context
===========================
An API-key middleware resolves the caller's tenant and stashes it here. Because
FastAPI handles each request in its own async context, the central
`broadcast_detection_event` function (and the identity/alert read endpoints) can
read the current tenant without every router having to pass it through.

Keyless requests (e.g. Apeilo's own bundled dashboard) fall back to the
DEFAULT_TENANT so nothing breaks when no key is supplied.
"""

from contextvars import ContextVar
from typing import Optional, Dict, Any

DEFAULT_TENANT_ID = "apeilo"

DEFAULT_TENANT: Dict[str, Any] = {
    "tenant_id":      DEFAULT_TENANT_ID,
    "name":           "Apeilo (default)",
    "webhook_url":    "",
    "webhook_secret": "",
    "active":         True,
}

_current_tenant: ContextVar[Dict[str, Any]] = ContextVar("current_tenant", default=DEFAULT_TENANT)


def set_current_tenant(tenant: Optional[Dict[str, Any]]) -> None:
    _current_tenant.set(tenant or DEFAULT_TENANT)


def get_current_tenant() -> Dict[str, Any]:
    return _current_tenant.get()


def current_tenant_id() -> str:
    return get_current_tenant().get("tenant_id", DEFAULT_TENANT_ID)


def scope_user_id(user_id: str) -> str:
    """Namespace a raw user id with the current tenant for storage isolation.

    e.g. tenant 'soda' + user 'alice'  ->  'soda#alice'. This keeps DynamoDB
    partition keys unique across tenants while the SDK keeps sending plain ids.
    """
    tid = current_tenant_id()
    uid = user_id or "anonymous"
    # Avoid double-prefixing if an already-scoped id comes back through.
    if uid.startswith(f"{tid}#"):
        return uid
    return f"{tid}#{uid}"


def unscope_user_id(scoped_user_id: str) -> str:
    """Strip the tenant prefix for display (inverse of scope_user_id)."""
    tid = current_tenant_id()
    prefix = f"{tid}#"
    if scoped_user_id.startswith(prefix):
        return scoped_user_id[len(prefix):]
    return scoped_user_id
