"""
Local DynamoDB bootstrap
========================
Creates the DynamoDB tables Apeilo needs, then seeds a bootstrap API key so a
freshly-started stack is immediately usable by an integrating app (e.g. SODA).

This runs on FastAPI startup and is a no-op when:
  - not in local mode, or
  - the tables/key already exist.

Tables (all PAY_PER_REQUEST / on-demand):
  Apeilo-events    PK=user_id (S)   SK=event_id (S)
  apeilo-users-profile  PK=user_id (S)
  Apeilo-alerts    PK=user_id (S)   SK=alert_id (S)
  Apeilo-apikeys   PK=key_hash (S)
"""

import os
import time
import logging

from src.utils.aws_utils import (
    LOCAL_MODE,
    DYNAMODB_ENDPOINT_URL,
    DYNAMODB_EVENTS_TABLE,
    DYNAMODB_PROFILES_TABLE,
    DYNAMODB_ALERTS_TABLE,
    DYNAMODB_APIKEYS_TABLE,
    get_resource,
)

logger = logging.getLogger(__name__)


# (table_name, partition_key, sort_key_or_None)
_TABLES = [
    (DYNAMODB_EVENTS_TABLE,   "user_id",  "event_id"),
    (DYNAMODB_PROFILES_TABLE, "user_id",  None),
    (DYNAMODB_ALERTS_TABLE,   "user_id",  "alert_id"),
    (DYNAMODB_APIKEYS_TABLE,  "key_hash", None),
]


def _create_table(resource, name: str, pk: str, sk):
    key_schema = [{"AttributeName": pk, "KeyType": "HASH"}]
    attr_defs  = [{"AttributeName": pk, "AttributeType": "S"}]
    if sk:
        key_schema.append({"AttributeName": sk, "KeyType": "RANGE"})
        attr_defs.append({"AttributeName": sk, "AttributeType": "S"})

    table = resource.create_table(
        TableName=name,
        KeySchema=key_schema,
        AttributeDefinitions=attr_defs,
        BillingMode="PAY_PER_REQUEST",
    )
    table.wait_until_exists()
    logger.info("Created DynamoDB table: %s", name)


def ensure_tables() -> bool:
    """Create any missing tables. Returns True if the backend is reachable."""
    if not (LOCAL_MODE and DYNAMODB_ENDPOINT_URL):
        return False

    resource = get_resource("dynamodb")
    if resource is None:
        logger.warning("DynamoDB resource unavailable; cannot bootstrap tables")
        return False

    # DynamoDB Local usually starts before the backend, but retry briefly in
    # case it's still coming up.
    existing = None
    for attempt in range(10):
        try:
            existing = set(resource.meta.client.list_tables().get("TableNames", []))
            break
        except Exception as e:
            if attempt == 9:
                logger.warning("Could not reach DynamoDB after retries: %s", e)
                return False
            time.sleep(1.5)
    if existing is None:
        return False

    for name, pk, sk in _TABLES:
        if name in existing:
            continue
        try:
            _create_table(resource, name, pk, sk)
        except Exception as e:
            # ResourceInUseException = created by a racing worker; safe to ignore.
            if "ResourceInUseException" in str(e):
                logger.info("Table already being created: %s", name)
            else:
                logger.error("Failed to create table %s: %s", name, e)
    return True


def seed_bootstrap_key() -> None:
    """Seed an API key from env so `docker compose up` yields a working key.

    Controlled by APEILO_BOOTSTRAP_API_KEY (+ tenant/webhook). Idempotent —
    re-running updates the same key record rather than duplicating it.
    """
    raw_key = os.getenv("APEILO_BOOTSTRAP_API_KEY", "").strip()
    if not raw_key:
        return

    from src.utils.api_keys import upsert_api_key

    tenant_id   = os.getenv("APEILO_BOOTSTRAP_TENANT", "soda").strip() or "soda"
    tenant_name = os.getenv("APEILO_BOOTSTRAP_TENANT_NAME", "SODA Corporate Dashboard").strip()
    webhook_url = os.getenv("APEILO_BOOTSTRAP_WEBHOOK_URL", "").strip()
    webhook_secret = os.getenv("APEILO_BOOTSTRAP_WEBHOOK_SECRET", "").strip()

    try:
        upsert_api_key(
            raw_key=raw_key,
            tenant_id=tenant_id,
            name=tenant_name,
            webhook_url=webhook_url or None,
            webhook_secret=webhook_secret or None,
        )
        logger.info("Seeded bootstrap API key for tenant '%s'", tenant_id)
    except Exception as e:
        logger.warning("Could not seed bootstrap API key: %s", e)


def bootstrap() -> None:
    """Full startup bootstrap: tables + seed key."""
    if ensure_tables():
        seed_bootstrap_key()
