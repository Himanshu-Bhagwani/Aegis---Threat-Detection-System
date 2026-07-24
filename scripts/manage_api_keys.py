#!/usr/bin/env python3
"""
Apeilo API-key manager
======================
Create, list, and revoke tenant API keys for the Apeilo threat-detection
service. Each key belongs to a *tenant* (an integrating app such as SODA) and
may carry a webhook URL that Apeilo POSTs to when a threat is detected.

Run it inside the backend container (so it shares the same DynamoDB endpoint):

  docker compose exec backend python scripts/manage_api_keys.py create \
      --tenant soda --name "SODA Dashboard" \
      --webhook http://host.docker.internal:3001/api/apeilo/webhook

  docker compose exec backend python scripts/manage_api_keys.py list
  docker compose exec backend python scripts/manage_api_keys.py revoke --key apeilo_sk_xxx

The raw key is shown ONCE at creation — store it somewhere safe. Only its hash
is persisted, so a lost key cannot be recovered (create a new one instead).
"""

import sys
import argparse

from src.utils.api_keys import (
    generate_key,
    generate_webhook_secret,
    upsert_api_key,
    list_api_keys,
    revoke_api_key,
)
from src.utils.local_bootstrap import ensure_tables


def cmd_create(args):
    ensure_tables()
    raw_key = args.key or generate_key()
    secret  = args.webhook_secret or generate_webhook_secret()
    rec = upsert_api_key(
        raw_key=raw_key,
        tenant_id=args.tenant,
        name=args.name or args.tenant,
        webhook_url=args.webhook,
        webhook_secret=secret,
    )
    if not rec:
        print("ERROR: could not store key — is DynamoDB reachable?", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  API key created — copy the raw key now, it won't be shown again")
    print("=" * 60)
    print(f"  Tenant         : {rec['tenant_id']}")
    print(f"  Name           : {rec['name']}")
    print(f"  API key        : {raw_key}")
    print(f"  Webhook URL    : {rec['webhook_url'] or '(none)'}")
    print(f"  Webhook secret : {rec['webhook_secret']}")
    print("=" * 60)
    print("  Send this header on every request from your app:")
    print(f"    X-Api-Key: {raw_key}")
    print("=" * 60)


def cmd_list(args):
    keys = list_api_keys()
    if not keys:
        print("No API keys found.")
        return
    print(f"{'TENANT':<16} {'NAME':<28} {'ACTIVE':<7} WEBHOOK")
    print("-" * 90)
    for k in keys:
        print(
            f"{k.get('tenant_id',''):<16} "
            f"{(k.get('name') or ''):<28} "
            f"{str(k.get('active', True)):<7} "
            f"{k.get('webhook_url') or '(none)'}"
        )


def cmd_revoke(args):
    ok = revoke_api_key(args.key)
    print("Revoked." if ok else "Revoke failed (key not found or DynamoDB unreachable).")


def main():
    p = argparse.ArgumentParser(description="Manage Apeilo tenant API keys")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("create", help="Create a new API key for a tenant")
    c.add_argument("--tenant", required=True, help="Tenant id, e.g. 'soda'")
    c.add_argument("--name", default="", help="Human-readable name")
    c.add_argument("--webhook", default="", help="Webhook URL for threat notifications")
    c.add_argument("--webhook-secret", default="", help="Override the generated webhook secret")
    c.add_argument("--key", default="", help="Override the generated raw key (advanced)")
    c.set_defaults(func=cmd_create)

    l = sub.add_parser("list", help="List all API keys")
    l.set_defaults(func=cmd_list)

    r = sub.add_parser("revoke", help="Revoke (deactivate) an API key")
    r.add_argument("--key", required=True, help="The raw API key to revoke")
    r.set_defaults(func=cmd_revoke)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
