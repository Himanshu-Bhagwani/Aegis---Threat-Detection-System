#!/usr/bin/env python3
"""
AEGIS AWS Infrastructure Provisioner
=====================================
Run this ONCE to create all required AWS resources:
  - S3 buckets (models, logs)
  - DynamoDB tables (events, profiles, alerts)
  - CloudWatch log group
  - SNS topic
  - Cognito User Pool + App Client

Usage:
  python scripts/provision_aws.py [--region us-east-1] [--dry-run]

After running, copy the printed values into your .env file.
"""

import sys
import json
import argparse
import logging
from typing import Optional

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("provision")


def parse_args():
    p = argparse.ArgumentParser(description="Provision AEGIS AWS resources")
    p.add_argument("--region",  default="us-east-1", help="AWS region")
    p.add_argument("--profile", default=None,         help="AWS CLI profile")
    p.add_argument("--dry-run", action="store_true",  help="Print what would be created, don't create")
    return p.parse_args()


def get_session(region: str, profile: Optional[str]):
    if profile:
        return boto3.Session(profile_name=profile, region_name=region)
    return boto3.Session(region_name=region)


# ─────────────────────────────────────────────
# S3
# ─────────────────────────────────────────────
def create_s3_bucket(s3, bucket_name: str, region: str, dry_run: bool) -> bool:
    if dry_run:
        log.info("[DRY-RUN] Would create S3 bucket: %s", bucket_name)
        return True
    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
        # Block all public access
        s3.put_public_access_block(
            Bucket=bucket_name,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls":       True,
                "IgnorePublicAcls":      True,
                "BlockPublicPolicy":     True,
                "RestrictPublicBuckets": True,
            },
        )
        # Enable versioning for model bucket
        s3.put_bucket_versioning(
            Bucket=bucket_name,
            VersioningConfiguration={"Status": "Enabled"},
        )
        log.info("✅  S3 bucket created: %s", bucket_name)
        return True
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            log.info("⏭️   S3 bucket already exists: %s", bucket_name)
            return True
        log.error("❌  S3 bucket creation failed (%s): %s", bucket_name, e)
        return False


# ─────────────────────────────────────────────
# DynamoDB
# ─────────────────────────────────────────────
def create_dynamodb_table(
    dynamodb,
    table_name: str,
    pk: str,
    sk: Optional[str],
    dry_run: bool,
) -> bool:
    if dry_run:
        log.info("[DRY-RUN] Would create DynamoDB table: %s (PK=%s SK=%s)", table_name, pk, sk)
        return True

    key_schema = [{"AttributeName": pk, "KeyType": "HASH"}]
    attr_defs  = [{"AttributeName": pk, "AttributeType": "S"}]

    if sk:
        key_schema.append({"AttributeName": sk, "KeyType": "RANGE"})
        attr_defs.append({"AttributeName": sk, "AttributeType": "S"})

    try:
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=key_schema,
            AttributeDefinitions=attr_defs,
            BillingMode="PAY_PER_REQUEST",  # On-demand, no capacity planning needed
            TimeToLiveSpecification={       # TTL support (used for auto-expiry)
                "Enabled":       True,
                "AttributeName": "ttl",
            },
        )
        table.wait_until_exists()
        log.info("✅  DynamoDB table created: %s", table_name)
        return True
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "ResourceInUseException":
            log.info("⏭️   DynamoDB table already exists: %s", table_name)
            return True
        log.error("❌  DynamoDB table creation failed (%s): %s", table_name, e)
        return False


# ─────────────────────────────────────────────
# CloudWatch Log Group
# ─────────────────────────────────────────────
def create_log_group(logs, group_name: str, retention_days: int, dry_run: bool) -> bool:
    if dry_run:
        log.info("[DRY-RUN] Would create CloudWatch log group: %s", group_name)
        return True
    try:
        logs.create_log_group(logGroupName=group_name)
        logs.put_retention_policy(logGroupName=group_name, retentionInDays=retention_days)
        log.info("✅  CloudWatch log group created: %s (%d-day retention)", group_name, retention_days)
        return True
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "ResourceAlreadyExistsException":
            log.info("⏭️   CloudWatch log group already exists: %s", group_name)
            return True
        log.error("❌  CloudWatch log group creation failed: %s", e)
        return False


# ─────────────────────────────────────────────
# SNS
# ─────────────────────────────────────────────
def create_sns_topic(sns, topic_name: str, dry_run: bool) -> Optional[str]:
    if dry_run:
        log.info("[DRY-RUN] Would create SNS topic: %s", topic_name)
        return f"arn:aws:sns:us-east-1:123456789:{ topic_name}"
    try:
        resp = sns.create_topic(Name=topic_name)
        arn  = resp["TopicArn"]
        log.info("✅  SNS topic created: %s → %s", topic_name, arn)
        return arn
    except ClientError as e:
        log.error("❌  SNS topic creation failed: %s", e)
        return None


# ─────────────────────────────────────────────
# Cognito
# ─────────────────────────────────────────────
def create_cognito_user_pool(cognito, pool_name: str, dry_run: bool) -> Optional[dict]:
    if dry_run:
        log.info("[DRY-RUN] Would create Cognito User Pool: %s", pool_name)
        return {"UserPool": {"Id": "us-east-1_DRYRUN", "Name": pool_name}}

    try:
        resp = cognito.create_user_pool(
            PoolName=pool_name,
            Policies={
                "PasswordPolicy": {
                    "MinimumLength":                 8,
                    "RequireUppercase":              True,
                    "RequireLowercase":              True,
                    "RequireNumbers":                True,
                    "RequireSymbols":                False,
                    "TemporaryPasswordValidityDays": 7,
                }
            },
            AutoVerifiedAttributes=["email"],
            UsernameAttributes=["email"],
            Schema=[
                {
                    "Name":       "email",
                    "AttributeDataType": "String",
                    "Required":   True,
                    "Mutable":    True,
                },
                {
                    "Name":       "given_name",
                    "AttributeDataType": "String",
                    "Required":   False,
                    "Mutable":    True,
                },
            ],
            UserPoolTags={"Project": "AEGIS", "Environment": "production"},
            AccountRecoverySetting={
                "RecoveryMechanisms": [{"Priority": 1, "Name": "verified_email"}]
            },
        )
        pool_id = resp["UserPool"]["Id"]
        log.info("✅  Cognito User Pool created: %s → %s", pool_name, pool_id)
        return resp
    except ClientError as e:
        log.error("❌  Cognito User Pool creation failed: %s", e)
        return None


def create_cognito_app_client(cognito, pool_id: str, client_name: str, dry_run: bool) -> Optional[dict]:
    if dry_run:
        log.info("[DRY-RUN] Would create Cognito App Client: %s", client_name)
        return {
            "UserPoolClient": {
                "ClientId":     "DRYRUN_CLIENT_ID",
                "ClientSecret": "DRYRUN_SECRET",
                "ClientName":   client_name,
            }
        }
    try:
        resp = cognito.create_user_pool_client(
            UserPoolId=pool_id,
            ClientName=client_name,
            GenerateSecret=False,          # Set True for server-side apps
            ExplicitAuthFlows=[
                "ALLOW_USER_PASSWORD_AUTH",
                "ALLOW_REFRESH_TOKEN_AUTH",
                "ALLOW_USER_SRP_AUTH",
            ],
            SupportedIdentityProviders=["COGNITO"],
            PreventUserExistenceErrors="ENABLED",
            AccessTokenValidity=  1,       # hours
            IdTokenValidity=      1,       # hours
            RefreshTokenValidity= 30,      # days
            TokenValidityUnits={
                "AccessToken":  "hours",
                "IdToken":      "hours",
                "RefreshToken": "days",
            },
        )
        client_id = resp["UserPoolClient"]["ClientId"]
        log.info("✅  Cognito App Client created: %s → %s", client_name, client_id)
        return resp
    except ClientError as e:
        log.error("❌  Cognito App Client creation failed: %s", e)
        return None


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    args    = parse_args()
    region  = args.region
    dry_run = args.dry_run
    session = get_session(region, args.profile)

    log.info("═" * 55)
    log.info("  AEGIS AWS Infrastructure Provisioner")
    log.info("  Region: %s  |  Dry-run: %s", region, dry_run)
    log.info("═" * 55)

    s3       = session.client("s3")
    dynamo   = session.resource("dynamodb")
    logs     = session.client("logs")
    sns      = session.client("sns")
    cognito  = session.client("cognito-idp")

    results: dict = {}

    # ── S3 ───────────────────────────────────
    log.info("\n[S3 Buckets]")
    create_s3_bucket(s3, "aegis-ml-models",  region, dry_run)
    create_s3_bucket(s3, "aegis-event-logs", region, dry_run)

    # ── DynamoDB ──────────────────────────────
    log.info("\n[DynamoDB Tables]")
    create_dynamodb_table(dynamo, "aegis-events",        pk="user_id", sk="sk",       dry_run=dry_run)
    create_dynamodb_table(dynamo, "aegis-user-profiles", pk="user_id", sk=None,       dry_run=dry_run)
    create_dynamodb_table(dynamo, "aegis-alerts",        pk="alert_id", sk=None,      dry_run=dry_run)

    # ── CloudWatch ────────────────────────────
    log.info("\n[CloudWatch Log Groups]")
    create_log_group(logs, "/aegis/api",       retention_days=30, dry_run=dry_run)
    create_log_group(logs, "/aegis/detection", retention_days=90, dry_run=dry_run)

    # ── SNS ───────────────────────────────────
    log.info("\n[SNS Topics]")
    sns_arn = create_sns_topic(sns, "aegis-alerts", dry_run)
    if sns_arn:
        results["SNS_ALERTS_TOPIC_ARN"] = sns_arn

    # ── Cognito ───────────────────────────────
    log.info("\n[Cognito User Pool]")
    pool_resp = create_cognito_user_pool(cognito, "AEGISUserPool", dry_run)
    if pool_resp:
        pool_id = pool_resp["UserPool"]["Id"]
        results["COGNITO_USER_POOL_ID"] = pool_id

        client_resp = create_cognito_app_client(
            cognito, pool_id, "aegis-web-client", dry_run
        )
        if client_resp:
            results["COGNITO_CLIENT_ID"] = client_resp["UserPoolClient"]["ClientId"]
            if client_resp["UserPoolClient"].get("ClientSecret"):
                results["COGNITO_CLIENT_SECRET"] = client_resp["UserPoolClient"]["ClientSecret"]

    # ── Print .env additions ───────────────────
    if results:
        log.info("\n" + "═" * 55)
        log.info("  Copy these values into your .env file:")
        log.info("═" * 55)
        for k, v in results.items():
            print(f"{k}={v}")
    else:
        log.info("\nAll resources were already configured or dry-run mode was used.")

    log.info("\n✅  Provisioning complete.")


if __name__ == "__main__":
    main()
