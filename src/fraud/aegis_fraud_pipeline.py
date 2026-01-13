#!/usr/bin/env python3
"""
aegis_fraud_pipeline.py

End-to-end Fraud Detection pipeline for the Aegis project.

Usage:
  python3 src/fraud/aegis_fraud_pipeline.py --raw-dir data/raw/fraud --out-dir data/processed/fraud --train
  python3 src/fraud/aegis_fraud_pipeline.py --raw-dir data/raw/fraud --out-dir data/processed/fraud --generate-synthetic --n-synth 3000 --train --smote
"""

import os
import argparse
import pandas as pd
import numpy as np
import joblib
import json
import time
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import StandardScaler

try:
    from imblearn.over_sampling import SMOTE
except:
    SMOTE = None

import xgboost as xgb


DEFAULT_RAW_MAP = {
    "creditcard": "creditcard.csv",
    "ieee": "ieee.csv",
    "paysim": "paysim.csv"
}

RANDOM_STATE = 42


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_csv_if_exists(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"WARNING: {path} not found.")
    return pd.DataFrame()


def load_datasets(raw_dir, map_files=DEFAULT_RAW_MAP):
    dfs = {}
    for key, fname in map_files.items():
        fpath = os.path.join(raw_dir, fname)
        print(f"Loading {key} from {fpath}")
        dfs[key] = read_csv_if_exists(fpath)
        print(f"{key} rows: {len(dfs[key])}")
    return dfs


def unify_schema_creditcard(df):
    if df.empty:
        return df
    out = pd.DataFrame()
    out["transaction_id"] = df.index.astype(str)
    out["user_id"] = (df["Time"] // (24 * 3600)).astype(str) + "_" + (df.index % 1000).astype(str)
    out["amount"] = df["Amount"]

    anchor = datetime(2013, 1, 1)
    out["timestamp"] = df["Time"].apply(lambda s: anchor + timedelta(seconds=float(s)))

    out["merchant"] = "merchant_" + (df.index % 50).astype(str)
    out["device"] = "device_" + (df.index % 20).astype(str)
    out["is_fraud"] = df["Class"].astype(int)

    for col in df.columns:
        if col.startswith("V") or col in ["Time", "Amount"]:
            out[col] = df[col]

    return out


def unify_schema_paysim(df):
    if df.empty:
        return df
    out = pd.DataFrame()
    out["transaction_id"] = df.index.astype(str)
    out["user_id"] = df["nameOrig"]
    out["amount"] = df["amount"].astype(float)
    out["timestamp"] = pd.date_range(start="2018-01-01", periods=len(df), freq="S")
    out["merchant"] = df["nameDest"]
    out["device"] = "paysim_device"

    if "isFraud" in df.columns:
        out["is_fraud"] = df["isFraud"]
    else:
        out["is_fraud"] = 0

    for c in df.columns:
        out[c] = df[c]

    return out


def unify_schema_ieee(df):
    if df.empty:
        return df

    out = pd.DataFrame()
    out["transaction_id"] = df.index.astype(str)

    if "TransactionDT" in df.columns:
        anchor = datetime(2017, 1, 1)
        out["timestamp"] = df["TransactionDT"].apply(lambda s: anchor + timedelta(seconds=float(s)))
    else:
        out["timestamp"] = pd.date_range(start="2017-01-01", periods=len(df), freq="S")

    out["user_id"] = df.get("card1", df.index.astype(str))
    out["amount"] = df.get("TransactionAmt", df.get("amount", 0)).astype(float)
    out["merchant"] = df.get("addr1", "merchant_x").astype(str)
    out["device"] = df.get("DeviceType", "device_ieee")
    out["is_fraud"] = df.get("isFraud", 0)

    for c in df.columns:
        out[c] = df[c]

    return out


def unify_all(dfs):
    parts = []
    if "creditcard" in dfs:
        parts.append(unify_schema_creditcard(dfs["creditcard"]))
    if "paysim" in dfs:
        parts.append(unify_schema_paysim(dfs["paysim"]))
    if "ieee" in dfs:
        parts.append(unify_schema_ieee(dfs["ieee"]))

    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    return combined

def basic_feature_engineering(df):
    """
    Feature engineering including:
     - hour, dayofweek
     - user_avg_amount
     - time_since_last_tx
     - tx_count_1h (transactions in the last 1 hour for the same user)
     - amount_ratio, merchant_freq_user, device_changed
    """
    if df.empty:
        return df

    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek

    # user average amount
    df['user_avg_amount'] = df.groupby('user_id')['amount'].transform('mean')

    # sort by user and timestamp for time-diff calculations
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    # time since last transaction (seconds)
    df['time_since_last_tx'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds().fillna(1e6)

    # tx_count_1h: number of transactions by the same user within the past 1 hour
    # Approach: set timestamp as index temporarily per user, apply rolling count on transaction_id
    def count_1h_per_user(subdf):
        # subdf is sorted by timestamp already
        tmp = subdf.set_index('timestamp')
        # rolling count of transaction_id in last 1 hour
        cnt = tmp['transaction_id'].rolling('1H').count()
        return cnt.reset_index(drop=True)

    # apply per-group and assign back
    counts = df.groupby('user_id', group_keys=False).apply(lambda g: count_1h_per_user(g))
    # counts has same index as grouped subframe (but resets indices), so align by position:
    # Rebuild a Series aligned to df index
    counts_series = counts.reset_index(drop=True)
    # If lengths mismatch (shouldn't), fill with zeros
    if len(counts_series) != len(df):
        counts_series = pd.Series(np.zeros(len(df)))
    df['tx_count_1h'] = counts_series.fillna(0).astype(int)

    # other features
    df['amount_ratio'] = df['amount'] / (df['user_avg_amount'] + 1e-6)
    df['merchant_freq_user'] = df.groupby(['user_id', 'merchant'])['transaction_id'].transform('count')

    df['prev_device'] = df.groupby('user_id')['device'].shift(1)
    df['device_changed'] = (df['prev_device'] != df['device']).astype(int).fillna(0)

    # final fill
    df = df.fillna(0)
    return df



def generate_synthetic_frauds(df, n_synth=2000, seed=RANDOM_STATE):
    np.random.seed(seed)
    if df.empty:
        return df

    df = df.copy()
    users = df["user_id"].unique()
    synths = []

    for i in range(n_synth):
        user = np.random.choice(users)
        user_rows = df[df["user_id"] == user]
        if user_rows.empty:
            continue

        last_row = user_rows.iloc[-1].copy()
        tx = last_row.copy()

        tx["transaction_id"] = f"synth_{i}"
        tx["amount"] = last_row["amount"] * np.random.choice([10, 20, 50, 100])
        tx["timestamp"] = last_row["timestamp"] + pd.Timedelta(seconds=np.random.randint(1, 3600))
        tx["device"] = f"synth_device_{np.random.randint(10000)}"
        tx["is_fraud"] = 1

        synths.append(tx)

    synth_df = pd.DataFrame(synths)
    combined = pd.concat([df, synth_df]).sort_values("timestamp").reset_index(drop=True)
    print(f"Injected {len(synth_df)} synthetic fraud rows")
    return combined


def prepare_ml_dataset(df):
    drop_cols = ["transaction_id", "timestamp", "user_id", "merchant", "device", "prev_device"]
    label_col = "is_fraud"

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in drop_cols and c != label_col]

    X = df[features].fillna(0).astype(float)
    y = df[label_col].astype(int)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    return Xs, y, scaler, features


def train_xgb(X_train, y_train, X_val, y_val, rounds=200):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "eta": 0.1,
        "max_depth": 6,
        "scale_pos_weight": max(1, (len(y_train) - sum(y_train)) / (sum(y_train) + 1))
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(params, dtrain, rounds, evals=[(dtrain, "train"), (dval, "val")])
    return model


def evaluate_model(model, X, y):
    dmat = xgb.DMatrix(X)
    preds = model.predict(dmat)
    y_pred = (preds >= 0.5).astype(int)

    auc = roc_auc_score(y, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary")
    report = classification_report(y, y_pred)

    return auc, prec, rec, f1, report


def save_artifacts(out_dir, model, scaler, features):
    ensure_dir(out_dir)

    model.save_model(os.path.join(out_dir, "xgb_model.bst"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    json.dump(features, open(os.path.join(out_dir, "features.json"), "w"))

    print("Saved model, scaler, features")


def run_pipe(args):
    dfs = load_datasets(args.raw_dir)
    unified = unify_all(dfs)

    if args.generate_synthetic:
        unified = generate_synthetic_frauds(unified, args.n_synth)

    engineered = basic_feature_engineering(unified)
    ensure_dir(args.out_dir)
    engineered.to_csv(os.path.join(args.out_dir, "fraud_processed.csv"), index=False)

    if not args.train:
        print("Preprocessing done.")
        return

    X, y, scaler, features = prepare_ml_dataset(engineered)

    n = len(X)
    t1, t2 = int(0.7 * n), int(0.85 * n)

    X_train, y_train = X[:t1], y[:t1]
    X_val, y_val = X[t1:t2], y[t1:t2]
    X_test, y_test = X[t2:], y[t2:]

    if args.smote:
        if SMOTE is not None:
            sm = SMOTE()
            X_train, y_train = sm.fit_resample(X_train, y_train)
        else:
            print("SMOTE not available")

    model = train_xgb(X_train, y_train, X_val, y_val)

    auc, prec, rec, f1, rep = evaluate_model(model, X_test, y_test)
    print("TEST METRICS:")
    print("AUC:", auc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print(rep)

    save_artifacts(args.out_dir, model, scaler, features)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw-dir",
        default="data/raw/fraud",
        help="Folder containing input CSV files"
    )
    ap.add_argument(
        "--out-dir",
        default="data/processed/fraud",
        help="Folder to save processed + trained model"
    )
    ap.add_argument("--generate-synthetic", action="store_true")
    ap.add_argument("--n-synth", type=int, default=2000)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--smote", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipe(args)

