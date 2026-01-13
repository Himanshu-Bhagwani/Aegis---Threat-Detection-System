#!/usr/bin/env python3
"""
Login anomaly pipeline for Aegis.

Place at: src/login/login_pipeline.py

Usage examples (from project root):
  # Preprocess only
  python3 src/login/login_pipeline.py --raw-dir data/raw/login --out-dir data/processed/login

  # Preprocess + inject synthetic attacks + train unsupervised anomaly model
  python3 src/login/login_pipeline.py --raw-dir data/raw/login --out-dir data/processed/login --generate-synthetic --n-synth 5000 --train --method isolation

  # Train supervised classifier if labels exist:
  python3 src/login/login_pipeline.py --raw-dir data/raw/login --out-dir data/processed/login --train --method supervised
"""
import os, argparse, json, time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support

RANDOM_STATE = 42

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def read_csv_if_exists(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"WARNING: {path} not found")
    return pd.DataFrame()

def load_datasets(raw_dir):
    # Expect files: lanl.csv or auth_logs.csv (generic)
    files = {
        "lanl": os.path.join(raw_dir, "lanl_auth.csv"),
        "cert": os.path.join(raw_dir, "cert_logs.csv"),
        "small": os.path.join(raw_dir, "authentication-logs.csv")
    }
    dfs = {}
    for k, p in files.items():
        dfs[k] = read_csv_if_exists(p)
        print(f"Loaded {k}: rows={len(dfs[k])}")
    return dfs

def unify_login_schema(df):
    """
    Map various login CSVs into unified schema:
    Required columns after unify:
      - event_id (unique)
      - user_id
      - timestamp (datetime)
      - success (1/0)  # whether login succeeded
      - ip (optional)
      - country (optional)
      - device (optional)
    """
    if df.empty:
        return df
    df = df.copy()
    # heuristics to map common column names
    if "LogonTime" in df.columns or "time" in df.columns:
        # try multiple heuristics
        time_col = "LogonTime" if "LogonTime" in df.columns else ("time" if "time" in df.columns else df.columns[0])
        df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        # fallback: create synthetic timestamps
        df["timestamp"] = pd.date_range(start="2020-01-01", periods=len(df), freq="S")

    # map user id
    user_col = None
    for c in ["UserID", "user_id", "username", "Account"]:
        if c in df.columns:
            user_col = c
            break
    if user_col:
        df["user_id"] = df[user_col].astype(str)
    else:
        df["user_id"] = "user_" + (df.index % 1000).astype(str)

    # success / failure flag
    if "Success" in df.columns or "success" in df.columns:
        col = "Success" if "Success" in df.columns else "success"
        df["success"] = df[col].apply(lambda v: 1 if str(v).lower() in ["1","true","yes","success"] else 0)
    elif "status" in df.columns:
        df["success"] = df["status"].apply(lambda v: 1 if str(v).lower() in ["success","ok","200"] else 0)
    else:
        # default: assume success unless explicitly failed
        df["success"] = 1

    # ip / country / device mapping
    for c in ["src_ip", "ip", "IPAddress"]:
        if c in df.columns:
            df["ip"] = df[c].astype(str)
            break
    for c in ["country", "Country", "geo_country"]:
        if c in df.columns:
            df["country"] = df[c].astype(str)
            break
    if "device" in df.columns:
        df["device"] = df["device"].astype(str)
    else:
        df["device"] = "device_" + (df.index % 50).astype(str)

    # event id
    df["event_id"] = df.index.astype(str)
    outcols = ["event_id","user_id","timestamp","success","ip","country","device"]
    for c in outcols:
        if c not in df.columns:
            df[c] = None
    return df[outcols + [c for c in df.columns if c not in outcols]]

def generate_synthetic_logins(df, n_synth=2000, seed=RANDOM_STATE):
    """
    Inject synthetic anomalies:
     - many failed logins in short window (brute-force)
     - login from new country
     - impossible travel: two logins for same user far apart in short time (simulate by assigning country hops)
    """
    np.random.seed(seed)
    if df.empty:
        return df
    df = df.copy()
    users = df["user_id"].unique()
    synth = []
    for i in range(n_synth):
        u = np.random.choice(users)
        base = df[df["user_id"]==u]
        if base.empty:
            continue
        sample = base.sample(1).iloc[0].copy()
        sample["event_id"] = f"synth_{i}"
        t0 = sample["timestamp"]
        # choose an attack type
        typ = np.random.choice(["fail_burst","new_country","impossible_travel"], p=[0.5,0.25,0.25])
        if typ=="fail_burst":
            sample["success"] = 0
            sample["timestamp"] = t0 + pd.Timedelta(seconds=np.random.randint(1,60))
        elif typ=="new_country":
            sample["country"] = "ZZ_" + str(np.random.randint(100))
            sample["timestamp"] = t0 + pd.Timedelta(seconds=np.random.randint(1,3600))
        else: # impossible_travel: set timestamp close but country far
            # set one login very close in time but different country
            sample["timestamp"] = t0 + pd.Timedelta(seconds=np.random.randint(1,300))
            sample["country"] = "ZZ_" + str(np.random.randint(100,200))
        synth.append(sample)
    if not synth:
        return df
    synth_df = pd.DataFrame(synth)
    out = pd.concat([df, synth_df], ignore_index=True, sort=False)
    out = out.sort_values("timestamp").reset_index(drop=True)
    print(f"Injected {len(synth_df)} synthetic login events")
    return out

def engineer_login_features(df):
    if df.empty:
        return df
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id","timestamp"]).reset_index(drop=True)
    # hour, day
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    # time since last login (sec)
    df["time_since_last_login"] = df.groupby("user_id")["timestamp"].diff().dt.total_seconds().fillna(1e6)
    # failed attempts in last 10 minutes per user
    def count_failed(sub):
        tmp = sub.set_index("timestamp")
        return tmp["success"].apply(lambda x: 1)  # placeholder
    # simplified: count failures in rolling 10min
    counts = df.groupby("user_id", group_keys=False).apply(lambda g: g.set_index("timestamp")["success"].rolling("10min").apply(lambda s: (s==0).sum()).reset_index(drop=True))
    counts = counts.reset_index(drop=True)
    if len(counts) == len(df):
        df["failed_10min"] = counts.fillna(0).astype(int)
    else:
        df["failed_10min"] = 0
    # unique countries in last 7 days
    df["country_last_7d"] = df.groupby("user_id")["country"].apply(lambda s: s.fillna("NA"))  # placeholder; keep raw
    # device change flag
    df["prev_device"] = df.groupby("user_id")["device"].shift(1)
    df["device_changed"] = (df["prev_device"] != df["device"]).astype(int).fillna(0)
    # impossible travel heuristic: if time_since_last_login < X and country changed -> flag
    df["prev_country"] = df.groupby("user_id")["country"].shift(1)
    df["impossible_travel"] = ((df["time_since_last_login"] < 3600) & (df["country"].notna()) & (df["prev_country"].notna()) & (df["country"] != df["prev_country"])).astype(int)
    # fillna numeric
    df["failed_10min"] = df["failed_10min"].fillna(0).astype(float)
    df["time_since_last_login"] = df["time_since_last_login"].fillna(1e6)
    return df

def prepare_ml_dataset(df, label_col="label"):
    """
    Select numeric features for modeling. If label_col exists in df (0/1), supervised mode is possible.
    """
    if df.empty:
        return None, None, None, None
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # remove internal cols
    drop = ["event_id"]
    features = [c for c in numeric_cols if c not in drop and c!="label"]
    X = df[features].fillna(0).astype(float).values
    y = df[label_col].astype(int).values if label_col in df.columns else None
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, y, scaler, features

def train_isolation_forest(X, n_estimators=200):
    clf = IsolationForest(n_estimators=n_estimators, contamination=0.01, random_state=RANDOM_STATE)
    clf.fit(X)
    return clf

def train_supervised(X, y):
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE)
    clf.fit(X,y)
    return clf

def save_artifacts(out_dir, model, scaler, features, prefix="login"):
    ensure_dir(out_dir)
    joblib.dump(model, os.path.join(out_dir, f"{prefix}_model.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, f"{prefix}_scaler.pkl"))
    with open(os.path.join(out_dir, f"{prefix}_features.json"), "w") as f:
        json.dump(features, f)
    print("Saved artifacts to", out_dir)

def run_pipeline(args):
    ensure_dir(args.out_dir)
    dfs = load_datasets(args.raw_dir)
    # unify and concat
    parts=[]
    for k,df in dfs.items():
        if df is None or df.empty: continue
        u = unify_login_schema(df)
        parts.append(u)
    if not parts:
        print("No login data found in raw dir")
        return
    all_df = pd.concat(parts, ignore_index=True, sort=False).sort_values("timestamp").reset_index(drop=True)
    if args.generate_synthetic:
        all_df = generate_synthetic_logins(all_df, n_synth=args.n_synth)
    engineered = engineer_login_features(all_df)
    engineered.to_csv(os.path.join(args.out_dir, "login_processed.csv"), index=False)
    print("Saved processed login CSV")
    if not args.train:
        return
    # prepare ML data
    Xs, y, scaler, features = prepare_ml_dataset(engineered)
    if Xs is None:
        print("No features prepared")
        return
    if args.method == "isolation":
        model = train_isolation_forest(Xs, n_estimators=args.n_estimators)
        # compute anomaly scores: negative scores more anomalous in IsolationForest
        scores = model.decision_function(Xs)  # higher -> less anomalous
        # convert to anomaly prob (simple)
        anomaly_prob = 1 - ((scores - scores.min()) / (scores.max() - scores.min()+1e-9))
        engineered["anomaly_score"] = anomaly_prob
        save_artifacts(args.out_dir, model, scaler, features, prefix="login_isolation")
    else:
        if y is None or len(y)==0:
            print("No labels for supervised training. Provide label column to train supervised model.")
            return
        model = train_supervised(Xs, y)
        preds = model.predict_proba(Xs)[:,1]
        engineered["pred_prob"] = preds
        save_artifacts(args.out_dir, model, scaler, features, prefix="login_supervised")
        if y is not None:
            print("Supervised training report:")
            print(classification_report(y, (preds>0.5).astype(int)))
            try:
                print("AUC:", roc_auc_score(y, preds))
            except:
                pass

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="data/raw/login", help="raw login csv folder")
    p.add_argument("--out-dir", default="data/processed/login", help="where to store processed and models")
    p.add_argument("--generate-synthetic", action="store_true")
    p.add_argument("--n-synth", type=int, default=2000)
    p.add_argument("--train", action="store_true")
    p.add_argument("--method", choices=["isolation","supervised"], default="isolation")
    p.add_argument("--n-estimators", dest="n_estimators", type=int, default=200)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
