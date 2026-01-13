"""Train fraud model (XGBoost)"""

#!/usr/bin/env python3
"""
aegis_fraud_pipeline.py

One-file pipeline for Fraud dataset handling and model training:
- load raw datasets (local CSVs)
- basic cleaning & unified schema
- feature engineering (time/velocity/amount features)
- optional synthetic fraud injection
- imbalance handling (SMOTE or class_weight)
- train XGBoost model, evaluate (ROC AUC, precision/recall)
- save model and artifacts
- optional upload to S3

Usage examples:
  python aegis_fraud_pipeline.py --raw-dir ./data/raw --out-dir ./data/out --train
  python aegis_fraud_pipeline.py --generate-synthetic --n-synth 5000
"""

import os
import argparse
import pandas as pd
import numpy as np
import joblib
import json
import time
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import boto3

# --- CONFIG ---
DEFAULT_RAW_MAP = {
    "creditcard": "creditcard.csv",   # Kaggle credit card dataset
    "ieee": "ieee.csv",               # placeholder name for IEEE-CIS (downloaded & preprocessed)
    "paysim": "paysim.csv"            # PaySim dataset
}
RANDOM_STATE = 42

# -------------------------
# Utilities
# -------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def read_csv_if_exists(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        print(f"WARNING: {path} not found.")
        return pd.DataFrame()

# -------------------------
# 1) Loading / basic unification
# -------------------------
def load_datasets(raw_dir, map_files=DEFAULT_RAW_MAP):
    """
    Expect user to have downloaded datasets into raw_dir with names matching DEFAULT_RAW_MAP.
    Returns dictionary of dataframes.
    """
    dfs = {}
    for key, fname in map_files.items():
        fpath = os.path.join(raw_dir, fname)
        print(f"Loading {key} from {fpath} ...")
        dfs[key] = read_csv_if_exists(fpath)
        print(f" -> {key} rows: {len(dfs[key])}")
    return dfs

def unify_schema_creditcard(df):
    """
    Convert Kaggle creditcard dataset (V14..V28 + Amount + Time + Class) to unified schema.
    creditcard contains: 'Time','V1'..'V28','Amount','Class'
    We'll create fields: transaction_id,user_id,amount,timestamp,merchant,device,is_fraud
    NOTE: user_id, merchant, device generated/synthesized if missing.
    """
    if df.empty:
        return df
    out = pd.DataFrame()
    out['transaction_id'] = df.index.astype(str)
    # no user_id available, synthesize a pseudo user id by bucketing Time
    out['user_id'] = (df['Time'] // (24*3600)).astype(str) + "_" + (df.index % 1000).astype(str)
    out['amount'] = df['Amount']
    # time -> convert seconds to a datetime anchored at 2013-01-01
    anchor = datetime(2013,1,1)
    out['timestamp'] = df['Time'].apply(lambda s: anchor + timedelta(seconds=float(s)))
    # simple merchant/device placeholders
    out['merchant'] = 'merchant_' + ((df.index % 50).astype(str))
    out['device'] = 'device_' + ((df.index % 20).astype(str))
    out['is_fraud'] = df['Class'].astype(int)
    # carry over principal features as numeric columns (optional)
    for col in df.columns:
        if col.startswith('V') or col in ['Time','Amount']:
            out[col] = df[col]
    return out

def unify_schema_paysim(df):
    """
    Paysim has: type, amount, nameOrig, nameDest, oldbalanceOrg, newbalanceOrig, isFraud etc.
    We'll map fields to our unified schema.
    """
    if df.empty:
        return df
    out = pd.DataFrame()
    out['transaction_id'] = df.index.astype(str)
    out['user_id'] = df['nameOrig'].astype(str)
    out['amount'] = df['amount'].astype(float)
    # Paysim doesn't have timestamp => create synthetic incremental timestamp
    out['timestamp'] = pd.date_range(start='2018-01-01', periods=len(df), freq='S')
    out['merchant'] = df['nameDest'].astype(str)
    out['device'] = 'paysim_device'
    # Some paysim variants use 'isFraud' or 'isfraud' or 'isFlaggedFraud'
    if 'isFraud' in df.columns:
        out['is_fraud'] = df['isFraud'].astype(int)
    elif 'isfraud' in df.columns:
        out['is_fraud'] = df['isfraud'].astype(int)
    else:
        # fallback: treat flagged fraud as 0
        out['is_fraud'] = 0
    # bring raw columns
    for c in df.columns:
        out[c] = df[c]
    return out

def unify_schema_ieee(df):
    """
    IEEE dataset has many columns. We'll do a naive mapping for demonstration.
    Expect user to preprocess the big ieee dataset offline or adjust mapping.
    """
    if df.empty:
        return df
    out = pd.DataFrame()
    out['transaction_id'] = df.index.astype(str)
    # Attempt to map common fields
    # If there is TransactionDT and TransactionAmt like the competition dataset:
    if 'TransactionDT' in df.columns:
        anchor = datetime(2017,1,1)
        out['timestamp'] = df['TransactionDT'].apply(lambda s: anchor + timedelta(seconds=float(s)))
    else:
        out['timestamp'] = pd.date_range(start='2017-01-01', periods=len(df), freq='S')
    out['user_id'] = df.get('card1', df.index.astype(str)).astype(str)
    out['amount'] = df.get('TransactionAmt', df.get('amount', 0)).astype(float)
    out['merchant'] = df.get('addr1', 'merchant_x').astype(str)
    out['device'] = df.get('DeviceType', 'device_ieee').astype(str)
    # Many versions don't have labels; if 'isFraud' exists, use it else zero
    out['is_fraud'] = df.get('isFraud', df.get('isFraud', 0)).astype(int)
    # copy other columns for potential use
    for c in df.columns:
        out[c] = df[c]
    return out

def unify_all(dfs):
    """
    Accepts dict of raw dataframes. Returns a concatenated unified DataFrame.
    """
    parts = []
    if 'creditcard' in dfs:
        parts.append(unify_schema_creditcard(dfs['creditcard']))
    if 'paysim' in dfs:
        parts.append(unify_schema_paysim(dfs['paysim']))
    if 'ieee' in dfs:
        parts.append(unify_schema_ieee(dfs['ieee']))
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True, sort=False)
    # Ensure timestamp is datetime
    combined['timestamp'] = pd.to_datetime(combined['timestamp'])
    # Sort by timestamp
    combined = combined.sort_values('timestamp').reset_index(drop=True)
    return combined

# -------------------------
# 2) Feature Engineering
# -------------------------
def basic_feature_engineering(df):
    """
    Adds:
      - time-based features (hour, dayofweek)
      - user-level aggregates: user_avg_amount, tx_count_last_24h, time_since_last_tx
      - velocity features
    """
    if df.empty:
        return df
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    # user avg amount
    user_avg = df.groupby('user_id')['amount'].transform('mean').rename('user_avg_amount')
    df['user_avg_amount'] = user_avg
    # time difference between consecutive transactions per user (seconds)
    df = df.sort_values(['user_id','timestamp'])
    df['time_since_last_tx'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds().fillna(1e6)
    # transactions in last 1 hour per user
    df['tx_count_1h'] = df.groupby('user_id')['timestamp'].rolling('1H', on='timestamp').count().reset_index(level=0, drop=True).fillna(0)
    # amount ratio
    df['amount_ratio'] = df['amount'] / (df['user_avg_amount'] + 1e-6)
    # merchant frequency per user
    df['merchant_freq_user'] = df.groupby(['user_id','merchant'])['transaction_id'].transform('count')
    # device change flag: whether device is same as previous tx for user
    df['prev_device'] = df.groupby('user_id')['device'].shift(1)
    df['device_changed'] = (df['prev_device'] != df['device']).astype(int).fillna(0)
    # fill na
    df = df.fillna(0)
    return df

# -------------------------
# 3) Synthetic fraud injection
# -------------------------
def generate_synthetic_frauds(df, n_synth=2000, seed=RANDOM_STATE):
    """
    Simple synthetic fraud generator:
    - picks random users and inserts atypical transactions:
      * very high amounts (multiplier)
      * impossible travel pairs simulated via large hops in merchant location (if location available)
      * new device flag
    Returns dataframe augmented with new flagged synthetic rows (is_fraud=1).
    """
    np.random.seed(seed)
    if df.empty:
        return df
    df = df.copy()
    synth_rows = []
    users = df['user_id'].unique()
    for i in range(n_synth):
        user = np.random.choice(users)
        user_rows = df[df['user_id'] == user]
        if user_rows.empty:
            continue
        last_row = user_rows.sort_values('timestamp').iloc[-1]
        tx = last_row.copy()
        tx['transaction_id'] = f'synth_{int(time.time()*1000)}_{i}'
        # make amount anomalous
        multiplier = np.random.choice([10,20,50,100], p=[0.5,0.25,0.15,0.1])
        tx['amount'] = max(1.0, float(last_row['amount']) * multiplier)
        # change device to new device
        tx['device'] = 'synth_device_' + str(np.random.randint(10000))
        # push timestamp forward slightly
        tx['timestamp'] = last_row['timestamp'] + pd.Timedelta(seconds=np.random.randint(1, 3600))
        tx['is_fraud'] = 1
        synth_rows.append(tx)
    if not synth_rows:
        return df
    synth_df = pd.DataFrame(synth_rows)
    out = pd.concat([df, synth_df], ignore_index=True, sort=False)
    out = out.sort_values('timestamp').reset_index(drop=True)
    print(f"Injected {len(synth_df)} synthetic fraud transactions.")
    return out

# -------------------------
# 4) Prepare ML dataset
# -------------------------
def prepare_ml_dataset(df, features=None, label_col='is_fraud', drop_cols=None):
    """
    Select features, fill NA, scale numeric features as required.
    Returns X, y, scaler (fitted).
    """
    if df.empty:
        return np.array([]), np.array([]), None
    df = df.copy()
    if drop_cols is None:
        drop_cols = ['transaction_id','timestamp','user_id','merchant','device','prev_device']
    # Default features to include
    if features is None:
        # numeric columns discovered heuristically
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # remove label
        numeric_cols = [c for c in numeric_cols if c != label_col]
        # and remove things user might not want
        features = [c for c in numeric_cols if c not in drop_cols]
    # ensure label exists
    if label_col not in df.columns:
        df[label_col] = 0
    X = df[features].fillna(0).astype(float)
    y = df[label_col].astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y.values, scaler, features

# -------------------------
# 5) Train / Evaluate
# -------------------------
def train_xgboost(X_train, y_train, X_val=None, y_val=None, params=None, num_boost_round=200):
    if params is None:
        params = {
            'objective':'binary:logistic',
            'eval_metric':'auc',
            'use_label_encoder':False,
            'n_jobs':8,
            'tree_method':'hist',
            'max_depth':6,
            'eta':0.1,
            'scale_pos_weight': max(1, (len(y_train)-sum(y_train))/ (sum(y_train)+1e-9))
        }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    evallist = []
    if X_val is not None and y_val is not None:
        dval = xgb.DMatrix(X_val, label=y_val)
        evallist = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=evallist, verbose_eval=10)
    return model

def evaluate_model_xgb(model, X, y, threshold=0.5):
    dmat = xgb.DMatrix(X)
    y_pred_prob = model.predict(dmat)
    y_pred = (y_pred_prob >= threshold).astype(int)
    auc = roc_auc_score(y, y_pred_prob) if len(np.unique(y))>1 else None
    prf = precision_recall_fscore_support(y, y_pred, average='binary', zero_division=0)
    report = classification_report(y, y_pred, zero_division=0)
    return {
        'auc': auc,
        'precision': prf[0],
        'recall': prf[1],
        'f1': prf[2],
        'report': report,
        'y_prob': y_pred_prob
    }

# -------------------------
# 6) Persistence & S3 upload
# -------------------------
def save_artifacts(out_dir, model, scaler, features):
    ensure_dir(out_dir)
    model_path = os.path.join(out_dir, 'xgb_model.bst')
    scaler_path = os.path.join(out_dir, 'scaler.pkl')
    features_path = os.path.join(out_dir, 'features.json')
    model.save_model(model_path)
    joblib.dump(scaler, scaler_path)
    with open(features_path, 'w') as f:
        json.dump(features, f)
    print(f"Saved model -> {model_path}, scaler -> {scaler_path}, features -> {features_path}")
    return model_path, scaler_path, features_path

def upload_to_s3(local_path, bucket_name, s3_key, aws_profile=None, region_name=None):
    session_args = {}
    if aws_profile:
        session_args['profile_name'] = aws_profile
    session = boto3.Session(**session_args)
    s3 = session.client('s3', region_name=region_name)
    s3.upload_file(local_path, bucket_name, s3_key)
    print(f"Uploaded {local_path} to s3://{bucket_name}/{s3_key}")

# -------------------------
# CLI + Orchestration
# -------------------------
def run_pipeline(args):
    ensure_dir(args.out_dir)
    dfs = load_datasets(args.raw_dir)
    unified = unify_all(dfs)
    print(f"Unified dataset rows: {len(unified)}")

    if args.generate_synthetic:
        unified = generate_synthetic_frauds(unified, n_synth=args.n_synth)

    # feature engineering
    engineered = basic_feature_engineering(unified)
    # Save processed
    processed_path = os.path.join(args.out_dir, 'fraud_processed.csv')
    engineered.to_csv(processed_path, index=False)
    print("Saved processed dataset to", processed_path)

    if not args.train:
        print("Train flag not set. Exiting after preprocessing.")
        return

    # prepare ML dataset
    X, y, scaler, features = prepare_ml_dataset(engineered)
    if X.size == 0:
        print("No training data prepared! Exiting.")
        return

    # time-based split: last 15% as test, previous 15% as val, rest train
    n = X.shape[0]
    idx_train_end = int(0.7 * n)
    idx_val_end = int(0.85 * n)
    X_train, y_train = X[:idx_train_end], y[:idx_train_end]
    X_val, y_val = X[idx_train_end:idx_val_end], y[idx_train_end:idx_val_end]
    X_test, y_test = X[idx_val_end:], y[idx_val_end:]

    print("Train/Val/Test sizes:", X_train.shape[0], X_val.shape[0], X_test.shape[0])

    # Optional: SMOTE to balance (use only on train)
    if args.smote:
        sampler = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = sampler.fit_resample(X_train, y_train)
        print("After SMOTE train size:", X_train.shape[0])

    # Train XGBoost
    model = train_xgboost(X_train, y_train, X_val, y_val, num_boost_round=args.boost_round)

    # Evaluate
    eval_train = evaluate_model_xgb(model, X_train, y_train)
    eval_val = evaluate_model_xgb(model, X_val, y_val)
    eval_test = evaluate_model_xgb(model, X_test, y_test)

    print("Train eval:", eval_train)
    print("Val eval:", eval_val)
    print("Test eval:", eval_test['report'])

    # Save artifacts
    model_path, scaler_path, features_path = save_artifacts(args.out_dir, model, scaler, features)

    # Optionally upload to S3
    if args.s3_bucket:
        upload_to_s3(model_path, args.s3_bucket, os.path.join(args.s3_prefix, os.path.basename(model_path)), aws_profile=args.aws_profile)
        upload_to_s3(scaler_path, args.s3_bucket, os.path.join(args.s3_prefix, os.path.basename(scaler_path)), aws_profile=args.aws_profile)
        upload_to_s3(features_path, args.s3_bucket, os.path.join(args.s3_prefix, os.path.basename(features_path)), aws_profile=args.aws_profile)
    print("Pipeline complete.")


def parse_args():
    p = argparse.ArgumentParser(description="Aegis Fraud Module Pipeline")
    p.add_argument('--raw-dir', default='./data/raw', help='directory with raw csv files (creditcard.csv etc.)')
    p.add_argument('--out-dir', default='./data/out', help='directory to write processed files and models')
    p.add_argument('--train', action='store_true', help='run training')
    p.add_argument('--generate-synthetic', action='store_true', help='inject synthetic frauds into processed dataset')
    p.add_argument('--n-synth', type=int, default=2000, help='number of synthetic fraud rows to inject')
    p.add_argument('--smote', action='store_true', help='use SMOTE on training set')
    p.add_argument('--boost-round', type=int, default=200, help='xgboost num_boost_round')
    p.add_argument('--s3-bucket', default=None, help='upload artifacts to this s3 bucket')
    p.add_argument('--s3-prefix', default='aegis/fraud', help='s3 key prefix')
    p.add_argument('--aws-profile', default=None, help='aws profile name for boto3 session')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_pipeline(args)
