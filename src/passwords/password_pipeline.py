#!/usr/bin/env python3
"""
Password breach / risk pipeline for Aegis.

Reads a CSV of passwords, engineers strength features, and trains a classifier
to predict whether a password is breached/weak (1) or safe (0).

Expected input (flexible):
  - password column: one of ["password", "pwd", "pass"]
  - label column (optional): one of ["label", "breached", "pwned", "is_breached"]
      * if absent, treat all rows as breached (1) and generate synthetic safe passwords (0)

Usage examples (from project root):

  # Train using passwords.csv, auto-detect columns, generate 50k safe examples
  python3 src/passwords/password_pipeline.py \
    --raw-file data/raw/passwords/passwords.csv \
    --out-dir data/processed/passwords \
    --n-synth-safe 50000 \
    --train

"""

import os
import argparse
import json
import math
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ------------- helpers for loading data -------------

def find_column(df: pd.DataFrame, candidates: List[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_password_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"Raw password file not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df


# ------------- feature engineering for passwords -------------

SEQUENTIAL_PATTERNS = [
    "1234", "2345", "3456", "4567", "5678", "6789",
    "abcd", "bcde", "cdef", "qwerty", "asdf", "zxcv"
]


def estimate_entropy(pw: str) -> float:
    """Very rough entropy estimate based on unique char set size and length."""
    if not pw:
        return 0.0
    charset = set(pw)
    N = len(charset)
    if N <= 1:
        return 0.0
    return len(pw) * math.log2(N)


def has_seq_pattern(pw: str) -> int:
    pw_lower = pw.lower()
    for pat in SEQUENTIAL_PATTERNS:
        if pat in pw_lower:
            return 1
    return 0


def password_to_features(pw: str) -> dict:
    length = len(pw)
    n_lower = sum(c.islower() for c in pw)
    n_upper = sum(c.isupper() for c in pw)
    n_digit = sum(c.isdigit() for c in pw)
    n_symbol = sum(not c.isalnum() for c in pw)

    has_lower = 1 if n_lower > 0 else 0
    has_upper = 1 if n_upper > 0 else 0
    has_digit = 1 if n_digit > 0 else 0
    has_symbol = 1 if n_symbol > 0 else 0

    unique_chars = len(set(pw))
    entropy_est = estimate_entropy(pw)
    seq = has_seq_pattern(pw)
    repeat_ratio = 0.0
    if length > 0:
        most_freq = max([pw.count(c) for c in set(pw)])
        repeat_ratio = most_freq / length

    return {
        "length": length,
        "n_lower": n_lower,
        "n_upper": n_upper,
        "n_digit": n_digit,
        "n_symbol": n_symbol,
        "has_lower": has_lower,
        "has_upper": has_upper,
        "has_digit": has_digit,
        "has_symbol": has_symbol,
        "unique_chars": unique_chars,
        "entropy_est": entropy_est,
        "has_seq_pattern": seq,
        "repeat_ratio": repeat_ratio,
    }


def df_password_features(passwords: pd.Series) -> pd.DataFrame:
    feats = [password_to_features(str(pw)) for pw in passwords.fillna("")]
    return pd.DataFrame(feats)


# ------------- synthetic safe password generation -------------

LOWER = "abcdefghijklmnopqrstuvwxyz"
UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
SYMBOLS = "!@#$%^&*()-_=+[]{};:,.?/|"


def random_str(charset: str, length: int) -> str:
    return "".join(random.choice(charset) for _ in range(length))


def generate_strong_password() -> str:
    # guarantee composition: at least one of each
    base = [
        random.choice(LOWER),
        random.choice(UPPER),
        random.choice(DIGITS),
        random.choice(SYMBOLS),
    ]
    extra_len = random.randint(4, 12)
    all_chars = LOWER + UPPER + DIGITS + SYMBOLS
    base.extend(random.choice(all_chars) for _ in range(extra_len))
    random.shuffle(base)
    return "".join(base)


def generate_safe_passwords(n: int) -> pd.Series:
    return pd.Series([generate_strong_password() for _ in range(n)])


# ------------- main pipeline -------------

def build_dataset(df_raw: pd.DataFrame, n_synth_safe: int) -> Tuple[pd.DataFrame, pd.Series]:
    pw_col = find_column(df_raw, ["password", "pwd", "pass"])
    if pw_col is None:
        raise SystemExit("Could not find password column (expected one of: password, pwd, pass)")
    print(f"Using password column: {pw_col}")

    # detect label column
    label_col = find_column(df_raw, ["label", "breached", "pwned", "is_breached"])
    if label_col is not None:
        print(f"Using label column: {label_col}")
        labels_raw = df_raw[label_col]
        # map different types to 0/1
        def to_label(v):
            s = str(v).strip().lower()
            if s in ["1", "true", "yes", "y", "breached", "pwned"]:
                return 1
            if s in ["0", "false", "no", "n", "safe", "clean"]:
                return 0
            # default to breached if unknown
            return 1
        y = labels_raw.apply(to_label).astype(int)
        pw_series = df_raw[pw_col].astype(str)
    else:
        print("No label column found. Treating all provided passwords as breached (1) and generating safe (0).")
        pw_series = df_raw[pw_col].astype(str)
        y = pd.Series(np.ones(len(pw_series), dtype=int))
        if n_synth_safe > 0:
            safe_pw = generate_safe_passwords(n_synth_safe)
            pw_series = pd.concat([pw_series, safe_pw], ignore_index=True)
            y = pd.concat([y, pd.Series(np.zeros(len(safe_pw), dtype=int))], ignore_index=True)
            print(f"Generated {len(safe_pw)} synthetic safe passwords.")

    X_df = df_password_features(pw_series)
    print("Feature columns:", list(X_df.columns))
    return X_df, y


def run_pipeline(args):
    os.makedirs(args.out_dir, exist_ok=True)
    df_raw = load_password_dataset(args.raw_file)
    X_df, y = build_dataset(df_raw, args.n_synth_safe)

    X = X_df.values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # save processed dataset
    processed_path = os.path.join(args.out_dir, "password_features.csv")
    full_df = X_df.copy()
    full_df["label"] = y.values
    full_df.to_csv(processed_path, index=False)
    print("Saved processed feature CSV to", processed_path)

    if not args.train:
        print("train flag not set; skipping model training.")
        return

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y.values, test_size=0.2, random_state=RANDOM_STATE, stratify=y.values
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    print("\nClassification report (test):")
    print(classification_report(y_test, preds))

    try:
        auc = roc_auc_score(y_test, probs)
        print("ROC AUC:", auc)
    except Exception as e:
        print("Could not compute AUC:", e)

    # save artifacts
    import joblib
    joblib.dump(clf, os.path.join(args.out_dir, "password_model_rf.pkl"))
    joblib.dump(scaler, os.path.join(args.out_dir, "password_scaler.pkl"))
    with open(os.path.join(args.out_dir, "password_features.json"), "w") as f:
        json.dump(list(X_df.columns), f)

    print("Saved model + scaler + feature list to", args.out_dir)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-file", required=True, help="input CSV with passwords")
    p.add_argument("--out-dir", default="data/processed/passwords", help="output directory")
    p.add_argument("--n-synth-safe", type=int, default=50000, help="synthetic safe passwords to generate if labels missing")
    p.add_argument("--train", action="store_true", help="train model")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
