"""
train_lanl_login_models.py

Train multiple anomaly-detection models on the LANL login feature chunks:

- Rule-based baseline (simple heuristics)
- Isolation Forest (unsupervised)
- Gradient Boosting (GBM-style) on pseudo-labels from Isolation Forest
- Dense Autoencoder (unsupervised)

Input: feature CSV chunks created by lanl_prepare_features.py, e.g.:

  ../data/processed/login/features/lanl_features_chunk_000.csv
  ../data/processed/login/features/lanl_features_chunk_001.csv
  ...

Each chunk must have columns:

  time, user, computer,
  user_deg, comp_deg,
  time_since_user_last, time_since_comp_last,
  hour_of_day, is_new_user, is_new_comp

Output models are saved under:

  ../models/login/
"""

import os
import glob
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from joblib import dump

import tensorflow as tf
from tensorflow import keras

layers = keras.layers
models = keras.models
callbacks = keras.callbacks


np.random.seed(42)
tf.random.set_seed(42)

# ---------- CONFIG ----------

# Where your feature chunks are
FEATURE_DIR = "data/processed/login/features"

# Where to save trained models
MODEL_DIR = "../models/login"

# Max rows to load in memory for training (you can increase if you have more RAM)
MAX_ROWS = 2_000_000

# ----------------------------


def load_feature_chunks(feature_dir: str, max_rows: int | None = None) -> pd.DataFrame:
    """
    Load up to max_rows rows from all lanl_features_chunk_*.csv files.
    Concatenates them into a single DataFrame.
    """
    pattern = os.path.join(feature_dir, "lanl_features_chunk_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No feature chunks found matching: {pattern}")

    print(f"[INFO] Found {len(files)} feature chunk files.")
    dfs = []
    total = 0

    for path in files:
        print(f"[INFO] Loading chunk: {path}")
        df_chunk = pd.read_csv(path)

        dfs.append(df_chunk)
        total += len(df_chunk)

        if max_rows is not None and total >= max_rows:
            print(f"[INFO] Reached max_rows={max_rows}, stopping load.")
            break

    df = pd.concat(dfs, ignore_index=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.iloc[:max_rows].reset_index(drop=True)

    print(f"[INFO] Loaded total rows: {len(df)}")
    return df


def make_feature_matrix(df: pd.DataFrame):
    """
    Build numeric feature matrix X from the DataFrame.
    We drop raw IDs and keep numeric behaviour features.
    """
    feature_cols = [
        "user_deg",
        "comp_deg",
        "time_since_user_last",
        "time_since_comp_last",
        "hour_of_day",
        "is_new_user",
        "is_new_comp",
    ]

    for col in feature_cols:
        if col not in df.columns:
            raise SystemExit(f"Missing expected column in features: {col}")

    X_df = df[feature_cols].copy()

    # Replace -1 in time_since_* with a large value (first event)
    for col in ["time_since_user_last", "time_since_comp_last"]:
        mask = X_df[col] < 0
        if mask.any():
            # use median of non-negative values
            med = X_df.loc[~mask, col].median()
            X_df.loc[mask, col] = med

    X_df = X_df.fillna(0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values.astype(np.float32))

    return X_scaled, feature_cols, scaler


# ---------------- RULE-BASED BASELINE ----------------

def rule_based_scores(df: pd.DataFrame) -> np.ndarray:
    """
    Very simple heuristic:
    - Flag as anomalous (1) if:
        * user_deg is very small but comp_deg is very large, OR
        * user is new (first time seen) and computer is high-degree, OR
        * time_since_user_last is extremely large compared to others
    Returns an array of 0/1 pseudo labels.
    """
    u = df["user_deg"].values
    c = df["comp_deg"].values
    dt_u = df["time_since_user_last"].values
    new_u = df["is_new_user"].values

    # thresholds chosen heuristically, you can tune
    c_high = c > np.percentile(c, 95)
    dt_u_high = dt_u > np.percentile(dt_u, 95)

    suspicious = (
        ((u <= 2) & c_high)
        | ((new_u == 1) & c_high)
        | dt_u_high
    )

    return suspicious.astype(int)


# ---------------- ISOLATION FOREST ----------------

def train_isolation_forest(X: np.ndarray):
    print("\n[IsolationForest] Training on", X.shape[0], "events ...")

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,  # adjust if you want more/fewer anomalies
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    iso.fit(X)

    scores = -iso.decision_function(X)  # higher = more anomalous
    return iso, scores


# ---------------- GBM ON PSEUDO-LABELS ----------------

def train_gbm_from_pseudolabels(X: np.ndarray, scores: np.ndarray):
    """
    Create pseudo labels from anomaly scores:
      - top 5% most anomalous -> label 1
      - bottom 50% least anomalous -> label 0
    Middle region is ignored to keep training data 'clean'.
    """
    n = len(scores)
    p95 = np.percentile(scores, 95)
    p50 = np.percentile(scores, 50)

    y_pseudo = np.full(n, -1, dtype=int)  # -1 = ignore

    y_pseudo[scores >= p95] = 1
    y_pseudo[scores <= p50] = 0

    mask = y_pseudo != -1
    X_train = X[mask]
    y_train = y_pseudo[mask]

    print(
        f"\n[GBM] Training on {X_train.shape[0]} pseudo-labeled samples "
        f"({(y_train == 1).sum()} anomalies, {(y_train == 0).sum()} normal)"
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    gbm = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
    )
    gbm.fit(X_tr, y_tr)

    y_pred = gbm.predict(X_val)
    print("\n===== GBM on pseudo-labels (validation) =====")
    print(classification_report(y_val, y_pred, digits=4))

    return gbm


# ---------------- AUTOENCODER ----------------

def build_autoencoder(input_dim: int):
    inp = keras.Input(shape=(input_dim,), name="input")

    x = keras.layers.Dense(128, activation="relu")(inp)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    encoded = keras.layers.Dense(16, activation="relu", name="bottleneck")(x)

    x = keras.layers.Dense(32, activation="relu")(encoded)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    out = keras.layers.Dense(input_dim, activation="linear")(x)

    model = keras.Model(inputs=inp, outputs=out, name="lanl_dense_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def train_autoencoder(X: np.ndarray):
    print("\n[Autoencoder] Training dense AE on", X.shape[0], "events ...")

    model = build_autoencoder(X.shape[1])
    model.summary()

    es = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    history = model.fit(
        X,
        X,
        epochs=30,
        batch_size=1024,
        validation_split=0.1,
        callbacks=[es],
        verbose=2,
    )

    # reconstruction error
    recon = model.predict(X, batch_size=4096, verbose=0)
    mse = np.mean((X - recon) ** 2, axis=1)

    return model, mse, history


# ---------------- MAIN PIPELINE ----------------

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("[STEP] Loading feature chunks ...")
    df = load_feature_chunks(FEATURE_DIR, max_rows=MAX_ROWS)

    print("[STEP] Building feature matrix ...")
    X_scaled, feature_cols, scaler = make_feature_matrix(df)
    print("[INFO] Feature matrix shape:", X_scaled.shape)

    # Save feature column names + scaler for later use
    meta = {
        "feature_cols": feature_cols,
        "scaler_path": os.path.join(MODEL_DIR, "lanl_scaler.joblib"),
    }
    dump(scaler, meta["scaler_path"])
    with open(os.path.join(MODEL_DIR, "lanl_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Saved scaler and metadata in {MODEL_DIR}")

    # -------- Rule-based baseline --------
    print("\n[STEP] Computing rule-based baseline scores ...")
    rb_labels = rule_based_scores(df)
    print(f"[INFO] Rule-based: flagged {rb_labels.sum()} / {len(rb_labels)} events as suspicious.")

    # -------- Isolation Forest --------
    iso, iso_scores = train_isolation_forest(X_scaled)
    iso_path = os.path.join(MODEL_DIR, "lanl_isolation_forest.joblib")
    dump(iso, iso_path)
    print(f"[INFO] Saved Isolation Forest -> {iso_path}")

    # -------- GBM on pseudo-labels --------
    gbm = train_gbm_from_pseudolabels(X_scaled, iso_scores)
    gbm_path = os.path.join(MODEL_DIR, "lanl_gbm_pseudo.joblib")
    dump(gbm, gbm_path)
    print(f"[INFO] Saved GBM (pseudo-label) -> {gbm_path}")

    # -------- Autoencoder --------
    ae, ae_mse, _ = train_autoencoder(X_scaled)
    ae_path = os.path.join(MODEL_DIR, "lanl_autoencoder.h5")
    ae.save(ae_path)
    print(f"[INFO] Saved Autoencoder -> {ae_path}")

    # Save example anomaly thresholds (you can tune later)
    thresholds = {
        "iso_score_p95": float(np.percentile(iso_scores, 95)),
        "ae_mse_p95": float(np.percentile(ae_mse, 95)),
    }
    with open(os.path.join(MODEL_DIR, "lanl_thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"[INFO] Saved example thresholds -> lanl_thresholds.json")

    print("\n[DONE] All models trained and saved in:", MODEL_DIR)


if __name__ == "__main__":
    main()
