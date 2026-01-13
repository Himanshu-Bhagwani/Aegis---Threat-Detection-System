"""
train_gps_spoof_models.py

Train multiple models for GPS spoofing detection on windowed data:

- Rule-based baseline
- Isolation Forest (unsupervised anomaly detection)
- Gradient Boosting (GBM style; can later swap to XGBoost)
- Autoencoder (unsupervised, Keras)
- 1D-CNN + BiLSTM (supervised, Keras)

Assumes you already ran a window-prep script so you have:

  data/windows/gps/X.npy
  data/windows/gps/y.npy
  data/windows/gps/feature_names.json
"""

import os
import json
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from joblib import dump

# Deep learning (comment these imports if you don't want DL models)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

np.random.seed(42)
tf.random.set_seed(42)

DATA_DIR = "data/windows/gps"
MODEL_DIR = "models/gps"
os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------
# Helpers
# -----------------------
def load_windows():
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    with open(os.path.join(DATA_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)
    print("Loaded windows:", X.shape, "labels:", y.shape)
    print("Positive windows:", int(y.sum()))
    return X, y, feature_names


def make_window_level_features(X: np.ndarray) -> np.ndarray:
    """
    Aggregate over time dimension to get fixed-length features per window:
    mean, std, max for each original feature.
    X: (n_windows, T, F) -> (n_windows, F*3)
    """
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    maxv = X.max(axis=1)
    return np.concatenate([mean, std, maxv], axis=1)


# -----------------------
# 1) Rule-based baseline
# -----------------------
def rule_based_baseline(X, y, feature_names):
    """
    Simple heuristic:
    - If any timestep in the window has sudden_jump==1 or impossible_speed_flag==1 => spoof
    """
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    sj_idx = name_to_idx.get("sudden_jump", None)
    is_idx = name_to_idx.get("impossible_speed_flag", None)

    if sj_idx is None and is_idx is None:
        print("\n[Rule-based] sudden_jump / impossible_speed_flag not in features, skipping.")
        return

    N, T, F = X.shape
    preds = np.zeros(N, dtype=int)

    for i in range(N):
        win = X[i]  # (T, F)
        spoof_flag = False
        if sj_idx is not None and (win[:, sj_idx] > 0.5).any():
            spoof_flag = True
        if is_idx is not None and (win[:, is_idx] > 0.5).any():
            spoof_flag = True
        preds[i] = 1 if spoof_flag else 0

    print("\n===== Rule-based baseline =====")
    print(classification_report(y, preds, digits=4))
    try:
        print("ROC-AUC:", roc_auc_score(y, preds))
    except Exception as e:
        print("Could not compute ROC-AUC:", e)


# -----------------------
# 2) Isolation Forest
# -----------------------
def train_isolation_forest(X, y):
    """
    Unsupervised anomaly detection:
    - Train only on normal windows (y == 0)
    - Evaluate on all windows
    """
    X_feat = make_window_level_features(X)
    X_normal = X_feat[y == 0]

    print("\n[IsolationForest] Training on", X_normal.shape[0], "normal windows")

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.1,  # tweak based on your spoof fraction
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_normal)

    scores = -iso.decision_function(X_feat)  # higher = more anomalous
    threshold = np.percentile(scores, 90)    # top 10% most anomalous -> spoof
    preds = (scores >= threshold).astype(int)

    print("\n===== Isolation Forest =====")
    print(classification_report(y, preds, digits=4))
    try:
        print("ROC-AUC (score):", roc_auc_score(y, scores))
    except Exception as e:
        print("Could not compute ROC-AUC:", e)

    dump(iso, os.path.join(MODEL_DIR, "gps_isolation_forest.joblib"))
    print("Saved Isolation Forest ->", os.path.join(MODEL_DIR, "gps_isolation_forest.joblib"))


# -----------------------
# 3) Gradient Boosting (GBM style / XGBoost-like)
# -----------------------
def train_gbm(X, y):
    """
    Classic GBM using sklearn GradientBoostingClassifier.
    If you install xgboost, you can later replace this with XGBClassifier.
    """
    X_feat = make_window_level_features(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_feat, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n[GBM] Training GradientBoostingClassifier...")
    gbm = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )
    gbm.fit(X_train, y_train)
    y_pred = gbm.predict(X_val)
    y_proba = gbm.predict_proba(X_val)[:, 1]

    print("\n===== Gradient Boosting (GBM) =====")
    print(classification_report(y_val, y_pred, digits=4))
    try:
        print("ROC-AUC:", roc_auc_score(y_val, y_proba))
    except Exception as e:
        print("Could not compute ROC-AUC:", e)

    dump(gbm, os.path.join(MODEL_DIR, "gps_gbm.joblib"))
    print("Saved GBM model ->", os.path.join(MODEL_DIR, "gps_gbm.joblib"))


# -----------------------
# 4) Autoencoder (unsupervised)
# -----------------------
def build_autoencoder(time_steps, feat_dim):
    """
    Simple fully connected autoencoder over flattened window: (T*F)
    """
    input_dim = time_steps * feat_dim
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(input_dim, activation="linear")(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model


def train_autoencoder(X, y):
    """
    Train AE only on normal windows.
    Then evaluate reconstruction error as anomaly score.
    """
    N, T, F = X.shape
    X_flat = X.reshape(N, T * F)

    X_normal = X_flat[y == 0]
    X_train, X_val = train_test_split(
        X_normal, test_size=0.2, random_state=42
    )

    ae = build_autoencoder(T, F)
    ae.summary()

    ckpt = callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "gps_ae_best.h5"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    ae.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=30,
        batch_size=64,
        callbacks=[ckpt],
        verbose=2,
    )

    # evaluate on all windows
    X_rec = ae.predict(X_flat, batch_size=128, verbose=0)
    mse = np.mean((X_flat - X_rec) ** 2, axis=1)

    thresh = np.percentile(mse[y == 0], 90)
    preds = (mse >= thresh).astype(int)

    print("\n===== Autoencoder (unsupervised) =====")
    print(classification_report(y, preds, digits=4))
    try:
        print("ROC-AUC (MSE):", roc_auc_score(y, mse))
    except Exception as e:
        print("Could not compute ROC-AUC:", e)

    ae.save(os.path.join(MODEL_DIR, "gps_autoencoder.h5"))
    print("Saved Autoencoder ->", os.path.join(MODEL_DIR, "gps_autoencoder.h5"))


# -----------------------
# 5) 1D-CNN + BiLSTM (supervised)
# -----------------------
def build_cnn_rnn_model(time_steps, feat_dim):
    """
    Simple 1D-CNN + BiLSTM model for sequence classification.
    Input: (T, F)
    """
    inp = layers.Input(shape=(time_steps, feat_dim))
    x = layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters=128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_cnn_rnn(X, y):
    N, T, F = X.shape
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_cnn_rnn_model(T, F)
    model.summary()

    ckpt = callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "gps_cnn_rnn_best.h5"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=64,
        callbacks=[ckpt, es],
        verbose=2,
    )

    y_proba = model.predict(X_val, batch_size=128).ravel()
    preds = (y_proba >= 0.5).astype(int)

    print("\n===== 1D-CNN + BiLSTM (supervised) =====")
    print(classification_report(y_val, preds, digits=4))
    try:
        print("ROC-AUC:", roc_auc_score(y_val, y_proba))
    except Exception as e:
        print("Could not compute ROC-AUC:", e)

    model.save(os.path.join(MODEL_DIR, "gps_cnn_rnn.h5"))
    print("Saved CNN/RNN model ->", os.path.join(MODEL_DIR, "gps_cnn_rnn.h5"))


# -----------------------
# Main
# -----------------------
def main():
    X, y, feature_names = load_windows()

    # 1) Rule-based
    rule_based_baseline(X, y, feature_names)

    # 2) Isolation Forest
    train_isolation_forest(X, y)

    # 3) GBM / XGBoost-style
    train_gbm(X, y)

    # 4) Autoencoder
    train_autoencoder(X, y)

    # 5) 1D-CNN + BiLSTM
    train_cnn_rnn(X, y)

    print("\nAll models trained. Check models/gps/ for saved files.")


if __name__ == "__main__":
    main()
