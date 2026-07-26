# Trained models

Machine-learning artifacts used by the Apeilo detection modules. Loaded at
runtime by the scoring code in `src/` and produced by the training scripts in
`tools/`.

## Login anomaly — `models/login/`
Ensemble trained on the **LANL authentication dataset** (a real-world corporate
auth-log benchmark).

| File | Model |
|------|-------|
| `lanl_isolation_forest.joblib` | Isolation Forest — unsupervised outlier detection |
| `lanl_gbm_pseudo.joblib` | Gradient Boosting Machine (pseudo-labelled) |
| `lanl_autoencoder.h5` | Keras autoencoder — reconstruction-error scoring |
| `lanl_scaler.joblib` | Feature scaler shared by the ensemble |

Used by `src/login/score_login.py`, blended with behavioural rules
(brute-force, off-hours, new-device, impossible-travel).

## GPS / trajectory — `models/gps/`
| File | Model |
|------|-------|
| `gps_isolation_forest.joblib` | Isolation Forest over trajectory features |
| `gps_gbm.joblib` | Gradient Boosting Machine |
| `gps_autoencoder.h5` / `gps_ae_best.h5` | Autoencoder — trajectory reconstruction |
| `gps_cnn_rnn.h5` / `gps_cnn_rnn_best.h5` | CNN-RNN sequence model for spoofed-path detection |

Used by `src/gps/score_gps.py`, alongside the impossible-travel check.

## Fraud
The transaction fraud model (XGBoost) lives under `data/processed/fraud/`
(`xgb_model.bst` + scaler + feature list) and is loaded by
`src/api/routers/fraud.py`, with rule-based scoring relative to each user's own
spending history as the primary signal.

---
*Retraining:* see `tools/train_lanl_login_models.py` and
`tools/train_gps_spoof_models.py`.
