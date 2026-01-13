"""
prepare_windows_single.py

Use ONE GPS CSV (e.g. B_spoofed.csv that already has is_spoof 0/1),
compute trajectory features, and generate sliding windows.

This avoids concatenating A + B (which is too big for very large files).

Usage example:

  python tools/prepare_windows_single.py \
    --file data/raw/gps/B_spoofed.csv \
    --out-csv data/processed/gps/merged_spoofed_single.csv \
    --out-windows data/windows/gps \
    --window-size 32 \
    --stride 8 \
    --max-rows 400000
"""

import os
import argparse
import json
import math
import random

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)


# -------------------------
# Geodesic helpers
# -------------------------
def haversine(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance in meters between two points.
    """
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def bearing(lat1, lon1, lat2, lon2):
    """
    Bearing in degrees from point 1 to point 2.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    brng = math.degrees(math.atan2(x, y))
    return (brng + 360.0) % 360.0


def angdiff(a, b):
    """
    Minimal absolute angular difference between two bearings (0..180).
    """
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


# -------------------------
# Feature computation
# -------------------------
def compute_point_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # time delta
    df["dt"] = df.groupby("user_id")["timestamp"].diff().dt.total_seconds().fillna(0.0)

    # previous coordinates
    df["lat_prev"] = df.groupby("user_id")["latitude"].shift(1)
    df["lon_prev"] = df.groupby("user_id")["longitude"].shift(1)

    # distance
    df["dist_m"] = df.apply(
        lambda r: haversine(r["lat_prev"], r["lon_prev"], r["latitude"], r["longitude"])
        if pd.notnull(r["lat_prev"])
        else 0.0,
        axis=1,
    )

    # speed
    df["speed_m_s"] = df["dist_m"] / df["dt"].replace(0, np.nan)
    df["speed_m_s"] = df["speed_m_s"].fillna(0.0)

    # bearing & bearing diff
    df["bearing"] = df.apply(
        lambda r: bearing(r["lat_prev"], r["lon_prev"], r["latitude"], r["longitude"])
        if pd.notnull(r["lat_prev"])
        else 0.0,
        axis=1,
    )
    df["bearing_prev"] = df.groupby("user_id")["bearing"].shift(1).fillna(0.0)
    df["bearing_diff"] = df.apply(
        lambda r: angdiff(r["bearing"], r["bearing_prev"]), axis=1
    )

    # acceleration
    df["speed_prev"] = df.groupby("user_id")["speed_m_s"].shift(1).fillna(0.0)
    df["accel"] = (df["speed_m_s"] - df["speed_prev"]) / df["dt"].replace(0, np.nan)
    df["accel"] = df["accel"].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # time-of-day features
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    # simple rule flags
    df["sudden_jump"] = ((df["dist_m"] > 1000.0) & (df["dt"] < 60.0)).astype(int)
    df["impossible_speed_flag"] = (df["speed_m_s"] > 100.0).astype(int)

    for c in ["dist_m", "speed_m_s", "bearing_diff", "accel", "dt"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    return df


# -------------------------
# Window generation
# -------------------------
def make_windows(df: pd.DataFrame, feature_cols, window_size=32, stride=8):
    X = []
    y = []

    # group by user to keep sequences contiguous
    for _, g in df.groupby("user_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        arr = g[feature_cols].fillna(0.0).values
        labels = (
            g["is_spoof"].astype(int).values
            if "is_spoof" in g.columns
            else np.zeros(len(g), dtype=int)
        )

        n = len(g)
        i = 0
        while i + window_size <= n:
            win = arr[i : i + window_size]
            lab = 1 if labels[i : i + window_size].sum() > 0 else 0
            X.append(win)
            y.append(lab)
            i += stride

    if not X:
        return None, None

    X = np.stack(X).astype(np.float32)
    y = np.array(y).astype(np.int64)
    return X, y


# -------------------------
# CSV loader
# -------------------------
def load_csv_single(path: str, max_rows: int | None = None) -> pd.DataFrame:
    print(f"Loading CSV: {path}")
    df = pd.read_csv(path, low_memory=False)

    # map common alternative column names to our expected names
    rename_map = {}
    if "lat" in df.columns and "latitude" not in df.columns:
        rename_map["lat"] = "latitude"
    if "lon" in df.columns and "longitude" not in df.columns:
        rename_map["lon"] = "longitude"
    if "Latitude" in df.columns and "latitude" not in df.columns:
        rename_map["Latitude"] = "latitude"
    if "Longitude" in df.columns and "longitude" not in df.columns:
        rename_map["Longitude"] = "longitude"
    if "datetime" in df.columns and "timestamp" not in df.columns:
        rename_map["datetime"] = "timestamp"
    if "time" in df.columns and "timestamp" not in df.columns:
        rename_map["time"] = "timestamp"

    if rename_map:
        df = df.rename(columns=rename_map)

    # ensure timestamp column
    if "timestamp" not in df.columns:
        for c in ["time", "utc", "datetime", "date"]:
            if c in df.columns:
                df["timestamp"] = df[c]
                break

    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise SystemExit(
            f"CSV {path} must contain latitude and longitude columns (or lat/lon alternatives)"
        )

    if "user_id" not in df.columns:
        df["user_id"] = df.index.astype(str)

    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)

    df = df.dropna(subset=["timestamp", "latitude", "longitude"]).reset_index(drop=True)

    # IMPORTANT: limit rows to avoid memory explosion
    if max_rows is not None and len(df) > max_rows:
        print(f"Truncating from {len(df)} rows to first {max_rows} rows for memory safety.")
        df = df.iloc[:max_rows].copy()

    print("Final rows after cleaning:", len(df))
    return df


# -------------------------
# Main
# -------------------------
def main(args):
    # create output dirs
    if os.path.dirname(args.out_csv):
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(args.out_windows, exist_ok=True)

    df = load_csv_single(args.file, max_rows=args.max_rows)

    # if is_spoof missing, default to all normal
    if "is_spoof" not in df.columns:
        print("Warning: is_spoof not in columns, defaulting all to 0.")
        df["is_spoof"] = 0

    df = compute_point_features(df)

    # save merged CSV with features
    df.to_csv(args.out_csv, index=False)
    print("Saved feature CSV:", args.out_csv)

    candidate = [
        "speed_m_s",
        "accel",
        "bearing_diff",
        "dist_m",
        "dt",
        "hour",
        "dayofweek",
        "sudden_jump",
        "impossible_speed_flag",
    ]
    feature_cols = [c for c in candidate if c in df.columns]
    if not feature_cols:
        raise SystemExit("No numeric feature columns found after feature engineering.")

    print("Feature columns:", feature_cols)

    X, y = make_windows(
        df,
        feature_cols,
        window_size=args.window_size,
        stride=args.stride,
    )
    if X is None:
        raise SystemExit(
            "No windows created. Try smaller window_size or stride, or check data length per user."
        )

    np.save(os.path.join(args.out_windows, "X.npy"), X)
    np.save(os.path.join(args.out_windows, "y.npy"), y)
    with open(os.path.join(args.out_windows, "feature_names.json"), "w") as f:
        json.dump(feature_cols, f)

    print("Saved windows ->", args.out_windows)
    print("X shape:", X.shape, "y shape:", y.shape)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="Path to single GPS CSV (e.g. B_spoofed.csv)")
    p.add_argument(
        "--out-csv",
        dest="out_csv",
        default="data/processed/gps/merged_spoofed_single.csv",
        help="Path to save feature CSV",
    )
    p.add_argument(
        "--out-windows",
        dest="out_windows",
        default="data/windows/gps",
        help="Directory to save X.npy / y.npy / feature_names.json",
    )
    p.add_argument(
        "--window-size",
        dest="window_size",
        type=int,
        default=32,
        help="Sequence length per window",
    )
    p.add_argument(
        "--stride",
        dest="stride",
        type=int,
        default=8,
        help="Step size between window starts",
    )
    p.add_argument(
        "--max-rows",
        dest="max_rows",
        type=int,
        default=400000,
        help="Max rows to keep from CSV to avoid memory issues (default: 400000).",
    )
    args = p.parse_args()
    main(args)
