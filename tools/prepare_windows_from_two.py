"""
prepare_windows_from_two.py

Merge two GPS CSVs (A and B), label them (A -> normal / 0, B -> spoof / 1 unless is_spoof exists),
compute trajectory features, and generate sliding windows for sequence models.

Usage:
  python3 tools/prepare_windows_from_two.py \
    --fileA data/raw/gps/A.csv \
    --fileB data/raw/gps/B.csv \
    --out-csv data/processed/gps/merged_spoofed.csv \
    --out-windows data/windows \
    --window-size 32 --stride 8
"""
import os, argparse, json, math, random
import numpy as np
import pandas as pd
from datetime import datetime
random.seed(42); np.random.seed(42)

# -------------------------
# Geodesic helpers
# -------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)*2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)*2
    return 2*R*math.asin(math.sqrt(a))

def bearing(lat1, lon1, lat2, lon2):
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlambda)
    brng = math.degrees(math.atan2(x, y))
    return (brng + 360) % 360

def angdiff(a,b):
    d = abs(a-b) % 360
    return min(d, 360-d)

# -------------------------
# Feature computation
# -------------------------
def compute_point_features(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['user_id','timestamp']).reset_index(drop=True)
    df['dt'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds().fillna(0)
    df['lat_prev'] = df.groupby('user_id')['latitude'].shift(1)
    df['lon_prev'] = df.groupby('user_id')['longitude'].shift(1)
    df['dist_m'] = df.apply(lambda r: haversine(r['lat_prev'], r['lon_prev'], r['latitude'], r['longitude']) if pd.notnull(r['lat_prev']) else 0.0, axis=1)
    df['speed_m_s'] = df['dist_m'] / df['dt'].replace(0, np.nan)
    df['speed_m_s'] = df['speed_m_s'].fillna(0.0)
    df['bearing'] = df.apply(lambda r: bearing(r['lat_prev'], r['lon_prev'], r['latitude'], r['longitude']) if pd.notnull(r['lat_prev']) else 0.0, axis=1)
    df['bearing_prev'] = df.groupby('user_id')['bearing'].shift(1).fillna(0.0)
    df['bearing_diff'] = df.apply(lambda r: angdiff(r['bearing'], r['bearing_prev']), axis=1)
    df['speed_prev'] = df.groupby('user_id')['speed_m_s'].shift(1).fillna(0.0)
    df['accel'] = (df['speed_m_s'] - df['speed_prev']) / df['dt'].replace(0, np.nan)
    df['accel'] = df['accel'].replace([np.inf, -np.inf], 0).fillna(0)
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['sudden_jump'] = ((df['dist_m'] > 1000) & (df['dt'] < 60)).astype(int)
    df['impossible_speed_flag'] = (df['speed_m_s'] > 100).astype(int)
    for c in ['dist_m','speed_m_s','bearing_diff','accel','dt']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df

# -------------------------
# Window generation
# -------------------------
def make_windows(df, feature_cols, window_size=32, stride=8):
    X=[]
    y=[]
    # group by user to keep sequences contiguous
    for user, g in df.groupby('user_id'):
        g = g.sort_values('timestamp').reset_index(drop=True)
        arr = g[feature_cols].fillna(0).values
        labels = g['is_spoof'].astype(int).values if 'is_spoof' in g.columns else np.zeros(len(g))
        n = len(g)
        i = 0
        while i + window_size <= n:
            win = arr[i:i+window_size]
            lab = 1 if labels[i:i+window_size].sum() > 0 else 0
            X.append(win)
            y.append(lab)
            i += stride
    if not X:
        return None, None
    X = np.stack(X).astype(np.float32)
    y = np.array(y).astype(np.int64)
    return X, y

# -------------------------
# Main flow
# -------------------------
def load_csv(path):
    df = pd.read_csv(path)
    # map common alternative column names to our expected names
    rename_map = {}
    if 'lat' in df.columns and 'longitude' not in df.columns:
        rename_map['lat'] = 'latitude'
    if 'lon' in df.columns and 'longitude' not in df.columns:
        rename_map['lon'] = 'longitude'
    if 'Latitude' in df.columns and 'latitude' not in df.columns:
        rename_map['Latitude'] = 'latitude'
    if 'Longitude' in df.columns and 'longitude' not in df.columns:
        rename_map['Longitude'] = 'longitude'
    if 'datetime' in df.columns and 'timestamp' not in df.columns:
        rename_map['datetime'] = 'timestamp'
    if 'time' in df.columns and 'timestamp' not in df.columns:
        rename_map['time'] = 'timestamp'
    if rename_map:
        df = df.rename(columns=rename_map)

    # now validate
    if 'timestamp' not in df.columns:
        for c in ['time','utc','datetime','date']:
            if c in df.columns:
                df['timestamp'] = df[c]
                break
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        raise SystemExit(f"CSV {path} must contain latitude and longitude columns (or lat/lon alternatives)")
    if 'user_id' not in df.columns:
        df['user_id'] = df.index.astype(str)
    df['latitude'] = df['latitude'].astype(float)
    df['longitude'] = df['longitude'].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp','latitude','longitude']).reset_index(drop=True)
    return df


def main(args):
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    os.makedirs(args.out_windows, exist_ok=True)
    # load
    print("Loading A:", args.fileA)
    A = load_csv(args.fileA)
    print("Loading B:", args.fileB)
    B = load_csv(args.fileB)
    # label if not present
    if 'is_spoof' not in A.columns:
        A['is_spoof'] = 0
    if 'is_spoof' not in B.columns:
        B['is_spoof'] = 1
    # prefix user ids so they remain distinct
    A['user_id'] = A['user_id'].astype(str).apply(lambda s: f"A_{s}")
    B['user_id'] = B['user_id'].astype(str).apply(lambda s: f"B_{s}")
    combined = pd.concat([A, B], ignore_index=True, sort=False)
    combined = combined.sort_values(['user_id','timestamp']).reset_index(drop=True)
    # compute point features
    combined = compute_point_features(combined)
    # save merged CSV
    combined.to_csv(args.out_csv, index=False)
    print("Saved merged CSV:", args.out_csv)
    # choose feature columns for windows (only numeric)
    candidate = ['speed_m_s','accel','bearing_diff','dist_m','dt','hour','dayofweek','sudden_jump','impossible_speed_flag']
    feature_cols = [c for c in candidate if c in combined.columns]
    if not feature_cols:
        raise SystemExit("No numeric feature columns found after feature engineering.")
    print("Feature columns:", feature_cols)
    X, y = make_windows(combined, feature_cols, window_size=args.window_size, stride=args.stride)
    if X is None:
        raise SystemExit("No windows created. Try smaller window_size or smaller stride, or check data length per user.")
    np.save(os.path.join(args.out_windows, 'X.npy'), X)
    np.save(os.path.join(args.out_windows, 'y.npy'), y)
    with open(os.path.join(args.out_windows, 'feature_names.json'), 'w') as f:
        json.dump(feature_cols, f)
    print("Saved windows ->", args.out_windows)
    print("X shape:", X.shape, "y shape:", y.shape)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--fileA', required=True)
    p.add_argument('--fileB', required=True)
    p.add_argument('--out-csv', dest='out_csv', default='data/processed/gps/merged_spoofed.csv')
    p.add_argument('--out-windows', dest='out_windows', default='data/windows')
    p.add_argument('--window-size', dest='window_size', type=int, default=32)
    p.add_argument('--stride', dest='stride', type=int, default=8)
    args = p.parse_args()
    main(args)