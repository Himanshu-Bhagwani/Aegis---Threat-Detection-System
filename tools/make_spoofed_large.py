"""
make_spoofed_large.py

Optimized spoof injection for LARGE GPS CSVs.

- Reads ONE big real GPS CSV
- Injects spoof attacks (jump / static / drift) in-place
- Writes ONE spoofed CSV (B_spoofed.csv)
- You then use:
    fileA = original CSV
    fileB = spoofed CSV
  with prepare_windows_from_two.py

Usage:
  python tools/make_spoofed_large.py \
    --in-csv data/raw/gps/geolife_trajectories.csv \
    --outB data/raw/gps/B_spoofed.csv \
    --attack-rate 0.02
"""

import argparse
import math
import random

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)


def load_gps(path: str) -> pd.DataFrame:
    """
    Load a big GPS CSV and normalize column names:
    - user_id
    - timestamp
    - latitude
    - longitude
    """
    print(f"[load] reading {path} ...")
    df = pd.read_csv(path, low_memory=False)

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

    if "timestamp" not in df.columns:
        raise SystemExit("CSV must contain timestamp/datetime column (timestamp/datetime/time).")

    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise SystemExit("CSV must contain latitude/longitude (or lat/lon/Latitude/Longitude).")

    if "user_id" not in df.columns:
        # if no user_id, treat entire file as one user
        df["user_id"] = "user_0"

    # keep only what we need plus any extra columns
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)

    # we don't actually need to parse timestamp to datetime for spoofing logic,
    # so we leave it as string to save time
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    print("[load] rows:", len(df))
    return df


def random_offset_deg(min_km=50.0, max_km=300.0):
    """
    Convert a random distance in km to approximate lat/lon degree offsets.
    1 degree ~ 111 km.
    """
    dist_km = random.uniform(min_km, max_km)
    deg = dist_km / 111.0
    angle = random.uniform(0.0, 2.0 * math.pi)
    dlat = deg * math.cos(angle)
    dlon = deg * math.sin(angle)
    return dlat, dlon


def inject_spoof_attacks(df: pd.DataFrame, attack_rate: float = 0.02) -> pd.DataFrame:
    """
    attack_rate = fraction of points per user that will be starts of attacks.
    Optimized for large files:
      - we operate per-user but only touch small blocks
      - use numpy operations as much as possible
    """
    df = df.copy()
    n_total = len(df)
    print(f"[spoof] total rows: {n_total}")

    df["is_spoof"] = 0
    df["spoof_type"] = "none"

    # Precompute groups once (dict: user_id -> indices array)
    groups = df.groupby("user_id", sort=False).indices

    for uid, idx_array in groups.items():
        idxs = np.fromiter(idx_array, dtype=np.int64)
        n = len(idxs)
        if n < 10:
            continue

        # number of attacks for this user (cap to avoid insane counts)
        n_attacks = int(n * attack_rate)
        if n_attacks <= 0:
            n_attacks = 1
        # optional cap per user so some users don't dominate
        n_attacks = min(n_attacks, 200)

        # choose start indices without replacement, away from tail (for blocks)
        valid_start_idxs = idxs[idxs < (idxs[-1] - 10)]
        if len(valid_start_idxs) == 0:
            continue

        if n_attacks > len(valid_start_idxs):
            n_attacks = len(valid_start_idxs)

        starts = np.random.choice(valid_start_idxs, size=n_attacks, replace=False)

        for start_idx in starts:
            attack_type = random.choice(["jump", "static", "drift"])

            if attack_type == "jump":
                # modify exactly one point far away
                lat = df.at[start_idx, "latitude"]
                lon = df.at[start_idx, "longitude"]
                dlat, dlon = random_offset_deg()
                df.at[start_idx, "latitude"] = lat + dlat
                df.at[start_idx, "longitude"] = lon + dlon
                df.at[start_idx, "is_spoof"] = 1
                df.at[start_idx, "spoof_type"] = "jump"

            elif attack_type == "static":
                # a small block stays stuck at same coordinate
                block_len = random.randint(3, 8)
                block_idx = np.arange(start_idx, start_idx + block_len)
                block_idx = block_idx[block_idx <= idxs[-1]]
                if len(block_idx) == 0:
                    continue
                lat = df.at[block_idx[0], "latitude"]
                lon = df.at[block_idx[0], "longitude"]
                df.loc[block_idx, "latitude"] = lat
                df.loc[block_idx, "longitude"] = lon
                df.loc[block_idx, "is_spoof"] = 1
                df.loc[block_idx, "spoof_type"] = "static"

            elif attack_type == "drift":
                # smooth linear drift over a small block
                block_len = random.randint(5, 12)
                block_idx = np.arange(start_idx, start_idx + block_len)
                block_idx = block_idx[block_idx <= idxs[-1]]
                if len(block_idx) < 2:
                    continue

                base_lat = float(df.at[block_idx[0], "latitude"])
                base_lon = float(df.at[block_idx[0], "longitude"])
                dlat, dlon = random_offset_deg(min_km=5.0, max_km=30.0)

                t = np.linspace(0.0, 1.0, num=len(block_idx), dtype=np.float64)
                lat_series = base_lat + t * dlat
                lon_series = base_lon + t * dlon

                df.loc[block_idx, "latitude"] = lat_series
                df.loc[block_idx, "longitude"] = lon_series
                df.loc[block_idx, "is_spoof"] = 1
                df.loc[block_idx, "spoof_type"] = "drift"

    spoof_count = int(df["is_spoof"].sum())
    print(f"[spoof] total spoofed points: {spoof_count}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", required=True, help="Real GPS CSV (large)")
    ap.add_argument("--outB", required=True, help="Output path for spoofed CSV (B_spoofed)")
    ap.add_argument(
        "--attack-rate",
        type=float,
        default=0.02,
        help="Fraction of points per user to use as starts of attacks (default: 0.02)",
    )
    args = ap.parse_args()

    df_real = load_gps(args.in_csv)
    df_spoofed = inject_spoof_attacks(df_real, attack_rate=args.attack_rate)

    print(f"[save] writing spoofed CSV to {args.outB} ...")
    df_spoofed.to_csv(args.outB, index=False)
    print("[done]")


if __name__ == "__main__":
    main()
