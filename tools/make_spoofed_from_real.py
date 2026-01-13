"""
make_spoofed_from_real.py

Given ONE real GPS CSV, create:
  - A_normal.csv  (real trajectories)
  - B_spoofed.csv (same trajectories + injected spoof attacks, with is_spoof column)

Usage:
  python3 tools/make_spoofed_from_real.py \
    --in-csv data/raw/gps/real_gps.csv \
    --outA data/raw/gps/A_normal.csv \
    --outB data/raw/gps/B_spoofed.csv \
    --attack-rate 0.05
"""

import argparse, math, random
import numpy as np
import pandas as pd
from datetime import timedelta

random.seed(42)
np.random.seed(42)

# ------------- helpers -------------
def load_gps(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # map common names to expected ones
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
        raise SystemExit("CSV must contain a timestamp/datetime column")
    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise SystemExit("CSV must contain latitude/longitude (or lat/lon/Latitude/Longitude)")

    if "user_id" not in df.columns:
        # if you don't have user_id, treat each file as one user
        df["user_id"] = "user_0"

    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert(None)  # make tz‑naive

    df = df.dropna(subset=["timestamp", "latitude", "longitude"]).reset_index(drop=True)
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return df


def random_offset_deg(min_km=50, max_km=500):
    """
    Convert a random distance in km to approx lat/lon degree offsets.
    1 degree ~ 111 km.
    """
    dist_km = random.uniform(min_km, max_km)
    deg = dist_km / 111.0
    # random direction
    angle = random.uniform(0, 2 * math.pi)
    dlat = deg * math.cos(angle)
    dlon = deg * math.sin(angle)
    return dlat, dlon


# ------------- main spoof injection -------------
def inject_spoof_attacks(df: pd.DataFrame, attack_rate: float = 0.05) -> pd.DataFrame:
    """
    attack_rate = fraction of points per user that will be the *start* of an attack
    For each chosen index we randomly pick an attack type:
      - teleport jump (single point)
      - static spoof (small block)
      - drift spoof (small block)
    """
    df = df.copy()
    df["is_spoof"] = 0
    df["spoof_type"] = "none"

    for uid, g_idx in df.groupby("user_id").groups.items():
        idxs = list(g_idx)
        n = len(idxs)
        if n < 10:
            continue

        n_attacks = max(1, int(n * attack_rate))
        attack_starts = np.random.choice(idxs[:-5], size=n_attacks, replace=False)

        for start_idx in attack_starts:
            attack_type = random.choice(["jump", "static", "drift"])

            if attack_type == "jump":
                # modify exactly one point to be far away
                lat = df.at[start_idx, "latitude"]
                lon = df.at[start_idx, "longitude"]
                dlat, dlon = random_offset_deg()
                df.at[start_idx, "latitude"] = lat + dlat
                df.at[start_idx, "longitude"] = lon + dlon
                df.at[start_idx, "is_spoof"] = 1
                df.at[start_idx, "spoof_type"] = "jump"

            elif attack_type == "static":
                # make a short block (e.g. 5 points) all same location
                block_len = random.randint(3, 7)
                block_idxs = [i for i in range(start_idx, start_idx + block_len) if i in idxs]
                if not block_idxs:
                    continue
                lat = df.at[block_idxs[0], "latitude"]
                lon = df.at[block_idxs[0], "longitude"]
                for i in block_idxs:
                    df.at[i, "latitude"] = lat
                    df.at[i, "longitude"] = lon
                    df.at[i, "is_spoof"] = 1
                    df.at[i, "spoof_type"] = "static"

            elif attack_type == "drift":
                # generate a very smooth unrealistic drift over a block
                block_len = random.randint(5, 10)
                block_idxs = [i for i in range(start_idx, start_idx + block_len) if i in idxs]
                if len(block_idxs) < 2:
                    continue
                base_lat = df.at[block_idxs[0], "latitude"]
                base_lon = df.at[block_idxs[0], "longitude"]
                dlat, dlon = random_offset_deg(min_km=5, max_km=20)
                for j, i in enumerate(block_idxs):
                    t = j / max(1, len(block_idxs) - 1)
                    df.at[i, "latitude"] = base_lat + t * dlat
                    df.at[i, "longitude"] = base_lon + t * dlon
                    df.at[i, "is_spoof"] = 1
                    df.at[i, "spoof_type"] = "drift"

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", required=True, help="Real GPS CSV")
    ap.add_argument("--outA", required=True, help="Output path for normal A.csv")
    ap.add_argument("--outB", required=True, help="Output path for spoofed B.csv")
    ap.add_argument("--attack-rate", type=float, default=0.05,
                    help="Fraction of points per user to turn into starts of attacks")
    args = ap.parse_args()

    print(f"Loading real GPS from {args.in_csv}")
    df_real = load_gps(args.in_csv)

    # A: normal copy (no spoof labels)
    df_real.to_csv(args.outA, index=False)
    print(f"Saved normal trajectories to {args.outA}")

    # B: spoofed copy with is_spoof column
    df_spoofed = inject_spoof_attacks(df_real, attack_rate=args.attack_rate)
    df_spoofed.to_csv(args.outB, index=False)
    print(f"Saved spoofed trajectories with is_spoof column to {args.outB}")


if __name__ == "__main__":
    main()
