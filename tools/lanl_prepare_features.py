"""
lanl_prepare_features.py

Stream the full LANL authentication dataset (13GB, 708M lines) and
produce per-event features in chunked CSV files.

Each raw line is:
    time,user,computer

Example:
    12,U8,C9

We compute features like:
    - user_deg: number of logins by this user so far
    - comp_deg: number of logins to this computer so far
    - time_since_user_last
    - time_since_comp_last
    - hour_of_day (time modulo 24h)
    - is_new_user
    - is_new_comp

Output:
    data/processed/login/lanl_features_chunk_000.csv
    data/processed/login/lanl_features_chunk_001.csv
    ...

You can control:
    --chunk-rows  (how many rows per output file)
    --max-rows    (optional cap on total rows processed, for testing)
"""

import os
import argparse
import math

import pandas as pd


def ensure_dirs(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def stream_lanl_with_features(
    in_file: str,
    out_dir: str,
    chunk_rows: int = 5_000_000,
    max_rows: int | None = None,
):
    """
    Stream LANL file, compute features on the fly, and write chunked CSVs.
    """

    os.makedirs(out_dir, exist_ok=True)

    # running stats
    user_deg = {}         # user -> count of logins so far
    comp_deg = {}         # computer -> count of logins so far
    user_last_time = {}   # user -> last time seen
    comp_last_time = {}   # computer -> last time seen

    # current batch buffers
    batch = {
        "time": [],
        "user": [],
        "computer": [],
        "user_deg": [],
        "comp_deg": [],
        "time_since_user_last": [],
        "time_since_comp_last": [],
        "hour_of_day": [],
        "is_new_user": [],
        "is_new_comp": [],
    }

    total_rows = 0
    chunk_idx = 0

    print(f"[INFO] Reading from: {in_file}")
    print(f"[INFO] Writing chunked features to: {out_dir}")
    print(f"[INFO] chunk_rows = {chunk_rows}, max_rows = {max_rows}")

    with open(in_file, "r", errors="ignore") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 3:
                # malformed line, skip
                continue

            t_str, u, c = parts
            try:
                t = int(t_str)
            except ValueError:
                # bad time, skip
                continue

            # basic stats
            is_new_u = 1 if u not in user_deg else 0
            is_new_c = 1 if c not in comp_deg else 0

            # degrees BEFORE including this event
            u_deg_prev = user_deg.get(u, 0)
            c_deg_prev = comp_deg.get(c, 0)

            # last times
            last_t_u = user_last_time.get(u, None)
            last_t_c = comp_last_time.get(c, None)

            if last_t_u is None:
                dt_u = -1  # or 0; -1 means "no previous login"
            else:
                dt_u = t - last_t_u

            if last_t_c is None:
                dt_c = -1
            else:
                dt_c = t - last_t_c

            # hour of day (pseudo, because epoch is anonymized, but OK for pattern)
            # 86400 seconds = 24*3600
            seconds_in_day = 86400
            sec_of_day = t % seconds_in_day
            hour_of_day = int(sec_of_day // 3600)

            # append features
            batch["time"].append(t)
            batch["user"].append(u)
            batch["computer"].append(c)
            batch["user_deg"].append(u_deg_prev)
            batch["comp_deg"].append(c_deg_prev)
            batch["time_since_user_last"].append(dt_u)
            batch["time_since_comp_last"].append(dt_c)
            batch["hour_of_day"].append(hour_of_day)
            batch["is_new_user"].append(is_new_u)
            batch["is_new_comp"].append(is_new_c)

            # update stats AFTER using them
            user_deg[u] = u_deg_prev + 1
            comp_deg[c] = c_deg_prev + 1
            user_last_time[u] = t
            comp_last_time[c] = t

            total_rows += 1

            # flush batch if full
            if len(batch["time"]) >= chunk_rows:
                out_path = os.path.join(
                    out_dir, f"lanl_features_chunk_{chunk_idx:03d}.csv"
                )
                df = pd.DataFrame(batch)
                df.to_csv(out_path, index=False)
                print(
                    f"[INFO] Wrote chunk {chunk_idx} with {len(df)} rows -> {out_path}"
                )
                chunk_idx += 1

                # reset batch
                for k in batch:
                    batch[k] = []

            # check max_rows
            if max_rows is not None and total_rows >= max_rows:
                print(f"[INFO] Reached max_rows={max_rows}, stopping.")
                break

    # flush remaining
    if batch["time"]:
        out_path = os.path.join(
            out_dir, f"lanl_features_chunk_{chunk_idx:03d}.csv"
        )
        df = pd.DataFrame(batch)
        df.to_csv(out_path, index=False)
        print(
            f"[INFO] Wrote final chunk {chunk_idx} with {len(df)} rows -> {out_path}"
        )

    print(f"[DONE] Total processed rows: {total_rows}")
    print("[DONE] Feature generation complete.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-file",
        required=True,
        help="data/raw/login/lanl_auth_dataset-1.txt",
    )
    ap.add_argument(
        "--out-dir",
        default="data/processed/login/features",
        help="Directory to store chunked feature CSVs",
    )
    ap.add_argument(
        "--chunk-rows",
        type=int,
        default=5_000_000,
        help="Number of rows per output chunk CSV (default: 5M)",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on total rows processed (for testing).",
    )
    args = ap.parse_args()

    stream_lanl_with_features(
        in_file=args.in_file,
        out_dir=args.out_dir,
        chunk_rows=args.chunk_rows,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
