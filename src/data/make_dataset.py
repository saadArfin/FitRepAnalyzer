# import pandas as pd
# from glob import glob
# import numpy as np

# # --------------------------------------------------------------
# # Read single CSV file
# # --------------------------------------------------------------

# single_file_acc = pd.read_csv(
#     "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
# )
# single_file_gyr = pd.read_csv(
#     "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
# )
# # --------------------------------------------------------------
# # List all data in data/raw/MetaMotion
# # --------------------------------------------------------------
# files = glob("../../data/raw/MetaMotion/*.csv")
# len(files)
# # --------------------------------------------------------------
# # Extract features from filename
# # --------------------------------------------------------------

# data_path = "../../data/raw/MetaMotion\\"
# f = files[1]

# participant = f.split("-")[0].replace(data_path, "")
# label = f.split("-")[1]
# category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

# df = pd.read_csv(f)
# df["participant"] = participant
# df["label"] = label
# df["category"] = category
# df
# # --------------------------------------------------------------
# # Read all files
# # --------------------------------------------------------------

# acc_df = pd.DataFrame()
# gyr_df = pd.DataFrame()

# acc_set = 1
# gyr_set = 1

# for f in files:
#     participant = f.split("-")[0].replace(data_path, "")
#     label = f.split("-")[1]
#     category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

#     df = pd.read_csv(f)

#     df["participant"] = participant
#     df["label"] = label
#     df["category"] = category

#     if "Accelerometer" in f:
#         df["set"] = acc_set
#         acc_set += 1
#         acc_df = pd.concat([acc_df, df])

#     if "Gyroscope" in f:
#         df["set"] = gyr_set
#         gyr_set += 1
#         gyr_df = pd.concat([gyr_df, df])

# # --------------------------------------------------------------
# # Working with datetimes
# # --------------------------------------------------------------

# acc_df.info()

# pd.to_datetime(df["epoch (ms)"], unit="ms")
# acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
# gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

# del acc_df["epoch (ms)"]
# del acc_df["time (01:00)"]
# del acc_df["elapsed (s)"]

# del gyr_df["epoch (ms)"]
# del gyr_df["time (01:00)"]
# del gyr_df["elapsed (s)"]


# # --------------------------------------------------------------
# # Turn into function
# # --------------------------------------------------------------
# files = files = glob("../../data/raw/MetaMotion/*.csv")


# def read_data_from_files(files):
#     acc_df = pd.DataFrame()
#     gyr_df = pd.DataFrame()

#     acc_set = 1
#     gyr_set = 1

#     for f in files:
#         participant = f.split("-")[0].replace(data_path, "")
#         label = f.split("-")[1]
#         category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

#         df = pd.read_csv(f)

#         df["participant"] = participant
#         df["label"] = label
#         df["category"] = category

#         if "Accelerometer" in f:
#             df["set"] = acc_set
#             acc_set += 1
#             acc_df = pd.concat([acc_df, df])

#         if "Gyroscope" in f:
#             df["set"] = gyr_set
#             gyr_set += 1
#             gyr_df = pd.concat([gyr_df, df])

#     acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
#     gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

#     del acc_df["epoch (ms)"]
#     del acc_df["time (01:00)"]
#     del acc_df["elapsed (s)"]

#     del gyr_df["epoch (ms)"]
#     del gyr_df["time (01:00)"]
#     del gyr_df["elapsed (s)"]

#     return acc_df, gyr_df


# acc_df, gyr_df = read_data_from_files(files)

# # --------------------------------------------------------------
# # Merging datasets
# # --------------------------------------------------------------
# data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)

# # Rename columns
# data_merged.columns = [
#     "acc_x",
#     "acc_y",
#     "acc_z",
#     "gyr_x",
#     "gyr_y",
#     "gyr_z",
#     "participant",
#     "label",
#     "category",
#     "set",
# ]


# # --------------------------------------------------------------
# # Resample data (frequency conversion)
# # --------------------------------------------------------------

# # Accelerometer:    12.500HZ
# # Gyroscope:        25.000Hz

# sampling = {
#     "acc_x": "mean",
#     "acc_y": "mean",
#     "acc_z": "mean",
#     "gyr_x": "mean",
#     "gyr_y": "mean",
#     "gyr_z": "mean",
#     "label": "last",
#     "category": "last",
#     "participant": "last",
#     "set": "last",
# }

# data_merged[:1000].resample(rule="200ms").apply(sampling)

# # Split by day
# days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
# data_resampled = pd.concat(
#     [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
# )
# data_resampled["set"] = data_resampled["set"].astype("int")
# data_resampled.info()


# # --------------------------------------------------------------
# # Export dataset
# # --------------------------------------------------------------
# data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")
#!/usr/bin/env python3
"""
Robust make_dataset script that works regardless of current working directory.
- Builds paths relative to this script's location (two parents up -> repo root).
- Optionally accepts --data-root to override the data directory.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import sys

def get_repo_root():
    try:
        # src/data/make_dataset.py -> parents[2] is repository root
        return Path(__file__).resolve().parents[2]
    except NameError:
        # fallback when __file__ is not defined (interactive environments)
        return Path.cwd()

def parse_filename(p: Path):
    """
    Extract participant, label, category from filename.
    Example filename:
    A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv
    """
    name = p.name
    parts = name.split("-")
    participant = parts[0] if len(parts) > 0 else ""
    label = parts[1] if len(parts) > 1 else ""
    category = ""
    if len(parts) > 2:
        # Take third chunk and remove any trailing '_MetaWear...' suffix
        raw_cat = parts[2]
        category = raw_cat.split("_MetaWear")[0]
        # remove trailing digits used in some filenames like 'heavy2' -> keep as is
        category = category.rstrip("123")
        category = category.rstrip("_")
    return participant, label, category

def read_files(files):
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()
    acc_set = 1
    gyr_set = 1

    for p in files:
        try:
            participant, label, category = parse_filename(p)
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}", file=sys.stderr)
            continue

        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if "Accelerometer" in p.name:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df], ignore_index=False)
        elif "Gyroscope" in p.name:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df], ignore_index=False)
        else:
            # If file doesn't have either string, we still try to infer by columns
            if {"acc x", "acc y", "acc z"} <= set(c.lower() for c in df.columns):
                df["set"] = acc_set
                acc_set += 1
                acc_df = pd.concat([acc_df, df], ignore_index=False)
            elif {"gyr x", "gyr y", "gyr z"} <= set(c.lower() for c in df.columns):
                df["set"] = gyr_set
                gyr_set += 1
                gyr_df = pd.concat([gyr_df, df], ignore_index=False)
            else:
                print(f"[INFO] Skipping file (unknown type): {p}", file=sys.stderr)

    # Convert epoch to datetime index if present
    if not acc_df.empty and "epoch (ms)" in acc_df.columns:
        acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    if not gyr_df.empty and "epoch (ms)" in gyr_df.columns:
        gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    # Safely drop columns if present
    for col in ["epoch (ms)", "time (01:00)", "elapsed (s)"]:
        if col in acc_df.columns:
            del acc_df[col]
        if col in gyr_df.columns:
            del gyr_df[col]

    return acc_df, gyr_df

def main(data_root: Path):
    meta_dir = data_root / "raw" / "MetaMotion"
    if not meta_dir.exists():
        raise FileNotFoundError(f"MetaMotion data directory does not exist: {meta_dir}")

    files = sorted(meta_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {meta_dir}")

    print(f"Found {len(files)} CSV files in {meta_dir}")

    acc_df, gyr_df = read_files(files)

    if acc_df.empty:
        print("[ERROR] No accelerometer files loaded.", file=sys.stderr)
    if gyr_df.empty:
        print("[ERROR] No gyroscope files loaded.", file=sys.stderr)

    # Attempt to merge side-by-side (assumes same timestamp index)
    # In original code: data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)
    # We will try the same approach but first ensure each side has expected columns.
    # Take first three numeric columns of acc_df as acc_x,acc_y,acc_z
    def first_n_numeric_cols(df, n=3):
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        return numeric_cols[:n]

    acc_cols = first_n_numeric_cols(acc_df, 3)
    gyr_cols = first_n_numeric_cols(gyr_df, 3)

    if len(acc_cols) < 3 or len(gyr_cols) < 3:
        raise ValueError("Could not find three numeric columns for accel/gyro to merge. "
                         f"acc_cols={acc_cols} gyr_cols={gyr_cols}")

    # Compose merged DataFrame
    data_merged = pd.concat([acc_df[acc_cols], gyr_df[gyr_cols],
                             acc_df[["participant", "label", "category", "set"]].fillna(method="ffill").iloc[:, :4]],
                            axis=1)

    # Rename columns to desired names
    # Ensure final length is 10
    expected_len = 10
    if data_merged.shape[1] != expected_len:
        # If there's mismatch, try to construct safe column list
        cols = []
        cols += [f"acc_{i}" for i in ["x", "y", "z"]]
        cols += [f"gyr_{i}" for i in ["x", "y", "z"]]
        cols += ["participant", "label", "category", "set"]
        # If columns match length, rename, else raise to make debugging explicit
        if data_merged.shape[1] == len(cols):
            data_merged.columns = cols
        else:
            raise ValueError(f"Unexpected number of columns after concat: {data_merged.shape[1]}, expected {len(cols)}")

    else:
        data_merged.columns = [
            "acc_x", "acc_y", "acc_z",
            "gyr_x", "gyr_y", "gyr_z",
            "participant", "label", "category", "set"
        ]

    # Resampling
    sampling = {
        "acc_x": "mean",
        "acc_y": "mean",
        "acc_z": "mean",
        "gyr_x": "mean",
        "gyr_y": "mean",
        "gyr_z": "mean",
        "label": "last",
        "category": "last",
        "participant": "last",
        "set": "last",
    }

    # Keep only first 1000 rows for a quick preview resample if needed (like original)
    try:
        preview = data_merged[:1000].resample("200ms").apply(sampling)
    except Exception as e:
        print(f"[WARN] Preview resample failed: {e}", file=sys.stderr)

    # Split by day and resample each day then concat, drop total-NaN rows
    days = [g for _, g in data_merged.groupby(pd.Grouper(freq="D")) if not g.empty]
    if not days:
        raise ValueError("No daily groups found in merged data (index may not be datetime).")

    resampled_list = []
    for day_df in days:
        rs = day_df.resample("200ms").apply(sampling).dropna()
        if not rs.empty:
            resampled_list.append(rs)

    if resampled_list:
        data_resampled = pd.concat(resampled_list)
        # ensure set is integer type
        if "set" in data_resampled.columns:
            data_resampled["set"] = data_resampled["set"].astype(int)
    else:
        data_resampled = pd.DataFrame()
        print("[WARN] No resampled data produced (all NaN after resampling).", file=sys.stderr)

    # Export
    out_dir = data_root.parent / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "01_data_processed.pkl"
    data_resampled.to_pickle(out_path)
    print(f"Saved processed data to {out_path}")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dataset from MetaMotion CSVs.")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Path to data folder (contains raw/MetaMotion). If omitted, inferred relative to script.")
    args = parser.parse_args()

    repo_root = get_repo_root()
    if args.data_root:
        data_root = Path(args.data_root).expanduser().resolve()
    else:
        data_root = repo_root / "data"

    print(f"Using data root: {data_root}")
    main(data_root)
