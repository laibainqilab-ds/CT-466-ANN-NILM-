"""
Stage 5 (pre-windowing) target cleaning (CT-466 Phase 2).

Reads data/processed/five_homes_aligned.parquet (left unmodified).
Applies exactly two approved transformations to a NEW derived file:
  A) targeted invalidation of the single extreme-anomaly row
     (home 7901, train split, utc_dt=2018-09-22 01:54:00):
     dishwasher1 and refrigerator1 set to NaN / valid=False.
  B) all remaining small negative appliance readings (air1, dishwasher1,
     refrigerator1 -- never aggregate/grid/solar) converted to 0.0,
     keeping their validity flag True.
Writes data/processed/five_homes_modeling.parquet and a full
documentation report. data/interim/five_homes.parquet and
data/processed/five_homes_aligned.parquet are never written to.
"""
import os
import time
import json
import numpy as np
import pandas as pd

ALIGNED_PATH = r"E:\CT-466\ANN_NILM\data\processed\five_homes_aligned.parquet"
RAW_PATH = r"E:\CT-466\ANN_NILM\data\interim\five_homes.parquet"
OUT_PATH = r"E:\CT-466\ANN_NILM\data\processed\five_homes_modeling.parquet"
REPORT_PATH = r"E:\CT-466\ANN_NILM\data\meta\stage5_target_cleaning_report.json"

HOMES = ["7901", "7951", "8386", "8565", "3538"]
APPLIANCES = ["air1", "dishwasher1", "refrigerator1"]

ANOMALY = {"dataid": "7901", "split": "train", "utc_dt": pd.Timestamp("2018-09-22 01:54:00")}
ANOMALY_FIELDS = ["dishwasher1", "refrigerator1"]

print(f"PID={os.getpid()}", flush=True)
t0 = time.time()

# --- capture preserved-file integrity BEFORE any work ---
def file_stat(path):
    return {"size": os.path.getsize(path), "mtime": os.path.getmtime(path)}

preserved_before = {
    "five_homes.parquet": file_stat(RAW_PATH),
    "five_homes_aligned.parquet": file_stat(ALIGNED_PATH),
}

df = pd.read_parquet(ALIGNED_PATH, engine="pyarrow")
n_rows_before = len(df)
cols_before = list(df.columns)

agg_neg_before = int((df["aggregate"] < 0).sum())
agg_valid_before = int(df["aggregate"].notna().sum())

report = {
    "source_file": ALIGNED_PATH,
    "output_file": OUT_PATH,
    "n_rows_before": n_rows_before,
    "columns_before": cols_before,
}

# --- Transformation A: targeted invalidation ---
mask_anom = (
    (df["dataid"] == ANOMALY["dataid"]) &
    (df["split"] == ANOMALY["split"]) &
    (df["utc_dt"] == ANOMALY["utc_dt"])
)
n_anom_rows = int(mask_anom.sum())
if n_anom_rows != 1:
    raise RuntimeError(f"STOP: expected exactly 1 anomaly row, found {n_anom_rows}")

original_values = {f: float(df.loc[mask_anom, f].iloc[0]) for f in ANOMALY_FIELDS}
for f in ANOMALY_FIELDS:
    df.loc[mask_anom, f] = np.nan
    df.loc[mask_anom, f"{f}_valid"] = False

report["transformation_A_invalidation"] = {
    "row": {k: str(v) for k, v in ANOMALY.items()},
    "original_values": original_values,
    "fields_invalidated": ANOMALY_FIELDS,
}

# --- Transformation B: small negative appliance readings -> 0.0 ---
zero_masks = {}
for a in APPLIANCES:
    valid_mask = df[f"{a}_valid"]
    zero_masks[a] = valid_mask & (df[a] < 0)

breakdown = []
for home in HOMES:
    for split in ["train", "val", "test"]:
        grp_mask = (df["dataid"] == home) & (df["split"] == split)
        row = {"home": home, "split": split}
        for a in APPLIANCES:
            combined = zero_masks[a] & grp_mask
            n_conv = int(combined.sum())
            min_val = float(df.loc[combined, a].min()) if n_conv else None
            row[f"{a}_n_converted"] = n_conv
            row[f"{a}_min_value_converted"] = min_val
        breakdown.append(row)

total_converted = {}
for a in APPLIANCES:
    n = int(zero_masks[a].sum())
    df.loc[zero_masks[a], a] = 0.0
    total_converted[a] = n

report["transformation_B_breakdown"] = breakdown
report["transformation_B_totals"] = total_converted

n_rows_after = len(df)
cols_after = list(df.columns)

# --- write new derived file only ---
df.to_parquet(OUT_PATH, engine="pyarrow", index=False)

# --- post-run verification ---
verification = {}

check_df = pd.read_parquet(OUT_PATH, engine="pyarrow")
verification["row_count_matches"] = (len(check_df) == n_rows_before)
verification["schema_matches"] = (list(check_df.columns) == cols_before)

invalid_rows = check_df[
    check_df["dishwasher1"].isna() & check_df["refrigerator1"].isna() &
    (~check_df["dishwasher1_valid"]) & (~check_df["refrigerator1_valid"])
]
verification["exactly_one_invalidated_row"] = (len(invalid_rows) == 1)
if len(invalid_rows) == 1:
    r = invalid_rows.iloc[0]
    verification["invalidated_row_matches_expected"] = (
        str(r["dataid"]) == ANOMALY["dataid"] and
        str(r["split"]) == ANOMALY["split"] and
        pd.Timestamp(r["utc_dt"]) == ANOMALY["utc_dt"]
    )
else:
    verification["invalidated_row_matches_expected"] = False

remaining_negatives = {}
for a in APPLIANCES:
    valid_mask = check_df[f"{a}_valid"]
    remaining_negatives[a] = int((valid_mask & (check_df[a] < 0)).sum())
verification["remaining_negative_counts"] = remaining_negatives
verification["zero_negatives_remaining"] = all(v == 0 for v in remaining_negatives.values())

agg_neg_after = int((check_df["aggregate"] < 0).sum())
agg_valid_after = int(check_df["aggregate"].notna().sum())
verification["aggregate_unchanged"] = (agg_neg_after == agg_neg_before and agg_valid_after == agg_valid_before)
verification["aggregate_neg_before"] = agg_neg_before
verification["aggregate_neg_after"] = agg_neg_after

preserved_after = {
    "five_homes.parquet": file_stat(RAW_PATH),
    "five_homes_aligned.parquet": file_stat(ALIGNED_PATH),
}
verification["preserved_files_unchanged"] = (preserved_before == preserved_after)
verification["preserved_before"] = preserved_before
verification["preserved_after"] = preserved_after

report["verification"] = verification

elapsed = time.time() - t0
report["elapsed_seconds"] = elapsed

with open(REPORT_PATH, "w") as fh:
    json.dump(report, fh, indent=2, default=str)

all_passed = (
    verification["row_count_matches"] and verification["schema_matches"] and
    verification["exactly_one_invalidated_row"] and verification["invalidated_row_matches_expected"] and
    verification["zero_negatives_remaining"] and verification["aggregate_unchanged"] and
    verification["preserved_files_unchanged"]
)

print(f"DONE elapsed={elapsed:.1f}s output_rows={n_rows_after:,} ALL_VERIFICATION_PASSED={all_passed}", flush=True)
if not all_passed:
    print("VERIFICATION_FAILURE_DETAILS:", json.dumps(verification, indent=2, default=str), flush=True)
