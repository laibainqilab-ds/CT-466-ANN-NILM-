"""
Stage 6: appliance ON/OFF label generation (CT-466 Phase 2).

Reads the three existing window-index files (read-only, never modified) and
produces separate label datasets carrying the SAME rows/columns plus three
new binary label columns (one per appliance). Continuous targets and their
validity flags are copied through unchanged -- never overwritten.

Approved thresholds (pooled 1D 2-means, fit on valid training-window
endpoint targets across all five homes -- see prior diagnosis):
    air1:          1.06026
    dishwasher1:   0.52378
    refrigerator1: 0.06482
Decision rule: label = 1 if valid non-null target >= threshold, else 0.
Invalid or null targets receive a null label (pandas nullable Int64 <NA>),
never an OFF (0) label.

Units: the physical unit of the source readings (kW vs W) could not be
verified from any available project documentation. Thresholds are applied
in the dataset's original, unconverted units -- no unit assumption or
conversion is made anywhere in this script.

No thresholds are fit here -- they are fixed, approved constants applied
identically to train, validation, and test.
"""
import os
import time
import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IN_PATHS = {
    "train": PROJECT_ROOT / "data" / "processed" / "window_index_train.parquet",
    "val": PROJECT_ROOT / "data" / "processed" / "window_index_val.parquet",
    "test": PROJECT_ROOT / "data" / "processed" / "window_index_test.parquet",
}
OUT_PATHS = {
    "train": PROJECT_ROOT / "data" / "processed" / "window_labels_train.parquet",
    "val": PROJECT_ROOT / "data" / "processed" / "window_labels_val.parquet",
    "test": PROJECT_ROOT / "data" / "processed" / "window_labels_test.parquet",
}
THRESHOLDS_PATH = PROJECT_ROOT / "data" / "meta" / "stage6_thresholds.json"
REPORT_PATH = PROJECT_ROOT / "data" / "meta" / "stage6_label_generation_report.json"

HOMES = ["7901", "7951", "8386", "8565", "3538"]
APPLIANCES = ["air1", "dishwasher1", "refrigerator1"]
THRESHOLDS = {"air1": 1.06026, "dishwasher1": 0.52378, "refrigerator1": 0.06482}

for p in list(OUT_PATHS.values()) + [THRESHOLDS_PATH, REPORT_PATH]:
    if os.path.exists(p):
        raise RuntimeError(f"STOP: output already exists, refusing to overwrite: {p}")
for p in IN_PATHS.values():
    if not os.path.exists(p):
        raise RuntimeError(f"STOP: required input missing: {p}")

print(f"PID={os.getpid()}", flush=True)
t0 = time.time()

thresholds_doc = {
    "method": "Pooled 1D 2-means (Lloyd's algorithm), fit on valid training-window endpoint targets pooled across all five homes",
    "decision_rule": "label = 1 (ON) if valid non-null target >= threshold; label = 0 (OFF) if valid non-null target < threshold; label = null if target is invalid or null",
    "thresholds": THRESHOLDS,
    "units_limitation": (
        "The physical unit of the source appliance readings (kW vs W) could not be verified "
        "from any documentation available in this project (archive contains only a directory "
        "entry, the main CSV, and metadata.csv; no unit string was found in any of them). "
        "Thresholds are expressed and applied in the dataset's original, unconverted measurement "
        "units. No unit conversion or unit assumption (kW or W) is made anywhere in this script."
    ),
    "source_files_for_threshold_derivation": ["data/processed/window_index_train.parquet"],
    "fit_on_validation_or_test": False,
    "approved_by": "explicit user approval, Approach B: shared pooled 2-means thresholds",
}
with open(THRESHOLDS_PATH, "w") as fh:
    json.dump(thresholds_doc, fh, indent=2)

report = {"splits": {}}

for split in ["train", "val", "test"]:
    df = pd.read_parquet(IN_PATHS[split], engine="pyarrow")
    n_rows_in = len(df)
    cols_in = list(df.columns)

    for a in APPLIANCES:
        thr = THRESHOLDS[a]
        valid_mask = df[f"{a}_valid"] & df[a].notna()
        label = pd.array([pd.NA] * len(df), dtype="Int64")
        label_values = (df.loc[valid_mask, a].values >= thr).astype(int)
        label[valid_mask.values] = label_values
        df[f"{a}_label"] = label

    df.to_parquet(OUT_PATHS[split], engine="pyarrow", index=False)

    split_report = {"n_rows": n_rows_in, "columns_in": cols_in, "per_appliance": {}, "per_home": {}}
    for a in APPLIANCES:
        on = int((df[f"{a}_label"] == 1).sum())
        off = int((df[f"{a}_label"] == 0).sum())
        invalid = int(df[f"{a}_label"].isna().sum())
        split_report["per_appliance"][a] = {"ON": on, "OFF": off, "invalid": invalid, "total": on + off + invalid}
        per_home = {}
        for home in HOMES:
            sub = df[df["dataid"] == home]
            h_on = int((sub[f"{a}_label"] == 1).sum())
            h_off = int((sub[f"{a}_label"] == 0).sum())
            h_inv = int(sub[f"{a}_label"].isna().sum())
            per_home[home] = {"ON": h_on, "OFF": h_off, "invalid": h_inv, "total": h_on + h_off + h_inv}
        split_report["per_home"][a] = per_home

    report["splits"][split] = split_report
    print(f"{split}: rows={n_rows_in:,} written to {OUT_PATHS[split]}", flush=True)

report["thresholds"] = THRESHOLDS
report["elapsed_seconds"] = time.time() - t0

with open(REPORT_PATH, "w") as fh:
    json.dump(report, fh, indent=2)

print(f"DONE elapsed={report['elapsed_seconds']:.1f}s", flush=True)
