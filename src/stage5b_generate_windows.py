"""
Stage 5B: 60-minute causal window generation (CT-466 Phase 2).

Reads data/processed/five_homes_modeling.parquet (read-only, never modified).
For each of the five homes, independently, per split (train/val/test):
  - verifies chronological integrity (strictly increasing utc_dt, exact
    1-minute spacing) as a build-time check, not an assumption
  - builds a FIXED absolute-position candidate grid from stride + segment
    length only (never shifted by missingness)
  - accepts a candidate only if all 60 aggregate samples ending there are
    non-null
  - records, per accepted window, the appliance target values and their
    independent per-appliance validity flags at the single prediction
    timestamp (never across the other 59 context samples)
No scaling, thresholding, or model training occurs here. Model input
(aggregate only) is never stored in the index -- only a reference
(endpoint position) sufficient to retrieve it later from the base file.
"""
import os
import time
import json
import numpy as np
import pandas as pd

MODELING_PATH = r"E:\CT-466\ANN_NILM\data\processed\five_homes_modeling.parquet"
OUT_PATHS = {
    "train": r"E:\CT-466\ANN_NILM\data\processed\window_index_train.parquet",
    "val": r"E:\CT-466\ANN_NILM\data\processed\window_index_val.parquet",
    "test": r"E:\CT-466\ANN_NILM\data\processed\window_index_test.parquet",
}
REPORT_PATH = r"E:\CT-466\ANN_NILM\data\meta\stage5b_window_generation_report.json"

HOMES = ["7901", "7951", "8386", "8565", "3538"]
APPLIANCES = ["air1", "dishwasher1", "refrigerator1"]
STRIDE = {"train": 5, "val": 1, "test": 1}
WINLEN = 60
EXPECTED_TOTALS = {"train": 392516, "val": 222905, "test": 439205, "grand_total": 1054626}

for p in list(OUT_PATHS.values()) + [REPORT_PATH]:
    if os.path.exists(p):
        raise RuntimeError(f"STOP: output already exists, refusing to overwrite: {p}")

print(f"PID={os.getpid()}", flush=True)
t0 = time.time()

df = pd.read_parquet(MODELING_PATH, engine="pyarrow")

report = {"homes": {}, "splits": {s: {"rows": [], "accepted_total": 0} for s in ["train", "val", "test"]}}
window_rows = {"train": [], "val": [], "test": []}

for home in HOMES:
    report["homes"][home] = {}
    for split in ["train", "val", "test"]:
        g = df[(df["dataid"] == home) & (df["split"] == split)].sort_values("utc_dt").reset_index(drop=True)
        n = len(g)

        diffs = g["utc_dt"].diff().dropna()
        chrono_ok = bool((diffs == pd.Timedelta(minutes=1)).all())
        monotonic_unique = bool(g["utc_dt"].is_monotonic_increasing and g["utc_dt"].is_unique)

        valid_agg = g["aggregate"].notna().values
        roll = pd.Series(valid_agg.astype(int)).rolling(WINLEN).sum()
        window_valid = (roll == WINLEN).values

        stride = STRIDE[split]
        candidates = np.arange(59, n, stride)
        valid_at_candidates = window_valid[candidates]
        n_rejected = int((~valid_at_candidates).sum())
        kept = candidates[valid_at_candidates]
        n_accepted = int(len(kept))

        home_split_report = {
            "n_rows": n,
            "chrono_strictly_1min_spacing": chrono_ok,
            "chrono_monotonic_unique": monotonic_unique,
            "n_candidates": int(len(candidates)),
            "n_rejected": n_rejected,
            "n_accepted": n_accepted,
        }
        if not (chrono_ok and monotonic_unique):
            raise RuntimeError(f"STOP: chronological integrity check failed for {home}/{split}")
        if n_accepted != home_split_report["n_candidates"] - n_rejected:
            raise RuntimeError(f"STOP: arithmetic mismatch for {home}/{split}")

        utc_dt_vals = g["utc_dt"].values[kept]
        local_dt_vals = g["local_dt"].values[kept]

        rec = {
            "dataid": home,
            "split": split,
            "utc_dt": utc_dt_vals,
            "local_dt": local_dt_vals,
            "endpoint_index": kept,
        }
        for a in APPLIANCES:
            rec[a] = g[a].values[kept]
            rec[f"{a}_valid"] = g[f"{a}_valid"].values[kept]

        n_appliance_invalid = {}
        for a in APPLIANCES:
            n_appliance_invalid[a] = int((~rec[f"{a}_valid"]).sum())
        home_split_report["n_appliance_invalid_at_endpoint"] = n_appliance_invalid

        report["homes"][home][split] = home_split_report
        report["splits"][split]["rows"].append({"dataid": home, **home_split_report})
        report["splits"][split]["accepted_total"] += n_accepted

        window_df = pd.DataFrame(rec)
        window_rows[split].append(window_df)

        print(f"{home}/{split}: rows={n:,} candidates={len(candidates):,} rejected={n_rejected:,} accepted={n_accepted:,}", flush=True)

final_totals = {}
for split in ["train", "val", "test"]:
    combined = pd.concat(window_rows[split], ignore_index=True)
    combined.to_parquet(OUT_PATHS[split], engine="pyarrow", index=False)
    final_totals[split] = int(len(combined))

final_totals["grand_total"] = sum(final_totals[s] for s in ["train", "val", "test"])

discrepancies = {k: {"expected": EXPECTED_TOTALS[k], "actual": final_totals[k]}
                  for k in EXPECTED_TOTALS if EXPECTED_TOTALS[k] != final_totals[k]}

report["final_totals"] = final_totals
report["expected_totals"] = EXPECTED_TOTALS
report["discrepancies"] = discrepancies
report["elapsed_seconds"] = time.time() - t0

with open(REPORT_PATH, "w") as fh:
    json.dump(report, fh, indent=2, default=str)

print(f"DONE elapsed={report['elapsed_seconds']:.1f}s final_totals={final_totals}", flush=True)
if discrepancies:
    print(f"DISCREPANCY_DETECTED: {json.dumps(discrepancies)}", flush=True)
else:
    print("ALL_TOTALS_MATCH_EXPECTED=True", flush=True)
