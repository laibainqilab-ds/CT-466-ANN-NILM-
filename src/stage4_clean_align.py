"""
Stage 4: cleaning and timestamp alignment (CT-466 Phase 2).

Reads data/interim/five_homes.parquet (left unmodified). For each of the
five approved homes, independently:
  - parses localminute into a local naive datetime and a UTC-normalized
    instant (using the row's own embedded offset) -- chronological order
    is established ONLY from these parsed values, never archive row order
  - constructs the aggregate input (grid, or grid+solar for 3538,
    null-propagating)
  - classifies rows into train/val/test by LOCAL calendar date
  - reindexes each home x split segment onto a complete 1-minute grid
    (fixed calendar boundaries converted to UTC via zoneinfo, DST-aware)
  - interpolates the aggregate column only, only for gaps <=5 minutes with
    valid data on both sides, never across a split boundary
  - marks per-appliance validity (no interpolation of appliance columns)
  - produces negative-reading diagnostics (no clipping)
No thresholding, windowing, scaling, or model training occurs here.
"""
import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

RAW_PATH = r"E:\CT-466\ANN_NILM\data\interim\five_homes.parquet"
OUT_PATH = r"E:\CT-466\ANN_NILM\data\processed\five_homes_aligned.parquet"
REPORT_PATH = r"E:\CT-466\ANN_NILM\data\meta\stage4_cleaning_report.json"

HOMES = ["7901", "7951", "8386", "8565", "3538"]
SOLAR_HOME = "3538"
APPLIANCES = ["air1", "dishwasher1", "refrigerator1"]
TZ = ZoneInfo("America/Chicago")
MAX_INTERP_GAP = 5

SPLIT_BOUNDS_LOCAL = {
    "train": (datetime(2018, 1, 1, 0, 0, 0), datetime(2018, 9, 30, 23, 59, 0)),
    "val":   (datetime(2018, 10, 1, 0, 0, 0), datetime(2018, 10, 31, 23, 59, 0)),
    "test":  (datetime(2018, 11, 1, 0, 0, 0), datetime(2018, 12, 31, 23, 59, 0)),
}

print(f"PID={os.getpid()}", flush=True)
t0 = time.time()

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

report = {"homes": {}}
all_out_frames = []


def classify(local_ts):
    for name, (start, end) in SPLIT_BOUNDS_LOCAL.items():
        if start <= local_ts <= end:
            return name
    return "out_of_range"


for home in HOMES:
    home_t0 = time.time()
    df = pd.read_parquet(RAW_PATH, engine="pyarrow", filters=[("dataid", "==", home)])
    n_raw = len(df)

    local_dt = pd.to_datetime(df["localminute"].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S")
    sign = df["localminute"].str.slice(19, 20)
    off_val = df["localminute"].str.slice(20).astype(int)
    offset_hours = np.where(sign.values == "+", off_val.values, -off_val.values)
    utc_dt = local_dt - pd.to_timedelta(offset_hours, unit="h")

    df = df.assign(local_dt=local_dt, utc_dt=utc_dt, offset_hours=offset_hours)

    # integrity: embedded offset vs. historically correct America/Chicago offset
    # (ambiguous/nonexistent local times from DST transitions are marked NaT,
    # not guessed, so they never create a spurious mismatch count)
    local_idx = pd.DatetimeIndex(df["local_dt"])
    localized = local_idx.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    expected_hours = np.array(
        [t.utcoffset().total_seconds() / 3600 if pd.notna(t) else np.nan for t in localized]
    )
    ambiguous_count = int(pd.isna(localized).sum())
    comparable = ~np.isnan(expected_hours)
    offset_mismatch = int((expected_hours[comparable] != df["offset_hours"].values[comparable]).sum())

    # chronological order established ONLY from parsed utc_dt
    df = df.sort_values("utc_dt").reset_index(drop=True)
    dup_count = int(df["utc_dt"].duplicated().sum())

    # aggregate construction (null-propagating)
    if home == SOLAR_HOME:
        df["aggregate"] = df["grid"] + df["solar"]
    else:
        df["aggregate"] = df["grid"]

    df["split"] = df["local_dt"].apply(classify)
    out_of_range_count = int((df["split"] == "out_of_range").sum())

    home_report = {
        "n_raw_rows": n_raw,
        "ambiguous_local_time_count": ambiguous_count,
        "offset_mismatch_count_excl_ambiguous": offset_mismatch,
        "duplicate_utc_count": dup_count,
        "out_of_range_count": out_of_range_count,
        "splits": {},
    }

    processed_segments = []

    for split_name, (start_local, end_local) in SPLIT_BOUNDS_LOCAL.items():
        seg = df[df["split"] == split_name].copy()

        start_utc = start_local.replace(tzinfo=TZ).astimezone(timezone.utc).replace(tzinfo=None)
        end_utc = end_local.replace(tzinfo=TZ).astimezone(timezone.utc).replace(tzinfo=None)
        full_grid = pd.date_range(start_utc, end_utc, freq="1min")
        span = len(full_grid)

        seg_indexed = seg.set_index("utc_dt")
        seg_indexed = seg_indexed[~seg_indexed.index.duplicated(keep="first")]

        aligned = seg_indexed.reindex(full_grid)
        aligned.index.name = "utc_dt"
        aligned = aligned.reset_index()
        aligned["dataid"] = home
        aligned["split"] = split_name

        agg_missing_before = int(aligned["aggregate"].isna().sum())

        for a in APPLIANCES:
            aligned[f"{a}_valid"] = aligned[a].notna()

        agg = aligned["aggregate"]
        is_na = agg.isna()
        run_id = (is_na != is_na.shift()).cumsum()
        gap_sizes = is_na.groupby(run_id).transform("sum")
        interpolated_mask = is_na & (gap_sizes <= MAX_INTERP_GAP)
        agg_interp = agg.interpolate(method="linear", limit=MAX_INTERP_GAP, limit_area="inside")
        aligned["aggregate_was_interpolated"] = interpolated_mask & agg_interp.notna()
        aligned["aggregate"] = np.where(aligned["aggregate_was_interpolated"], agg_interp, aligned["aggregate"])

        agg_missing_after = int(aligned["aggregate"].isna().sum())
        interpolated_count = int(aligned["aggregate_was_interpolated"].sum())

        neg_diag = {}
        for a in APPLIANCES:
            vals = aligned.loc[aligned[f"{a}_valid"], a]
            n_valid = len(vals)
            neg_vals = vals[vals < 0]
            neg_diag[a] = {
                "n_valid": int(n_valid),
                "n_negative": int(len(neg_vals)),
                "pct_negative": round(float(len(neg_vals) / n_valid * 100), 4) if n_valid else None,
                "neg_min": float(neg_vals.min()) if len(neg_vals) else None,
                "neg_max": float(neg_vals.max()) if len(neg_vals) else None,
                "neg_mean": float(neg_vals.mean()) if len(neg_vals) else None,
            }

        home_report["splits"][split_name] = {
            "local_range": [str(start_local), str(end_local)],
            "span_minutes": span,
            "present_before_interp": span - agg_missing_before,
            "missing_before_interp": agg_missing_before,
            "interpolated_count": interpolated_count,
            "preserved_gap_count_after_interp": agg_missing_after,
            "appliance_valid_counts": {a: int(aligned[f"{a}_valid"].sum()) for a in APPLIANCES},
            "appliance_invalid_counts": {a: int((~aligned[f"{a}_valid"]).sum()) for a in APPLIANCES},
            "negative_reading_diagnostics": neg_diag,
        }

        keep_cols = ["dataid", "split", "utc_dt", "local_dt", "aggregate", "aggregate_was_interpolated"]
        for a in APPLIANCES:
            keep_cols += [a, f"{a}_valid"]
        processed_segments.append(aligned[keep_cols])

    report["homes"][home] = home_report
    home_df = pd.concat(processed_segments, ignore_index=True)
    all_out_frames.append(home_df)

    print(f"HOME {home} done in {time.time()-home_t0:.1f}s raw_rows={n_raw:,}", flush=True)

final_df = pd.concat(all_out_frames, ignore_index=True)
final_df.to_parquet(OUT_PATH, engine="pyarrow", index=False)

with open(REPORT_PATH, "w") as fh:
    json.dump(report, fh, indent=2, default=str)

elapsed = time.time() - t0
print(f"DONE elapsed={elapsed:.1f}s output_rows={len(final_df):,} output_file={OUT_PATH}", flush=True)
