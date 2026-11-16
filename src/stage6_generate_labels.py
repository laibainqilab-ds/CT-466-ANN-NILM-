"""
Stage 6: appliance ON/OFF label generation (CT-466 Phase 2).

Reads the three existing window-index files (read-only, never modified) and
produces separate label datasets carrying the SAME rows/columns plus three
new binary label columns (one per appliance). Continuous targets and their
validity flags are copied through unchanged -- never overwritten.

Thresholds (pooled 1D 2-means, CONSTRAINED exact optimal partition -- see
fit_1d_2means_threshold) are fit HERE, at runtime, on ALL valid one-minute
TRAIN-split appliance readings from data/processed/five_homes_modeling.parquet
-- every valid minute, never the stride-5 window endpoints in
window_index_train.parquet -- pooled across the four development households
(7901, 7951, 8565, 3538) only. fit_pooled_thresholds() enforces this with an
explicit WHITELIST: after filtering to the train split, it raises if ANY
household outside the four approved development homes is observed --
including, but not limited to, the held-out household 8386 -- rather than
relying solely on Stage 4 already excluding it or on a blacklist of 8386
specifically.

The 2-means split is constrained so that neither resulting cluster may
contain fewer than MIN_CLUSTER_FRACTION (0.1%) of the valid readings for
that appliance -- see fit_1d_2means_threshold's docstring for why this is
necessary (an unconstrained exact solver can isolate a single extreme
outlier reading as its own degenerate "cluster") and for the documented
rationale/limitation of the 0.1% default.
Decision rule: label = 1 if valid non-null target >= threshold, else 0.
Invalid or null targets receive a null label (pandas nullable Int64 <NA>),
never an OFF (0) label.

Units: the physical unit of the source readings (kW vs W) could not be
verified from any available project documentation. Thresholds are applied
in the dataset's original, unconverted units -- no unit assumption or
conversion is made anywhere in this script.
"""
import os
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELING_PATH = PROJECT_ROOT / "data" / "processed" / "five_homes_modeling.parquet"
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
HELD_OUT_HOME = "8386"
DEV_HOMES = [h for h in HOMES if h != HELD_OUT_HOME]

SUPERSEDED_THRESHOLDS_ORIGINAL_5HOME_FIT = {
    "air1": 1.06026, "dishwasher1": 0.52378, "refrigerator1": 0.06482,
    "note": (
        "Original pooled 1D 2-means fit across all five homes (including the now-held-out "
        "household 8386) on window_index_train.parquet stride-5 window endpoints. Superseded "
        "because (a) it let 8386's training-period data influence a threshold now applied "
        "against 8386 as the unseen test household, and (b) it fit on subsampled window "
        "endpoints rather than the full per-minute training signal."
    ),
}


MIN_CLUSTER_FRACTION = 0.001  # 0.1% -- see fit_1d_2means_threshold docstring for rationale/limitation


def fit_1d_2means_threshold(values: np.ndarray, min_cluster_fraction: float = MIN_CLUSTER_FRACTION) -> float:
    """Exact (not heuristic) 1D 2-means, CONSTRAINED to a minimum cluster
    size on both sides.

    For one-dimensional data, the globally optimal 2-cluster partition
    minimizing total within-cluster sum of squares is always a contiguous
    split of the SORTED values (a standard property of 1D k-means / Jenks
    natural breaks). An UNCONSTRAINED exact search is not robust to
    outliers: a single extreme reading (or a small handful of them) can
    minimize total WCSS by being isolated into its own near-zero-variance
    cluster, producing a threshold that does not correspond to any
    meaningful ON/OFF boundary (observed on real appliance data: a lone
    sensor-glitch reading was isolated as a one-point "ON" cluster).

    To prevent this, both candidate cluster sizes must be at least
    `ceil(min_cluster_fraction * n)` (minimum 1), applied SYMMETRICALLY to
    both the low and high side of every candidate split -- this makes a
    single outlier, or a small group of them smaller than the floor,
    structurally impossible to isolate as its own cluster, without
    removing, clipping, or altering any reading. Subject to that
    constraint, the search is still exact and deterministic: every
    remaining valid split point is evaluated via vectorized prefix sums,
    and the one minimizing total WCSS is chosen.

    Rationale/limitation of the MIN_CLUSTER_FRACTION=0.001 (0.1%) default:
    empirically, across three real appliances with true minority ("ON")
    fractions ranging from 0.64% to 41.6% of valid training minutes, the
    resulting split was identical and stable across floors from 0.01% to
    0.5% (a 50x range), and only degraded once the floor (1.0%) exceeded
    the smallest true minority fraction observed. 0.1% was chosen as a
    value well inside that stable range: ~6x below the smallest observed
    true minority fraction, and ~10x above the smallest floor that already
    avoided the outlier-isolation failure. This is a documented ASSUMPTION
    based on the appliances examined, not a provably universal constant --
    a future appliance with a true ON fraction rarer than ~0.1% of valid
    training minutes would need this floor revisited.

    Deterministic: no randomness; ties in total WCSS are broken by the
    smallest valid split index (smallest resulting threshold). Raises
    ValueError if fewer than 2 distinct values are present, if the
    minimum cluster size leaves no valid split point for this n, and
    RuntimeError if the optimal constrained partition still produces two
    identical cluster means -- no case silently returns a degenerate
    threshold.
    """
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    if n < 2:
        raise ValueError(f"fit_1d_2means_threshold requires at least 2 values, got {n}.")
    if values[0] == values[-1]:
        raise ValueError(
            "fit_1d_2means_threshold received a degenerate input (all values identical) -- "
            "no threshold can separate two clusters from a single value."
        )

    min_size = max(1, int(np.ceil(min_cluster_fraction * n)))
    if 2 * min_size > n:
        raise ValueError(
            f"fit_1d_2means_threshold: minimum cluster size {min_size} (ceil({min_cluster_fraction} * "
            f"{n})) leaves no split point where both clusters can simultaneously satisfy the minimum."
        )

    cumsum = np.cumsum(values)
    cumsum_sq = np.cumsum(values ** 2)
    total_sum = cumsum[-1]
    total_sum_sq = cumsum_sq[-1]

    # Every split point where BOTH resulting clusters are at least
    # min_size -- the symmetric floor applied to both sides.
    ks = np.arange(min_size, n - min_size + 1)
    left_sum = cumsum[ks - 1]
    left_sum_sq = cumsum_sq[ks - 1]
    right_sum = total_sum - left_sum
    right_sum_sq = total_sum_sq - left_sum_sq
    right_sizes = n - ks

    left_wcss = left_sum_sq - (left_sum ** 2) / ks
    right_wcss = right_sum_sq - (right_sum ** 2) / right_sizes
    total_wcss = left_wcss + right_wcss

    best_pos = int(np.argmin(total_wcss))  # argmin: smallest k wins ties -> deterministic
    best_k = ks[best_pos]
    c0 = float(left_sum[best_pos] / best_k)
    c1 = float(right_sum[best_pos] / right_sizes[best_pos])

    if c0 == c1:
        raise RuntimeError(
            "fit_1d_2means_threshold: optimal constrained partition produced identical cluster "
            f"means ({c0}) -- refusing to return a degenerate (non-discriminative) threshold."
        )

    lo, hi = sorted((c0, c1))
    return float((lo + hi) / 2.0)


def fit_pooled_thresholds(modeling_df: pd.DataFrame, appliances=APPLIANCES,
                           split: str = "train", allowed_homes=DEV_HOMES) -> dict:
    """Pooled, CONSTRAINED exact 1D 2-means (see fit_1d_2means_threshold)
    over ALL valid, non-null, one-minute TRAIN-split appliance readings in
    `modeling_df` (five_homes_modeling.parquet) -- every valid minute,
    never the stride-5 window endpoints in window_index_train.parquet.

    Enforces an explicit WHITELIST, not a blacklist: after filtering to
    `split`, raises RuntimeError if ANY household outside `allowed_homes`
    is observed -- this rejects the held-out household 8386 (not a member
    of DEV_HOMES) and any other unexpected household ID, never silently
    including a household outside the approved development set. Only
    `split`-matching rows are ever read, so validation/test readings are
    never seen by this function regardless of `allowed_homes`.

    Also enforces EXACT-SET membership, not just a one-sided whitelist:
    raises RuntimeError if any approved development home is MISSING from
    the train-split rows, so a partial/incomplete pool (e.g. only 2 of 4
    dev homes present due to an upstream data problem) is never silently
    fit on fewer households than intended."""
    train_rows = modeling_df[modeling_df["split"] == split]
    observed_homes = set(train_rows["dataid"].unique())
    allowed_set = set(allowed_homes)
    unexpected_homes = observed_homes - allowed_set
    if unexpected_homes:
        raise RuntimeError(
            f"STOP: unexpected household(s) {sorted(unexpected_homes)} present in {split!r}-split rows "
            f"of five_homes_modeling.parquet -- only the approved development households "
            f"{sorted(allowed_set)} may contribute to threshold fitting."
        )
    missing_homes = allowed_set - observed_homes
    if missing_homes:
        raise RuntimeError(
            f"STOP: approved development household(s) {sorted(missing_homes)} are MISSING from "
            f"{split!r}-split rows of five_homes_modeling.parquet -- refusing to fit thresholds on "
            f"an incomplete pool of development households. Expected exactly {sorted(allowed_set)}, "
            f"observed {sorted(observed_homes)}."
        )
    thresholds = {}
    for a in appliances:
        valid_mask = train_rows[f"{a}_valid"] & train_rows[a].notna()
        values = train_rows.loc[valid_mask, a].values
        thresholds[a] = fit_1d_2means_threshold(values)
    return thresholds


if __name__ == "__main__":
    # Executable pipeline body, gated behind __main__ (matching
    # stage7_preprocess.py's convention) so this module can be imported
    # for unit testing fit_1d_2means_threshold/fit_pooled_thresholds
    # without triggering file-existence guards or any file I/O.
    for p in list(OUT_PATHS.values()) + [THRESHOLDS_PATH, REPORT_PATH]:
        if os.path.exists(p):
            raise RuntimeError(f"STOP: output already exists, refusing to overwrite: {p}")
    for p in list(IN_PATHS.values()) + [MODELING_PATH]:
        if not os.path.exists(p):
            raise RuntimeError(f"STOP: required input missing: {p}")

    print(f"PID={os.getpid()}", flush=True)
    t0 = time.time()

    modeling_df = pd.read_parquet(MODELING_PATH, engine="pyarrow")
    THRESHOLDS = fit_pooled_thresholds(modeling_df)

    thresholds_doc = {
        "method": (
            "Pooled 1D 2-means, CONSTRAINED exact optimal contiguous-partition solution "
            "(not Lloyd's algorithm / not percentile-initialized): for one-dimensional data "
            "the optimal 2-cluster WCSS-minimizing partition is always a contiguous split of "
            "sorted values; found here via exhaustive vectorized search over every split point "
            "satisfying a minimum-cluster-size constraint on BOTH sides. Fit on ALL valid, "
            "non-null, one-minute TRAIN-split appliance readings (not window-subsampled) from "
            "the four development households (7901, 7951, 8565, 3538), enforced by an explicit "
            "whitelist. Household 8386 (held-out unseen test household) excluded."
        ),
        "minimum_cluster_fraction": MIN_CLUSTER_FRACTION,
        "minimum_cluster_size_calculation": (
            "min_size = max(1, ceil(minimum_cluster_fraction * n_valid)), computed "
            "independently per appliance from that appliance's own valid-reading count "
            "(n_valid); applied SYMMETRICALLY -- neither the low nor the high cluster may "
            "contain fewer than min_size readings. Rounds UP (ceiling), never down, so the "
            "constraint is never accidentally weaker than requested."
        ),
        "tie_breaking": (
            "Deterministic: among all split points tied for minimum total within-cluster sum "
            "of squares, the smallest split index (ascending order) is selected -- i.e. the "
            "smallest resulting threshold on ties. No randomness anywhere in the computation."
        ),
        "minimum_cluster_fraction_rationale_and_limitation": (
            "0.1% was chosen after a read-only sensitivity sweep (0.01%-1.0%) on the real "
            "four-development-household training data: results were identical and stable "
            "across 0.01%-0.5% (a 50x range) for all three appliances (true minority ON "
            "fractions observed: 0.64%-41.6%), and only degraded at 1.0% once it exceeded the "
            "smallest observed minority fraction (dishwasher1, 0.64%). 0.1% sits inside that "
            "stable range with margin on both sides. This is a documented ASSUMPTION based on "
            "the appliances examined here, not a provably universal constant -- a future "
            "appliance with a true ON fraction rarer than ~0.1% of valid training minutes "
            "would require this floor to be revisited."
        ),
        "decision_rule": "label = 1 (ON) if valid non-null target >= threshold; label = 0 (OFF) if valid non-null target < threshold; label = null if target is invalid or null",
        "thresholds": THRESHOLDS,
        "held_out_home": HELD_OUT_HOME,
        "development_homes": DEV_HOMES,
        "units_limitation": (
            "The physical unit of the source appliance readings (kW vs W) could not be verified "
            "from any documentation available in this project (archive contains only a directory "
            "entry, the main CSV, and metadata.csv; no unit string was found in any of them). "
            "Thresholds are expressed and applied in the dataset's original, unconverted measurement "
            "units. No unit conversion or unit assumption (kW or W) is made anywhere in this script."
        ),
        "source_files_for_threshold_derivation": ["data/processed/five_homes_modeling.parquet"],
        "fit_on_validation_or_test": False,
        "approved_by": "explicit user approval, revised split: dev-homes-only pooled 2-means thresholds",
        "superseded_thresholds_original_5home_fit": SUPERSEDED_THRESHOLDS_ORIGINAL_5HOME_FIT,
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
