"""
Stage 7: preprocessing statistics (CT-466 Phase 2).

Fits two kinds of scaler, both strictly from TRAINING data only:
  - aggregate input scaler: mean/std over unique valid training-MINUTE
    aggregate readings from five_homes_modeling.parquet (NOT reconstructed
    window contents, which would duplicate most minutes non-uniformly --
    see prior diagnosis).
  - per-appliance regression target scaler: mean/std over valid training
    targets at window endpoints (window_labels_train.parquet), since those
    are exactly the values that will actually be used for the regression
    loss.

Both fit functions internally filter to split == 'train' even though the
caller is expected to already pass train-only data -- a defensive,
auditable guarantee against accidental validation/test leakage, not just
caller discipline.

No scaler here is ever fit on validation or test data.

Under the revised evaluation design, the train split contains only the
four development households (7901, 7951, 8565, 3538); household 8386 is
the held-out, unseen test household and contributes nothing to either
scaler. save_preprocessing_params() persists the SAME scaler dict objects
already fit above (never independently refit) to
data/meta/stage7_preprocessing_params.json, together with explicit
source_split/source_homes provenance, so the on-disk metadata accurately
documents what was actually fit.
"""
import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELING_PATH = PROJECT_ROOT / "data" / "processed" / "five_homes_modeling.parquet"
LABELS_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "window_labels_train.parquet"
PARAMS_PATH = PROJECT_ROOT / "data" / "meta" / "stage7_preprocessing_params.json"

APPLIANCES = ["air1", "dishwasher1", "refrigerator1"]
DEV_HOMES = ["7901", "7951", "8565", "3538"]


def fit_aggregate_scaler(modeling_df: pd.DataFrame) -> dict:
    """Mean/std over unique valid TRAIN-split per-minute aggregate readings.

    modeling_df must be the full five_homes_modeling.parquet frame (all
    splits) -- this function does the train-only filtering itself.
    """
    train_only = modeling_df[modeling_df["split"] == "train"]
    vals = train_only.loc[train_only["aggregate"].notna(), "aggregate"]
    if len(vals) == 0:
        raise RuntimeError("No valid training aggregate readings found -- refusing to fit a degenerate scaler.")
    mean = float(vals.mean())
    std = float(vals.std())
    if not (std > 0):
        raise RuntimeError(f"Aggregate training std is not positive ({std}) -- refusing to fit a degenerate scaler.")
    return {"mean": mean, "std": std, "n": int(len(vals))}


def fit_target_scalers(labels_train_df: pd.DataFrame, appliances=APPLIANCES) -> dict:
    """Per-appliance mean/std over valid TRAIN-split window-endpoint targets.

    labels_train_df must already be the train-split label file; this
    function additionally asserts split == 'train' for every row as a
    defensive check (it does not silently filter -- a labels_train_df
    containing non-train rows indicates a caller error and should fail
    loudly, not be masked).
    """
    if not (labels_train_df["split"] == "train").all():
        raise RuntimeError("fit_target_scalers received rows outside the train split -- refusing to fit.")
    scalers = {}
    for a in appliances:
        vals = labels_train_df.loc[labels_train_df[f"{a}_valid"], a]
        if len(vals) == 0:
            raise RuntimeError(f"No valid training targets found for {a} -- refusing to fit a degenerate scaler.")
        mean = float(vals.mean())
        std = float(vals.std())
        if not (std > 0):
            raise RuntimeError(f"Training target std for {a} is not positive ({std}) -- refusing to fit.")
        scalers[a] = {"mean": mean, "std": std, "n": int(len(vals))}
    return scalers


def standardize(x, mean: float, std: float):
    """(x - mean) / std. NaN propagates through untouched -- no special-casing needed."""
    return (x - mean) / std


def inverse_standardize(x, mean: float, std: float):
    """Inverse of standardize(): x * std + mean."""
    return x * std + mean


def build_preprocessing_params_doc(agg_scaler: dict, target_scalers: dict,
                                    source_homes=DEV_HOMES, source_split: str = "train") -> dict:
    """Assembles the on-disk metadata document from ALREADY-FIT scaler
    dicts -- `agg_scaler`/`target_scalers` are used exactly as given, never
    refit or recomputed here. Every value is a plain Python float/int/str/
    list/bool (the scalers' own fit functions already cast via float()/
    int()), so the result is directly json.dump-able with no custom
    encoder needed."""
    return {
        "aggregate_scaler": {
            "description": (
                "Mean/std over unique valid TRAIN-split per-minute aggregate readings from "
                "five_homes_modeling.parquet (NOT reconstructed window contents -- avoids "
                "non-uniform duplication bias near segment edges/gaps). Pooled across the "
                "four development households only -- household 8386 (held-out unseen test "
                "household) is absent from the train split by construction and contributes "
                "nothing to this fit."
            ),
            "mean": agg_scaler["mean"], "std": agg_scaler["std"], "n": agg_scaler["n"],
            "units_note": "Original, unconverted dataset measurement units. Physical unit (kW vs W) not independently verified.",
        },
        "target_scalers": target_scalers,
        "target_scaler_description": (
            "Per-appliance mean/std over valid TRAIN-split window-endpoint targets "
            "(window_labels_train.parquet) -- matches the population actually used by the "
            "regression loss. Same four-development-household, train-only population as the "
            "aggregate scaler above."
        ),
        "fit_on_validation_or_test": False,
        "source_split": source_split,
        "source_homes": list(source_homes),
        "held_out_home": "8386",
        "source_files": [
            "data/processed/five_homes_modeling.parquet",
            "data/processed/window_labels_train.parquet",
        ],
        "generated_by": "src/stage7_preprocess.py",
    }


def save_preprocessing_params(agg_scaler: dict, target_scalers: dict, output_path=PARAMS_PATH,
                               source_homes=DEV_HOMES, source_split: str = "train") -> dict:
    """Writes build_preprocessing_params_doc(...)'s result to `output_path`
    as JSON and returns the document that was written (so a caller can
    verify the exact content without re-reading the file). Uses the given
    `agg_scaler`/`target_scalers` as-is -- does not fit or refit anything."""
    doc = build_preprocessing_params_doc(agg_scaler, target_scalers, source_homes=source_homes,
                                          source_split=source_split)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(doc, fh, indent=2)
    return doc


if __name__ == "__main__":
    # Manual run (not part of automated tests): fit the real statistics
    # from the current (four-development-household) train split and
    # persist them to data/meta/stage7_preprocessing_params.json.
    modeling_df = pd.read_parquet(MODELING_PATH, engine="pyarrow")
    labels_train_df = pd.read_parquet(LABELS_TRAIN_PATH, engine="pyarrow")

    agg_scaler = fit_aggregate_scaler(modeling_df)
    target_scalers = fit_target_scalers(labels_train_df)

    print("aggregate_scaler:", agg_scaler)
    print("target_scalers:", target_scalers)

    written_doc = save_preprocessing_params(agg_scaler, target_scalers)
    print(f"Preprocessing parameters written to {PARAMS_PATH}")
    print("source_split:", written_doc["source_split"], " source_homes:", written_doc["source_homes"])
