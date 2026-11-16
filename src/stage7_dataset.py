"""
Stage 7: memory-efficient PyTorch Dataset for 60-minute causal windows
(CT-466 Phase 2).

Design (matches the independently-verified Stage 5B/6 reconstruction
method -- see prior diagnosis, 45/45 sampled windows passed):
  - A single (dataid, split) -> sorted float32 aggregate array lookup is
    built ONCE from five_homes_modeling.parquet and shared across all
    three split Datasets. No overlapping window is ever materialized in
    bulk; each __getitem__ call slices exactly one 60-sample window on
    demand.
  - Model input (X) is built from the `aggregate` column ONLY. Targets,
    labels, validity flags, timestamps, endpoint indices, and home IDs
    are never placed in X -- they are returned as separate tensors/metadata
    for loss computation and evaluation, not as model input.
  - Both the STANDARDIZED regression target (for training) and the
    ORIGINAL, unstandardized regression target (for evaluation in
    original units) are returned directly -- the original value is never
    recomputed by inverse-transforming the standardized one; it is read
    straight from the source column, since it is already available.
  - Classification and regression validity masks are kept as separate
    tensors per appliance, even though (by the current Stage 6 design)
    they are numerically identical -- this is deliberate, not an
    oversight, so the two tasks can diverge independently in the future
    without a data-model change.
  - Nullable pandas Int64 labels are converted via
    `to_numpy(dtype="float32", na_value=np.nan)` so a missing label
    becomes NaN, never 0 (OFF).
  - Every `__getitem__` call validates `endpoint_index` before slicing
    (must be >= 59 and within the home/split array's bounds) and
    explicitly verifies the reconstructed window is exactly 60 samples
    long, raising a descriptive error otherwise.
  - `__init__` validates that every row of the loaded label file actually
    belongs to the requested split, raising a clear error otherwise.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELING_PATH = PROJECT_ROOT / "data" / "processed" / "five_homes_modeling.parquet"
LABELS_PATHS = {
    "train": PROJECT_ROOT / "data" / "processed" / "window_labels_train.parquet",
    "val": PROJECT_ROOT / "data" / "processed" / "window_labels_val.parquet",
    "test": PROJECT_ROOT / "data" / "processed" / "window_labels_test.parquet",
}

HOMES = ["7901", "7951", "8386", "8565", "3538"]
APPLIANCES = ["air1", "dishwasher1", "refrigerator1"]
WINLEN = 60


def build_aggregate_lookup(modeling_df: pd.DataFrame) -> dict:
    """(dataid, split) -> sorted float32 aggregate array.

    Reproduces EXACTLY the sort used by Stage 5B to define endpoint_index:
    filter by dataid+split, then sort_values('utc_dt').reset_index(drop=True).
    """
    lookup = {}
    for home in HOMES:
        for split in ["train", "val", "test"]:
            g = (
                modeling_df[(modeling_df["dataid"] == home) & (modeling_df["split"] == split)]
                .sort_values("utc_dt")
                .reset_index(drop=True)
            )
            lookup[(home, split)] = g["aggregate"].to_numpy(dtype=np.float32)
    return lookup


def safe_nullable_to_float32(series: pd.Series) -> np.ndarray:
    """Convert a pandas nullable Int64 (or similar) column to float32,
    mapping <NA> to NaN -- never to 0."""
    return series.to_numpy(dtype="float32", na_value=np.nan)


class NILMWindowDataset(Dataset):
    """One sample = one accepted 60-minute causal window.

    __getitem__ returns:
        x            : FloatTensor, shape (60,) -- standardized aggregate input ONLY
        y_reg        : FloatTensor, shape (3,)  -- STANDARDIZED regression targets (NaN where invalid)
        y_reg_raw    : FloatTensor, shape (3,)  -- ORIGINAL, unstandardized regression targets
                       (NaN where invalid) -- read directly from the source column, not
                       derived by inverse-transforming y_reg
        y_cls        : FloatTensor, shape (3,)  -- binary labels (NaN where invalid, never 0)
        mask_reg     : BoolTensor,  shape (3,)  -- per-appliance regression validity
        mask_cls     : BoolTensor,  shape (3,)  -- per-appliance classification validity
        meta         : dict with 'dataid' (str), 'split' (str), 'utc_dt' (pd.Timestamp),
                       'endpoint_index' (int) -- for evaluation/tracing ONLY, never model input
    """

    def __init__(self, split: str, agg_lookup: dict, agg_mean: float, agg_std: float, target_scalers: dict):
        assert split in ("train", "val", "test")
        self.split = split
        self.agg_lookup = agg_lookup
        self.agg_mean = agg_mean
        self.agg_std = agg_std
        self.target_scalers = target_scalers

        df = pd.read_parquet(LABELS_PATHS[split], engine="pyarrow")
        if not (df["split"] == split).all():
            bad_values = sorted(df.loc[df["split"] != split, "split"].unique().tolist())
            raise ValueError(
                f"Label file for split='{split}' contains row(s) with mismatched split "
                f"value(s): {bad_values}. Refusing to construct a Dataset over mismatched data."
            )
        self.dataid = df["dataid"].to_numpy()
        self.endpoint_index = df["endpoint_index"].to_numpy()
        self.utc_dt = df["utc_dt"].to_numpy()

        self.target_raw = {a: df[a].to_numpy(dtype="float32") for a in APPLIANCES}
        self.valid = {a: df[f"{a}_valid"].to_numpy(dtype=bool) for a in APPLIANCES}
        self.label = {a: safe_nullable_to_float32(df[f"{a}_label"]) for a in APPLIANCES}

        self.n = len(df)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        home = self.dataid[idx]
        ep = int(self.endpoint_index[idx])

        if ep < WINLEN - 1:
            raise ValueError(
                f"Invalid endpoint_index={ep} for dataid={home} split={self.split} (row {idx}): "
                f"must be >= {WINLEN - 1} to have {WINLEN} preceding samples available."
            )
        key = (home, self.split)
        if key not in self.agg_lookup:
            raise KeyError(f"No aggregate array found for (dataid={home}, split={self.split}).")
        agg_arr = self.agg_lookup[key]
        if ep >= len(agg_arr):
            raise ValueError(
                f"endpoint_index={ep} is out of bounds for dataid={home} split={self.split} "
                f"(aggregate array length is {len(agg_arr)})."
            )

        window = agg_arr[ep - (WINLEN - 1): ep + 1]
        if len(window) != WINLEN:
            raise RuntimeError(
                f"Reconstructed window for dataid={home} split={self.split} row={idx} ep={ep} "
                f"has length {len(window)}, expected exactly {WINLEN}."
            )
        x = torch.from_numpy(((window - self.agg_mean) / self.agg_std).astype(np.float32))

        y_reg = np.empty(len(APPLIANCES), dtype=np.float32)
        y_reg_raw = np.empty(len(APPLIANCES), dtype=np.float32)
        y_cls = np.empty(len(APPLIANCES), dtype=np.float32)
        mask_reg = np.empty(len(APPLIANCES), dtype=bool)
        mask_cls = np.empty(len(APPLIANCES), dtype=bool)
        for i, a in enumerate(APPLIANCES):
            raw = self.target_raw[a][idx]
            sc = self.target_scalers[a]
            y_reg_raw[i] = raw
            y_reg[i] = (raw - sc["mean"]) / sc["std"]  # NaN propagates if raw is NaN
            y_cls[i] = self.label[a][idx]
            mask_reg[i] = self.valid[a][idx]
            mask_cls[i] = self.valid[a][idx]

        meta = {
            "dataid": str(home),
            "split": self.split,
            "utc_dt": pd.Timestamp(self.utc_dt[idx]),
            "endpoint_index": ep,
        }
        return (
            x,
            torch.from_numpy(y_reg),
            torch.from_numpy(y_reg_raw),
            torch.from_numpy(y_cls),
            torch.from_numpy(mask_reg),
            torch.from_numpy(mask_cls),
            meta,
        )


def collate_fn(batch):
    """Stacks tensors; keeps metadata as plain Python lists (not tensorized)."""
    xs = torch.stack([b[0] for b in batch])
    y_reg = torch.stack([b[1] for b in batch])
    y_reg_raw = torch.stack([b[2] for b in batch])
    y_cls = torch.stack([b[3] for b in batch])
    mask_reg = torch.stack([b[4] for b in batch])
    mask_cls = torch.stack([b[5] for b in batch])
    meta = {
        "dataid": [b[6]["dataid"] for b in batch],
        "split": [b[6]["split"] for b in batch],
        "utc_dt": [b[6]["utc_dt"] for b in batch],
        "endpoint_index": torch.tensor([b[6]["endpoint_index"] for b in batch], dtype=torch.int64),
    }
    return xs, y_reg, y_reg_raw, y_cls, mask_reg, mask_cls, meta
