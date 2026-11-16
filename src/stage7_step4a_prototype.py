"""
Stage 7 Step 4A: minimal CPU-only training-pipeline prototype (smoke test).

Wires together the EXISTING, already-implemented and reviewed Stage 7
components -- dataset (stage7_dataset.py), preprocessing
(stage7_preprocess.py), models (models/mlp.py, models/cnn1d.py), and
losses (stage7_losses.py) -- end to end, on a small, deterministic subset
of TRAINING data only, to verify the pipeline actually runs: forward
pass, masked loss computation, backward pass, and exactly one optimizer
step.

This is explicitly NOT a training script: no epochs, no validation/test
data, no checkpointing, no evaluation, no class-weight comparison. None
of the existing dataset/preprocessing/model/loss files are modified by
this module.

Loss combination: total_loss = lambda_cls * cls_loss + lambda_reg * reg_loss.
Both coefficients are plain function parameters (default 1.0 each) --
explicitly NOT a tuned or selected final weighting. The actual
train/validation-based selection among candidate weights (and among
unweighted / capped-20 / capped-50 class weighting, per stage7_losses.
compute_pos_weights) is deferred to a later step, not decided here.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "models"))

import stage7_preprocess as pp  # noqa: E402
import stage7_dataset as ds  # noqa: E402
import stage7_losses as sl  # noqa: E402
from mlp import MLP  # noqa: E402
from cnn1d import CNN1D  # noqa: E402

MODEL_CLASSES = {"MLP": MLP, "CNN1D": CNN1D}

SUBSET_PER_HOME = 40   # deterministic: first N indices per home, train split only
BATCH_SIZE = 64
SEED = 2024
LAMBDA_CLS = 1.0        # configurable combination coefficient -- prototype default, NOT tuned
LAMBDA_REG = 1.0        # configurable combination coefficient -- prototype default, NOT tuned


def build_train_dataset():
    """Loads training-only data and fits preprocessing using the existing,
    unmodified Stage 7 functions. Returns the NILMWindowDataset for split='train'."""
    modeling_df = pd.read_parquet(ds.MODELING_PATH, engine="pyarrow")
    labels_train_df = pd.read_parquet(ds.LABELS_PATHS["train"], engine="pyarrow")

    agg_scaler = pp.fit_aggregate_scaler(modeling_df)
    target_scalers = pp.fit_target_scalers(labels_train_df)
    agg_lookup = ds.build_aggregate_lookup(modeling_df)

    train_ds = ds.NILMWindowDataset("train", agg_lookup, agg_scaler["mean"], agg_scaler["std"], target_scalers)
    return train_ds


def build_small_deterministic_train_subset(train_ds, per_home=SUBSET_PER_HOME):
    """First `per_home` dataset indices for each of the five homes, in
    existing dataset order -- fully deterministic, no randomness involved.
    Operates only on the already-train-only `train_ds`."""
    indices = []
    for home in ds.HOMES:
        home_idx = np.where(train_ds.dataid == home)[0]
        indices.extend(home_idx[:per_home].tolist())
    return sorted(indices)


def run_prototype_step(model_name="MLP", lambda_cls=LAMBDA_CLS, lambda_reg=LAMBDA_REG,
                        per_home=SUBSET_PER_HOME, batch_size=BATCH_SIZE, seed=SEED):
    """Runs exactly one forward + masked-loss + backward + optimizer.step()
    on a small deterministic TRAINING subset, using the existing model and
    loss implementations unchanged. Returns a dict of results for
    inspection/testing."""
    train_ds = build_train_dataset()
    subset_indices = build_small_deterministic_train_subset(train_ds, per_home=per_home)
    subset = Subset(train_ds, subset_indices)

    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=ds.collate_fn)
    x, y_reg, y_reg_raw, y_cls, mask_reg, mask_cls, meta = next(iter(loader))

    torch.manual_seed(seed)
    model = MODEL_CLASSES[model_name]()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    cls_probs, reg_out = model(x)  # x: aggregate-only input, unchanged from the dataset

    cls_loss = sl.masked_bce_loss(cls_probs, y_cls, mask_cls)
    reg_loss = sl.masked_mse_loss(reg_out, y_reg, mask_reg)  # y_reg = STANDARDIZED (never y_reg_raw)
    total_loss = lambda_cls * cls_loss + lambda_reg * reg_loss

    optimizer.zero_grad()
    total_loss.backward()

    grads_finite = all(
        torch.all(torch.isfinite(p.grad)).item()
        for p in model.parameters() if p.grad is not None
    )
    all_params_have_grad = all(p.grad is not None for p in model.parameters())

    params_before = [p.detach().clone() for p in model.parameters()]
    optimizer.step()
    any_param_changed = any(
        not torch.equal(before, after.detach())
        for before, after in zip(params_before, model.parameters())
    )

    return {
        "model_name": model_name,
        "n_subset": len(subset_indices),
        "batch_size_used": int(x.shape[0]),
        "cls_loss": cls_loss.item(),
        "reg_loss": reg_loss.item(),
        "total_loss": total_loss.item(),
        "lambda_cls": lambda_cls,
        "lambda_reg": lambda_reg,
        "loss_finite": bool(torch.isfinite(total_loss).item()),
        "all_params_have_grad": all_params_have_grad,
        "grads_finite": grads_finite,
        "any_param_changed": any_param_changed,
    }


if __name__ == "__main__":
    for name in ["MLP", "CNN1D"]:
        result = run_prototype_step(name)
        print(f"--- {name} ---")
        for k, v in result.items():
            print(f"  {k}: {v}")
