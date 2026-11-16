"""
Stage 7 Step 3: masked multi-task losses (CT-466 Phase 2).

Matches the ACTUAL existing interfaces (confirmed by direct inspection,
not assumed):
  - src/models/mlp.py / src/models/cnn1d.py forward() returns
    (cls_probs, reg_out), each (batch, 3) -- cls_probs are already SIGMOID
    PROBABILITIES in [0,1], NOT raw logits. The loss functions here are
    written for probabilities, matching this -- the model architecture is
    not modified to accommodate the loss.
  - src/stage7_dataset.py NILMWindowDataset / collate_fn produce y_cls,
    y_reg (standardized), mask_cls, mask_reg, each (batch, 3) float32 /
    bool. Invalid/missing positions are NaN in y_cls/y_reg and False in
    the corresponding mask -- never 0/OFF.

Masking strategy (as explicitly required): invalid entries are REMOVED
via boolean selection BEFORE any loss arithmetic touches them -- NaN
values are never passed into log()/MSE, and are never relied upon to be
"zeroed out" after the fact (a per-element loss on a NaN target would
itself be NaN, and NaN * 0 == NaN, not 0 -- that approach is deliberately
NOT used here).

Each of the three appliances is masked and averaged independently. An
appliance with zero valid entries in a given batch is excluded from that
batch's appliance-average (not included as a diluting zero); if ALL
appliances have zero valid entries, a differentiable zero (still
connected to the autograd graph via the model's own output tensor) is
returned instead of NaN or a disconnected plain float.

No optimizer, training loop, or evaluation code is included here.
"""
import torch

APPLIANCES = ["air1", "dishwasher1", "refrigerator1"]
EPS = 1e-7


def masked_bce_loss(cls_probs: torch.Tensor, y_cls: torch.Tensor, mask_cls: torch.Tensor,
                     pos_weight: torch.Tensor = None) -> torch.Tensor:
    """Binary cross-entropy on PROBABILITIES (cls_probs already has sigmoid
    applied by the model), masked independently per appliance.

    cls_probs : (batch, n_appliances) in [0,1]
    y_cls     : (batch, n_appliances) in {0.0, 1.0}; NaN at invalid positions
                (never read -- excluded via mask_cls before any arithmetic)
    mask_cls  : (batch, n_appliances) bool, independent per appliance
    pos_weight: optional (n_appliances,) tensor; None == unweighted (all 1.0)

    Returns a scalar loss, averaged over appliances that have >=1 valid
    entry in this batch. If no appliance has any valid entry, returns a
    differentiable zero.
    """
    n_app = cls_probs.shape[1]
    per_appliance = []
    for i in range(n_app):
        valid = mask_cls[:, i]
        if valid.sum() == 0:
            continue
        p = cls_probs[valid, i].clamp(EPS, 1.0 - EPS)
        y = y_cls[valid, i]
        w = 1.0 if pos_weight is None else pos_weight[i]
        elem_loss = -(w * y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))
        per_appliance.append(elem_loss.mean())
    if len(per_appliance) == 0:
        return cls_probs.sum() * 0.0
    return torch.stack(per_appliance).mean()


def masked_mse_loss(reg_out: torch.Tensor, y_reg: torch.Tensor, mask_reg: torch.Tensor) -> torch.Tensor:
    """Mean squared error on STANDARDIZED regression targets, masked
    independently per appliance.

    reg_out  : (batch, n_appliances) linear, unconstrained
    y_reg    : (batch, n_appliances) -- MUST be the STANDARDIZED regression
               targets, produced using the training-only preprocessing
               statistics fit in stage7_preprocess.fit_target_scalers (see
               NILMWindowDataset's `y_reg` field). NaN at invalid positions
               (never read -- excluded via mask_reg before any arithmetic).
               Do NOT pass `y_reg_raw` (the original, unstandardized
               targets also returned by the dataset) to this function --
               it is intended only for evaluation/inverse-transform, not
               for this loss. This requirement is documented here, not
               enforced at runtime: no heuristic is used to guess whether
               a given tensor is standardized, since any such check could
               itself be wrong (e.g. a genuinely standardized appliance
               with an unusual distribution could look "raw", and vice
               versa) -- correctness here depends on the caller passing
               the right tensor, not on this function detecting misuse.
    mask_reg : (batch, n_appliances) bool, independent per appliance

    Returns a scalar loss, averaged over appliances that have >=1 valid
    entry in this batch. If no appliance has any valid entry, returns a
    differentiable zero.
    """
    n_app = reg_out.shape[1]
    per_appliance = []
    for i in range(n_app):
        valid = mask_reg[:, i]
        if valid.sum() == 0:
            continue
        pred = reg_out[valid, i]
        target = y_reg[valid, i]
        elem_loss = (pred - target) ** 2
        per_appliance.append(elem_loss.mean())
    if len(per_appliance) == 0:
        return reg_out.sum() * 0.0
    return torch.stack(per_appliance).mean()


def compute_pos_weights(labels_train_df, appliances=APPLIANCES, cap: float = None) -> dict:
    """Per-appliance positive-class weight (n_OFF / n_ON), from valid
    TRAINING labels only. Does not alter Stage 6 thresholds or labels --
    read-only computation over the already-generated label column.

    labels_train_df must already be the train-split label file; this
    function additionally asserts split == 'train' for every row (same
    defensive pattern as stage7_preprocess.fit_target_scalers).

    cap: if given, the raw ratio is clipped to this maximum (e.g. 20 or
    50), to compare against the uncapped ratio and against unweighted
    (pos_weight=None in masked_bce_loss) during later, separate model
    selection on validation data -- no selection is made here.
    """
    if not (labels_train_df["split"] == "train").all():
        raise RuntimeError("compute_pos_weights received rows outside the train split -- refusing to compute.")
    weights = {}
    for a in appliances:
        valid = labels_train_df[f"{a}_valid"]
        y = labels_train_df.loc[valid, f"{a}_label"]
        n_pos = int((y == 1).sum())
        n_off = int((y == 0).sum())
        if n_pos == 0:
            raise RuntimeError(f"No positive training examples for {a} -- cannot compute a positive-class weight.")
        if n_off == 0:
            raise ValueError(
                f"No negative (OFF) training examples for {a} -- cannot compute a meaningful "
                f"positive-class weight (the ratio n_off/n_pos would be 0, which would silently "
                f"zero out the positive-class loss term entirely)."
            )
        ratio = n_off / n_pos
        if cap is not None:
            ratio = min(ratio, cap)
        weights[a] = {"weight": ratio, "raw_ratio": n_off / n_pos, "capped": cap is not None and (n_off / n_pos) > cap,
                      "n_pos": n_pos, "n_off": n_off}
    return weights


def pos_weight_dict_to_tensor(weights: dict, appliances=APPLIANCES) -> torch.Tensor:
    """Convert the dict returned by compute_pos_weights into the ordered
    tensor expected by masked_bce_loss's pos_weight argument."""
    return torch.tensor([weights[a]["weight"] for a in appliances], dtype=torch.float32)
