"""
Stage 7: test-set evaluation for the two CCP-selected checkpoints
(CT-466 Phase 2).

Reuses the EXISTING, unmodified Stage 7 components -- stage7_dataset.py
(NILMWindowDataset, collate_fn, build_aggregate_lookup), stage7_preprocess.py
(inverse_standardize), models/mlp.py, models/cnn1d.py -- plus the
checkpoint format produced by stage7_train.train_model (model_state_dict,
config, preprocessing_reference). No training, preprocessing,
label-generation, or checkpoint-producing file is modified or re-run by
this module. Stage 6 ground-truth ON/OFF labels and thresholds
(data/meta/stage6_thresholds.json) are read as-is and never regenerated.

Two-phase design (this is the structural guarantee, not just a comment):

  PHASE 1 -- select_validation_thresholds() -- VALIDATION ONLY.
    Loads a checkpoint, runs inference on the VAL split only, and picks one
    classification probability threshold per appliance by grid search,
    maximizing validation Macro-F1 (deterministic tie-break: the smallest
    threshold achieving the maximum). This function never opens
    LABELS_PATHS["test"] and never filters MODELING_PATH to split='test'.

  PHASE 2 -- evaluate_on_test() -- TEST ONLY, uses ALREADY-FROZEN thresholds.
    Takes a `frozen_thresholds` dict as a required argument (produced by
    Phase 1 and, in run_full_evaluation(), round-tripped through a JSON
    file on disk before being handed to this function) and performs NO
    threshold search of its own -- it only applies the thresholds it is
    given. Loads the checkpoint again, runs inference on the TEST split,
    and computes all reported metrics.

  run_full_evaluation() enforces the ordering for real: it runs Phase 1 for
  every checkpoint, writes ALL thresholds to
  data/meta/stage7_validation_thresholds.json, re-reads that file back from
  disk, and only then calls Phase 2 -- so Phase 2 can never see an
  in-memory threshold that wasn't actually persisted before any test data
  was touched.

Classification probability threshold != Stage 6 ON/OFF power threshold.
The latter (data/meta/stage6_thresholds.json) was already applied, once,
to generate the ground-truth labels in window_labels_{train,val,test}.parquet
-- it is read-only input here. The former is a new, separate decision (this
model's own probability cutoff for calling its sigmoid output "ON"),
selected per appliance per model on validation data only, exactly as this
module does.

Metrics are implemented from scratch (no scikit-learn in this environment,
and no new package may be installed): confusion-matrix counts, Accuracy,
Balanced Accuracy, Macro-F1, and regression MAE/RMSE. Every ratio that
would require dividing by zero (e.g. no actual ON samples for an appliance
in a given home -- confirmed to occur in the real test data for air1 at
homes 8386/3538) is returned as `None` (-> JSON null), never silently 0.

Plots are standalone, hand-written SVG files (no matplotlib available) --
see render_actual_vs_predicted_svg().
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "models"))

import stage7_dataset as ds  # noqa: E402
import stage7_preprocess as pp  # noqa: E402
from mlp import MLP  # noqa: E402
from cnn1d import CNN1D  # noqa: E402

MODEL_CLASSES = {"MLP": MLP, "CNN1D": CNN1D}
APPLIANCES = ds.APPLIANCES
HOMES = ds.HOMES

# Fixed, documented, reproducible candidate grid for threshold selection.
# 99 evenly spaced candidates; no randomness, no dependence on any data.
CANDIDATE_THRESHOLDS = [round(0.01 * i, 2) for i in range(1, 100)]

# CCP benchmark constants, as supplied by the project owner (not found
# anywhere in this repository -- there is no other source of truth for
# these numbers). Classification-only; the CCP specifies no MAE/RMSE
# benchmark, so regression metrics are reported without a pass/fail status.
CCP_MINIMUM = {"accuracy": 0.85, "balanced_accuracy": 0.75, "macro_f1": 0.70}
CCP_HIGHER = {"accuracy": 0.90, "balanced_accuracy": 0.85, "macro_f1": 0.80}

# Fixed plot settings -- the SAME home and window are used for every model
# so actual-vs-predicted plots are directly comparable across models.
# PLOT_HOME is fixed to the official unseen test household (8386) under the
# revised evaluation design -- the four development households have zero
# test-split rows, so plotting any of them would raise in
# render_actual_vs_predicted_svg() (no test rows found).
OFFICIAL_TEST_HOME = "8386"
PLOT_HOME = OFFICIAL_TEST_HOME
PLOT_WINDOW_MINUTES = 2880  # ~48 hours of consecutive test windows

DEFAULT_DATA_META_DIR = PROJECT_ROOT / "data" / "meta"
DEFAULT_THRESHOLDS_FILENAME = "stage7_validation_thresholds.json"
DEFAULT_RESULTS_FILENAME = "stage7_test_evaluation.json"
DEFAULT_PLOTS_DIRNAME = "stage7_evaluation_plots"


# ---------------------------------------------------------------------------
# Checkpoint loading / architecture reconstruction
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path):
    """Loads a stage7_train.train_model() checkpoint and reconstructs the
    exact architecture it was trained with. weights_only=False is required
    -- the checkpoint contains plain dicts (config, preprocessing_reference),
    not only tensors. Returns (model, checkpoint) with model in eval() mode."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model_name = checkpoint["config"]["model_name"]
    if model_name not in MODEL_CLASSES:
        raise ValueError(
            f"Checkpoint {checkpoint_path!r} has unknown config.model_name={model_name!r}; "
            f"expected one of {list(MODEL_CLASSES)}."
        )
    model = MODEL_CLASSES[model_name]()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


# ---------------------------------------------------------------------------
# Dataset construction (train-fitted scalers only; never refit here)
# ---------------------------------------------------------------------------

def build_split_dataset(split: str, agg_scaler: dict, target_scalers: dict):
    """Loads ONLY the requested split's aggregate rows (pyarrow row filter,
    same pattern as stage7_train.build_datasets()) and builds a
    NILMWindowDataset over it, using the given (already train-fitted,
    never refit here) scalers. A runtime guard re-checks the observed split
    values, matching stage7_train.build_datasets()'s own defensive pattern."""
    modeling_df = pd.read_parquet(
        ds.MODELING_PATH, engine="pyarrow", filters=[("split", "in", [split])],
    )
    observed_splits = set(modeling_df["split"].unique())
    if not observed_splits <= {split}:
        raise RuntimeError(
            f"build_split_dataset({split!r}) loaded unexpected split value(s) "
            f"{sorted(observed_splits - {split})} -- refusing to proceed."
        )
    agg_lookup = ds.build_aggregate_lookup(modeling_df)
    return ds.NILMWindowDataset(split, agg_lookup, agg_scaler["mean"], agg_scaler["std"], target_scalers)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, dataset, target_scalers: dict, batch_size: int = 256):
    """Runs model.eval() + torch.no_grad() inference over the WHOLE dataset,
    in dataset order (shuffle=False). Regression predictions are inverse-
    standardized back to original units here (via stage7_preprocess.
    inverse_standardize), using the SAME target_scalers the model's targets
    were standardized with during training -- never refit. Returns a dict
    of concatenated numpy arrays, one row per dataset sample."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=ds.collate_fn)

    dataid_parts, utc_parts = [], []
    y_cls_parts, mask_cls_parts, cls_probs_parts = [], [], []
    y_reg_raw_parts, mask_reg_parts, reg_pred_raw_parts = [], [], []

    model.eval()
    with torch.no_grad():
        for x, y_reg, y_reg_raw, y_cls, mask_reg, mask_cls, meta in loader:
            cls_probs, reg_out = model(x)

            reg_pred_raw = np.empty(reg_out.shape, dtype=np.float32)
            for i, a in enumerate(APPLIANCES):
                sc = target_scalers[a]
                reg_pred_raw[:, i] = pp.inverse_standardize(reg_out[:, i].numpy(), sc["mean"], sc["std"])

            dataid_parts.extend(meta["dataid"])
            utc_parts.extend(meta["utc_dt"])
            y_cls_parts.append(y_cls.numpy())
            mask_cls_parts.append(mask_cls.numpy())
            cls_probs_parts.append(cls_probs.numpy())
            y_reg_raw_parts.append(y_reg_raw.numpy())
            mask_reg_parts.append(mask_reg.numpy())
            reg_pred_raw_parts.append(reg_pred_raw)

    return {
        "dataid": np.array(dataid_parts),
        "utc_dt": np.array(utc_parts),
        "y_cls": np.concatenate(y_cls_parts, axis=0),
        "mask_cls": np.concatenate(mask_cls_parts, axis=0).astype(bool),
        "cls_probs": np.concatenate(cls_probs_parts, axis=0),
        "y_reg_raw": np.concatenate(y_reg_raw_parts, axis=0),
        "mask_reg": np.concatenate(mask_reg_parts, axis=0).astype(bool),
        "reg_pred_raw": np.concatenate(reg_pred_raw_parts, axis=0),
    }


# ---------------------------------------------------------------------------
# Metrics -- hand-rolled (no scikit-learn available), null-safe
# ---------------------------------------------------------------------------

def _safe_div(num, den):
    """Plain division that returns None (-> JSON null) instead of raising
    or silently returning 0/NaN when the denominator is zero."""
    return (num / den) if den > 0 else None


def _f1(precision, recall):
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return None
    return 2.0 * precision * recall / (precision + recall)


def confusion_counts(y_true, y_pred, valid_mask):
    """(tp, fp, tn, fn) as plain Python ints, computed ONLY over positions
    where valid_mask is True -- invalid/missing-label positions are
    excluded entirely, never counted as any class."""
    v = valid_mask.astype(bool)
    yt = y_true[v]
    yp = y_pred[v]
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    return tp, fp, tn, fn


def classification_metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
    """Accuracy, Balanced Accuracy, Macro-F1, and ON/OFF counts from
    confusion-matrix counts. Every metric that depends on a zero
    denominator (e.g. no actual ON samples -> recall_on undefined -> both
    balanced_accuracy and macro_f1 undefined) is None, never fabricated as
    0 or 1."""
    n_on = tp + fn
    n_off = tn + fp
    total = n_on + n_off

    accuracy = _safe_div(tp + tn, total)
    recall_on = _safe_div(tp, n_on)
    recall_off = _safe_div(tn, n_off)
    precision_on = _safe_div(tp, tp + fp)
    precision_off = _safe_div(tn, tn + fn)
    f1_on = _f1(precision_on, recall_on)
    f1_off = _f1(precision_off, recall_off)
    balanced_accuracy = None if (recall_on is None or recall_off is None) else (recall_on + recall_off) / 2.0
    macro_f1 = None if (f1_on is None or f1_off is None) else (f1_on + f1_off) / 2.0

    return {
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "n_on": n_on, "n_off": n_off,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "precision_on": precision_on, "recall_on": recall_on, "f1_on": f1_on,
        "precision_off": precision_off, "recall_off": recall_off, "f1_off": f1_off,
    }


def regression_metrics(y_true_raw: np.ndarray, y_pred_raw: np.ndarray, valid_mask: np.ndarray) -> dict:
    """MAE/RMSE in ORIGINAL measurement units, over valid entries only.
    None (not 0) if there are zero valid entries."""
    v = valid_mask.astype(bool)
    n = int(np.sum(v))
    if n == 0:
        return {"mae": None, "rmse": None, "n_valid": 0}
    diff = y_pred_raw[v] - y_true_raw[v]
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "n_valid": n,
    }


def ccp_status(cls_metrics: dict, benchmark: dict):
    """True/False against `benchmark` (CCP_MINIMUM or CCP_HIGHER), or None
    if ANY underlying metric is undefined -- an undetermined benchmark
    status must never be reported as a false pass or a false fail."""
    vals = {k: cls_metrics[k] for k in ("accuracy", "balanced_accuracy", "macro_f1")}
    if any(v is None for v in vals.values()):
        return None
    return bool(
        vals["accuracy"] >= benchmark["accuracy"]
        and vals["balanced_accuracy"] >= benchmark["balanced_accuracy"]
        and vals["macro_f1"] >= benchmark["macro_f1"]
    )


def _build_metrics_block(y_true, y_pred, mask_cls, y_reg_true_raw, y_reg_pred_raw, mask_reg) -> dict:
    """One classification+regression metrics block (used identically for
    pooled and every per-home slice, so there is exactly one code path)."""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred, mask_cls)
    block = classification_metrics_from_counts(tp, fp, tn, fn)
    block["ccp_minimum_met"] = ccp_status(block, CCP_MINIMUM)
    block["ccp_higher_met"] = ccp_status(block, CCP_HIGHER)
    block["regression"] = regression_metrics(y_reg_true_raw, y_reg_pred_raw, mask_reg)
    return block


# ---------------------------------------------------------------------------
# PHASE 1 -- VALIDATION ONLY -- threshold selection
# ---------------------------------------------------------------------------

def select_threshold_for_appliance(y_true, cls_probs, valid_mask, candidate_thresholds=CANDIDATE_THRESHOLDS) -> dict:
    """Grid search over `candidate_thresholds`, maximizing Macro-F1.
    Deterministic tie-break: candidates are scanned in ascending order and
    only replace the current best on a STRICT improvement, so the smallest
    threshold achieving the maximum Macro-F1 is the one kept. Candidates
    with an undefined Macro-F1 (e.g. would require dividing by zero) are
    skipped, never treated as a winning or losing score."""
    best = None
    for t in candidate_thresholds:
        y_pred = (cls_probs >= t).astype(np.float32)
        tp, fp, tn, fn = confusion_counts(y_true, y_pred, valid_mask)
        metrics = classification_metrics_from_counts(tp, fp, tn, fn)
        mf1 = metrics["macro_f1"]
        if mf1 is None:
            continue
        if best is None or mf1 > best["val_metrics"]["macro_f1"]:
            best = {"threshold": t, "val_metrics": metrics}
    if best is None:
        raise RuntimeError(
            "No candidate threshold produced a defined validation Macro-F1 -- "
            "cannot select a classification threshold for this appliance."
        )
    return best


def select_validation_thresholds(checkpoint_path, candidate_thresholds=CANDIDATE_THRESHOLDS,
                                  batch_size: int = 256, val_ds=None) -> dict:
    """VALIDATION-ONLY. Never opens LABELS_PATHS['test'] or filters
    MODELING_PATH to split='test'. Returns a dict that is the single
    source of truth for "the frozen thresholds" -- callers (notably
    run_full_evaluation()) are expected to persist this to disk and
    re-load it before ever calling evaluate_on_test().

    val_ds: injectable (tests pass a small synthetic Dataset); defaults to
    the real val split built from the checkpoint's own train-fitted scalers.
    """
    model, checkpoint = load_checkpoint(checkpoint_path)
    preprocessing_reference = checkpoint["preprocessing_reference"]
    agg_scaler = preprocessing_reference["aggregate_scaler"]
    target_scalers = preprocessing_reference["target_scalers"]

    if val_ds is None:
        val_ds = build_split_dataset("val", agg_scaler, target_scalers)
    val_result = run_inference(model, val_ds, target_scalers, batch_size=batch_size)

    thresholds = {}
    for i, a in enumerate(APPLIANCES):
        best = select_threshold_for_appliance(
            val_result["y_cls"][:, i], val_result["cls_probs"][:, i], val_result["mask_cls"][:, i],
            candidate_thresholds,
        )
        thresholds[a] = {
            "threshold": best["threshold"],
            "selection_rule": (
                "argmax validation Macro-F1 over a fixed grid of candidate thresholds "
                f"({candidate_thresholds[0]} to {candidate_thresholds[-1]}, step "
                f"{round(candidate_thresholds[1] - candidate_thresholds[0], 4)}); deterministic "
                "tie-break = smallest threshold achieving the maximum"
            ),
            "val_metrics": best["val_metrics"],
        }

    return {
        "model_name": checkpoint["config"]["model_name"],
        "class_weight_mode": checkpoint["config"]["class_weight_mode"],
        "checkpoint_path": str(checkpoint_path),
        "candidate_thresholds": list(candidate_thresholds),
        "thresholds": thresholds,
    }


# ---------------------------------------------------------------------------
# PHASE 2 -- TEST ONLY -- frozen-threshold evaluation
# ---------------------------------------------------------------------------

def render_actual_vs_predicted_svg(result: dict, appliance_index: int, appliance: str, home: str,
                                    window_minutes: int, model_label: str, out_path) -> str:
    """Hand-written, dependency-free SVG line plot (no matplotlib
    available): actual vs predicted power for ONE appliance at ONE home,
    over a representative chronological slice of `window_minutes`
    consecutive test windows (NOT all ~440k test windows at once).
    PLOT_HOME/PLOT_WINDOW_MINUTES are fixed module constants so the exact
    same slice is used for every model, making plots comparable."""
    home_sel = np.where(result["dataid"] == home)[0]
    if len(home_sel) == 0:
        raise RuntimeError(f"No test rows found for home={home!r} -- cannot render a plot.")
    order = np.argsort(result["utc_dt"][home_sel])
    idx = home_sel[order][:window_minutes]

    actual = result["y_reg_raw"][idx, appliance_index]
    valid = result["mask_reg"][idx, appliance_index].astype(bool)
    predicted = result["reg_pred_raw"][idx, appliance_index]
    times = result["utc_dt"][idx]
    n = len(idx)

    width, height = 960, 420
    margin_l, margin_r, margin_t, margin_b = 70, 20, 50, 60
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    finite_vals = np.concatenate([actual[valid], predicted[np.isfinite(predicted)]])
    if len(finite_vals) == 0:
        y_min, y_max = 0.0, 1.0
    else:
        y_min, y_max = float(np.min(finite_vals)), float(np.max(finite_vals))
        if y_max <= y_min:
            y_max = y_min + 1.0

    def xy(i, v):
        x = margin_l + (i / max(n - 1, 1)) * plot_w
        y = margin_t + (1 - (v - y_min) / (y_max - y_min)) * plot_h
        return x, y

    def polyline_points(values, value_valid=None):
        pts = []
        for i, v in enumerate(values):
            if value_valid is not None and not value_valid[i]:
                continue
            if not np.isfinite(v):
                continue
            x, y = xy(i, v)
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    actual_pts = polyline_points(actual, valid)
    pred_pts = polyline_points(predicted)
    t0, t1 = pd.Timestamp(times[0]), pd.Timestamp(times[-1])

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="sans-serif">
  <rect x="0" y="0" width="{width}" height="{height}" fill="white"/>
  <text x="{width / 2}" y="20" text-anchor="middle" font-size="16" font-weight="bold">{model_label} -- {appliance} -- home {home}</text>
  <text x="{width / 2}" y="38" text-anchor="middle" font-size="11" fill="#555">{t0} to {t1} ({n} consecutive test windows)</text>
  <line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="black"/>
  <line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="black"/>
  <text x="{margin_l - 8}" y="{margin_t + 4}" text-anchor="end" font-size="10">{y_max:.3f}</text>
  <text x="{margin_l - 8}" y="{margin_t + plot_h}" text-anchor="end" font-size="10">{y_min:.3f}</text>
  <text x="15" y="{margin_t + plot_h / 2}" text-anchor="middle" font-size="12" transform="rotate(-90 15 {margin_t + plot_h / 2})">Power (original units)</text>
  <text x="{margin_l + plot_w / 2}" y="{height - 10}" text-anchor="middle" font-size="12">Time (chronological test windows)</text>
  <polyline points="{actual_pts}" fill="none" stroke="#1f77b4" stroke-width="1.5"/>
  <polyline points="{pred_pts}" fill="none" stroke="#d62728" stroke-width="1.5" stroke-dasharray="4,2"/>
  <rect x="{width - 180}" y="{margin_t}" width="12" height="12" fill="#1f77b4"/>
  <text x="{width - 162}" y="{margin_t + 10}" font-size="11">Actual</text>
  <rect x="{width - 180}" y="{margin_t + 18}" width="12" height="12" fill="#d62728"/>
  <text x="{width - 162}" y="{margin_t + 28}" font-size="11">Predicted</text>
</svg>'''
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")
    return str(out_path)


def evaluate_on_test(checkpoint_path, frozen_thresholds: dict, batch_size: int = 256,
                      test_ds=None, plots_dir=None, plot_home: str = PLOT_HOME,
                      plot_window_minutes: int = PLOT_WINDOW_MINUTES) -> dict:
    """TEST-ONLY. `frozen_thresholds` MUST already be finalized (the dict
    returned by select_validation_thresholds(), ideally round-tripped
    through disk first -- see run_full_evaluation()). This function
    performs NO threshold search of its own; it only applies the
    thresholds it is given.

    test_ds: injectable (tests pass a small synthetic Dataset); defaults
    to the real test split built from the checkpoint's own train-fitted
    scalers.
    """
    model, checkpoint = load_checkpoint(checkpoint_path)
    preprocessing_reference = checkpoint["preprocessing_reference"]
    agg_scaler = preprocessing_reference["aggregate_scaler"]
    target_scalers = preprocessing_reference["target_scalers"]

    if test_ds is None:
        test_ds = build_split_dataset("test", agg_scaler, target_scalers)
    test_result = run_inference(model, test_ds, target_scalers, batch_size=batch_size)

    model_label = f"{frozen_thresholds['model_name']}_{frozen_thresholds['class_weight_mode']}"
    dataid_arr = test_result["dataid"]

    appliances_report = {}
    plots = {}
    for i, a in enumerate(APPLIANCES):
        threshold = frozen_thresholds["thresholds"][a]["threshold"]
        y_true = test_result["y_cls"][:, i]
        mask_cls = test_result["mask_cls"][:, i]
        probs = test_result["cls_probs"][:, i]
        y_pred = (probs >= threshold).astype(np.float32)
        y_reg_true = test_result["y_reg_raw"][:, i]
        y_reg_pred = test_result["reg_pred_raw"][:, i]
        mask_reg = test_result["mask_reg"][:, i]

        pooled = _build_metrics_block(y_true, y_pred, mask_cls, y_reg_true, y_reg_pred, mask_reg)

        per_home = {}
        for home in HOMES:
            sel = dataid_arr == home
            per_home[home] = _build_metrics_block(
                y_true[sel], y_pred[sel], mask_cls[sel], y_reg_true[sel], y_reg_pred[sel], mask_reg[sel],
            )

        appliances_report[a] = {"threshold_used": threshold, "pooled": pooled, "per_home": per_home}

        if plots_dir is not None:
            svg_path = Path(plots_dir) / model_label / f"{a}.svg"
            render_actual_vs_predicted_svg(
                test_result, appliance_index=i, appliance=a, home=plot_home,
                window_minutes=plot_window_minutes, model_label=model_label, out_path=svg_path,
            )
            plots[a] = str(svg_path)

    return {
        "model_name": frozen_thresholds["model_name"],
        "class_weight_mode": frozen_thresholds["class_weight_mode"],
        "checkpoint_path": str(checkpoint_path),
        "appliances": appliances_report,
        "plots": plots,
    }


# ---------------------------------------------------------------------------
# Orchestration: enforces the freeze-before-test ordering for real
# ---------------------------------------------------------------------------

def run_full_evaluation(checkpoint_paths: dict, data_meta_dir=None, make_plots: bool = True,
                         log_fn=print) -> dict:
    """checkpoint_paths: {label -> checkpoint path}, e.g.
    {"MLP_unweighted": ".../MLP_unweighted_best.pt", "CNN1D_unweighted": ".../CNN1D_unweighted_best.pt"}.
    Loads ONLY these checkpoints -- no other model file is ever opened.

    Phase ordering is enforced, not just documented: ALL validation
    thresholds for ALL checkpoints are computed and written to
    `stage7_validation_thresholds.json` first; that file is then re-read
    from disk into a fresh variable, and ONLY THAT re-read copy is ever
    passed into evaluate_on_test() (phase 2). No test-split file is opened
    anywhere before the phase-1 write completes.

    Never touches data/meta/stage7_experiment_summary.json (the six-
    experiment training summary) -- results are written to separate,
    new files under `data_meta_dir`.
    """
    data_meta_dir = Path(data_meta_dir) if data_meta_dir is not None else DEFAULT_DATA_META_DIR
    thresholds_path = data_meta_dir / DEFAULT_THRESHOLDS_FILENAME
    results_path = data_meta_dir / DEFAULT_RESULTS_FILENAME
    plots_dir = (data_meta_dir / DEFAULT_PLOTS_DIRNAME) if make_plots else None

    for label, path in checkpoint_paths.items():
        prefix = label.split("_")[0]
        if prefix not in MODEL_CLASSES:
            raise ValueError(f"Checkpoint label {label!r} does not start with a known model name; expected one of {list(MODEL_CLASSES)}.")

    # ---- PHASE 1: VALIDATION ONLY. No test file is opened above this line. ----
    all_thresholds = {}
    for label, path in checkpoint_paths.items():
        log_fn(f"[phase 1/2] selecting validation-only thresholds for {label} ...")
        frozen = select_validation_thresholds(path)
        expected_prefix = label.split("_")[0]
        if frozen["model_name"] != expected_prefix:
            raise ValueError(
                f"Checkpoint label {label!r} implies model_name={expected_prefix!r} but the "
                f"checkpoint's own config.model_name is {frozen['model_name']!r} -- refusing to proceed."
            )
        all_thresholds[label] = frozen

    data_meta_dir.mkdir(parents=True, exist_ok=True)
    with open(thresholds_path, "w") as f:
        json.dump(all_thresholds, f, indent=2)
    log_fn(f"Frozen validation thresholds written to {thresholds_path}")

    # Re-read what was actually persisted -- the in-memory `all_thresholds`
    # above is never reused for phase 2, only this disk-round-tripped copy.
    with open(thresholds_path) as f:
        frozen_thresholds_by_label = json.load(f)

    # ---- PHASE 2: TEST ONLY, using the frozen thresholds read back above. ----
    models_report = {}
    for label, path in checkpoint_paths.items():
        log_fn(f"[phase 2/2] evaluating {label} on the test split ...")
        models_report[label] = evaluate_on_test(
            path, frozen_thresholds_by_label[label], plots_dir=plots_dir,
        )

    all_results = {
        "generated_by": "src/stage7_evaluate.py",
        "official_test_household": OFFICIAL_TEST_HOME,
        "test_period_note": (
            "Full calendar year 2018 (2018-01-01 to 2018-12-31) test split for the official, "
            f"unseen test household ({OFFICIAL_TEST_HOME}) only (data/processed/window_labels_test.parquet). "
            "The four development households (7901, 7951, 8565, 3538) have zero rows in this "
            "split -- they were never held out from training/validation, so pooled test metrics "
            "are computed exclusively on the unseen household."
        ),
        "ccp_benchmarks": {
            "minimum": CCP_MINIMUM, "higher": CCP_HIGHER,
            "note": "Classification benchmarks only; the CCP specifies no MAE/RMSE threshold.",
        },
        "models": models_report,
    }
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log_fn(f"Test evaluation results written to {results_path}")

    return {
        "thresholds_path": str(thresholds_path),
        "results_path": str(results_path),
        "thresholds": frozen_thresholds_by_label,
        "results": all_results,
    }


if __name__ == "__main__":
    run_full_evaluation({
        "MLP_unweighted": PROJECT_ROOT / "models" / "MLP_unweighted_best.pt",
        "CNN1D_unweighted": PROJECT_ROOT / "models" / "CNN1D_unweighted_best.pt",
    })
