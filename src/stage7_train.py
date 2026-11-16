"""
Stage 7 Step 4B: reusable CPU-compatible training pipeline (CT-466 Phase 2).

This module is NOT launched here -- it provides reusable functions/classes
(TrainConfig, run_epoch, train_model) that a future, explicitly-approved
step would call to actually run full training. Nothing in this file is
executed as a script in this implementation step.

Built entirely from the EXISTING, unmodified Stage 7 components:
  - stage7_preprocess.py  (train-only scaler fitting)
  - stage7_dataset.py     (NILMWindowDataset, collate_fn, build_aggregate_lookup)
  - stage7_losses.py      (masked_bce_loss, masked_mse_loss, compute_pos_weights)
  - models/mlp.py, models/cnn1d.py

Design points:
  - Model input remains aggregate-only (x), exactly as produced by the
    existing dataset -- this module never constructs or modifies model
    input tensors itself.
  - Preprocessing is fit ONCE, on the training split only, via the
    existing fit_aggregate_scaler / fit_target_scalers functions, and
    applied unchanged to validation (and, in a later step, test).
  - Validation NEVER updates gradients or parameters: run_epoch(train=False)
    wraps the forward pass in torch.no_grad() and never calls .backward()
    or optimizer.step().
  - The test split is never referenced anywhere in this module. As of the
    Step 5A correction, build_datasets() reads five_homes_modeling.parquet
    with an explicit pyarrow row filter excluding split='test' rows, plus
    a runtime guard re-checking the observed split values -- so no
    test-period aggregate reading or test appliance target column ever
    enters this pipeline's memory, not even transiently.
  - Checkpoints are saved only under the ROOT-LEVEL `models/` directory
    (git-ignored via the project's `/models/` rule), never under
    `src/models/` (which holds source code, not output artifacts).
    TrainConfig.__post_init__ actively enforces this: any checkpoint_dir
    that resolves to `src/models/` itself, or to any path nested inside
    it, raises ValueError -- this is a runtime guard, not merely a
    documented default.
  - Class-imbalance handling (unweighted / capped-20 / capped-50) is
    implemented as a configuration switch using the existing
    stage7_losses.compute_pos_weights -- no Stage 6 threshold or label is
    touched; only the loss's positive-class weight changes.
"""
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "models"))

import stage7_preprocess as pp  # noqa: E402
import stage7_dataset as ds  # noqa: E402
import stage7_losses as sl  # noqa: E402
from mlp import MLP  # noqa: E402
from cnn1d import CNN1D  # noqa: E402

MODEL_CLASSES = {"MLP": MLP, "CNN1D": CNN1D}
CLASS_WEIGHT_MODES = ("unweighted", "cap20", "cap50")

# Root-level checkpoint directory -- NOT src/models/. Matches the
# project's .gitignore `/models/` rule (anchored to repo root).
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "models"

# The source model-definition directory (mlp.py, cnn1d.py) -- holds code,
# never training output. TrainConfig rejects any checkpoint_dir that
# resolves to this directory or to anything nested inside it.
UNSAFE_CHECKPOINT_DIR = (SRC_DIR / "models").resolve()


@dataclass
class TrainConfig:
    model_name: str = "MLP"
    batch_size: int = 256
    epochs: int = 10
    lr: float = 1e-3
    lambda_cls: float = 1.0
    lambda_reg: float = 1.0
    class_weight_mode: str = "unweighted"  # one of CLASS_WEIGHT_MODES
    patience: int = 5
    num_workers: int = 0
    device: str = "cpu"
    seed: int = 42
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR

    def __post_init__(self):
        if self.model_name not in MODEL_CLASSES:
            raise ValueError(f"Unknown model_name={self.model_name!r}; expected one of {list(MODEL_CLASSES)}")
        if self.class_weight_mode not in CLASS_WEIGHT_MODES:
            raise ValueError(f"Unknown class_weight_mode={self.class_weight_mode!r}; expected one of {CLASS_WEIGHT_MODES}")
        if self.device != "cpu":
            raise ValueError("This pipeline is CPU-only; TrainConfig.device must be 'cpu'.")
        self.checkpoint_dir = Path(self.checkpoint_dir)
        resolved_checkpoint_dir = self.checkpoint_dir.resolve()
        if resolved_checkpoint_dir == UNSAFE_CHECKPOINT_DIR or UNSAFE_CHECKPOINT_DIR in resolved_checkpoint_dir.parents:
            raise ValueError(
                f"checkpoint_dir={self.checkpoint_dir!r} resolves to {resolved_checkpoint_dir}, which is "
                f"the source model directory ({UNSAFE_CHECKPOINT_DIR}) or a location inside it. "
                "That directory holds model-definition source code, not training output. "
                "Use a different directory, e.g. the default root-level 'models/'."
            )


def get_memory_usage_mb():
    """Best-effort current-process memory in MB. Returns None if unavailable
    on this platform (no new package is installed to support this)."""
    try:
        import resource  # POSIX only
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return usage / 1024.0  # ru_maxrss is KB on Linux
    except ImportError:
        pass
    try:
        import ctypes
        import ctypes.wintypes as wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        # Explicit argtypes/restype are required here: without them, ctypes
        # falls back to a default 32-bit-int call signature for both
        # functions, which causes GetProcessMemoryInfo to fail (return
        # FALSE) SILENTLY on 64-bit Windows -- caught below by `if ...:`
        # evaluating falsy, not by an exception. This was the exact defect
        # identified during the Step 4C feasibility assessment (confirmed:
        # the identical call succeeds once these types are declared).
        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        psapi = ctypes.WinDLL("psapi.dll", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / (1024.0 * 1024.0)
    except Exception:
        pass
    return None


def build_datasets():
    """Loads TRAIN and VALIDATION datasets using the existing, unmodified
    preprocessing and dataset functions. Preprocessing is fit on the
    training split only. The test split is never loaded by this function:
    five_homes_modeling.parquet is read with a pyarrow row filter that
    excludes split='test' rows at read time (verified against the real
    file: filtered read returns exactly the train+val row count, matching
    a manual post-hoc filter of the unfiltered file), and a runtime guard
    immediately re-checks the observed split values, so no test-period
    aggregate reading or test appliance target column ever becomes part
    of modeling_df -- not even transiently -- before fit_aggregate_scaler
    or build_aggregate_lookup see it. Stage 1-6 source files themselves
    are untouched; only what this function reads from them changes."""
    modeling_df = pd.read_parquet(
        ds.MODELING_PATH, engine="pyarrow", filters=[("split", "in", ["train", "val"])],
    )
    observed_splits = set(modeling_df["split"].unique())
    if not observed_splits <= {"train", "val"}:
        raise RuntimeError(
            f"build_datasets() loaded unexpected split value(s) {sorted(observed_splits - {'train', 'val'})} "
            "from five_homes_modeling.parquet -- refusing to proceed. Test-period data must never "
            "enter the training pipeline."
        )
    labels_train_df = pd.read_parquet(ds.LABELS_PATHS["train"], engine="pyarrow")

    agg_scaler = pp.fit_aggregate_scaler(modeling_df)
    target_scalers = pp.fit_target_scalers(labels_train_df)
    agg_lookup = ds.build_aggregate_lookup(modeling_df)

    train_ds = ds.NILMWindowDataset("train", agg_lookup, agg_scaler["mean"], agg_scaler["std"], target_scalers)
    val_ds = ds.NILMWindowDataset("val", agg_lookup, agg_scaler["mean"], agg_scaler["std"], target_scalers)

    preprocessing_reference = {
        "aggregate_scaler": agg_scaler,
        "target_scalers": target_scalers,
        "params_file": str(PROJECT_ROOT / "data" / "meta" / "stage7_preprocessing_params.json"),
    }
    return train_ds, val_ds, labels_train_df, preprocessing_reference


def compute_pos_weight_tensor(labels_train_df, mode: str):
    """Builds the pos_weight tensor for masked_bce_loss according to the
    requested class-imbalance configuration. Reads training labels only;
    never alters Stage 6 thresholds or label columns."""
    if mode not in CLASS_WEIGHT_MODES:
        raise ValueError(f"Unknown class_weight_mode={mode!r}")
    if mode == "unweighted":
        return None
    cap = {"cap20": 20.0, "cap50": 50.0}[mode]
    weights = sl.compute_pos_weights(labels_train_df, appliances=sl.APPLIANCES, cap=cap)
    return sl.pos_weight_dict_to_tensor(weights, appliances=sl.APPLIANCES)


def _bce_elementwise_sums(cls_probs: torch.Tensor, y_cls: torch.Tensor, mask_cls: torch.Tensor, pos_weight):
    """Per-appliance (sum_of_elementwise_bce_loss, valid_count) for ONE
    batch. Uses the IDENTICAL masking/selection/formula as
    stage7_losses.masked_bce_loss (same EPS clamp, same weighted-BCE
    expression) -- replicated here ONLY to accumulate epoch-level
    statistics; stage7_losses.masked_bce_loss itself remains the sole,
    unmodified function used to compute the actual per-batch training
    loss passed to .backward(). Returns plain Python floats/ints (no
    grad tracking needed for reporting statistics)."""
    n_app = cls_probs.shape[1]
    sums_counts = []
    for i in range(n_app):
        valid = mask_cls[:, i]
        n = int(valid.sum().item())
        if n == 0:
            sums_counts.append((0.0, 0))
            continue
        p = cls_probs[valid, i].clamp(sl.EPS, 1.0 - sl.EPS)
        y = y_cls[valid, i]
        w = 1.0 if pos_weight is None else pos_weight[i]
        elem_loss = -(w * y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))
        sums_counts.append((float(elem_loss.sum().item()), n))
    return sums_counts


def _mse_elementwise_sums(reg_out: torch.Tensor, y_reg: torch.Tensor, mask_reg: torch.Tensor):
    """Per-appliance (sum_of_squared_error, valid_count) for ONE batch.
    Identical masking/formula to stage7_losses.masked_mse_loss, replicated
    only for epoch-level statistics -- see _bce_elementwise_sums docstring."""
    n_app = reg_out.shape[1]
    sums_counts = []
    for i in range(n_app):
        valid = mask_reg[:, i]
        n = int(valid.sum().item())
        if n == 0:
            sums_counts.append((0.0, 0))
            continue
        pred = reg_out[valid, i]
        target = y_reg[valid, i]
        elem_loss = (pred - target) ** 2
        sums_counts.append((float(elem_loss.sum().item()), n))
    return sums_counts


def _aggregate_epoch_appliance_means(sum_by_app, count_by_app):
    """Final epoch-level reduction: each appliance's mean over ALL its
    valid entries across the WHOLE epoch, then the mean across appliances
    that had at least one valid entry anywhere in the epoch (appliances
    with zero valid entries all epoch are excluded, not treated as zero --
    matching stage7_losses' own per-batch reduction design, just applied
    over accumulated epoch totals instead of a single batch)."""
    appliance_means = [s / c for s, c in zip(sum_by_app, count_by_app) if c > 0]
    if len(appliance_means) == 0:
        return 0.0
    return sum(appliance_means) / len(appliance_means)


def run_epoch(model, loader, optimizer, lambda_cls: float, lambda_reg: float, pos_weight, train: bool):
    """Runs one pass over `loader`.

    train=True : model.train(), gradients enabled, backward()+optimizer.step()
                 called once per batch, using the OFFICIAL per-batch
                 stage7_losses.masked_bce_loss / masked_mse_loss scalars
                 (unchanged by this correction).
    train=False: model.eval(), wrapped in torch.no_grad() -- .backward() is
                 never called and optimizer.step() is never called, so no
                 gradient or parameter update can occur. (optimizer is
                 accepted but unused in this branch, kept only for a
                 uniform call signature.)

    Epoch-level reporting (this is what was corrected): rather than
    averaging each batch's already-reduced scalar loss uniformly (which
    implicitly gives every batch equal weight regardless of its size or
    how many valid labels it contained), this function accumulates each
    appliance's (sum_of_elementwise_loss, valid_count) across every batch
    in the epoch, then computes each appliance's true mean over all its
    valid entries for the whole epoch, then averages across appliances
    that had any valid entries anywhere in the epoch. A batch in which an
    appliance has zero valid entries contributes (0, 0) to that
    appliance's running total -- which leaves the running mean completely
    unaffected, so such batches cannot distort the epoch-level result.
    """
    model.train() if train else model.eval()

    cls_sum_by_app, cls_count_by_app = None, None
    reg_sum_by_app, reg_count_by_app = None, None
    n_batches = 0
    grad_context = torch.enable_grad() if train else torch.no_grad()

    with grad_context:
        for batch in loader:
            x, y_reg, y_reg_raw, y_cls, mask_reg, mask_cls, meta = batch
            cls_probs, reg_out = model(x)  # x: aggregate-only input, unmodified

            # Official per-batch losses -- UNCHANGED, still the only values
            # ever used for .backward()/optimizer.step().
            cls_loss = sl.masked_bce_loss(cls_probs, y_cls, mask_cls, pos_weight=pos_weight)
            reg_loss = sl.masked_mse_loss(reg_out, y_reg, mask_reg)  # standardized y_reg, never y_reg_raw
            combined = lambda_cls * cls_loss + lambda_reg * reg_loss

            if train:
                optimizer.zero_grad()
                combined.backward()
                optimizer.step()

            # Epoch-level statistics -- detached, reporting only, no effect
            # on gradients/parameters either way (and validation already
            # runs this whole block under torch.no_grad() regardless).
            cls_stats = _bce_elementwise_sums(cls_probs.detach(), y_cls, mask_cls, pos_weight)
            reg_stats = _mse_elementwise_sums(reg_out.detach(), y_reg, mask_reg)
            if cls_sum_by_app is None:
                n_app = len(cls_stats)
                cls_sum_by_app, cls_count_by_app = [0.0] * n_app, [0] * n_app
                reg_sum_by_app, reg_count_by_app = [0.0] * n_app, [0] * n_app
            for i, (s, c) in enumerate(cls_stats):
                cls_sum_by_app[i] += s
                cls_count_by_app[i] += c
            for i, (s, c) in enumerate(reg_stats):
                reg_sum_by_app[i] += s
                reg_count_by_app[i] += c

            n_batches += 1

    if cls_sum_by_app is None:
        # loader produced zero batches -- degenerate but must not crash
        return {"cls_loss": 0.0, "reg_loss": 0.0, "total_loss": 0.0, "n_batches": 0}

    epoch_cls_loss = _aggregate_epoch_appliance_means(cls_sum_by_app, cls_count_by_app)
    epoch_reg_loss = _aggregate_epoch_appliance_means(reg_sum_by_app, reg_count_by_app)
    epoch_total_loss = lambda_cls * epoch_cls_loss + lambda_reg * epoch_reg_loss

    return {"cls_loss": epoch_cls_loss, "reg_loss": epoch_reg_loss, "total_loss": epoch_total_loss, "n_batches": n_batches}


def is_improvement(current_val_loss: float, best_val_loss: float) -> bool:
    return current_val_loss < best_val_loss


def should_stop_early(epochs_without_improvement: int, patience: int) -> bool:
    return epochs_without_improvement >= patience


def train_model(config: TrainConfig, train_ds=None, val_ds=None, labels_train_df=None,
                 preprocessing_reference=None, log_fn=print):
    """Trains one model under one configuration, with validation-based
    early stopping and best-checkpoint saving. Never references the test
    split. Intended to be called by a future, separately-approved step --
    not invoked here.

    train_ds/val_ds/labels_train_df/preprocessing_reference may be supplied
    externally (e.g. small subsets for testing); if omitted, the full
    training/validation splits are loaded via build_datasets().
    """
    torch.manual_seed(config.seed)

    if train_ds is None or val_ds is None or labels_train_df is None:
        train_ds, val_ds, labels_train_df, preprocessing_reference = build_datasets()
    elif preprocessing_reference is None:
        preprocessing_reference = {"note": "externally supplied datasets; no preprocessing_reference provided"}

    pos_weight = compute_pos_weight_tensor(labels_train_df, config.class_weight_mode)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers,
        collate_fn=ds.collate_fn, generator=torch.Generator().manual_seed(config.seed),
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers,
        collate_fn=ds.collate_fn,
    )

    model = MODEL_CLASSES[config.model_name]()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.checkpoint_dir / f"{config.model_name}_{config.class_weight_mode}_best.pt"

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history = []

    for epoch in range(config.epochs):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, config.lambda_cls, config.lambda_reg, pos_weight, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, config.lambda_cls, config.lambda_reg, pos_weight, train=False)
        duration_s = time.time() - t0
        mem_mb = get_memory_usage_mb()

        log_fn(f"epoch={epoch} train={train_metrics} val={val_metrics} duration_s={duration_s:.2f} mem_mb={mem_mb}")
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics, "duration_s": duration_s, "memory_mb": mem_mb})

        if is_improvement(val_metrics["total_loss"], best_val_loss):
            best_val_loss = val_metrics["total_loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "config": {**asdict(config), "checkpoint_dir": str(config.checkpoint_dir)},
                "preprocessing_reference": preprocessing_reference,
                "validation_metrics": val_metrics,
            }
            torch.save(checkpoint, checkpoint_path)
        else:
            epochs_without_improvement += 1

        if should_stop_early(epochs_without_improvement, config.patience):
            log_fn(f"Early stopping at epoch {epoch} (no improvement for {config.patience} epochs).")
            break

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path.exists() else None,
        "history": history,
    }
