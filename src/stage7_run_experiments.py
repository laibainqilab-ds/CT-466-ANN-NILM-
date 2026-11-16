"""
Stage 7 Step 5A: six-experiment runner (CT-466 Phase 2).

This module is NOT launched by this implementation step -- it provides
reusable orchestration functions that a future, explicitly-approved step
would call to actually run the six experiments. Nothing in this file is
executed as a script here.

Runs, sequentially, all six (model, class-imbalance-handling) experiment
configurations:
    MLP:   unweighted, cap20, cap50
    CNN1D: unweighted, cap20, cap50

Built entirely from the EXISTING, unmodified stage7_train.py components
(TrainConfig, build_datasets, train_model) -- this module adds only
orchestration (shared-dataset reuse, per-experiment config construction,
summary recording), no new training/loss/model/dataset logic.

Design points:
  - build_datasets() is called exactly ONCE; the returned train_ds,
    val_ds, labels_train_df, preprocessing_reference are reused,
    unmodified, across all six train_model() calls -- avoiding six
    redundant parquet reads / aggregate-lookup rebuilds (flagged in the
    Step 4E review as a recommended efficiency fix, not a correctness
    defect).
  - A fresh TrainConfig is constructed per experiment; model_name and
    class_weight_mode vary, batch_size=256 and num_workers=0 are set
    explicitly on every config (matching the existing TrainConfig
    defaults, stated explicitly here for clarity rather than relying on
    defaults alone).
  - train_model() itself is never modified: early stopping, best-epoch
    checkpoint selection, and checkpoint-directory safety are exactly as
    implemented there. Validation-based model selection (is_improvement /
    should_stop_early inside train_model()) is untouched by this module.
  - Each experiment's model/optimizer/DataLoaders are local to its own
    train_model() call and are never referenced by this runner's loop
    beyond that call; this module additionally drops its own per-
    iteration `result`/`config` local references and forces a GC pass
    before starting the next experiment, as an explicit (not merely
    incidental) guarantee.
  - The test split is never loaded or referenced anywhere in this module
    -- it only ever touches what build_datasets() returns, and
    build_datasets() itself excludes test rows (Step 5A correction in
    stage7_train.py).
  - train_fn is an injectable parameter (default: the real
    stage7_train.train_model) so this orchestration logic can be tested
    without launching real training.
  - Step 5C correction (failure handling and summary persistence): each
    experiment's train_fn(...) call is wrapped in its own try/except, so
    one experiment raising an exception is recorded as a "failed" entry
    (with the exception type/message, no fabricated metrics or
    checkpoint path) and execution continues to the next experiment,
    rather than aborting the whole run. The summary is written to disk
    after EVERY experiment (success or failure), not only once at the
    end, so already-completed experiments' results survive a later
    experiment's crash.
"""
import gc
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "models"))

import stage7_train as st  # noqa: E402

EXPERIMENTS = [
    {"model_name": "MLP", "class_weight_mode": "unweighted"},
    {"model_name": "MLP", "class_weight_mode": "cap20"},
    {"model_name": "MLP", "class_weight_mode": "cap50"},
    {"model_name": "CNN1D", "class_weight_mode": "unweighted"},
    {"model_name": "CNN1D", "class_weight_mode": "cap20"},
    {"model_name": "CNN1D", "class_weight_mode": "cap50"},
]

# Machine-readable summary location. The project has no top-level
# results/ or evidence/ directory; the established convention (used by
# every prior stage -- stage4_cleaning_report.json, stage6_thresholds.json,
# stage7_preprocessing_params.json, etc.) is data/meta/*.json, so this
# summary follows that same convention rather than inventing a new
# top-level directory.
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data" / "meta" / "stage7_experiment_summary.json"


def build_experiment_configs(checkpoint_dir=None):
    """Returns 6 fresh TrainConfig instances, one per (model_name,
    class_weight_mode) combination in EXPERIMENTS. batch_size=256 and
    num_workers=0 are set explicitly (matching, not overriding, the
    existing TrainConfig defaults). checkpoint_dir is left at
    TrainConfig's own default (root-level models/, safety-guarded) unless
    explicitly overridden -- override only used by tests, to avoid ever
    writing under the real checkpoint directory."""
    configs = []
    for exp in EXPERIMENTS:
        kwargs = dict(exp, batch_size=256, num_workers=0)
        if checkpoint_dir is not None:
            kwargs["checkpoint_dir"] = checkpoint_dir
        configs.append(st.TrainConfig(**kwargs))
    return configs


def _best_epoch_val_metrics(result):
    """Extracts the val cls_loss/reg_loss recorded at the checkpointed
    best_epoch from a train_model() result's history. Returns
    (cls_loss, reg_loss), each None if best_epoch is unavailable (e.g. no
    improvement was ever recorded)."""
    best_epoch = result.get("best_epoch", -1)
    history = result.get("history", [])
    if best_epoch is None or best_epoch < 0 or best_epoch >= len(history):
        return None, None
    val_metrics = history[best_epoch]["val"]
    return val_metrics.get("cls_loss"), val_metrics.get("reg_loss")


def _success_entry(model_name, class_weight_mode, result):
    """Summary entry for an experiment whose train_fn(...) call returned
    normally. Records the real best-epoch validation metrics and
    checkpoint path -- never fabricated."""
    best_val_cls_loss, best_val_reg_loss = _best_epoch_val_metrics(result)
    return {
        "model_name": model_name,
        "class_weight_mode": class_weight_mode,
        "status": "completed",
        "best_epoch": result.get("best_epoch"),
        "best_val_cls_loss": best_val_cls_loss,
        "best_val_reg_loss": best_val_reg_loss,
        "best_val_total_loss": result.get("best_val_loss"),
        "checkpoint_path": result.get("checkpoint_path"),
        "epochs_run": result.get("epochs_run"),
        "error_type": None,
        "error_message": None,
    }


def _failure_entry(model_name, class_weight_mode, exc):
    """Summary entry for an experiment whose train_fn(...) call raised.
    All metric/checkpoint fields are explicitly None -- never fabricated
    or inherited from a previous experiment -- and the exception type and
    message are recorded so the failure is diagnosable from the summary
    alone."""
    return {
        "model_name": model_name,
        "class_weight_mode": class_weight_mode,
        "status": "failed",
        "best_epoch": None,
        "best_val_cls_loss": None,
        "best_val_reg_loss": None,
        "best_val_total_loss": None,
        "checkpoint_path": None,
        "epochs_run": None,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def run_all_experiments(train_fn=None, log_fn=print, summary_path=None, checkpoint_dir=None):
    """Runs all six experiments sequentially, reusing ONE build_datasets()
    call's outputs across all of them. Returns the list of per-experiment
    summary dicts (also written as JSON to summary_path after EVERY
    experiment, not only at the end).

    train_fn: injectable, defaults to the real stage7_train.train_model.
    Tests pass a stub here to avoid launching real training.
    summary_path: defaults to DEFAULT_SUMMARY_PATH; tests pass a temp path.
    checkpoint_dir: forwarded to build_experiment_configs(); left at the
    TrainConfig default (root-level models/) unless explicitly overridden.

    Each experiment's train_fn(...) call is wrapped in its own
    try/except: a successful call is recorded via _success_entry() with
    status="completed" and its real metrics/checkpoint path; a raised
    exception is recorded via _failure_entry() with status="failed", the
    exception type/message, and no fabricated metrics or checkpoint path
    -- execution then CONTINUES to the next experiment rather than
    aborting the whole run. The summary is persisted to summary_path
    after every experiment, so a later experiment's failure cannot lose
    an earlier experiment's already-completed, already-persisted result.

    Never loads or references the test split -- build_datasets() itself
    already excludes it. Performs no model-selection across the six
    experiments (and does not change train_model()'s own validation-based
    is_improvement/should_stop_early selection); it only trains each and
    records its own validation metrics (or its failure). Selecting the
    single best configuration across all six, and any subsequent
    test-split evaluation, are later, separate, not-yet-authorized steps.
    """
    if train_fn is None:
        train_fn = st.train_model
    if summary_path is None:
        summary_path = DEFAULT_SUMMARY_PATH

    train_ds, val_ds, labels_train_df, preprocessing_reference = st.build_datasets()  # called exactly ONCE

    configs = build_experiment_configs(checkpoint_dir=checkpoint_dir)
    summary = []
    for config in configs:
        model_name = config.model_name
        class_weight_mode = config.class_weight_mode
        log_fn(f"Starting experiment model={model_name} class_weight_mode={class_weight_mode}")

        try:
            result = train_fn(
                config,
                train_ds=train_ds, val_ds=val_ds,
                labels_train_df=labels_train_df, preprocessing_reference=preprocessing_reference,
                log_fn=log_fn,
            )
        except Exception as exc:
            log_fn(f"Experiment model={model_name} class_weight_mode={class_weight_mode} FAILED: "
                   f"{type(exc).__name__}: {exc}")
            entry = _failure_entry(model_name, class_weight_mode, exc)
        else:
            entry = _success_entry(model_name, class_weight_mode, result)
            del result

        summary.append(entry)
        _write_summary(summary, summary_path)

        # Explicitly release this experiment's config before the next one
        # starts -- it holds no model/optimizer/DataLoader reference
        # (train_model's own locals already go out of scope on return),
        # but dropping it here and forcing a GC pass makes the guarantee
        # explicit rather than incidental.
        del config, entry
        gc.collect()

    return summary


def _write_summary(summary, summary_path):
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"experiments": summary}, f, indent=2)


if __name__ == "__main__":
    run_all_experiments()
