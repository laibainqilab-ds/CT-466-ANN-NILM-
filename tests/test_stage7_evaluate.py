"""
Tests for Stage 7 test-set evaluation (CT-466 Phase 2).

Run with:  python -m unittest tests.test_stage7_evaluate -v
(from the project root, using the project's .venv interpreter)

Metrics/threshold-selection logic is tested against hand-calculated
expected values using small synthetic datasets/models (no real parquet
files touched, no sklearn dependency). A couple of lightweight tests load
the two REAL selected checkpoints (skipped if absent) to confirm
architecture reconstruction against the actual files, without running
full inference over the full real dataset.

Phase separation (validation-only threshold selection vs test-only
evaluation) is tested explicitly: TestValidationThresholdSelection* never
constructs or references anything resembling the test split; TestEvaluateOnTest*
always supplies a hand-written frozen_thresholds dict (never one produced by
select_validation_thresholds() in the same test), to prove evaluate_on_test()
works given ANY compliant frozen thresholds and performs no selection itself.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

import stage7_evaluate as se  # noqa: E402

APPLIANCES = se.APPLIANCES
HOMES = se.HOMES


def _identity_scalers():
    return {
        "aggregate_scaler": {"mean": 0.0, "std": 1.0},
        "target_scalers": {a: {"mean": 0.0, "std": 1.0} for a in APPLIANCES},
    }


def _fake_checkpoint(model_name="MLP", class_weight_mode="unweighted"):
    return {
        "config": {"model_name": model_name, "class_weight_mode": class_weight_mode},
        "preprocessing_reference": _identity_scalers(),
    }


def _make_row(y_reg_raw, y_cls, mask_reg, mask_cls, dataid, utc_dt, endpoint_index=59):
    x = torch.zeros(60, dtype=torch.float32)
    y_reg_raw_t = torch.tensor(y_reg_raw, dtype=torch.float32)
    y_reg_t = y_reg_raw_t.clone()  # unused by run_inference's downstream logic
    y_cls_t = torch.tensor(y_cls, dtype=torch.float32)
    mask_reg_t = torch.tensor(mask_reg, dtype=torch.bool)
    mask_cls_t = torch.tensor(mask_cls, dtype=torch.bool)
    meta = {"dataid": dataid, "split": "val", "utc_dt": utc_dt, "endpoint_index": endpoint_index}
    return x, y_reg_t, y_reg_raw_t, y_cls_t, mask_reg_t, mask_cls_t, meta


class _FakeDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class _FixedOutputModel(torch.nn.Module):
    """Returns pre-specified (cls_probs, reg_out) tensors per call, in
    sequence, regardless of input -- matches the pattern already used in
    tests/test_stage7_training.py's _FixedOutputModel."""

    def __init__(self, outputs):
        super().__init__()
        self._outputs = outputs
        self._i = 0
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out = self._outputs[self._i]
        self._i += 1
        return out


# ---------------------------------------------------------------------------
# Pure metrics functions -- hand-calculated expectations
# ---------------------------------------------------------------------------

class TestConfusionCounts(unittest.TestCase):
    def test_basic_counts(self):
        y_true = np.array([1, 1, 0, 0, 1], dtype=np.float32)
        y_pred = np.array([1, 0, 0, 1, 1], dtype=np.float32)
        mask = np.array([True, True, True, True, True])
        tp, fp, tn, fn = se.confusion_counts(y_true, y_pred, mask)
        self.assertEqual((tp, fp, tn, fn), (2, 1, 1, 1))

    def test_invalid_positions_excluded_entirely(self):
        y_true = np.array([1, 1, 0, 0], dtype=np.float32)
        y_pred = np.array([1, 0, 1, 0], dtype=np.float32)
        mask = np.array([True, False, True, False])  # only indices 0,2 count
        tp, fp, tn, fn = se.confusion_counts(y_true, y_pred, mask)
        self.assertEqual((tp, fp, tn, fn), (1, 1, 0, 0))


class TestClassificationMetricsFromCounts(unittest.TestCase):
    def test_normal_case_hand_calculated(self):
        m = se.classification_metrics_from_counts(tp=3, fp=1, tn=4, fn=2)
        self.assertEqual(m["n_on"], 5)
        self.assertEqual(m["n_off"], 5)
        self.assertAlmostEqual(m["accuracy"], 7 / 10)
        self.assertAlmostEqual(m["recall_on"], 3 / 5)
        self.assertAlmostEqual(m["recall_off"], 4 / 5)
        self.assertAlmostEqual(m["precision_on"], 3 / 4)
        self.assertAlmostEqual(m["precision_off"], 4 / 6)
        self.assertAlmostEqual(m["balanced_accuracy"], (3 / 5 + 4 / 5) / 2)
        expected_f1_on = 2 * (3 / 4) * (3 / 5) / ((3 / 4) + (3 / 5))
        expected_f1_off = 2 * (4 / 6) * (4 / 5) / ((4 / 6) + (4 / 5))
        self.assertAlmostEqual(m["f1_on"], expected_f1_on)
        self.assertAlmostEqual(m["f1_off"], expected_f1_off)
        self.assertAlmostEqual(m["macro_f1"], (expected_f1_on + expected_f1_off) / 2)

    def test_absent_positive_class_yields_null_not_zero(self):
        # No actual ON samples at all (tp=0, fn=0): recall_on, balanced_accuracy,
        # f1_on and macro_f1 must all be None -- accuracy must still be defined.
        m = se.classification_metrics_from_counts(tp=0, fp=0, tn=5, fn=0)
        self.assertEqual(m["n_on"], 0)
        self.assertEqual(m["n_off"], 5)
        self.assertAlmostEqual(m["accuracy"], 1.0)
        self.assertIsNone(m["recall_on"])
        self.assertIsNone(m["precision_on"])
        self.assertIsNone(m["f1_on"])
        self.assertIsNone(m["balanced_accuracy"])
        self.assertIsNone(m["macro_f1"])

    def test_zero_predicted_positive_yields_null_precision_and_f1(self):
        # Actual ON samples exist, but the model predicted none of them.
        m = se.classification_metrics_from_counts(tp=0, fp=0, tn=3, fn=2)
        self.assertIsNone(m["precision_on"])
        self.assertIsNone(m["f1_on"])
        self.assertIsNone(m["macro_f1"])
        self.assertEqual(m["recall_on"], 0.0)  # recall IS defined (0 of 2 actual ON caught)

    def test_totally_empty_slice_is_all_null_not_an_error(self):
        m = se.classification_metrics_from_counts(tp=0, fp=0, tn=0, fn=0)
        self.assertIsNone(m["accuracy"])
        self.assertIsNone(m["balanced_accuracy"])
        self.assertIsNone(m["macro_f1"])


class TestRegressionMetrics(unittest.TestCase):
    def test_mae_rmse_hand_calculated(self):
        y_true = np.array([1.0, 2.0, 3.0, 100.0], dtype=np.float32)
        y_pred = np.array([1.5, 2.0, 5.0, -999.0], dtype=np.float32)
        mask = np.array([True, True, True, False])  # last entry excluded
        m = se.regression_metrics(y_true, y_pred, mask)
        # float32 to match the dtype regression_metrics actually computes in (y_true/y_pred above are float32)
        diffs = np.array([0.5, 0.0, 2.0], dtype=np.float32)
        self.assertAlmostEqual(m["mae"], float(np.mean(np.abs(diffs))), places=5)
        self.assertAlmostEqual(m["rmse"], float(np.sqrt(np.mean(diffs ** 2))), places=5)
        self.assertEqual(m["n_valid"], 3)

    def test_zero_valid_entries_returns_null(self):
        y_true = np.array([1.0, 2.0], dtype=np.float32)
        y_pred = np.array([9.0, 9.0], dtype=np.float32)
        mask = np.array([False, False])
        m = se.regression_metrics(y_true, y_pred, mask)
        self.assertIsNone(m["mae"])
        self.assertIsNone(m["rmse"])
        self.assertEqual(m["n_valid"], 0)


class TestCcpStatus(unittest.TestCase):
    def test_meets_both_benchmarks(self):
        m = {"accuracy": 0.95, "balanced_accuracy": 0.90, "macro_f1": 0.85}
        self.assertTrue(se.ccp_status(m, se.CCP_MINIMUM))
        self.assertTrue(se.ccp_status(m, se.CCP_HIGHER))

    def test_fails_minimum(self):
        m = {"accuracy": 0.5, "balanced_accuracy": 0.5, "macro_f1": 0.5}
        self.assertFalse(se.ccp_status(m, se.CCP_MINIMUM))

    def test_undefined_metric_yields_null_not_false(self):
        m = {"accuracy": 0.95, "balanced_accuracy": None, "macro_f1": 0.85}
        self.assertIsNone(se.ccp_status(m, se.CCP_MINIMUM))
        self.assertIsNone(se.ccp_status(m, se.CCP_HIGHER))


# ---------------------------------------------------------------------------
# Threshold selection -- deterministic tie-break
# ---------------------------------------------------------------------------

class TestSelectThresholdForAppliance(unittest.TestCase):
    def test_smallest_perfect_threshold_is_selected(self):
        # Perfectly separable: y=0 at probs {0.1,0.3}, y=1 at probs {0.6,0.9}.
        # Every grid threshold in (0.30, 0.60] yields perfect Macro-F1=1.0;
        # the deterministic tie-break must pick the SMALLEST such threshold.
        y_true = np.array([0, 0, 1, 1], dtype=np.float32)
        probs = np.array([0.1, 0.3, 0.6, 0.9], dtype=np.float32)
        mask = np.array([True, True, True, True])
        best = se.select_threshold_for_appliance(y_true, probs, mask)
        self.assertAlmostEqual(best["threshold"], 0.31)
        self.assertAlmostEqual(best["val_metrics"]["macro_f1"], 1.0)

    def test_raises_when_no_candidate_has_defined_macro_f1(self):
        # Every row is OFF -> n_on == 0 at every threshold -> macro_f1 always None.
        y_true = np.array([0, 0, 0], dtype=np.float32)
        probs = np.array([0.2, 0.5, 0.8], dtype=np.float32)
        mask = np.array([True, True, True])
        with self.assertRaises(RuntimeError):
            se.select_threshold_for_appliance(y_true, probs, mask)


# ---------------------------------------------------------------------------
# PHASE 1 -- validation-only threshold selection (never touches test split)
# ---------------------------------------------------------------------------

class TestSelectValidationThresholds(unittest.TestCase):
    def _val_rows(self):
        pattern = [(0.0, 0.0), (0.0, 0.0), (1.0, 1.0), (1.0, 1.0)]  # (y_cls value reused per appliance)
        rows = []
        for c, _ in pattern:
            rows.append(_make_row(
                y_reg_raw=(0.0, 0.0, 0.0), y_cls=(c, c, c),
                mask_reg=(True, True, True), mask_cls=(True, True, True),
                dataid="7901", utc_dt=pd.Timestamp("2018-06-01"),
            ))
        return rows

    def test_selected_thresholds_and_metrics_match_hand_calculation(self):
        probs_batch = torch.tensor([[0.1, 0.1, 0.1], [0.3, 0.3, 0.3], [0.6, 0.6, 0.6], [0.9, 0.9, 0.9]])
        reg_batch = torch.zeros(4, 3)
        fake_model = _FixedOutputModel([(probs_batch, reg_batch)])
        val_ds = _FakeDataset(self._val_rows())

        with mock.patch.object(se, "load_checkpoint", return_value=(fake_model, _fake_checkpoint())):
            frozen = se.select_validation_thresholds("dummy_checkpoint.pt", val_ds=val_ds)

        self.assertEqual(frozen["model_name"], "MLP")
        self.assertEqual(frozen["class_weight_mode"], "unweighted")
        for a in APPLIANCES:
            self.assertAlmostEqual(frozen["thresholds"][a]["threshold"], 0.31)
            self.assertAlmostEqual(frozen["thresholds"][a]["val_metrics"]["macro_f1"], 1.0)
            self.assertIn("selection_rule", frozen["thresholds"][a])

    def test_never_requests_the_test_split(self):
        probs_batch = torch.tensor([[0.1, 0.1, 0.1], [0.3, 0.3, 0.3], [0.6, 0.6, 0.6], [0.9, 0.9, 0.9]])
        reg_batch = torch.zeros(4, 3)
        fake_model = _FixedOutputModel([(probs_batch, reg_batch)])
        val_ds = _FakeDataset(self._val_rows())
        fake_checkpoint = _fake_checkpoint()

        def fake_build_split_dataset(split, agg_scaler, target_scalers):
            self.assertEqual(split, "val", "select_validation_thresholds must only ever request the val split")
            return val_ds

        with mock.patch.object(se, "load_checkpoint", return_value=(fake_model, fake_checkpoint)):
            with mock.patch.object(se, "build_split_dataset", side_effect=fake_build_split_dataset) as mocked:
                se.select_validation_thresholds("dummy_checkpoint.pt")  # no val_ds given -> must call build_split_dataset
                mocked.assert_called_once()
                self.assertEqual(mocked.call_args.args[0], "val")


# ---------------------------------------------------------------------------
# PHASE 2 -- test-only evaluation given an ALREADY-frozen thresholds dict
# ---------------------------------------------------------------------------

def _hand_written_frozen_thresholds(threshold=0.5):
    return {
        "model_name": "MLP",
        "class_weight_mode": "unweighted",
        "checkpoint_path": "irrelevant_for_this_test.pt",
        "candidate_thresholds": [threshold],
        "thresholds": {a: {"threshold": threshold, "selection_rule": "hand-written for this test",
                            "val_metrics": {}} for a in APPLIANCES},
    }


def _test_rows_two_homes():
    rows = []
    # home 7901: 4 rows, perfectly separable at threshold 0.5 -> tp=2,fp=0,tn=2,fn=0
    for c, t in zip([0.0, 0.0, 1.0, 1.0], ["2018-11-01 00:00", "2018-11-01 00:01", "2018-11-01 00:02", "2018-11-01 00:03"]):
        rows.append(_make_row(
            y_reg_raw=(0.0, 0.0, 0.0), y_cls=(c, c, c),
            mask_reg=(True, True, True), mask_cls=(True, True, True),
            dataid="7901", utc_dt=pd.Timestamp(t),
        ))
    # home 7951: 3 rows, appliance NEVER ON here (n_on == 0 for this home) -> recall_on/macro_f1 must be null
    for t in ["2018-11-01 00:10", "2018-11-01 00:11", "2018-11-01 00:12"]:
        rows.append(_make_row(
            y_reg_raw=(0.0, 0.0, 0.0), y_cls=(0.0, 0.0, 0.0),
            mask_reg=(True, True, True), mask_cls=(True, True, True),
            dataid="7951", utc_dt=pd.Timestamp(t),
        ))
    return rows


class TestEvaluateOnTest(unittest.TestCase):
    def _run(self, plots_dir=None):
        rows = _test_rows_two_homes()
        test_ds = _FakeDataset(rows)
        # probs matching the y_cls pattern above (all 3 appliance columns identical);
        # reg_out standardized=1.0 everywhere, identity scalers -> reg_pred_raw=1.0 everywhere,
        # y_reg_raw=0.0 everywhere -> diff=1.0 everywhere -> MAE=RMSE=1.0 wherever n_valid>0.
        probs_batch = torch.tensor([
            [0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.6, 0.6, 0.6], [0.9, 0.9, 0.9],
            [0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.05, 0.05, 0.05],
        ])
        reg_batch = torch.ones(7, 3)
        fake_model = _FixedOutputModel([(probs_batch, reg_batch)])
        frozen = _hand_written_frozen_thresholds(threshold=0.5)

        with mock.patch.object(se, "load_checkpoint", return_value=(fake_model, _fake_checkpoint())):
            return se.evaluate_on_test("dummy_checkpoint.pt", frozen, test_ds=test_ds, plots_dir=plots_dir)

    def test_pooled_metrics_match_hand_calculation(self):
        result = self._run()
        for a in APPLIANCES:
            pooled = result["appliances"][a]["pooled"]
            self.assertEqual(pooled["confusion"], {"tp": 2, "fp": 0, "tn": 5, "fn": 0})
            self.assertEqual(pooled["n_on"], 2)
            self.assertEqual(pooled["n_off"], 5)
            self.assertAlmostEqual(pooled["accuracy"], 1.0)
            self.assertAlmostEqual(pooled["balanced_accuracy"], 1.0)
            self.assertAlmostEqual(pooled["macro_f1"], 1.0)
            self.assertTrue(pooled["ccp_minimum_met"])
            self.assertTrue(pooled["ccp_higher_met"])
            self.assertAlmostEqual(pooled["regression"]["mae"], 1.0)
            self.assertAlmostEqual(pooled["regression"]["rmse"], 1.0)
            self.assertEqual(pooled["regression"]["n_valid"], 7)
            self.assertEqual(result["appliances"][a]["threshold_used"], 0.5)

    def test_per_home_absent_on_class_is_null_not_zero(self):
        result = self._run()
        for a in APPLIANCES:
            home_7951 = result["appliances"][a]["per_home"]["7951"]
            self.assertEqual(home_7951["n_on"], 0)
            self.assertEqual(home_7951["n_off"], 3)
            self.assertAlmostEqual(home_7951["accuracy"], 1.0)  # all 3 correctly predicted OFF
            self.assertIsNone(home_7951["balanced_accuracy"])
            self.assertIsNone(home_7951["macro_f1"])
            self.assertIsNone(home_7951["ccp_minimum_met"])
            self.assertIsNone(home_7951["ccp_higher_met"])

    def test_per_home_normal_case(self):
        result = self._run()
        for a in APPLIANCES:
            home_7901 = result["appliances"][a]["per_home"]["7901"]
            self.assertEqual(home_7901["confusion"], {"tp": 2, "fp": 0, "tn": 2, "fn": 0})
            self.assertAlmostEqual(home_7901["accuracy"], 1.0)
            self.assertAlmostEqual(home_7901["macro_f1"], 1.0)

    def test_homes_with_zero_rows_are_present_and_all_null(self):
        result = self._run()
        for a in APPLIANCES:
            for home in HOMES:
                if home in ("7901", "7951"):
                    continue
                block = result["appliances"][a]["per_home"][home]
                self.assertIsNone(block["accuracy"])
                self.assertIsNone(block["macro_f1"])
                self.assertEqual(block["regression"]["n_valid"], 0)

    def test_plots_are_written_as_valid_svg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(plots_dir=tmpdir)
            for a in APPLIANCES:
                svg_path = Path(result["plots"][a])
                self.assertTrue(svg_path.exists())
                content = svg_path.read_text(encoding="utf-8")
                self.assertIn("<svg", content)
                self.assertIn("Actual", content)
                self.assertIn("Predicted", content)
                self.assertIn(a, content)

    def test_applies_given_threshold_literally_without_reoptimizing(self):
        # A deliberately bad threshold (0.95) -- above every prediction probability
        # in the home-7901 slice -- must be APPLIED AS-IS, not silently improved.
        rows = _test_rows_two_homes()
        test_ds = _FakeDataset(rows)
        probs_batch = torch.tensor([
            [0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.6, 0.6, 0.6], [0.9, 0.9, 0.9],
            [0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.05, 0.05, 0.05],
        ])
        reg_batch = torch.ones(7, 3)
        fake_model = _FixedOutputModel([(probs_batch, reg_batch)])
        frozen = _hand_written_frozen_thresholds(threshold=0.95)

        with mock.patch.object(se, "load_checkpoint", return_value=(fake_model, _fake_checkpoint())):
            result = se.evaluate_on_test("dummy_checkpoint.pt", frozen, test_ds=test_ds)

        for a in APPLIANCES:
            pooled = result["appliances"][a]["pooled"]
            # every probability < 0.95 -> every prediction is OFF -> both true ONs are missed
            self.assertEqual(pooled["confusion"], {"tp": 0, "fp": 0, "tn": 5, "fn": 2})
            self.assertIsNone(pooled["macro_f1"])  # precision_on undefined (tp+fp == 0)


# ---------------------------------------------------------------------------
# Orchestration: freeze-before-test ordering, label/model_name safety check
# ---------------------------------------------------------------------------

class TestRunFullEvaluationOrdering(unittest.TestCase):
    def test_thresholds_written_and_reread_before_any_test_evaluation(self):
        call_order = []

        def fake_select(path, **kwargs):
            call_order.append(("select", path))
            label = "MLP_unweighted" if "MLP" in str(path) else "CNN1D_unweighted"
            model_name = "MLP" if "MLP" in str(path) else "CNN1D"
            return {
                "model_name": model_name, "class_weight_mode": "unweighted",
                "checkpoint_path": str(path), "candidate_thresholds": [0.5],
                "thresholds": {a: {"threshold": 0.5, "selection_rule": "r", "val_metrics": {}} for a in APPLIANCES},
            }

        received_by_evaluate = {}

        def fake_evaluate(path, frozen_thresholds, **kwargs):
            label = "MLP_unweighted" if "MLP" in str(path) else "CNN1D_unweighted"
            call_order.append(("evaluate", path))
            received_by_evaluate[label] = frozen_thresholds
            return {"model_name": frozen_thresholds["model_name"], "class_weight_mode": "unweighted",
                    "checkpoint_path": str(path), "appliances": {}, "plots": {}}

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(se, "select_validation_thresholds", side_effect=fake_select), \
                 mock.patch.object(se, "evaluate_on_test", side_effect=fake_evaluate):
                out = se.run_full_evaluation(
                    {"MLP_unweighted": "models/MLP_unweighted_best.pt",
                     "CNN1D_unweighted": "models/CNN1D_unweighted_best.pt"},
                    data_meta_dir=tmpdir, make_plots=False, log_fn=lambda *a, **k: None,
                )

            # all selections happen before any evaluation
            kinds = [k for k, _ in call_order]
            self.assertEqual(kinds.count("select"), 2)
            self.assertEqual(kinds.count("evaluate"), 2)
            self.assertLess(max(i for i, (k, _) in enumerate(call_order) if k == "select"),
                             min(i for i, (k, _) in enumerate(call_order) if k == "evaluate"))

            # the thresholds file on disk matches exactly what evaluate_on_test received
            with open(Path(tmpdir) / se.DEFAULT_THRESHOLDS_FILENAME) as f:
                on_disk = json.load(f)
            self.assertEqual(received_by_evaluate["MLP_unweighted"], on_disk["MLP_unweighted"])
            self.assertEqual(received_by_evaluate["CNN1D_unweighted"], on_disk["CNN1D_unweighted"])

            self.assertTrue((Path(tmpdir) / se.DEFAULT_RESULTS_FILENAME).exists())
            self.assertEqual(out["results"]["ccp_benchmarks"]["minimum"], se.CCP_MINIMUM)

    def test_label_model_name_mismatch_raises(self):
        def fake_select_mismatched(path, **kwargs):
            # claims to be a CNN1D checkpoint regardless of the label used to call it
            return {
                "model_name": "CNN1D", "class_weight_mode": "unweighted",
                "checkpoint_path": str(path), "candidate_thresholds": [0.5],
                "thresholds": {a: {"threshold": 0.5, "selection_rule": "r", "val_metrics": {}} for a in APPLIANCES},
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(se, "select_validation_thresholds", side_effect=fake_select_mismatched):
                with self.assertRaises(ValueError):
                    se.run_full_evaluation(
                        {"MLP_unweighted": "models/MLP_unweighted_best.pt"},
                        data_meta_dir=tmpdir, make_plots=False, log_fn=lambda *a, **k: None,
                    )

    def test_never_writes_to_the_six_experiment_summary_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            def fake_select(path, **kwargs):
                return {
                    "model_name": "MLP", "class_weight_mode": "unweighted",
                    "checkpoint_path": str(path), "candidate_thresholds": [0.5],
                    "thresholds": {a: {"threshold": 0.5, "selection_rule": "r", "val_metrics": {}} for a in APPLIANCES},
                }

            def fake_evaluate(path, frozen_thresholds, **kwargs):
                return {"model_name": "MLP", "class_weight_mode": "unweighted",
                        "checkpoint_path": str(path), "appliances": {}, "plots": {}}

            with mock.patch.object(se, "select_validation_thresholds", side_effect=fake_select), \
                 mock.patch.object(se, "evaluate_on_test", side_effect=fake_evaluate):
                se.run_full_evaluation({"MLP_unweighted": "models/MLP_unweighted_best.pt"},
                                        data_meta_dir=tmpdir, make_plots=False, log_fn=lambda *a, **k: None)

            self.assertFalse((Path(tmpdir) / "stage7_experiment_summary.json").exists())


# ---------------------------------------------------------------------------
# Real checkpoints -- architecture reconstruction only (skipped if absent)
# ---------------------------------------------------------------------------

class TestRealCheckpointArchitectureReconstruction(unittest.TestCase):
    def test_mlp_checkpoint_reconstructs_and_runs_forward(self):
        path = PROJECT_ROOT / "models" / "MLP_unweighted_best.pt"
        if not path.exists():
            self.skipTest(f"{path} not present")
        model, checkpoint = se.load_checkpoint(path)
        self.assertEqual(checkpoint["config"]["model_name"], "MLP")
        cls_probs, reg_out = model(torch.zeros(2, 60))
        self.assertEqual(tuple(cls_probs.shape), (2, 3))
        self.assertEqual(tuple(reg_out.shape), (2, 3))
        self.assertTrue(torch.all((cls_probs >= 0) & (cls_probs <= 1)))

    def test_cnn1d_checkpoint_reconstructs_and_runs_forward(self):
        path = PROJECT_ROOT / "models" / "CNN1D_unweighted_best.pt"
        if not path.exists():
            self.skipTest(f"{path} not present")
        model, checkpoint = se.load_checkpoint(path)
        self.assertEqual(checkpoint["config"]["model_name"], "CNN1D")
        cls_probs, reg_out = model(torch.zeros(2, 60))
        self.assertEqual(tuple(cls_probs.shape), (2, 3))
        self.assertEqual(tuple(reg_out.shape), (2, 3))
        self.assertTrue(torch.all((cls_probs >= 0) & (cls_probs <= 1)))


if __name__ == "__main__":
    unittest.main()
