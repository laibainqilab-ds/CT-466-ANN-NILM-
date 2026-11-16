"""
Tests for Stage 7 Step 4B training pipeline (CT-466 Phase 2).

Run with:  python -m unittest tests.test_stage7_training -v
(from the project root, using the project's .venv interpreter)

Uses a mix of small synthetic batches (for pure logic checks like the
validation/no-grad guarantee) and small, deterministic, TRAINING-ONLY
real-data subsets (for end-to-end checkpoint/early-stopping checks) --
never launches full training, never touches the test split.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import stage7_train as st  # noqa: E402
import stage7_dataset as ds  # noqa: E402
import stage7_losses as sl  # noqa: E402
import stage7_step4a_prototype as proto  # noqa: E402
from mlp import MLP  # noqa: E402


def make_synthetic_batch(n=8, n_app=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 60, generator=g)
    y_reg = torch.randn(n, n_app, generator=g)
    y_reg_raw = y_reg.clone() * 2.0 + 1.0  # irrelevant for these tests, just distinct values
    y_cls = torch.randint(0, 2, (n, n_app), generator=g).float()
    mask_reg = torch.ones(n, n_app, dtype=torch.bool)
    mask_cls = torch.ones(n, n_app, dtype=torch.bool)
    meta = {"dataid": ["synthetic"] * n, "split": ["train"] * n, "utc_dt": [None] * n,
            "endpoint_index": torch.arange(n)}
    return x, y_reg, y_reg_raw, y_cls, mask_reg, mask_cls, meta


class TestRunEpochValidationNeverUpdatesParameters(unittest.TestCase):
    def test_eval_mode_leaves_parameters_unchanged(self):
        torch.manual_seed(0)
        model = MLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = make_synthetic_batch()

        params_before = [p.detach().clone() for p in model.parameters()]
        st.run_epoch(model, [batch], optimizer, lambda_cls=1.0, lambda_reg=1.0, pos_weight=None, train=False)
        params_after = list(model.parameters())

        for before, after in zip(params_before, params_after):
            self.assertTrue(torch.equal(before, after.detach()))

    def test_eval_mode_does_not_populate_optimizer_state(self):
        torch.manual_seed(0)
        model = MLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = make_synthetic_batch()

        st.run_epoch(model, [batch], optimizer, lambda_cls=1.0, lambda_reg=1.0, pos_weight=None, train=False)
        self.assertEqual(len(optimizer.state), 0)

    def test_eval_mode_leaves_grad_none(self):
        torch.manual_seed(0)
        model = MLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = make_synthetic_batch()

        st.run_epoch(model, [batch], optimizer, lambda_cls=1.0, lambda_reg=1.0, pos_weight=None, train=False)
        for p in model.parameters():
            self.assertIsNone(p.grad)


class TestRunEpochTrainingUpdatesParameters(unittest.TestCase):
    def test_train_mode_changes_parameters(self):
        torch.manual_seed(0)
        model = MLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        batch = make_synthetic_batch()

        params_before = [p.detach().clone() for p in model.parameters()]
        st.run_epoch(model, [batch], optimizer, lambda_cls=1.0, lambda_reg=1.0, pos_weight=None, train=True)
        params_after = list(model.parameters())

        self.assertTrue(any(not torch.equal(b, a.detach()) for b, a in zip(params_before, params_after)))

    def test_returned_metrics_are_finite(self):
        torch.manual_seed(0)
        model = MLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = make_synthetic_batch()
        metrics = st.run_epoch(model, [batch], optimizer, 1.0, 1.0, None, train=True)
        for v in metrics.values():
            self.assertTrue(np.isfinite(v))


class TestEarlyStoppingLogic(unittest.TestCase):
    def test_is_improvement(self):
        self.assertTrue(st.is_improvement(0.5, 1.0))
        self.assertFalse(st.is_improvement(1.0, 1.0))
        self.assertFalse(st.is_improvement(1.5, 1.0))

    def test_should_stop_early(self):
        self.assertFalse(st.should_stop_early(2, patience=3))
        self.assertTrue(st.should_stop_early(3, patience=3))
        self.assertTrue(st.should_stop_early(5, patience=3))

    def test_simulated_training_loop_stops_at_expected_epoch(self):
        # Simulate a sequence of validation losses with no improvement
        # after epoch 1, patience=2 -> should stop after epoch 3 (indices 0,1,2,3).
        val_losses = [1.0, 0.8, 0.8, 0.8, 0.8, 0.8]
        best = float("inf")
        epochs_without_improvement = 0
        stopped_at = None
        for epoch, v in enumerate(val_losses):
            if st.is_improvement(v, best):
                best = v
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if st.should_stop_early(epochs_without_improvement, patience=2):
                stopped_at = epoch
                break
        self.assertEqual(stopped_at, 3)


class TestClassWeightConfigurations(unittest.TestCase):
    def test_unweighted_mode_returns_none(self):
        import pandas as pd
        df = pd.DataFrame({"split": ["train"] * 4,
                            "air1_valid": [True] * 4, "air1_label": pd.array([1, 0, 0, 0], dtype="Int64"),
                            "dishwasher1_valid": [True] * 4, "dishwasher1_label": pd.array([1, 0, 0, 0], dtype="Int64"),
                            "refrigerator1_valid": [True] * 4, "refrigerator1_label": pd.array([1, 0, 0, 0], dtype="Int64")})
        result = st.compute_pos_weight_tensor(df, "unweighted")
        self.assertIsNone(result)

    def test_cap20_mode_caps_at_20(self):
        import pandas as pd
        # raw ratio 99 for each appliance (1 positive, 99 negative)
        n_pos, n_off = 1, 99
        df = pd.DataFrame({
            "split": ["train"] * (n_pos + n_off),
            **{f"{a}_valid": [True] * (n_pos + n_off) for a in ["air1", "dishwasher1", "refrigerator1"]},
            **{f"{a}_label": pd.array([1] * n_pos + [0] * n_off, dtype="Int64") for a in ["air1", "dishwasher1", "refrigerator1"]},
        })
        result = st.compute_pos_weight_tensor(df, "cap20")
        self.assertIsNotNone(result)
        self.assertTrue(torch.all(result <= 20.0))
        self.assertTrue(torch.all(result == 20.0))  # raw ratio 99 > 20, so capped exactly at 20

    def test_cap50_mode_caps_at_50(self):
        import pandas as pd
        n_pos, n_off = 1, 99
        df = pd.DataFrame({
            "split": ["train"] * (n_pos + n_off),
            **{f"{a}_valid": [True] * (n_pos + n_off) for a in ["air1", "dishwasher1", "refrigerator1"]},
            **{f"{a}_label": pd.array([1] * n_pos + [0] * n_off, dtype="Int64") for a in ["air1", "dishwasher1", "refrigerator1"]},
        })
        result = st.compute_pos_weight_tensor(df, "cap50")
        self.assertTrue(torch.all(result == 50.0))

    def test_cap_has_no_effect_below_threshold(self):
        import pandas as pd
        n_pos, n_off = 10, 20  # raw ratio 2.0, below both caps
        df = pd.DataFrame({
            "split": ["train"] * (n_pos + n_off),
            **{f"{a}_valid": [True] * (n_pos + n_off) for a in ["air1", "dishwasher1", "refrigerator1"]},
            **{f"{a}_label": pd.array([1] * n_pos + [0] * n_off, dtype="Int64") for a in ["air1", "dishwasher1", "refrigerator1"]},
        })
        r20 = st.compute_pos_weight_tensor(df, "cap20")
        r50 = st.compute_pos_weight_tensor(df, "cap50")
        torch.testing.assert_close(r20, r50)
        self.assertTrue(torch.all(r20 == 2.0))

    def test_invalid_mode_raises(self):
        import pandas as pd
        df = pd.DataFrame({"split": ["train"]})
        with self.assertRaises(ValueError):
            st.compute_pos_weight_tensor(df, "bogus_mode")


class TestMemoryUsageMeasurement(unittest.TestCase):
    """Regression tests for the Windows GetProcessMemoryInfo ctypes defect
    found during the Stage 7 Step 4C feasibility assessment: without
    explicit argtypes/restype, the Windows API call failed silently and
    get_memory_usage_mb() always returned None on this machine. Fixed in
    Step 4D."""

    def test_returns_a_value_not_none_on_this_platform(self):
        result = st.get_memory_usage_mb()
        self.assertIsNotNone(result, "get_memory_usage_mb() returned None -- regression of the Step 4C defect")

    def test_returned_value_is_a_positive_float(self):
        result = st.get_memory_usage_mb()
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.0)

    def test_returned_value_is_plausible(self):
        # Sanity bound, not a tight assertion: a running test process uses
        # well over 1 MB and well under 50 GB of working-set memory.
        result = st.get_memory_usage_mb()
        self.assertGreater(result, 1.0)
        self.assertLess(result, 50_000.0)

    def test_repeated_calls_are_consistent_in_type_and_sign(self):
        for _ in range(3):
            result = st.get_memory_usage_mb()
            self.assertIsInstance(result, float)
            self.assertGreater(result, 0.0)


class TestBuildDatasetsExcludesTestSplit(unittest.TestCase):
    """Stage 7 Step 5A: build_datasets() must exclude test-period data at
    read time (pyarrow row filter + runtime guard), not merely fail to use
    it. Verified against the real pipeline -- fast (train+val only, no
    test data loaded), consistent with every other real-data test in this
    suite."""

    @classmethod
    def setUpClass(cls):
        cls.train_ds, cls.val_ds, cls.labels_train_df, cls.preprocessing_reference = st.build_datasets()

    def test_agg_lookup_test_entries_are_empty_for_every_home(self):
        for home in ds.HOMES:
            test_arr = self.train_ds.agg_lookup[(home, "test")]
            self.assertEqual(len(test_arr), 0,
                              f"agg_lookup[({home!r}, 'test')] is not empty -- test aggregate data leaked in")

    DEV_HOMES = ("7901", "7951", "8565", "3538")
    HELD_OUT_HOME = "8386"

    def test_agg_lookup_train_and_val_entries_are_nonempty_for_development_homes(self):
        for home in self.DEV_HOMES:
            self.assertGreater(len(self.train_ds.agg_lookup[(home, "train")]), 0)
            self.assertGreater(len(self.train_ds.agg_lookup[(home, "val")]), 0)

    def test_agg_lookup_train_and_val_entries_are_empty_for_held_out_home(self):
        # Household 8386 is the official unseen test household under the
        # revised evaluation design -- it contributes zero train/val rows,
        # by construction, not merely "untested" by the loop above.
        self.assertEqual(len(self.train_ds.agg_lookup[(self.HELD_OUT_HOME, "train")]), 0)
        self.assertEqual(len(self.train_ds.agg_lookup[(self.HELD_OUT_HOME, "val")]), 0)

    def test_train_and_val_window_counts_match_regenerated_revised_split(self):
        # Pre-established, independently-verified exact counts for the
        # REVISED split (four development homes only, Jan-Sep train /
        # Oct val) -- confirms no regression against the Stage 5B/6
        # verification already performed on the regenerated data. These
        # differ from the original 5-home-split counts (392516/222905) by
        # design: household 8386 no longer contributes to train/val.
        self.assertEqual(len(self.train_ds), 313938)
        self.assertEqual(len(self.val_ds), 178324)

    def test_train_ds_and_val_ds_share_the_same_agg_lookup_object(self):
        self.assertIs(self.train_ds.agg_lookup, self.val_ds.agg_lookup)

    def test_labels_train_df_contains_only_train_rows(self):
        self.assertTrue((self.labels_train_df["split"] == "train").all())

    def test_labels_train_df_excludes_held_out_household(self):
        # The official unseen test household (8386) must never appear in
        # the training labels used to fit preprocessing/compute pos-weights.
        self.assertNotIn("8386", set(self.labels_train_df["dataid"].unique()))
        self.assertEqual(set(self.labels_train_df["dataid"].unique()), set(self.DEV_HOMES))


class TestTrainConfigValidation(unittest.TestCase):
    def test_checkpoint_dir_defaults_to_root_level_models_not_src_models(self):
        cfg = st.TrainConfig()
        self.assertEqual(cfg.checkpoint_dir, PROJECT_ROOT / "models")
        self.assertNotIn("src", cfg.checkpoint_dir.parts)

    def test_rejects_unknown_model_name(self):
        with self.assertRaises(ValueError):
            st.TrainConfig(model_name="NOT_A_MODEL")

    def test_rejects_unknown_class_weight_mode(self):
        with self.assertRaises(ValueError):
            st.TrainConfig(class_weight_mode="not_a_mode")

    def test_rejects_non_cpu_device(self):
        with self.assertRaises(ValueError):
            st.TrainConfig(device="cuda")

    def test_explicit_root_level_alternative_is_accepted(self):
        # An explicit, different root-level (outside src/models/) directory
        # must be accepted -- the safety check is about src/models/, not
        # about forbidding any non-default path.
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = st.TrainConfig(checkpoint_dir=Path(tmpdir) / "alt_checkpoints")
            self.assertEqual(cfg.checkpoint_dir, Path(tmpdir) / "alt_checkpoints")

    def test_rejects_src_models_directory(self):
        with self.assertRaises(ValueError):
            st.TrainConfig(checkpoint_dir=PROJECT_ROOT / "src" / "models")

    def test_rejects_nested_path_inside_src_models_directory(self):
        with self.assertRaises(ValueError):
            st.TrainConfig(checkpoint_dir=PROJECT_ROOT / "src" / "models" / "nested" / "subdir")


class TestTrainModelEndToEndSmallSubset(unittest.TestCase):
    """Small, deterministic, TRAINING-ONLY (+validation, never test) real-data
    subset -- not full training. Uses a temporary checkpoint directory so no
    real artifact is left under the project's actual models/ directory."""

    @classmethod
    def setUpClass(cls):
        full_train_ds, full_val_ds, labels_train_df, preprocessing_reference = st.build_datasets()
        train_indices = proto.build_small_deterministic_train_subset(full_train_ds, per_home=20)
        # small, deterministic val subset: first 20 rows per home from val
        val_indices = []
        for home in ds.HOMES:
            home_idx = np.where(full_val_ds.dataid == home)[0]
            val_indices.extend(home_idx[:20].tolist())
        val_indices = sorted(val_indices)

        cls.train_subset = torch.utils.data.Subset(full_train_ds, train_indices)
        cls.val_subset = torch.utils.data.Subset(full_val_ds, val_indices)
        cls.labels_train_df = labels_train_df
        cls.preprocessing_reference = preprocessing_reference

    def test_checkpoint_created_with_required_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = st.TrainConfig(model_name="MLP", batch_size=16, epochs=3, patience=5,
                                     checkpoint_dir=Path(tmpdir))
            result = st.train_model(config, train_ds=self.train_subset, val_ds=self.val_subset,
                                     labels_train_df=self.labels_train_df,
                                     preprocessing_reference=self.preprocessing_reference,
                                     log_fn=lambda *a, **k: None)

            self.assertIsNotNone(result["checkpoint_path"])
            self.assertTrue(Path(result["checkpoint_path"]).exists())
            # checkpoint must be under the TEMP root passed in, not under src/models
            self.assertNotIn("src", Path(result["checkpoint_path"]).parts)

            checkpoint = torch.load(result["checkpoint_path"], weights_only=False)
            for key in ["model_state_dict", "optimizer_state_dict", "epoch", "config",
                        "preprocessing_reference", "validation_metrics"]:
                self.assertIn(key, checkpoint)

    def test_training_runs_for_at_most_configured_epochs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = st.TrainConfig(model_name="MLP", batch_size=16, epochs=3, patience=100,
                                     checkpoint_dir=Path(tmpdir))
            result = st.train_model(config, train_ds=self.train_subset, val_ds=self.val_subset,
                                     labels_train_df=self.labels_train_df,
                                     preprocessing_reference=self.preprocessing_reference,
                                     log_fn=lambda *a, **k: None)
            self.assertLessEqual(result["epochs_run"], 3)
            self.assertEqual(len(result["history"]), result["epochs_run"])

    def test_all_three_class_weight_modes_run_successfully(self):
        for mode in st.CLASS_WEIGHT_MODES:
            with tempfile.TemporaryDirectory() as tmpdir:
                config = st.TrainConfig(model_name="MLP", batch_size=16, epochs=2, patience=5,
                                         class_weight_mode=mode, checkpoint_dir=Path(tmpdir))
                result = st.train_model(config, train_ds=self.train_subset, val_ds=self.val_subset,
                                         labels_train_df=self.labels_train_df,
                                         preprocessing_reference=self.preprocessing_reference,
                                         log_fn=lambda *a, **k: None)
                self.assertTrue(np.isfinite(result["best_val_loss"]))
                self.assertIsNotNone(result["checkpoint_path"])

    def test_cnn1d_also_runs_successfully(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = st.TrainConfig(model_name="CNN1D", batch_size=16, epochs=2, patience=5,
                                     checkpoint_dir=Path(tmpdir))
            result = st.train_model(config, train_ds=self.train_subset, val_ds=self.val_subset,
                                     labels_train_df=self.labels_train_df,
                                     preprocessing_reference=self.preprocessing_reference,
                                     log_fn=lambda *a, **k: None)
            self.assertTrue(np.isfinite(result["best_val_loss"]))


class _FixedOutputModel(torch.nn.Module):
    """Returns pre-specified (cls_probs, reg_out) tensors per call, in
    sequence, regardless of the input x -- lets tests control model output
    exactly so epoch-level aggregation can be hand-verified precisely.
    Has one real parameter so it behaves like a normal nn.Module for
    .parameters()/optimizer purposes."""

    def __init__(self, outputs):
        super().__init__()
        self._outputs = outputs
        self._i = 0
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        cls_probs, reg_out = self._outputs[self._i]
        self._i += 1
        return cls_probs, reg_out


def _make_batch(n, cls_probs, y_cls, mask_cls, reg_out, y_reg, mask_reg):
    x = torch.zeros(n, 60)
    y_reg_raw = y_reg.clone()
    meta = {"dataid": ["x"] * n, "split": ["train"] * n, "utc_dt": [None] * n, "endpoint_index": torch.arange(n)}
    return x, y_reg, y_reg_raw, y_cls, mask_reg, mask_cls, meta


class TestEpochLevelAggregationCorrectness(unittest.TestCase):
    """Independently hand-calculated reference for epoch-level loss
    aggregation across two UNEVEN-sized batches (3 then 1) with different
    missing-label patterns per appliance, including one appliance that is
    entirely invalid within a single batch (but valid in the other)."""

    def setUp(self):
        import math
        self.math = math

        # --- classification batches ---
        # Batch 1 (n=3): appliance0 all valid; appliance1 only 1/3 valid;
        # appliance2 entirely invalid in this batch.
        cls_probs_b1 = torch.tensor([[0.9, 0.0, 0.0], [0.2, 0.0, 0.0], [0.6, 0.7, 0.0]])
        y_cls_b1 = torch.tensor([[1.0, float("nan"), float("nan")],
                                  [0.0, float("nan"), float("nan")],
                                  [1.0, 1.0, float("nan")]])
        mask_cls_b1 = torch.tensor([[True, False, False], [True, False, False], [True, True, False]])

        # Batch 2 (n=1, the smaller/"final" batch): all three appliances valid.
        cls_probs_b2 = torch.tensor([[0.4, 0.3, 0.8]])
        y_cls_b2 = torch.tensor([[0.0, 0.0, 1.0]])
        mask_cls_b2 = torch.tensor([[True, True, True]])

        # --- regression batches (independent synthetic values) ---
        reg_out_b1 = torch.tensor([[1.0, 5.0, 0.0], [3.0, 0.0, 0.0]])
        y_reg_b1 = torch.tensor([[1.5, 4.0, float("nan")], [2.0, float("nan"), float("nan")]])
        mask_reg_b1 = torch.tensor([[True, True, False], [True, False, False]])

        reg_out_b2 = torch.tensor([[2.0, 0.0, 10.0]])
        y_reg_b2 = torch.tensor([[2.5, 1.0, 9.0]])
        mask_reg_b2 = torch.tensor([[True, True, True]])

        self.batch1 = _make_batch(3, cls_probs_b1, y_cls_b1, mask_cls_b1, reg_out_b1, y_reg_b1, mask_reg_b1)
        self.batch2 = _make_batch(1, cls_probs_b2, y_cls_b2, mask_cls_b2, reg_out_b2, y_reg_b2, mask_reg_b2)
        self.model = _FixedOutputModel([(cls_probs_b1, reg_out_b1), (cls_probs_b2, reg_out_b2)])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def test_classification_epoch_loss_matches_hand_calculation(self):
        log = self.math.log
        # appliance0: 4 valid entries total (3 in batch1 + 1 in batch2)
        sum_app0 = (-log(0.9)) + (-log(0.8)) + (-log(0.6)) + (-log(0.6))
        mean_app0 = sum_app0 / 4
        # appliance1: 2 valid entries total (1 in batch1 + 1 in batch2)
        sum_app1 = (-log(0.7)) + (-log(0.7))
        mean_app1 = sum_app1 / 2
        # appliance2: 1 valid entry total (0 in batch1 + 1 in batch2)
        sum_app2 = (-log(0.8))
        mean_app2 = sum_app2 / 1
        expected_epoch_cls = (mean_app0 + mean_app1 + mean_app2) / 3

        result = st.run_epoch(self.model, [self.batch1, self.batch2], self.optimizer,
                               lambda_cls=1.0, lambda_reg=1.0, pos_weight=None, train=False)
        self.assertAlmostEqual(result["cls_loss"], expected_epoch_cls, places=5)

        # Sanity: this must differ from the (incorrect) naive mean-of-per-batch-scalars
        naive_batch1 = sl.masked_bce_loss(torch.tensor([[0.9, 0.0, 0.0], [0.2, 0.0, 0.0], [0.6, 0.7, 0.0]]),
                                           torch.tensor([[1.0, float("nan"), float("nan")],
                                                         [0.0, float("nan"), float("nan")],
                                                         [1.0, 1.0, float("nan")]]),
                                           torch.tensor([[True, False, False], [True, False, False], [True, True, False]])).item()
        naive_batch2 = sl.masked_bce_loss(torch.tensor([[0.4, 0.3, 0.8]]), torch.tensor([[0.0, 0.0, 1.0]]),
                                           torch.tensor([[True, True, True]])).item()
        naive_mean = (naive_batch1 + naive_batch2) / 2
        self.assertNotAlmostEqual(result["cls_loss"], naive_mean, places=3)

    def test_regression_epoch_loss_matches_hand_calculation(self):
        # appliance0: 3 valid total (2 in batch1 + 1 in batch2)
        sum_app0 = (0.5 ** 2) + (1.0 ** 2) + (0.5 ** 2)
        mean_app0 = sum_app0 / 3
        # appliance1: 2 valid total (1 in batch1 + 1 in batch2)
        sum_app1 = (1.0 ** 2) + (1.0 ** 2)
        mean_app1 = sum_app1 / 2
        # appliance2: 1 valid total (0 in batch1 + 1 in batch2)
        sum_app2 = (1.0 ** 2)
        mean_app2 = sum_app2 / 1
        expected_epoch_reg = (mean_app0 + mean_app1 + mean_app2) / 3

        model = _FixedOutputModel([
            (torch.tensor([[0.9, 0.0, 0.0], [0.2, 0.0, 0.0], [0.6, 0.7, 0.0]]), torch.tensor([[1.0, 5.0, 0.0], [3.0, 0.0, 0.0]])),
            (torch.tensor([[0.4, 0.3, 0.8]]), torch.tensor([[2.0, 0.0, 10.0]])),
        ])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        result = st.run_epoch(model, [self.batch1, self.batch2], optimizer,
                               lambda_cls=1.0, lambda_reg=1.0, pos_weight=None, train=False)
        self.assertAlmostEqual(result["reg_loss"], expected_epoch_reg, places=5)

    def test_total_loss_equals_weighted_sum_of_corrected_epoch_losses(self):
        result = st.run_epoch(self.model, [self.batch1, self.batch2], self.optimizer,
                               lambda_cls=2.0, lambda_reg=0.5, pos_weight=None, train=False)
        expected_total = 2.0 * result["cls_loss"] + 0.5 * result["reg_loss"]
        self.assertAlmostEqual(result["total_loss"], expected_total, places=6)

    def test_appliance_invalid_for_entire_epoch_is_excluded_not_zeroed(self):
        # appliance 2 invalid in BOTH batches for classification -> must be
        # excluded from the appliance average (denominator 2, not 3), and
        # must not produce NaN/crash.
        cls_probs_b1 = torch.tensor([[0.9, 0.5, 0.0], [0.3, 0.6, 0.0]])
        y_cls_b1 = torch.tensor([[1.0, 0.0, float("nan")], [0.0, 1.0, float("nan")]])
        mask_cls_b1 = torch.tensor([[True, True, False], [True, True, False]])
        cls_probs_b2 = torch.tensor([[0.7, 0.2, 0.0]])
        y_cls_b2 = torch.tensor([[1.0, 0.0, float("nan")]])
        mask_cls_b2 = torch.tensor([[True, True, False]])

        reg_out_dummy = torch.zeros(2, 3)
        y_reg_dummy = torch.zeros(2, 3)
        mask_reg_dummy = torch.zeros(2, 3, dtype=torch.bool)
        reg_out_dummy2 = torch.zeros(1, 3)
        y_reg_dummy2 = torch.zeros(1, 3)
        mask_reg_dummy2 = torch.zeros(1, 3, dtype=torch.bool)

        batch1 = _make_batch(2, cls_probs_b1, y_cls_b1, mask_cls_b1, reg_out_dummy, y_reg_dummy, mask_reg_dummy)
        batch2 = _make_batch(1, cls_probs_b2, y_cls_b2, mask_cls_b2, reg_out_dummy2, y_reg_dummy2, mask_reg_dummy2)
        model = _FixedOutputModel([(cls_probs_b1, reg_out_dummy), (cls_probs_b2, reg_out_dummy2)])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        result = st.run_epoch(model, [batch1, batch2], optimizer, lambda_cls=1.0, lambda_reg=1.0,
                               pos_weight=None, train=False)
        self.assertTrue(np.isfinite(result["cls_loss"]))

        log = self.math.log
        # appliance0 (col 0): row0 y=1,p=0.9; row1 y=0,p=0.3; batch2 y=1,p=0.7
        mean_app0 = ((-log(0.9)) + (-log(1 - 0.3)) + (-log(0.7))) / 3
        # appliance1 (col 1): row0 y=0,p=0.5; row1 y=1,p=0.6; batch2 y=0,p=0.2
        mean_app1 = ((-log(1 - 0.5)) + (-log(0.6)) + (-log(1 - 0.2))) / 3
        expected = (mean_app0 + mean_app1) / 2  # appliance2 excluded entirely
        self.assertAlmostEqual(result["cls_loss"], expected, places=5)

    def test_training_mode_still_uses_official_batch_level_loss_for_backward(self):
        """Gradients/optimizer updates must still come from the UNCHANGED
        stage7_losses batch-level scalars, not the corrected epoch-level
        aggregation (which only affects reporting)."""
        model = MLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        batch_a = make_synthetic_batch(n=10, seed=1)
        batch_b = make_synthetic_batch(n=3, seed=2)  # uneven smaller second batch

        params_before = [p.detach().clone() for p in model.parameters()]
        st.run_epoch(model, [batch_a, batch_b], optimizer, 1.0, 1.0, None, train=True)
        self.assertTrue(any(not torch.equal(b, a.detach()) for b, a in zip(params_before, model.parameters())))

    def test_validation_with_uneven_batches_still_never_updates_parameters(self):
        model = MLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        batch_a = make_synthetic_batch(n=10, seed=1)
        batch_b = make_synthetic_batch(n=3, seed=2)

        params_before = [p.detach().clone() for p in model.parameters()]
        st.run_epoch(model, [batch_a, batch_b], optimizer, 1.0, 1.0, None, train=False)
        for before, after in zip(params_before, model.parameters()):
            self.assertTrue(torch.equal(before, after.detach()))
        for p in model.parameters():
            self.assertIsNone(p.grad)


if __name__ == "__main__":
    unittest.main()
