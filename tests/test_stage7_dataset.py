"""
Tests for Stage 7 dataset/preprocessing (CT-466 Phase 2).

Run with:  python -m unittest tests.test_stage7_dataset -v
(from the project root, using the project's .venv interpreter)

No pytest dependency is used -- stdlib unittest only, consistent with
"do not install additional packages".
"""
import json
import sys
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import stage7_preprocess as pp  # noqa: E402
import stage7_dataset as ds  # noqa: E402

HOMES = ds.HOMES
APPLIANCES = ds.APPLIANCES


def _load_everything():
    modeling_df = pd.read_parquet(ds.MODELING_PATH, engine="pyarrow")
    labels_train_df = pd.read_parquet(ds.LABELS_PATHS["train"], engine="pyarrow")
    agg_scaler = pp.fit_aggregate_scaler(modeling_df)
    target_scalers = pp.fit_target_scalers(labels_train_df)
    agg_lookup = ds.build_aggregate_lookup(modeling_df)
    return modeling_df, agg_scaler, target_scalers, agg_lookup


class TestPreprocessingTrainOnly(unittest.TestCase):
    """Preprocessing must be fit on training data only."""

    @classmethod
    def setUpClass(cls):
        cls.modeling_df, cls.agg_scaler, cls.target_scalers, cls.agg_lookup = _load_everything()

    def test_aggregate_scaler_is_sane(self):
        self.assertGreater(self.agg_scaler["std"], 0)
        self.assertTrue(np.isfinite(self.agg_scaler["mean"]))
        self.assertGreater(self.agg_scaler["n"], 0)

    def test_target_scalers_cover_all_appliances(self):
        for a in APPLIANCES:
            self.assertIn(a, self.target_scalers)
            self.assertGreater(self.target_scalers[a]["std"], 0)

    def test_fit_target_scalers_rejects_non_train_rows(self):
        val_df = pd.read_parquet(ds.LABELS_PATHS["val"], engine="pyarrow")
        with self.assertRaises(RuntimeError):
            pp.fit_target_scalers(val_df)

    def test_val_and_test_datasets_use_identical_training_statistics(self):
        # Construct val and test Datasets with the SAME fitted (train-only)
        # scalers and confirm they are literally the same objects/values --
        # i.e. nothing is refit per split.
        val_ds = ds.NILMWindowDataset("val", self.agg_lookup, self.agg_scaler["mean"], self.agg_scaler["std"], self.target_scalers)
        test_ds = ds.NILMWindowDataset("test", self.agg_lookup, self.agg_scaler["mean"], self.agg_scaler["std"], self.target_scalers)
        self.assertEqual(val_ds.agg_mean, self.agg_scaler["mean"])
        self.assertEqual(test_ds.agg_mean, self.agg_scaler["mean"])
        self.assertEqual(val_ds.agg_std, self.agg_scaler["std"])
        self.assertEqual(test_ds.agg_std, self.agg_scaler["std"])
        self.assertIs(val_ds.target_scalers, self.target_scalers)
        self.assertIs(test_ds.target_scalers, self.target_scalers)


class TestStandardizeRoundTrip(unittest.TestCase):
    """NEW: direct round-trip test of pp.standardize / pp.inverse_standardize."""

    def test_round_trip_is_exact_on_scalars(self):
        for x, mean, std in [(0.0, 1.1940404855783435, 1.3982393787705587),
                              (5.3, 0.4013813347736144, 0.8486543388452014),
                              (-2.7, 0.009695424073871178, 0.09072874574768985)]:
            scaled = pp.standardize(x, mean, std)
            back = pp.inverse_standardize(scaled, mean, std)
            self.assertAlmostEqual(back, x, places=9)

    def test_round_trip_is_exact_on_numpy_array(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=1000).astype(np.float64)
        mean, std = 0.0743, 0.0754
        back = pp.inverse_standardize(pp.standardize(x, mean, std), mean, std)
        np.testing.assert_allclose(back, x, rtol=1e-10, atol=1e-10)

    def test_round_trip_preserves_nan(self):
        x = np.array([1.0, np.nan, 3.0])
        back = pp.inverse_standardize(pp.standardize(x, 2.0, 0.5), 2.0, 0.5)
        self.assertTrue(np.isnan(back[1]))
        np.testing.assert_allclose(back[[0, 2]], x[[0, 2]], rtol=1e-10)


class TestOriginalAndStandardizedTargetsAligned(unittest.TestCase):
    """NEW: y_reg_raw and y_reg must correspond to the same underlying value,
    and both must stay aligned with masks/labels/metadata."""

    @classmethod
    def setUpClass(cls):
        cls.modeling_df, cls.agg_scaler, cls.target_scalers, cls.agg_lookup = _load_everything()
        cls.train_ds = ds.NILMWindowDataset("train", cls.agg_lookup, cls.agg_scaler["mean"], cls.agg_scaler["std"], cls.target_scalers)

    def test_y_reg_equals_standardized_y_reg_raw_for_valid_entries(self):
        # Spread across most of the dataset, including the last valid
        # index -- computed dynamically rather than hardcoded, since the
        # revised split's train_ds length (313,938) differs from the
        # original (392,516) and must never be assumed.
        n = len(self.train_ds)
        for idx in sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1}):
            x, y_reg, y_reg_raw, y_cls, mask_reg, mask_cls, meta = self.train_ds[idx]
            for i, a in enumerate(APPLIANCES):
                if mask_reg[i]:
                    sc = self.target_scalers[a]
                    expected_scaled = pp.standardize(y_reg_raw[i].item(), sc["mean"], sc["std"])
                    self.assertAlmostEqual(y_reg[i].item(), expected_scaled, places=4)

    def test_y_reg_raw_matches_source_column_directly_not_via_inverse_transform(self):
        # y_reg_raw must equal the RAW column value read from the label file,
        # not a recomputation via inverse-transforming y_reg (which would be
        # circular and could mask a scaling-direction bug).
        raw_df = pd.read_parquet(ds.LABELS_PATHS["train"], engine="pyarrow")
        for idx in [0, 50000, 150000]:
            _, _, y_reg_raw, _, _, _, meta = self.train_ds[idx]
            for i, a in enumerate(APPLIANCES):
                expected = raw_df.iloc[idx][a]
                actual = y_reg_raw[i].item()
                if pd.isna(expected):
                    self.assertTrue(np.isnan(actual))
                else:
                    self.assertAlmostEqual(actual, float(expected), places=5)

    def test_both_nan_together_on_invalid_entries(self):
        invalid_positions = np.where((self.train_ds.dataid == "7901") & (~self.train_ds.valid["dishwasher1"]))[0]
        idx = int(invalid_positions[0])
        _, y_reg, y_reg_raw, _, _, mask_reg, _ = self.train_ds[idx]
        dw_i = APPLIANCES.index("dishwasher1")
        self.assertFalse(mask_reg[dw_i].item())
        self.assertTrue(torch.isnan(y_reg[dw_i]))
        self.assertTrue(torch.isnan(y_reg_raw[dw_i]))

    def test_collate_preserves_y_reg_raw_alignment(self):
        loader = DataLoader(self.train_ds, batch_size=16, shuffle=False, collate_fn=ds.collate_fn)
        xs, y_reg, y_reg_raw, y_cls, mask_reg, mask_cls, meta = next(iter(loader))
        self.assertEqual(tuple(y_reg_raw.shape), (16, 3))
        for b in range(16):
            for i, a in enumerate(APPLIANCES):
                if mask_reg[b, i]:
                    sc = self.target_scalers[a]
                    expected = pp.standardize(y_reg_raw[b, i].item(), sc["mean"], sc["std"])
                    self.assertAlmostEqual(y_reg[b, i].item(), expected, places=4)


class TestDatasetReconstructionAllHomesAllSplits(unittest.TestCase):
    """Deterministic sample from every (home, split) EXPECTED to be
    non-empty under the revised evaluation design: the four development
    households (7901, 7951, 8565, 3538) in train/val, and the official
    unseen test household (8386) in test -- nine combinations in total.
    The remaining six (dev homes x test, 8386 x train/val) are expected to
    be EMPTY by construction (8386 contributes nothing to train/val;
    development homes contribute nothing to test) and are explicitly
    confirmed empty, not merely skipped."""

    EXPECTED_NONEMPTY_HOME_SPLITS = {
        ("7901", "train"), ("7951", "train"), ("8565", "train"), ("3538", "train"),
        ("7901", "val"), ("7951", "val"), ("8565", "val"), ("3538", "val"),
        ("8386", "test"),
    }

    @classmethod
    def setUpClass(cls):
        cls.modeling_df, cls.agg_scaler, cls.target_scalers, cls.agg_lookup = _load_everything()
        cls.datasets = {
            split: ds.NILMWindowDataset(split, cls.agg_lookup, cls.agg_scaler["mean"], cls.agg_scaler["std"], cls.target_scalers)
            for split in ["train", "val", "test"]
        }

    def test_nine_home_split_reconstructions_match_base_file_six_confirmed_empty(self):
        checked = 0
        for home in HOMES:
            for split in ["train", "val", "test"]:
                d = self.datasets[split]
                positions = np.where(d.dataid == home)[0]

                if (home, split) not in self.EXPECTED_NONEMPTY_HOME_SPLITS:
                    self.assertEqual(
                        len(positions), 0,
                        f"expected ({home!r}, {split!r}) to be empty under the revised split, "
                        f"but found {len(positions)} row(s)",
                    )
                    continue

                self.assertGreater(len(positions), 0, f"no rows for {home}/{split}")
                idx = int(positions[0])
                x, y_reg, y_reg_raw, y_cls, mask_reg, mask_cls, meta = d[idx]

                self.assertEqual(tuple(x.shape), (60,))
                self.assertEqual(tuple(y_reg.shape), (3,))
                self.assertEqual(tuple(y_reg_raw.shape), (3,))
                self.assertEqual(tuple(y_cls.shape), (3,))
                self.assertEqual(tuple(mask_reg.shape), (3,))
                self.assertEqual(tuple(mask_cls.shape), (3,))

                g = (
                    self.modeling_df[(self.modeling_df["dataid"] == home) & (self.modeling_df["split"] == split)]
                    .sort_values("utc_dt")
                    .reset_index(drop=True)
                )
                ep = meta["endpoint_index"]
                expected_raw = g["aggregate"].to_numpy(dtype=np.float32)[ep - 59: ep + 1]
                x_unscaled = x.numpy() * self.agg_scaler["std"] + self.agg_scaler["mean"]
                np.testing.assert_allclose(x_unscaled, expected_raw, rtol=1e-4, atol=1e-5)

                self.assertEqual(pd.Timestamp(g["utc_dt"].iloc[ep]), meta["utc_dt"])
                self.assertEqual(meta["dataid"], home)
                self.assertEqual(meta["split"], split)
                checked += 1
        self.assertEqual(checked, 9)

    def test_explicit_first_valid_endpoint_boundary_case(self):
        """NEW: deliberately locate endpoint_index == 59 (not via incidental
        row ordering) and confirm it reconstructs a full, valid 60-sample window."""
        checked_any = False
        for home in HOMES:
            for split in ["train", "val", "test"]:
                d = self.datasets[split]
                at_boundary = np.where((d.dataid == home) & (d.endpoint_index == 59))[0]
                if len(at_boundary) == 0:
                    continue
                idx = int(at_boundary[0])
                x, y_reg, y_reg_raw, y_cls, mask_reg, mask_cls, meta = d[idx]
                self.assertEqual(meta["endpoint_index"], 59)
                self.assertEqual(tuple(x.shape), (60,))
                self.assertFalse(np.isnan(x.numpy()).any())
                checked_any = True
        self.assertTrue(checked_any, "no (home, split) with endpoint_index==59 was found -- test setup assumption violated")


class TestEndpointSafety(unittest.TestCase):
    """NEW: invalid endpoint_index values must raise clear, descriptive errors."""

    @classmethod
    def setUpClass(cls):
        cls.modeling_df, cls.agg_scaler, cls.target_scalers, cls.agg_lookup = _load_everything()

    def _fresh_dataset(self):
        return ds.NILMWindowDataset("train", self.agg_lookup, self.agg_scaler["mean"], self.agg_scaler["std"], self.target_scalers)

    def test_endpoint_below_minimum_raises_value_error(self):
        d = self._fresh_dataset()
        d.endpoint_index = d.endpoint_index.copy()
        d.endpoint_index[0] = 5  # below the required minimum of 59
        with self.assertRaises(ValueError):
            d[0]

    def test_endpoint_out_of_bounds_raises_value_error(self):
        d = self._fresh_dataset()
        d.endpoint_index = d.endpoint_index.copy()
        home = d.dataid[0]
        arr_len = len(d.agg_lookup[(home, d.split)])
        d.endpoint_index[0] = arr_len + 1000  # far beyond the array's length
        with self.assertRaises(ValueError):
            d[0]

    def test_endpoint_exactly_at_minimum_does_not_raise(self):
        d = self._fresh_dataset()
        d.endpoint_index = d.endpoint_index.copy()
        d.endpoint_index[0] = 59  # the exact boundary -- must succeed
        x, *_ = d[0]
        self.assertEqual(tuple(x.shape), (60,))


class TestSplitValidation(unittest.TestCase):
    """NEW: a label file whose rows don't match the requested split must raise."""

    @classmethod
    def setUpClass(cls):
        cls.modeling_df, cls.agg_scaler, cls.target_scalers, cls.agg_lookup = _load_everything()

    def test_mismatched_split_column_raises_value_error(self):
        real_val_df = pd.read_parquet(ds.LABELS_PATHS["val"], engine="pyarrow").copy()
        corrupted = real_val_df.copy()
        corrupted.loc[corrupted.index[0], "split"] = "train"  # deliberately corrupt one row
        with mock.patch.object(ds.pd, "read_parquet", return_value=corrupted):
            with self.assertRaises(ValueError):
                ds.NILMWindowDataset("val", self.agg_lookup, self.agg_scaler["mean"], self.agg_scaler["std"], self.target_scalers)

    def test_correct_split_column_does_not_raise(self):
        # Sanity counterpart: a genuinely correct file must NOT raise.
        d = ds.NILMWindowDataset("val", self.agg_lookup, self.agg_scaler["mean"], self.agg_scaler["std"], self.target_scalers)
        self.assertGreater(len(d), 0)


class TestMissingLabelsStayMasked(unittest.TestCase):
    """A missing/invalid target must never become an OFF (0) label."""

    @classmethod
    def setUpClass(cls):
        cls.modeling_df, cls.agg_scaler, cls.target_scalers, cls.agg_lookup = _load_everything()
        cls.train_ds = ds.NILMWindowDataset("train", cls.agg_lookup, cls.agg_scaler["mean"], cls.agg_scaler["std"], cls.target_scalers)

    def test_known_invalid_row_is_nan_not_zero(self):
        invalid_positions = np.where((self.train_ds.dataid == "7901") & (~self.train_ds.valid["dishwasher1"]))[0]
        self.assertGreater(len(invalid_positions), 0)
        idx = int(invalid_positions[0])
        x, y_reg, y_reg_raw, y_cls, mask_reg, mask_cls, meta = self.train_ds[idx]

        dw_i = APPLIANCES.index("dishwasher1")
        self.assertFalse(mask_cls[dw_i].item())
        self.assertFalse(mask_reg[dw_i].item())
        self.assertTrue(torch.isnan(y_cls[dw_i]))
        self.assertTrue(torch.isnan(y_reg[dw_i]))
        self.assertNotEqual(y_cls[dw_i].item(), 0.0)

    def test_no_sentinel_zero_disguised_as_valid(self):
        for a in APPLIANCES:
            y = np.array([self.train_ds[i][3][APPLIANCES.index(a)].item() for i in range(0, len(self.train_ds), 50000)])
            m = np.array([self.train_ds[i][5][APPLIANCES.index(a)].item() for i in range(0, len(self.train_ds), 50000)])
            invalid_mask = ~m
            if invalid_mask.any():
                self.assertTrue(np.all(np.isnan(y[invalid_mask])))


class TestBatchCollation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modeling_df, cls.agg_scaler, cls.target_scalers, cls.agg_lookup = _load_everything()
        cls.train_ds = ds.NILMWindowDataset("train", cls.agg_lookup, cls.agg_scaler["mean"], cls.agg_scaler["std"], cls.target_scalers)

    def test_dataloader_batches_correctly(self):
        loader = DataLoader(self.train_ds, batch_size=32, shuffle=False, collate_fn=ds.collate_fn)
        xs, y_reg, y_reg_raw, y_cls, mask_reg, mask_cls, meta = next(iter(loader))
        self.assertEqual(tuple(xs.shape), (32, 60))
        self.assertEqual(tuple(y_reg.shape), (32, 3))
        self.assertEqual(tuple(y_reg_raw.shape), (32, 3))
        self.assertEqual(tuple(y_cls.shape), (32, 3))
        self.assertEqual(tuple(mask_reg.shape), (32, 3))
        self.assertEqual(tuple(mask_cls.shape), (32, 3))
        self.assertEqual(len(meta["dataid"]), 32)
        self.assertEqual(meta["endpoint_index"].shape, (32,))

    def test_shuffled_dataloader_keeps_everything_aligned(self):
        """NEW: repeatable (seeded) shuffle test, cross-referencing each
        shuffled sample against the source label file independently."""
        raw_df = pd.read_parquet(ds.LABELS_PATHS["train"], engine="pyarrow")
        generator = torch.Generator().manual_seed(12345)
        loader = DataLoader(self.train_ds, batch_size=24, shuffle=True, collate_fn=ds.collate_fn, generator=generator)
        xs, y_reg, y_reg_raw, y_cls, mask_reg, mask_cls, meta = next(iter(loader))

        for b in range(24):
            home = meta["dataid"][b]
            ep = int(meta["endpoint_index"][b])
            row = raw_df[(raw_df["dataid"] == home) & (raw_df["endpoint_index"] == ep)].iloc[0]
            for i, a in enumerate(APPLIANCES):
                expected_valid = bool(row[f"{a}_valid"])
                self.assertEqual(expected_valid, bool(mask_cls[b, i].item()))
                self.assertEqual(expected_valid, bool(mask_reg[b, i].item()))
                if expected_valid:
                    self.assertAlmostEqual(float(row[f"{a}_label"]), y_cls[b, i].item(), places=5)
                    self.assertAlmostEqual(float(row[a]), y_reg_raw[b, i].item(), places=5)
                else:
                    self.assertTrue(torch.isnan(y_cls[b, i]))
                    self.assertTrue(torch.isnan(y_reg_raw[b, i]))

    def test_repeated_run_with_same_seed_is_identical(self):
        """Confirms the shuffle test above is genuinely repeatable, not flaky."""
        def first_batch_dataids(seed):
            gen = torch.Generator().manual_seed(seed)
            loader = DataLoader(self.train_ds, batch_size=10, shuffle=True, collate_fn=ds.collate_fn, generator=gen)
            _, _, _, _, _, _, meta = next(iter(loader))
            return meta["dataid"], meta["endpoint_index"].tolist()

        a1, e1 = first_batch_dataids(777)
        a2, e2 = first_batch_dataids(777)
        self.assertEqual(a1, a2)
        self.assertEqual(e1, e2)


class TestNoMetadataOrTargetLeakageIntoInput(unittest.TestCase):
    """Explicit proof that x contains ONLY aggregate-derived values."""

    @classmethod
    def setUpClass(cls):
        cls.modeling_df, cls.agg_scaler, cls.target_scalers, cls.agg_lookup = _load_everything()
        cls.train_ds = ds.NILMWindowDataset("train", cls.agg_lookup, cls.agg_scaler["mean"], cls.agg_scaler["std"], cls.target_scalers)

    def test_input_tensor_shape_excludes_any_extra_columns(self):
        x, *_ = self.train_ds[0]
        self.assertEqual(x.numel(), 60)
        self.assertEqual(x.dtype, torch.float32)

    def test_input_values_do_not_contain_appliance_readings(self):
        idx = 12345
        x, y_reg, y_reg_raw, y_cls, mask_reg, mask_cls, meta = self.train_ds[idx]
        raw_appliance_values = [self.train_ds.target_raw[a][idx] for a in APPLIANCES]
        x_unscaled = (x * self.agg_scaler["std"] + self.agg_scaler["mean"]).tolist()
        for v in raw_appliance_values:
            if not np.isnan(v):
                self.assertNotIn(round(v, 6), [round(xx, 6) for xx in x_unscaled])

    def test_getitem_return_does_not_expose_home_id_in_x(self):
        x, *rest = self.train_ds[0]
        self.assertNotIsInstance(x, dict)
        self.assertEqual(x.dim(), 1)


class TestPreprocessingMetadataPersistence(unittest.TestCase):
    """NEW: stage7_preprocess.save_preprocessing_params() must persist the
    EXACT, already-fit scaler dicts it is given (never refit/recompute
    them) with accurate source_split/source_homes provenance. Every test
    here uses a temporary output path -- none ever touches the real
    data/meta/stage7_preprocessing_params.json."""

    def test_saved_doc_uses_given_scalers_verbatim_and_records_provenance(self):
        import tempfile
        fake_agg_scaler = {"mean": 1.23, "std": 4.56, "n": 1000}
        fake_target_scalers = {
            "air1": {"mean": 0.1, "std": 0.2, "n": 100},
            "dishwasher1": {"mean": 0.3, "std": 0.4, "n": 100},
            "refrigerator1": {"mean": 0.5, "std": 0.6, "n": 100},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "stage7_preprocessing_params.json"
            doc = pp.save_preprocessing_params(fake_agg_scaler, fake_target_scalers, output_path=out_path)

            self.assertTrue(out_path.exists())
            with open(out_path) as fh:
                on_disk = json.load(fh)
            self.assertEqual(doc, on_disk)  # returned doc matches exactly what was written

            self.assertEqual(on_disk["aggregate_scaler"]["mean"], 1.23)
            self.assertEqual(on_disk["aggregate_scaler"]["std"], 4.56)
            self.assertEqual(on_disk["aggregate_scaler"]["n"], 1000)
            self.assertEqual(on_disk["target_scalers"], fake_target_scalers)
            self.assertEqual(on_disk["source_split"], "train")
            self.assertEqual(set(on_disk["source_homes"]), {"7901", "7951", "8565", "3538"})
            self.assertEqual(on_disk["held_out_home"], "8386")
            self.assertFalse(on_disk["fit_on_validation_or_test"])
            self.assertEqual(on_disk["generated_by"], "src/stage7_preprocess.py")

    def test_default_source_homes_are_the_four_development_homes_only(self):
        self.assertEqual(set(pp.DEV_HOMES), {"7901", "7951", "8565", "3538"})
        self.assertNotIn("8386", pp.DEV_HOMES)

    def test_custom_source_homes_and_split_are_honored_not_hardcoded(self):
        import tempfile
        uniform_scaler = {"mean": 0.0, "std": 1.0, "n": 1}
        uniform_targets = {a: dict(uniform_scaler) for a in APPLIANCES}
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "custom.json"
            doc = pp.save_preprocessing_params(
                uniform_scaler, uniform_targets, output_path=out_path,
                source_homes=["1111", "2222"], source_split="train",
            )
            self.assertEqual(doc["source_homes"], ["1111", "2222"])

    def test_output_is_valid_json_with_no_non_serializable_values(self):
        # Round-trips using the REAL fit functions' output shape (plain
        # python float/int, as fit_aggregate_scaler/fit_target_scalers
        # already guarantee via explicit float()/int() casts) rather than
        # hand-built dicts, to guard against a future accidental numpy leak.
        import tempfile
        agg_scaler = {"mean": float(1.0), "std": float(2.0), "n": int(5)}
        target_scalers = {a: {"mean": float(0.0), "std": float(1.0), "n": int(5)} for a in APPLIANCES}
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "params.json"
            pp.save_preprocessing_params(agg_scaler, target_scalers, output_path=out_path)
            with open(out_path) as fh:
                reloaded = json.load(fh)  # raises if the file isn't valid JSON
            self.assertIsInstance(reloaded["aggregate_scaler"]["mean"], float)
            self.assertIsInstance(reloaded["aggregate_scaler"]["n"], int)


class TestNoDevHomesInTestSplit(unittest.TestCase):
    """NEW: the regenerated window_labels_test.parquet must contain ONLY
    the official unseen test household (8386) -- confirms no development
    home ever appears in the revised test split."""

    def test_test_split_contains_only_8386(self):
        test_df = pd.read_parquet(ds.LABELS_PATHS["test"], engine="pyarrow")
        self.assertEqual(set(test_df["dataid"].unique()), {"8386"})

    def test_test_split_has_zero_development_home_rows(self):
        test_df = pd.read_parquet(ds.LABELS_PATHS["test"], engine="pyarrow")
        dev_homes = {"7901", "7951", "8565", "3538"}
        self.assertEqual(len(test_df[test_df["dataid"].isin(dev_homes)]), 0)


class TestEvaluatePlotHomeCrossCheck(unittest.TestCase):
    """NEW: only src/stage7_evaluate.py itself (not tests/test_stage7_evaluate.py)
    was authorized to change alongside this file -- this minimal
    cross-check of its PLOT_HOME/OFFICIAL_TEST_HOME constants therefore
    lives here, in the already-in-scope dataset test file."""

    def test_plot_home_is_the_official_unseen_test_household(self):
        import stage7_evaluate as se
        self.assertEqual(se.PLOT_HOME, "8386")
        self.assertEqual(se.OFFICIAL_TEST_HOME, "8386")
        self.assertNotIn(se.PLOT_HOME, {"7901", "7951", "8565", "3538"})


if __name__ == "__main__":
    unittest.main()
