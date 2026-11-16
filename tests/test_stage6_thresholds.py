"""
Unit tests for Stage 6 pooled, CONSTRAINED exact 1D two-means threshold
fitting (fit_1d_2means_threshold, fit_pooled_thresholds).

These are synthetic-data unit tests of the fitting functions in isolation
-- they do not read the real project parquet files. End-to-end correctness
against the real five_homes_modeling.parquet (dev-homes-only counts, etc.)
is verified separately once Stage 4-6 are actually regenerated.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import stage6_generate_labels as s6  # noqa: E402


class TestFit1D2MeansThreshold(unittest.TestCase):
    # --- Normal two-cluster data ---
    def test_converges_between_two_well_separated_clusters(self):
        rng = np.random.RandomState(0)
        low = rng.normal(loc=0.0, scale=0.05, size=500)
        high = rng.normal(loc=10.0, scale=0.05, size=500)
        values = np.concatenate([low, high])
        threshold = s6.fit_1d_2means_threshold(values)
        self.assertGreater(threshold, 1.0)
        self.assertLess(threshold, 9.0)

    def test_handles_unbalanced_cluster_sizes(self):
        rng = np.random.RandomState(2)
        low = rng.normal(0.0, 0.05, size=950)
        high = rng.normal(10.0, 0.05, size=50)
        values = np.concatenate([low, high])
        threshold = s6.fit_1d_2means_threshold(values)
        self.assertGreater(threshold, 1.0)
        self.assertLess(threshold, 9.0)

    # --- A single extreme outlier ---
    def test_single_extreme_outlier_not_isolated_with_default_floor(self):
        rng = np.random.RandomState(9)
        bulk = rng.normal(0, 0.05, 1999)
        values = np.concatenate([bulk, [1000.0]])  # n=2000, min_size=ceil(0.001*2000)=2
        unconstrained = s6.fit_1d_2means_threshold(values, min_cluster_fraction=1e-9)
        constrained = s6.fit_1d_2means_threshold(values)  # default MIN_CLUSTER_FRACTION
        # Unconstrained isolates the lone outlier alone: threshold ~ (0+1000)/2 = 500.
        self.assertGreater(unconstrained, 400)
        # The floor (min_size=2 > outlier group size 1) forces a different, lower split.
        self.assertLess(constrained, 400)

    # --- A small group of extreme outliers ---
    def test_small_group_of_extreme_outliers_not_isolated(self):
        rng = np.random.RandomState(7)
        bulk = rng.normal(0, 0.05, 9995)
        group = np.array([1000.0, 1000.1, 999.9, 1000.2, 999.8])  # only 5 points, n=10000
        values = np.concatenate([bulk, group])
        # min_size = ceil(0.001*10000) = 10 > group size 5.
        unconstrained = s6.fit_1d_2means_threshold(values, min_cluster_fraction=1e-9)
        constrained = s6.fit_1d_2means_threshold(values)
        self.assertGreater(unconstrained, 400)  # isolates the 5-point group alone
        self.assertLess(constrained, 400)       # floor prevents that isolation

    # --- A cluster smaller than the 0.1% floor being excluded (exact boundary) ---
    def test_cluster_exactly_at_floor_size_is_allowed(self):
        rng = np.random.RandomState(42)
        bulk = rng.normal(0, 0.05, 9990)
        group_at_floor = np.full(10, 1000.0) + rng.normal(0, 0.01, 10)  # n=10000, min_size=10
        values = np.concatenate([bulk, group_at_floor])
        threshold = s6.fit_1d_2means_threshold(values)
        # A group exactly AT the floor (size 10 == min_size) is a legal, isolable cluster.
        self.assertGreater(threshold, 400)
        self.assertLess(threshold, 600)

    def test_cluster_one_below_floor_size_is_excluded(self):
        rng = np.random.RandomState(42)
        bulk = rng.normal(0, 0.05, 9990)
        group_below_floor = np.full(9, 1000.0) + rng.normal(0, 0.01, 9)  # n=9999, min_size=10
        values = np.concatenate([bulk, group_below_floor])
        unconstrained = s6.fit_1d_2means_threshold(values, min_cluster_fraction=1e-9)
        constrained = s6.fit_1d_2means_threshold(values)
        # Unconstrained isolates the 9-point group alone: threshold ~ (0+1000)/2 = 500.
        self.assertGreater(unconstrained, 450)
        # min_size=10 > group size 9 -- the floor forces at least 1 extra point in,
        # pulling the resulting threshold meaningfully below the unconstrained value.
        self.assertLess(constrained, 460)
        self.assertLess(constrained, unconstrained - 30)

    # --- A genuine minority cluster below 1% ---
    def test_rare_high_value_minority_cluster_under_one_percent(self):
        rng = np.random.RandomState(3)
        low = rng.normal(loc=0.0, scale=0.05, size=999)
        high = np.array([100.0])
        values = np.concatenate([low, high])  # n=1000, min_size=ceil(0.001*1000)=1
        threshold = s6.fit_1d_2means_threshold(values)
        self.assertGreater(threshold, float(low.max()))
        self.assertLess(threshold, 100.0)

    def test_genuine_minority_cluster_below_one_percent_large_n(self):
        # n=20000, minority=150 (0.75%) -- well above the 0.1% floor (min_size=20),
        # so the floor must NOT reject or distort this genuine minority cluster.
        rng = np.random.RandomState(11)
        bulk = rng.normal(0.0, 0.02, 19850)
        minority = rng.normal(5.0, 0.1, 150)
        values = np.concatenate([bulk, minority])
        threshold = s6.fit_1d_2means_threshold(values)
        self.assertGreater(threshold, float(bulk.max()))
        self.assertLess(threshold, float(minority.min()))

    def test_rare_high_value_minority_cluster_half_percent_with_spread(self):
        rng = np.random.RandomState(4)
        low = rng.normal(loc=0.0, scale=0.02, size=2000)
        high = rng.normal(loc=5.0, scale=0.1, size=10)  # 0.5% minority
        values = np.concatenate([low, high])
        threshold = s6.fit_1d_2means_threshold(values)
        self.assertGreater(threshold, float(low.max()))
        self.assertLess(threshold, float(high.min()))

    # --- All-identical readings / fewer than two distinct readings ---
    def test_raises_on_all_identical_values(self):
        values = np.full(100, 5.0)
        with self.assertRaises(ValueError):
            s6.fit_1d_2means_threshold(values)

    def test_raises_on_single_value(self):
        with self.assertRaises(ValueError):
            s6.fit_1d_2means_threshold(np.array([5.0]))

    def test_raises_on_empty_array(self):
        with self.assertRaises(ValueError):
            s6.fit_1d_2means_threshold(np.array([]))

    # --- Correct ceiling-based minimum cluster-size calculation ---
    def test_ceiling_rounding_of_minimum_cluster_size(self):
        # n=1001, default fraction 0.001 -> 0.001*1001=1.001 -> ceil=2 (NOT 1).
        # Data: 999 zeros + [50.0, 100.0]. If min_size were (incorrectly) 1, the
        # unconstrained-equivalent answer would isolate just {100.0} alone,
        # giving threshold=(0+100)/2=50.0. Because ceil(1.001)=2, the single
        # point 100.0 cannot be isolated alone -- it must merge with 50.0 to
        # reach size 2, giving threshold=(0 + mean(50,100))/2 = 37.5 exactly.
        values = [0.0] * 999 + [50.0, 100.0]
        threshold = s6.fit_1d_2means_threshold(np.array(values))
        self.assertEqual(threshold, 37.5)

    def test_raises_when_min_size_leaves_no_valid_split(self):
        # n=10, min_cluster_fraction=0.5 -> min_size=5, 2*5=10 -- still exactly
        # fits (boundary case), so bump the fraction further to force no split.
        values = np.array([float(i) for i in range(10)])
        with self.assertRaises(ValueError):
            s6.fit_1d_2means_threshold(values, min_cluster_fraction=0.6)  # min_size=6, 2*6=12>10

    # --- Deterministic thresholds and cluster assignments ---
    def test_deterministic_given_same_input(self):
        rng = np.random.RandomState(1)
        values = np.concatenate([rng.normal(0, 0.1, 200), rng.normal(5, 0.1, 200)])
        t1 = s6.fit_1d_2means_threshold(values)
        t2 = s6.fit_1d_2means_threshold(values)
        self.assertEqual(t1, t2)

    def test_deterministic_across_independent_calls_and_array_copies(self):
        rng = np.random.RandomState(5)
        values = np.concatenate([rng.normal(0, 0.05, 900), rng.normal(10, 0.05, 100)])
        results = {s6.fit_1d_2means_threshold(values.copy()) for _ in range(5)}
        self.assertEqual(len(results), 1)  # every repeated call produced the identical threshold


class TestFitPooledThresholds(unittest.TestCase):
    @staticmethod
    def _rows_for_home(home, split, values):
        return [
            {
                "dataid": home, "split": split,
                "air1": v, "air1_valid": True,
                "dishwasher1": v, "dishwasher1_valid": True,
                "refrigerator1": v, "refrigerator1_valid": True,
            }
            for v in values
        ]

    # --- Household 8386 rejected ---
    def test_raises_if_held_out_home_present_in_train_split(self):
        rows = self._rows_for_home("8386", "train", [0.0, 0.1, 10.0, 10.1])
        df = pd.DataFrame(rows)
        with self.assertRaises(RuntimeError):
            s6.fit_pooled_thresholds(df)

    # --- Unexpected household IDs ---
    def test_raises_on_any_unexpected_household_id_not_just_8386(self):
        rows = self._rows_for_home("9999", "train", [0.0, 0.1, 10.0, 10.1])
        df = pd.DataFrame(rows)
        with self.assertRaises(RuntimeError):
            s6.fit_pooled_thresholds(df)

    def test_raises_if_unexpected_household_mixed_with_approved_ones(self):
        rows = []
        for home in ["7901", "7951", "8565", "3538"]:
            rows += self._rows_for_home(home, "train", [0.0, 10.0])
        rows += self._rows_for_home("1234", "train", [5.0, 5.0])  # unexpected
        df = pd.DataFrame(rows)
        with self.assertRaises(RuntimeError):
            s6.fit_pooled_thresholds(df)

    def test_one_threshold_per_appliance_dev_homes_only(self):
        # All four approved development homes present (the valid case).
        rows = []
        for home in ["7901", "7951", "8565", "3538"]:
            rows += self._rows_for_home(home, "train", [0.0, 0.05, 10.0, 10.05])
        df = pd.DataFrame(rows)
        thresholds = s6.fit_pooled_thresholds(df)
        self.assertEqual(set(thresholds.keys()), set(s6.APPLIANCES))
        for a, t in thresholds.items():
            self.assertGreater(t, 1.0)
            self.assertLess(t, 9.0)

    # --- A missing development home ---
    def test_raises_if_a_development_home_is_missing(self):
        # Only 3 of the 4 approved development homes present (3538 missing)
        # -- must raise, never silently fit on an incomplete pool.
        rows = []
        for home in ["7901", "7951", "8565"]:
            rows += self._rows_for_home(home, "train", [0.0, 0.05, 10.0, 10.05])
        df = pd.DataFrame(rows)
        with self.assertRaises(RuntimeError):
            s6.fit_pooled_thresholds(df)

    def test_raises_if_only_one_development_home_present(self):
        rows = self._rows_for_home("7901", "train", [0.0, 0.05, 10.0, 10.05])
        df = pd.DataFrame(rows)
        with self.assertRaises(RuntimeError):
            s6.fit_pooled_thresholds(df)

    # --- Validation/test readings excluded ---
    def test_ignores_invalid_rows_and_non_train_splits(self):
        # All four approved dev homes present (required by the missing-home
        # check); the invalid/val-split contamination is injected via 7901.
        rows = self._rows_for_home("7901", "train", [0.0, 10.0])
        invalid_row = self._rows_for_home("7901", "train", [999.0])[0]
        invalid_row["air1_valid"] = False
        invalid_row["dishwasher1_valid"] = False
        invalid_row["refrigerator1_valid"] = False
        rows.append(invalid_row)
        rows += self._rows_for_home("7901", "val", [-999.0])
        for home in ["7951", "8565", "3538"]:
            rows += self._rows_for_home(home, "train", [0.0, 10.0])
        df = pd.DataFrame(rows)
        thresholds = s6.fit_pooled_thresholds(df)
        for a, t in thresholds.items():
            self.assertGreater(t, 0.0)
            self.assertLess(t, 10.0)

    def test_test_split_readings_excluded_even_for_approved_homes(self):
        rows = []
        for home in ["7901", "7951", "8565", "3538"]:
            rows += self._rows_for_home(home, "train", [0.0, 10.0])
        # Extreme test-split readings for an approved dev home -- must be
        # completely ignored, not pooled into the train-only fit.
        rows += self._rows_for_home("7901", "test", [-5000.0, 5000.0])
        df = pd.DataFrame(rows)
        thresholds = s6.fit_pooled_thresholds(df)
        for a, t in thresholds.items():
            self.assertGreater(t, 0.0)
            self.assertLess(t, 10.0)  # unaffected by the test-split extreme values

    def test_does_not_require_held_out_home_absent_from_other_splits(self):
        rows = []
        for home in ["7901", "7951", "8565", "3538"]:
            rows += self._rows_for_home(home, "train", [0.0, 10.0])
        rows += self._rows_for_home("8386", "test", [5.0, 5.0])
        df = pd.DataFrame(rows)
        thresholds = s6.fit_pooled_thresholds(df)  # must not raise
        self.assertEqual(set(thresholds.keys()), set(s6.APPLIANCES))

    def test_degenerate_identical_appliance_values_raise_not_silent(self):
        rows = []
        for home in ["7901", "7951", "8565", "3538"]:
            for i in range(10):
                rows.append({
                    "dataid": home, "split": "train",
                    "air1": 5.0, "air1_valid": True,  # identical for every row
                    "dishwasher1": 0.0 if i % 2 == 0 else 10.0, "dishwasher1_valid": True,
                    "refrigerator1": 0.0 if i % 2 == 0 else 10.0, "refrigerator1_valid": True,
                })
        df = pd.DataFrame(rows)
        with self.assertRaises(ValueError):
            s6.fit_pooled_thresholds(df)

    # --- Deterministic thresholds and cluster assignments (pooled level) ---
    def test_fit_pooled_thresholds_deterministic_across_repeated_calls(self):
        rows = []
        for home in ["7901", "7951", "8565", "3538"]:
            rows += self._rows_for_home(home, "train", [0.0, 0.05, 10.0, 10.05])
        df = pd.DataFrame(rows)
        result1 = s6.fit_pooled_thresholds(df)
        result2 = s6.fit_pooled_thresholds(df.copy())
        self.assertEqual(result1, result2)


if __name__ == "__main__":
    unittest.main()
