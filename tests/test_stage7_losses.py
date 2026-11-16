"""
Tests for Stage 7 Step 3 masked loss functions (CT-466 Phase 2).

Run with:  python -m unittest tests.test_stage7_losses -v
(from the project root, using the project's .venv interpreter)

Kept separate from tests/test_stage7_dataset.py and tests/test_stage7_models.py.
"""
import math
import sys
import unittest
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import stage7_losses as losses  # noqa: E402


class TestMaskedBCEManualCalculation(unittest.TestCase):
    def test_unweighted_matches_hand_calculation(self):
        cls_probs = torch.tensor([[0.8], [0.3]])
        y_cls = torch.tensor([[1.0], [0.0]])
        mask = torch.tensor([[True], [True]])

        expected = (-math.log(0.8) + -math.log(0.7)) / 2
        got = losses.masked_bce_loss(cls_probs, y_cls, mask).item()
        self.assertAlmostEqual(got, expected, places=6)

    def test_weighted_matches_hand_calculation(self):
        cls_probs = torch.tensor([[0.8], [0.3]])
        y_cls = torch.tensor([[1.0], [0.0]])
        mask = torch.tensor([[True], [True]])
        pos_weight = torch.tensor([2.0])

        elem0 = -(2.0 * 1.0 * math.log(0.8) + 0.0 * math.log(0.2))
        elem1 = -(2.0 * 0.0 * math.log(0.3) + 1.0 * math.log(0.7))
        expected = (elem0 + elem1) / 2
        got = losses.masked_bce_loss(cls_probs, y_cls, mask, pos_weight=pos_weight).item()
        self.assertAlmostEqual(got, expected, places=6)


class TestMaskedMSEManualCalculation(unittest.TestCase):
    def test_matches_hand_calculation(self):
        reg_out = torch.tensor([[1.5], [3.0]])
        y_reg = torch.tensor([[1.0], [2.0]])
        mask = torch.tensor([[True], [True]])

        expected = ((0.5 ** 2) + (1.0 ** 2)) / 2
        got = losses.masked_mse_loss(reg_out, y_reg, mask).item()
        self.assertAlmostEqual(got, expected, places=6)


class TestInvalidTargetsNeverAffectLoss(unittest.TestCase):
    def test_bce_unaffected_by_nan_target_value_at_invalid_position(self):
        cls_probs = torch.tensor([[0.8, 0.5], [0.3, 0.5]])
        mask = torch.tensor([[True, False], [True, False]])
        y_cls_a = torch.tensor([[1.0, float("nan")], [0.0, float("nan")]])
        y_cls_b = torch.tensor([[1.0, 999.0], [0.0, -999.0]])  # same valid cols, garbage at invalid positions

        loss_a = losses.masked_bce_loss(cls_probs, y_cls_a, mask)
        loss_b = losses.masked_bce_loss(cls_probs, y_cls_b, mask)
        self.assertTrue(torch.isfinite(loss_a))
        self.assertTrue(torch.isfinite(loss_b))
        self.assertAlmostEqual(loss_a.item(), loss_b.item(), places=6)

    def test_mse_unaffected_by_nan_target_value_at_invalid_position(self):
        reg_out = torch.tensor([[1.5, 2.0], [3.0, 2.0]])
        mask = torch.tensor([[True, False], [True, False]])
        y_reg_a = torch.tensor([[1.0, float("nan")], [2.0, float("nan")]])
        y_reg_b = torch.tensor([[1.0, 12345.0], [2.0, -12345.0]])

        loss_a = losses.masked_mse_loss(reg_out, y_reg_a, mask)
        loss_b = losses.masked_mse_loss(reg_out, y_reg_b, mask)
        self.assertTrue(torch.isfinite(loss_a))
        self.assertTrue(torch.isfinite(loss_b))
        self.assertAlmostEqual(loss_a.item(), loss_b.item(), places=6)


class TestIndependentPerApplianceMasking(unittest.TestCase):
    def test_bce_ignores_fully_invalid_appliance_column(self):
        cls_probs = torch.tensor([[0.8, 0.9, 0.2], [0.3, 0.1, 0.6]])
        y_cls = torch.tensor([[1.0, float("nan"), 0.0], [0.0, float("nan"), 1.0]])
        mask = torch.tensor([[True, False, True], [True, False, True]])

        # appliance 1 fully invalid -> loss must equal the loss computed with
        # that column removed entirely (2-appliance version)
        cls_probs_2 = cls_probs[:, [0, 2]]
        y_cls_2 = y_cls[:, [0, 2]]
        mask_2 = mask[:, [0, 2]]

        loss_3col = losses.masked_bce_loss(cls_probs, y_cls, mask)
        loss_2col = losses.masked_bce_loss(cls_probs_2, y_cls_2, mask_2)
        self.assertAlmostEqual(loss_3col.item(), loss_2col.item(), places=6)

    def test_mse_ignores_fully_invalid_appliance_column(self):
        reg_out = torch.tensor([[1.0, 5.0, 2.0], [2.0, 5.0, 3.0]])
        y_reg = torch.tensor([[1.5, float("nan"), 2.5], [1.5, float("nan"), 2.5]])
        mask = torch.tensor([[True, False, True], [True, False, True]])

        reg_out_2 = reg_out[:, [0, 2]]
        y_reg_2 = y_reg[:, [0, 2]]
        mask_2 = mask[:, [0, 2]]

        loss_3col = losses.masked_mse_loss(reg_out, y_reg, mask)
        loss_2col = losses.masked_mse_loss(reg_out_2, y_reg_2, mask_2)
        self.assertAlmostEqual(loss_3col.item(), loss_2col.item(), places=6)

    def test_partial_mask_within_single_appliance_column(self):
        # appliance 0: only row 0 valid; appliance 1: only row 1 valid
        cls_probs = torch.tensor([[0.7, 0.4], [0.6, 0.9]])
        y_cls = torch.tensor([[1.0, float("nan")], [float("nan"), 1.0]])
        mask = torch.tensor([[True, False], [False, True]])

        expected0 = -math.log(0.7)
        expected1 = -math.log(0.9)
        expected = (expected0 + expected1) / 2  # mean over the 2 appliance-level means
        got = losses.masked_bce_loss(cls_probs, y_cls, mask).item()
        self.assertAlmostEqual(got, expected, places=6)


class TestGradientsFiniteAndFlowing(unittest.TestCase):
    def test_bce_gradient_is_finite_and_nonzero(self):
        logits_like = torch.tensor([[0.2, 0.8, 0.5]], requires_grad=True)
        cls_probs = torch.sigmoid(logits_like)
        y_cls = torch.tensor([[1.0, 0.0, 1.0]])
        mask = torch.tensor([[True, True, True]])

        loss = losses.masked_bce_loss(cls_probs, y_cls, mask)
        loss.backward()
        self.assertTrue(torch.all(torch.isfinite(logits_like.grad)))
        self.assertTrue(torch.any(logits_like.grad != 0))

    def test_mse_gradient_is_finite_and_nonzero(self):
        reg_out = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        y_reg = torch.tensor([[1.5, 2.5, 2.5]])
        mask = torch.tensor([[True, True, True]])

        loss = losses.masked_mse_loss(reg_out, y_reg, mask)
        loss.backward()
        self.assertTrue(torch.all(torch.isfinite(reg_out.grad)))
        self.assertTrue(torch.any(reg_out.grad != 0))

    def test_gradient_only_flows_to_valid_positions(self):
        reg_out = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        y_reg = torch.tensor([[1.5, float("nan"), 2.5]])
        mask = torch.tensor([[True, False, True]])

        loss = losses.masked_mse_loss(reg_out, y_reg, mask)
        loss.backward()
        self.assertNotEqual(reg_out.grad[0, 0].item(), 0.0)
        self.assertEqual(reg_out.grad[0, 1].item(), 0.0)  # masked-out column gets exactly zero gradient
        self.assertNotEqual(reg_out.grad[0, 2].item(), 0.0)


class TestNoValidTargetsSafeZero(unittest.TestCase):
    def test_bce_all_invalid_returns_zero_and_is_differentiable(self):
        cls_probs = torch.tensor([[0.7, 0.4, 0.2]], requires_grad=True)
        y_cls = torch.full((1, 3), float("nan"))
        mask = torch.zeros((1, 3), dtype=torch.bool)

        loss = losses.masked_bce_loss(cls_probs, y_cls, mask)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()  # must not raise
        self.assertTrue(torch.all(torch.isfinite(cls_probs.grad)))

    def test_mse_all_invalid_returns_zero_and_is_differentiable(self):
        reg_out = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        y_reg = torch.full((1, 3), float("nan"))
        mask = torch.zeros((1, 3), dtype=torch.bool)

        loss = losses.masked_mse_loss(reg_out, y_reg, mask)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.all(torch.isfinite(reg_out.grad)))


class TestPositiveWeightCalculation(unittest.TestCase):
    def _make_df(self, n_pos, n_off, n_invalid, split="train"):
        rows = []
        rows += [{"split": split, "dishwasher1_valid": True, "dishwasher1_label": 1} for _ in range(n_pos)]
        rows += [{"split": split, "dishwasher1_valid": True, "dishwasher1_label": 0} for _ in range(n_off)]
        rows += [{"split": split, "dishwasher1_valid": False, "dishwasher1_label": pd.NA} for _ in range(n_invalid)]
        df = pd.DataFrame(rows)
        df["dishwasher1_label"] = df["dishwasher1_label"].astype("Int64")
        return df

    def test_uncapped_ratio_is_exact(self):
        df = self._make_df(n_pos=10, n_off=990, n_invalid=5)
        w = losses.compute_pos_weights(df, appliances=["dishwasher1"])
        self.assertEqual(w["dishwasher1"]["n_pos"], 10)
        self.assertEqual(w["dishwasher1"]["n_off"], 990)
        self.assertAlmostEqual(w["dishwasher1"]["weight"], 99.0, places=6)
        self.assertFalse(w["dishwasher1"]["capped"])

    def test_cap_of_20_is_applied(self):
        df = self._make_df(n_pos=10, n_off=990, n_invalid=0)  # raw ratio 99
        w = losses.compute_pos_weights(df, appliances=["dishwasher1"], cap=20)
        self.assertEqual(w["dishwasher1"]["weight"], 20.0)
        self.assertTrue(w["dishwasher1"]["capped"])
        self.assertAlmostEqual(w["dishwasher1"]["raw_ratio"], 99.0, places=6)

    def test_cap_of_50_is_applied(self):
        df = self._make_df(n_pos=10, n_off=990, n_invalid=0)
        w = losses.compute_pos_weights(df, appliances=["dishwasher1"], cap=50)
        self.assertEqual(w["dishwasher1"]["weight"], 50.0)
        self.assertTrue(w["dishwasher1"]["capped"])

    def test_cap_above_raw_ratio_has_no_effect(self):
        df = self._make_df(n_pos=100, n_off=200, n_invalid=0)  # raw ratio 2.0
        w = losses.compute_pos_weights(df, appliances=["dishwasher1"], cap=50)
        self.assertAlmostEqual(w["dishwasher1"]["weight"], 2.0, places=6)
        self.assertFalse(w["dishwasher1"]["capped"])

    def test_rejects_non_train_rows(self):
        df = self._make_df(n_pos=10, n_off=90, n_invalid=0, split="val")
        with self.assertRaises(RuntimeError):
            losses.compute_pos_weights(df, appliances=["dishwasher1"])

    def test_zero_positive_examples_raises_runtime_error(self):
        """NEW: permanent test for the existing n_pos==0 guard."""
        df = self._make_df(n_pos=0, n_off=50, n_invalid=0)
        with self.assertRaises(RuntimeError):
            losses.compute_pos_weights(df, appliances=["dishwasher1"])

    def test_zero_negative_examples_raises_value_error(self):
        """NEW: n_off==0 must raise ValueError, never silently return weight=0.0."""
        df = self._make_df(n_pos=50, n_off=0, n_invalid=0)
        with self.assertRaises(ValueError):
            losses.compute_pos_weights(df, appliances=["dishwasher1"])

    def test_zero_negative_does_not_silently_return_zero_weight(self):
        """Explicit negative check: confirm no dict is ever returned in this case."""
        df = self._make_df(n_pos=50, n_off=0, n_invalid=0)
        try:
            result = losses.compute_pos_weights(df, appliances=["dishwasher1"])
            self.fail(f"Expected ValueError, but got a result instead: {result}")
        except ValueError:
            pass

    def test_against_real_stage6_train_counts(self):
        """Cross-check against the ACTUAL dishwasher1 pooled train counts
        already on record (Stage 6 report: ON=2747, OFF=389720)."""
        import stage7_dataset as ds
        real_train_df = pd.read_parquet(ds.LABELS_PATHS["train"], engine="pyarrow")
        w = losses.compute_pos_weights(real_train_df, appliances=["dishwasher1"])
        self.assertEqual(w["dishwasher1"]["n_pos"], 2747)
        self.assertEqual(w["dishwasher1"]["n_off"], 389720)
        self.assertAlmostEqual(w["dishwasher1"]["weight"], 389720 / 2747, places=4)

    def test_pos_weight_dict_to_tensor_ordering(self):
        df_rows = {
            "split": ["train"] * 4,
            "air1_valid": [True, True, True, True],
            "air1_label": pd.array([1, 0, 1, 0], dtype="Int64"),
            "dishwasher1_valid": [True, True, True, True],
            "dishwasher1_label": pd.array([1, 0, 0, 0], dtype="Int64"),
            "refrigerator1_valid": [True, True, True, True],
            "refrigerator1_label": pd.array([1, 1, 0, 0], dtype="Int64"),
        }
        df = pd.DataFrame(df_rows)
        w = losses.compute_pos_weights(df)
        t = losses.pos_weight_dict_to_tensor(w)
        self.assertEqual(t.shape, (3,))
        self.assertAlmostEqual(t[0].item(), w["air1"]["weight"], places=6)
        self.assertAlmostEqual(t[1].item(), w["dishwasher1"]["weight"], places=6)
        self.assertAlmostEqual(t[2].item(), w["refrigerator1"]["weight"], places=6)


if __name__ == "__main__":
    unittest.main()
