"""
Tests for Stage 7 Step 4A CPU-only training-pipeline prototype (CT-466 Phase 2).

Run with:  python -m unittest tests.test_stage7_step4a_prototype -v
(from the project root, using the project's .venv interpreter)

Kept separate from the dataset, model, and loss test suites.
"""
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import stage7_step4a_prototype as proto  # noqa: E402
import stage7_dataset as ds  # noqa: E402


class TestDeterministicTrainOnlySubset(unittest.TestCase):
    """Under the revised evaluation design, the train split contains only
    the four development households (7901, 7951, 8565, 3538) -- the
    official unseen test household (8386) has zero train rows, so it
    contributes zero indices to this subset (build_small_deterministic_
    train_subset tolerates an empty home slice gracefully, by design)."""

    DEV_HOMES = ("7901", "7951", "8565", "3538")
    HELD_OUT_HOME = "8386"

    @classmethod
    def setUpClass(cls):
        cls.train_ds = proto.build_train_dataset()

    def test_dataset_is_train_split_only(self):
        self.assertEqual(self.train_ds.split, "train")

    def test_subset_size_matches_four_development_homes_times_per_home(self):
        indices = proto.build_small_deterministic_train_subset(self.train_ds, per_home=40)
        self.assertEqual(len(indices), 4 * 40)

    def test_subset_selection_is_deterministic_across_calls(self):
        indices_a = proto.build_small_deterministic_train_subset(self.train_ds, per_home=40)
        indices_b = proto.build_small_deterministic_train_subset(self.train_ds, per_home=40)
        self.assertEqual(indices_a, indices_b)

    def test_subset_covers_all_four_development_homes(self):
        indices = proto.build_small_deterministic_train_subset(self.train_ds, per_home=40)
        homes_present = {self.train_ds.dataid[i] for i in indices}
        self.assertEqual(homes_present, set(self.DEV_HOMES))

    def test_subset_excludes_held_out_household(self):
        indices = proto.build_small_deterministic_train_subset(self.train_ds, per_home=40)
        homes_present = {self.train_ds.dataid[i] for i in indices}
        self.assertNotIn(self.HELD_OUT_HOME, homes_present)


class TestPrototypeStepMLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = proto.run_prototype_step("MLP")

    def test_losses_are_finite_and_nonnegative(self):
        self.assertTrue(self.result["loss_finite"])
        self.assertGreaterEqual(self.result["cls_loss"], 0.0)
        self.assertGreaterEqual(self.result["reg_loss"], 0.0)
        self.assertGreaterEqual(self.result["total_loss"], 0.0)

    def test_combined_loss_equals_weighted_sum(self):
        expected = self.result["lambda_cls"] * self.result["cls_loss"] + self.result["lambda_reg"] * self.result["reg_loss"]
        self.assertAlmostEqual(self.result["total_loss"], expected, places=5)

    def test_all_parameters_received_finite_gradients(self):
        self.assertTrue(self.result["all_params_have_grad"])
        self.assertTrue(self.result["grads_finite"])

    def test_optimizer_step_changed_parameters(self):
        self.assertTrue(self.result["any_param_changed"])

    def test_batch_size_matches_configured_value(self):
        self.assertEqual(self.result["batch_size_used"], proto.BATCH_SIZE)

    def test_subset_size_matches_configured_value(self):
        # Four development homes only under the revised split -- 8386
        # contributes zero rows to the train split, hence zero indices.
        self.assertEqual(self.result["n_subset"], 4 * proto.SUBSET_PER_HOME)


class TestPrototypeStepCNN1D(unittest.TestCase):
    """Supplementary: confirm the same pipeline also works for the 1D-CNN
    (not strictly required by this step, but cheap and directly useful)."""

    @classmethod
    def setUpClass(cls):
        cls.result = proto.run_prototype_step("CNN1D")

    def test_losses_are_finite_and_nonnegative(self):
        self.assertTrue(self.result["loss_finite"])
        self.assertGreaterEqual(self.result["cls_loss"], 0.0)
        self.assertGreaterEqual(self.result["reg_loss"], 0.0)

    def test_all_parameters_received_finite_gradients(self):
        self.assertTrue(self.result["all_params_have_grad"])
        self.assertTrue(self.result["grads_finite"])

    def test_optimizer_step_changed_parameters(self):
        self.assertTrue(self.result["any_param_changed"])


class TestConfigurableLambdaCoefficients(unittest.TestCase):
    def test_different_lambdas_change_total_loss_but_not_component_losses(self):
        result_default = proto.run_prototype_step("MLP", lambda_cls=1.0, lambda_reg=1.0)
        result_custom = proto.run_prototype_step("MLP", lambda_cls=2.0, lambda_reg=0.5)

        # same seed/subset/model init -> component losses must be identical
        self.assertAlmostEqual(result_default["cls_loss"], result_custom["cls_loss"], places=5)
        self.assertAlmostEqual(result_default["reg_loss"], result_custom["reg_loss"], places=5)

        expected_custom_total = 2.0 * result_custom["cls_loss"] + 0.5 * result_custom["reg_loss"]
        self.assertAlmostEqual(result_custom["total_loss"], expected_custom_total, places=5)
        self.assertNotAlmostEqual(result_default["total_loss"], result_custom["total_loss"], places=5)


if __name__ == "__main__":
    unittest.main()
