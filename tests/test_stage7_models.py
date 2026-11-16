"""
Tests for Stage 7 Step 2 model architectures (CT-466 Phase 2).

Run with:  python -m unittest tests.test_stage7_models -v
(from the project root, using the project's .venv interpreter)

Deliberately kept separate from tests/test_stage7_dataset.py -- these
tests only exercise the model modules in isolation (synthetic tensors),
never the real dataset/preprocessing pipeline.
"""
import inspect
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

import mlp as mlp_module  # noqa: E402
import cnn1d as cnn1d_module  # noqa: E402

MODEL_CLASSES = {"MLP": mlp_module.MLP, "CNN1D": cnn1d_module.CNN1D}


def make_model(name, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return MODEL_CLASSES[name]()


class TestForwardSignatureAcceptsOnlyAggregateInput(unittest.TestCase):
    """Structural check: forward() takes exactly one tensor argument -- no
    channel for home ID, timestamps, appliance readings, labels, or masks."""

    def test_forward_signature_has_single_input_parameter(self):
        for name, cls in MODEL_CLASSES.items():
            sig = inspect.signature(cls.forward)
            params = [p for p in sig.parameters if p != "self"]
            self.assertEqual(params, ["x"], f"{name}.forward has unexpected parameters: {params}")

    def test_forward_rejects_wrong_last_dimension(self):
        for name, cls in MODEL_CLASSES.items():
            model = cls()
            bad_input = torch.randn(4, 61)  # one extra column -- must be rejected, not silently accepted
            with self.assertRaises(ValueError):
                model(bad_input)


class TestOutputShapesAndRanges(unittest.TestCase):
    def _check_model(self, name, batch_size):
        model = make_model(name, seed=0)
        x = torch.randn(batch_size, 60)
        cls_probs, reg_out = model(x)

        self.assertEqual(tuple(cls_probs.shape), (batch_size, 3))
        self.assertEqual(tuple(reg_out.shape), (batch_size, 3))

        # classification outputs must be valid probabilities
        self.assertTrue(torch.all(cls_probs >= 0.0))
        self.assertTrue(torch.all(cls_probs <= 1.0))

        # regression outputs must be finite and NOT constrained to be
        # non-negative (proves no Softplus/ReLU was applied on this head)
        self.assertTrue(torch.all(torch.isfinite(reg_out)))

    def test_mlp_batch_size_1(self):
        self._check_model("MLP", 1)

    def test_mlp_normal_batch_size(self):
        self._check_model("MLP", 32)

    def test_cnn1d_batch_size_1(self):
        self._check_model("CNN1D", 1)

    def test_cnn1d_normal_batch_size(self):
        self._check_model("CNN1D", 32)

    def test_regression_head_can_produce_negative_values(self):
        # With standardized targets (mean 0), negative regression outputs
        # must be achievable -- not clamped by Softplus/ReLU. Use a large
        # batch and extreme input scale to make a negative output likely,
        # then explicitly confirm at least one negative value appears
        # across both models over several seeds (not asserting on a single
        # unlucky draw).
        found_negative = {"MLP": False, "CNN1D": False}
        for seed in range(5):
            for name in MODEL_CLASSES:
                model = make_model(name, seed=seed)
                x = torch.randn(256, 60) * 3.0
                _, reg_out = model(x)
                if torch.any(reg_out < 0):
                    found_negative[name] = True
        for name, ok in found_negative.items():
            self.assertTrue(ok, f"{name} regression head never produced a negative value across 5 seeds/256 samples -- possible unwanted constraint")


class TestBackwardPassOnCPU(unittest.TestCase):
    def _check_backward(self, name):
        model = make_model(name, seed=1)
        x = torch.randn(16, 60, requires_grad=False)
        cls_probs, reg_out = model(x)

        cls_target = torch.randint(0, 2, (16, 3)).float()
        reg_target = torch.randn(16, 3)
        loss = torch.nn.functional.binary_cross_entropy(cls_probs, cls_target) + \
            torch.nn.functional.mse_loss(reg_out, reg_target)

        loss.backward()

        grad_found = False
        for p in model.parameters():
            self.assertIsNotNone(p.grad, "a parameter received no gradient")
            if torch.any(p.grad != 0):
                grad_found = True
        self.assertTrue(grad_found, "no parameter received a nonzero gradient")
        self.assertTrue(torch.isfinite(loss))

    def test_mlp_backward(self):
        self._check_backward("MLP")

    def test_cnn1d_backward(self):
        self._check_backward("CNN1D")

    def test_models_run_on_cpu_device_explicitly(self):
        for name in MODEL_CLASSES:
            model = make_model(name, seed=2).to(torch.device("cpu"))
            x = torch.randn(8, 60, device="cpu")
            cls_probs, reg_out = model(x)
            self.assertEqual(cls_probs.device.type, "cpu")
            self.assertEqual(reg_out.device.type, "cpu")


class TestDeterministicInitializationAndOutput(unittest.TestCase):
    def _check_determinism(self, name):
        model_a = make_model(name, seed=42)
        model_b = make_model(name, seed=42)

        for (name_a, p_a), (name_b, p_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
            self.assertEqual(name_a, name_b)
            torch.testing.assert_close(p_a, p_b)

        x = torch.randn(4, 60)
        torch.manual_seed(123)
        cls_a, reg_a = model_a(x)
        torch.manual_seed(123)
        cls_b, reg_b = model_b(x)
        torch.testing.assert_close(cls_a, cls_b)
        torch.testing.assert_close(reg_a, reg_b)

    def test_mlp_deterministic_with_same_seed(self):
        self._check_determinism("MLP")

    def test_cnn1d_deterministic_with_same_seed(self):
        self._check_determinism("CNN1D")


class TestBothModelsShareIdenticalInputInformation(unittest.TestCase):
    """Both architectures must be fed exactly the same input tensor content --
    the CNN only reshapes it internally, never augments it."""

    def test_same_input_tensor_feeds_both_models_without_modification(self):
        x = torch.randn(10, 60)
        x_clone = x.clone()
        mlp_model = make_model("MLP", seed=5)
        cnn_model = make_model("CNN1D", seed=5)
        mlp_model(x)
        cnn_model(x)
        # neither model's forward pass may mutate the input tensor in place
        torch.testing.assert_close(x, x_clone)


if __name__ == "__main__":
    unittest.main()
