"""
Stage 7 Step 2: basic MLP architecture (CT-466 Phase 2).

Receives ONLY the 60 standardized aggregate readings produced by
NILMWindowDataset (src/stage7_dataset.py). No home ID, timestamp,
appliance reading, label, or mask is ever part of this model's input --
the forward() signature accepts a single tensor of shape (batch, 60) and
nothing else.

Architecture (as specified):
    Input (batch, 60)
    -> Linear(60, 128) -> ReLU -> Dropout(0.2)
    -> Linear(128, 64) -> ReLU -> Dropout(0.2)
    -> classification head: Linear(64, 3) -> Sigmoid  (3 independent probabilities)
    -> regression head:     Linear(64, 3)             (3 linear, unconstrained outputs)

No Softplus on the regression head: targets are standardized (train-only
mean/std, see stage7_preprocess.py) and can be negative, so an
unconstrained linear output is required to match that target range.

No loss function, training loop, or evaluation code is included here.
"""
import torch
import torch.nn as nn

INPUT_LEN = 60
N_APPLIANCES = 3


class MLP(nn.Module):
    def __init__(self, input_len: int = INPUT_LEN, n_appliances: int = N_APPLIANCES, dropout: float = 0.2):
        super().__init__()
        self.input_len = input_len
        self.trunk = nn.Sequential(
            nn.Linear(input_len, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(64, n_appliances),
            nn.Sigmoid(),
        )
        self.regression_head = nn.Linear(64, n_appliances)

    def forward(self, x: torch.Tensor):
        """
        x: FloatTensor, shape (batch, 60) -- standardized aggregate readings ONLY.
        Returns: (cls_probs, reg_out), each shape (batch, 3).
        """
        if x.dim() != 2 or x.size(1) != self.input_len:
            raise ValueError(f"Expected input shape (batch, {self.input_len}), got {tuple(x.shape)}")
        h = self.trunk(x)
        cls_probs = self.classification_head(h)
        reg_out = self.regression_head(h)
        return cls_probs, reg_out
