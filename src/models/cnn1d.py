"""
Stage 7 Step 2: advanced 1D-CNN architecture (CT-466 Phase 2).

Receives the SAME 60 standardized aggregate readings as the MLP (see
src/models/mlp.py) -- identical input information, just reshaped
internally to a single channel for convolution. No home ID, timestamp,
appliance reading, label, or mask is ever part of this model's input.

Architecture (as specified):
    Input (batch, 60) -> reshaped internally to (batch, 1, 60)
    -> Conv1d(1, 8,  kernel=5, padding=2) -> ReLU -> MaxPool1d(2)   # 60 -> 30
    -> Conv1d(8, 16, kernel=5, padding=2) -> ReLU -> MaxPool1d(2)   # 30 -> 15
    -> Conv1d(16, 32, kernel=3, padding=1) -> ReLU -> AdaptiveAvgPool1d(1)  # -> 1
    -> flatten to (batch, 32)
    -> compact shared representation: Linear(32, 32) -> ReLU
    -> classification head: Linear(32, 3) -> Sigmoid  (3 independent probabilities)
    -> regression head:     Linear(32, 3)             (3 linear, unconstrained outputs)

No Softplus on the regression head, for the same reason as the MLP:
standardized targets can be negative.

No loss function, training loop, or evaluation code is included here.
"""
import torch
import torch.nn as nn

INPUT_LEN = 60
N_APPLIANCES = 3


class CNN1D(nn.Module):
    def __init__(self, input_len: int = INPUT_LEN, n_appliances: int = N_APPLIANCES):
        super().__init__()
        self.input_len = input_len
        self.conv_trunk = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.shared_dense = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(32, n_appliances),
            nn.Sigmoid(),
        )
        self.regression_head = nn.Linear(32, n_appliances)

    def forward(self, x: torch.Tensor):
        """
        x: FloatTensor, shape (batch, 60) -- standardized aggregate readings ONLY.
        Returns: (cls_probs, reg_out), each shape (batch, 3).
        """
        if x.dim() != 2 or x.size(1) != self.input_len:
            raise ValueError(f"Expected input shape (batch, {self.input_len}), got {tuple(x.shape)}")
        x = x.unsqueeze(1)  # (batch, 60) -> (batch, 1, 60): reshape only, no new information
        h = self.conv_trunk(x)         # (batch, 32, 1)
        h = h.squeeze(-1)              # (batch, 32)
        h = self.shared_dense(h)       # (batch, 32)
        cls_probs = self.classification_head(h)
        reg_out = self.regression_head(h)
        return cls_probs, reg_out
