from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from torch import nn

DEFAULT_INPUT_CHANNELS = 6
DEFAULT_NUM_CLASSES = 6


class HAR1DCNN(nn.Module):
    def __init__(self, input_channels: int = DEFAULT_INPUT_CHANNELS, num_classes: int = DEFAULT_NUM_CLASSES) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(f'Expected input with shape [batch, channels, time], received {tuple(inputs.shape)}')
        if inputs.shape[1] != self.input_channels:
            raise ValueError(f'Expected {self.input_channels} channels, received {inputs.shape[1]}')
        return self.classifier(self.features(inputs))


def build_model(
    seed: int,
    input_channels: int = DEFAULT_INPUT_CHANNELS,
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> HAR1DCNN:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return HAR1DCNN(input_channels=input_channels, num_classes=num_classes)


def get_model_parameters(model: nn.Module) -> list[np.ndarray]:
    return [tensor.detach().cpu().numpy().copy() for _, tensor in model.state_dict().items()]


def set_model_parameters(model: nn.Module, params: list[np.ndarray]) -> None:
    state_dict = model.state_dict()
    if len(state_dict) != len(params):
        raise ValueError(f'Parameter count mismatch: expected {len(state_dict)}, received {len(params)}')

    updated = OrderedDict()
    for (name, tensor), array in zip(state_dict.items(), params, strict=True):
        expected_shape = tuple(tensor.shape)
        actual_shape = tuple(array.shape)
        if expected_shape != actual_shape:
            raise ValueError(f"Shape mismatch for '{name}': expected {expected_shape}, received {actual_shape}")
        updated[name] = torch.tensor(array, dtype=tensor.dtype)

    model.load_state_dict(updated, strict=True)
