from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedsense.config import AppConfig


@dataclass(slots=True)
class TrainResult:
    loss: float
    accuracy: float
    num_examples: int
    epsilon: float | None = None


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_local_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    config: AppConfig,
    local_epochs: int | None = None,
) -> TrainResult:
    device = torch.device(config.training.device)
    model.to(device)
    model.train()

    loader = _make_loader(x, y, batch_size=config.training.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    criterion = nn.CrossEntropyLoss()

    privacy_engine = None
    epsilon = None
    if config.extensions.enable_dp:
        try:
            from opacus import PrivacyEngine
        except ImportError as exc:
            raise RuntimeError('DP mode requested but Opacus is not installed.') from exc

        privacy_engine = PrivacyEngine()
        model, optimizer, loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=config.extensions.dp_noise_multiplier,
            max_grad_norm=config.extensions.dp_max_grad_norm,
        )

    reference_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    epochs = local_epochs or config.training.local_epochs
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for _ in range(epochs):
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)

            if config.training.strategy == 'fedprox':
                prox_term = torch.zeros(1, device=device)
                for current, reference in zip(model.parameters(), reference_parameters, strict=True):
                    prox_term = prox_term + torch.sum((current - reference.to(device)) ** 2)
                loss = loss + 0.5 * config.training.proximal_mu * prox_term

            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_examples += batch_size

    if privacy_engine is not None:
        epsilon = float(privacy_engine.get_epsilon(delta=1e-5))

    return TrainResult(
        loss=total_loss / max(total_examples, 1),
        accuracy=total_correct / max(total_examples, 1),
        num_examples=total_examples,
        epsilon=epsilon,
    )


def evaluate_model(model: nn.Module, x: np.ndarray, y: np.ndarray, config: AppConfig) -> TrainResult:
    device = torch.device(config.training.device)
    model.to(device)
    model.eval()

    loader = _make_loader(x, y, batch_size=config.training.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_examples += batch_size

    return TrainResult(
        loss=total_loss / max(total_examples, 1),
        accuracy=total_correct / max(total_examples, 1),
        num_examples=total_examples,
    )
