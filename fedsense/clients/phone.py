from __future__ import annotations

from typing import Any

import flwr as fl
import numpy as np

from fedsense.config import AppConfig
from fedsense.contracts import build_client_payload
from fedsense.data.phone import load_phone_windows
from fedsense.model import build_model, get_model_parameters, set_model_parameters
from fedsense.training import evaluate_model, train_local_model


class PhoneHarClient(fl.client.NumPyClient):
    def __init__(self, config: AppConfig, normalization_mean: np.ndarray, normalization_std: np.ndarray) -> None:
        self.config = config
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.model = build_model(seed=config.training.seed)

    def get_parameters(self, config: dict[str, Any]) -> list[Any]:
        return get_model_parameters(self.model)

    def fit(self, parameters: list[Any], config: dict[str, Any]) -> tuple[list[Any], int, dict[str, Any]]:
        set_model_parameters(self.model, parameters)
        dataset = load_phone_windows(
            csv_path=self.config.phone.csv_path,
            label_column=self.config.phone.label_column,
            window_size=self.config.data.window_size,
            stride=self.config.data.stride,
            normalization_mean=self.normalization_mean,
            normalization_std=self.normalization_std,
            epsilon=self.config.data.normalization_epsilon,
        )

        if dataset.num_windows < self.config.training.min_phone_windows_per_round:
            metrics = {
                'client_type': 'phone',
                'skipped': 1,
                'train_loss': 0.0,
                'train_accuracy': 0.0,
                'skip_reason': 'insufficient_phone_windows',
            }
            current = get_model_parameters(self.model)
            build_client_payload(weights=current, num_examples=0, metrics=metrics)
            return current, 0, metrics

        result = train_local_model(self.model, dataset.windows, dataset.labels, self.config)
        updated = get_model_parameters(self.model)
        metrics = {
            'client_type': 'phone',
            'skipped': 0,
            'train_loss': float(result.loss),
            'train_accuracy': float(result.accuracy),
        }
        if result.epsilon is not None:
            metrics['epsilon'] = float(result.epsilon)

        build_client_payload(weights=updated, num_examples=result.num_examples, metrics=metrics)
        return updated, result.num_examples, metrics

    def evaluate(self, parameters: list[Any], config: dict[str, Any]) -> tuple[float, int, dict[str, Any]]:
        set_model_parameters(self.model, parameters)
        dataset = load_phone_windows(
            csv_path=self.config.phone.csv_path,
            label_column=self.config.phone.label_column,
            window_size=self.config.data.window_size,
            stride=self.config.data.stride,
            normalization_mean=self.normalization_mean,
            normalization_std=self.normalization_std,
            epsilon=self.config.data.normalization_epsilon,
        )
        if dataset.num_windows == 0:
            return 0.0, 0, {'accuracy': 0.0}
        result = evaluate_model(self.model, dataset.windows, dataset.labels, self.config)
        return float(result.loss), result.num_examples, {'accuracy': float(result.accuracy)}
