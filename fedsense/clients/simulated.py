from __future__ import annotations

from typing import Any

import flwr as fl

from fedsense.config import AppConfig
from fedsense.contracts import build_client_payload
from fedsense.data.uci_har import ClientPartition
from fedsense.model import build_model, get_model_parameters, set_model_parameters
from fedsense.training import evaluate_model, train_local_model


class SimulatedHarClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, data: ClientPartition, config: AppConfig) -> None:
        self.client_id = client_id
        self.data = data
        self.config = config
        self.model = build_model(seed=config.training.seed)

    def get_parameters(self, config: dict[str, Any]) -> list[Any]:
        return get_model_parameters(self.model)

    def fit(self, parameters: list[Any], config: dict[str, Any]) -> tuple[list[Any], int, dict[str, Any]]:
        set_model_parameters(self.model, parameters)
        result = train_local_model(self.model, self.data.train_x, self.data.train_y, self.config)
        updated = get_model_parameters(self.model)

        metrics = {
            'client_id': self.client_id,
            'client_type': 'simulated',
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
        result = evaluate_model(self.model, self.data.val_x, self.data.val_y, self.config)
        return float(result.loss), result.num_examples, {'accuracy': float(result.accuracy)}
