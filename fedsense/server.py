from __future__ import annotations

from typing import Any
import numpy as np

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from fedsense.config import AppConfig
from fedsense.contracts import ClientPayload, validate_client_payload
from fedsense.data.uci_har import FederatedDatasetBundle
from fedsense.model import build_model, get_model_parameters, set_model_parameters
from fedsense.runtime.metrics import MetricsRecorder
from fedsense.training import evaluate_model


def make_server_eval_fn(dataset: FederatedDatasetBundle, config: AppConfig, recorder: MetricsRecorder):
    def evaluate(server_round: int, parameters: fl.common.NDArrays, _: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        model = build_model(seed=config.training.seed)
        set_model_parameters(model, list(parameters))
        result = evaluate_model(model, dataset.test_x, dataset.test_y, config)
        recorder.record_server_evaluation(server_round=server_round, loss=result.loss, accuracy=result.accuracy)
        return float(result.loss), {'accuracy': float(result.accuracy)}

    return evaluate


class ValidatingFedAdam(fl.server.strategy.FedAdam):
    def __init__(self, recorder: MetricsRecorder, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.recorder = recorder

    def aggregate_fit(self, server_round: int, results: list[Any], failures: list[Any]):
        
        # Security Filter: Malicious Node Rejection (Byzantine Tolerance)
        magnitudes = []
        parsed_payloads = []
        
        for client, fit_res in results:
            payload = ClientPayload(
                weights=list(parameters_to_ndarrays(fit_res.parameters)),
                num_examples=int(fit_res.num_examples),
                metrics=dict(fit_res.metrics),
            )
            validate_client_payload(payload)
            parsed_payloads.append((client, fit_res, payload))
            
            # Calculate mathematical magnitude (Euclidean norm) of the client's weight updates
            # to check if it's wildly diverging from the norm.
            mag = sum(np.linalg.norm(w) for w in payload.weights)
            magnitudes.append(mag)
        
        security_rejections = 0
        valid_results = []
        round_results: list[tuple[int, dict[str, Any]]] = []
        
        if magnitudes and len(magnitudes) >= 2:
            median_mag = np.median(magnitudes)
            # Threshold to mathematically drop a malicious client
            # A client update larger than 1.8x the median is considered poison (Byzantine fault)
            threshold = median_mag * 1.8
            
            for (client, fit_res, payload), mag in zip(parsed_payloads, magnitudes):
                if mag > threshold and payload.num_examples > 0:
                    print(f"\n[SECURITY] 🛡️ Byzantine Threat Blocked! Client magnitude {mag:.2f} exceeded threshold {threshold:.2f}.")
                    failures.append((client, fit_res))
                    security_rejections += 1
                else:
                    valid_results.append((client, fit_res))
                    round_results.append((payload.num_examples, payload.metrics))
        else:
            valid_results = results
            for _, _, payload in parsed_payloads:
                round_results.append((payload.num_examples, payload.metrics))

        # Pass findings securely to the visual dashboard CSV Logger
        self.recorder.record_fit_round(server_round=server_round, results=round_results, security_rejections=security_rejections)
        
        # Continue FedAdam Adaptive global momentum update
        return super().aggregate_fit(server_round, valid_results, failures)


def create_strategy(dataset: FederatedDatasetBundle, config: AppConfig, recorder: MetricsRecorder) -> ValidatingFedAdam:
    initial_model = build_model(seed=config.training.seed)
    initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

    return ValidatingFedAdam(
        recorder=recorder,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=config.training.min_fit_clients,
        min_available_clients=config.training.min_available_clients,
        on_fit_config_fn=lambda round_number: {
            'server_round': round_number,
            'strategy': 'fedadam',
            'local_epochs': config.training.local_epochs,
            'learning_rate': config.training.learning_rate,
        },
        evaluate_fn=make_server_eval_fn(dataset=dataset, config=config, recorder=recorder),
        initial_parameters=initial_parameters,
        eta=1e-2,
        eta_l=1e-2,
        tau=1e-5,
    )
