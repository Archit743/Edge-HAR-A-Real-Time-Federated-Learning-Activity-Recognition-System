from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

Scalar = bool | float | int | str
FORBIDDEN_METRIC_KEYS = {'raw_sensor_data', 'sensor_buffer', 'sensor_windows', 'phone_path', 'file_path'}


@dataclass(slots=True)
class ClientPayload:
    weights: list[np.ndarray]
    num_examples: int
    metrics: dict[str, Scalar]


def _is_scalar_metric(value: Any) -> bool:
    return isinstance(value, (bool, float, int, str))


def _contains_path_like_value(value: str) -> bool:
    lowered = value.lower()
    return any(
        marker in lowered
        for marker in ('/storage/', '/sdcard/', 'termux', '\\users\\', 'c:\\', 'd:\\', '/data/data/')
    )


def validate_weights(weights: list[np.ndarray]) -> None:
    if not isinstance(weights, list) or not weights:
        raise ValueError('Model weights must be a non-empty list of NumPy arrays.')

    for index, array in enumerate(weights):
        if not isinstance(array, np.ndarray):
            raise TypeError(f'Weight index {index} is not a NumPy array.')


def validate_scalar_metrics(metrics: dict[str, Scalar]) -> None:
    for key, value in metrics.items():
        if key in FORBIDDEN_METRIC_KEYS:
            raise ValueError(f'Forbidden metric key encountered: {key}')
        if not _is_scalar_metric(value):
            raise TypeError(f"Metric '{key}' must be scalar, received {type(value).__name__}.")
        if isinstance(value, str) and _contains_path_like_value(value):
            raise ValueError(f"Metric '{key}' appears to leak a local path or device storage reference.")


def validate_client_payload(payload: ClientPayload) -> None:
    validate_weights(payload.weights)
    if payload.num_examples < 0:
        raise ValueError('num_examples cannot be negative.')
    validate_scalar_metrics(payload.metrics)


def build_client_payload(
    weights: list[np.ndarray],
    num_examples: int,
    metrics: dict[str, Scalar],
) -> ClientPayload:
    payload = ClientPayload(weights=weights, num_examples=num_examples, metrics=metrics)
    validate_client_payload(payload)
    return payload
