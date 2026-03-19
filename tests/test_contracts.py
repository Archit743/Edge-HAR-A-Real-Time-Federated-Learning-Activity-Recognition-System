import numpy as np
import pytest

from fedsense.contracts import (
    ClientPayload,
    build_client_payload,
    validate_scalar_metrics,
    validate_weights,
)


def test_validate_weights_valid():
    weights = [np.array([1.0, 2.0]), np.array([[3.0]])]
    validate_weights(weights)  # Should not raise

def test_validate_weights_invalid():
    with pytest.raises(ValueError):
        validate_weights([])

    with pytest.raises(TypeError):
        validate_weights([np.array([1.0]), "not-an-array"]) # type: ignore

def test_validate_scalar_metrics_valid():
    metrics = {"loss": 0.5, "acc": 0.9, "is_training": True, "status": "ok"}
    validate_scalar_metrics(metrics)

def test_validate_scalar_metrics_forbidden_keys():
    metrics = {"loss": 0.5, "sensor_buffer": 123}
    with pytest.raises(ValueError, match="Forbidden metric key"):
        validate_scalar_metrics(metrics)

def test_validate_scalar_metrics_invalid_type():
    metrics = {"loss": 0.5, "list_metric": [1, 2, 3]}
    with pytest.raises(TypeError):
        validate_scalar_metrics(metrics) # type: ignore

def test_validate_scalar_metrics_path_leak():
    metrics = {"loss": 0.5, "error_log": "Dumped to /data/data/com.termux/files/home"}
    with pytest.raises(ValueError, match="leak a local path"):
        validate_scalar_metrics(metrics)

def test_build_client_payload():
    weights = [np.array([1.0])]
    payload = build_client_payload(weights=weights, num_examples=100, metrics={"loss": 0.1})
    assert isinstance(payload, ClientPayload)
    assert payload.num_examples == 100

def test_build_client_payload_negative_examples():
    weights = [np.array([1.0])]
    with pytest.raises(ValueError, match="num_examples cannot be negative"):
        build_client_payload(weights=weights, num_examples=-5, metrics={})
