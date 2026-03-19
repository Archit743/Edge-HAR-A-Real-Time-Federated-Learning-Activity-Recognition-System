import numpy as np
import pytest
import torch

from fedsense.model import (
    build_model,
    get_model_parameters,
    set_model_parameters,
)


def test_build_model_reproducibility():
    model1 = build_model(seed=42)
    model2 = build_model(seed=42)

    params1 = get_model_parameters(model1)
    params2 = get_model_parameters(model2)

    for p1, p2 in zip(params1, params2):
        np.testing.assert_array_equal(p1, p2)

def test_har_1dcnn_forward_shape():
    model = build_model(seed=42, input_channels=6, num_classes=6)
    # Batch size 8, 6 channels, 128 sequence length
    x = torch.randn(8, 6, 128)
    out = model(x)
    assert out.shape == (8, 6)

def test_har_1dcnn_forward_invalid_shape():
    model = build_model(seed=42)

    # Missing channel dim
    with pytest.raises(ValueError, match="Expected input with shape"):
        model(torch.randn(8, 128))

    # Wrong number of channels
    with pytest.raises(ValueError, match="Expected 6 channels"):
        model(torch.randn(8, 3, 128))

def test_get_and_set_model_parameters():
    model = build_model(seed=1)
    params = get_model_parameters(model)

    # Modify params
    new_params = [p + 1.0 for p in params]

    set_model_parameters(model, new_params)
    updated_params = get_model_parameters(model)

    for p_new, p_upd in zip(new_params, updated_params):
        np.testing.assert_array_equal(p_new, p_upd)

def test_set_model_parameters_mismatch():
    model = build_model(seed=1)
    params = get_model_parameters(model)

    # Remove one layer
    with pytest.raises(ValueError, match="Parameter count mismatch"):
        set_model_parameters(model, params[:-1])

    # Corrupt shape
    corrupted_params = list(params)
    corrupted_params[0] = np.zeros((1, 1, 1))

    with pytest.raises(ValueError, match="Shape mismatch"):
        set_model_parameters(model, corrupted_params)
