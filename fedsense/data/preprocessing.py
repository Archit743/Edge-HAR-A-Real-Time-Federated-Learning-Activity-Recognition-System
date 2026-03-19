from __future__ import annotations

import numpy as np


def normalize_windows(
    windows: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if windows.ndim != 3:
        raise ValueError(f'Expected windows with shape [samples, channels, time], received {windows.shape}')

    if mean is None:
        mean = windows.mean(axis=(0, 2), dtype=np.float64)
    if std is None:
        std = windows.std(axis=(0, 2), dtype=np.float64)

    safe_std = np.maximum(std, epsilon)
    normalized = (windows - mean[None, :, None]) / safe_std[None, :, None]
    return normalized.astype(np.float32), mean.astype(np.float32), safe_std.astype(np.float32)
