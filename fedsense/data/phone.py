from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocessing import normalize_windows

ACTIVITY_LABELS = [
    'WALKING',
    'WALKING_UPSTAIRS',
    'WALKING_DOWNSTAIRS',
    'SITTING',
    'STANDING',
    'LAYING',
]
ACTIVITY_TO_INDEX = {label: index for index, label in enumerate(ACTIVITY_LABELS)}
SENSOR_COLUMNS = ('acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z')


@dataclass(slots=True)
class PhoneDatasetBundle:
    windows: np.ndarray
    labels: np.ndarray
    num_windows: int


def encode_activity_label(label: str | int) -> int:
    if isinstance(label, str):
        normalized = label.strip().upper()
        if normalized.isdigit():
            value = int(normalized)
            return value - 1 if 1 <= value <= len(ACTIVITY_LABELS) else value
        if normalized not in ACTIVITY_TO_INDEX:
            raise ValueError(f"Unknown activity label '{label}'.")
        return ACTIVITY_TO_INDEX[normalized]

    value = int(label)
    if 1 <= value <= len(ACTIVITY_LABELS):
        return value - 1
    if 0 <= value < len(ACTIVITY_LABELS):
        return value
    raise ValueError(f'Activity label {label} is outside the supported UCI HAR label space.')


def _majority_label(window_labels: np.ndarray) -> int:
    return Counter(window_labels.tolist()).most_common(1)[0][0]


def load_phone_windows(
    csv_path: Path,
    label_column: str,
    window_size: int,
    stride: int,
    normalization_mean: np.ndarray,
    normalization_std: np.ndarray,
    epsilon: float,
) -> PhoneDatasetBundle:
    if not csv_path.exists():
        return PhoneDatasetBundle(
            windows=np.empty((0, len(SENSOR_COLUMNS), window_size), dtype=np.float32),
            labels=np.empty((0,), dtype=np.int64),
            num_windows=0,
        )

    frame = pd.read_csv(csv_path)
    required_columns = {label_column, *SENSOR_COLUMNS}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f'Phone CSV is missing required columns: {sorted(missing)}')

    labels = np.array([encode_activity_label(value) for value in frame[label_column].tolist()], dtype=np.int64)
    features = frame.loc[:, SENSOR_COLUMNS].to_numpy(dtype=np.float32)

    if len(features) < window_size:
        return PhoneDatasetBundle(
            windows=np.empty((0, len(SENSOR_COLUMNS), window_size), dtype=np.float32),
            labels=np.empty((0,), dtype=np.int64),
            num_windows=0,
        )

    windows: list[np.ndarray] = []
    window_labels: list[int] = []
    for start in range(0, len(features) - window_size + 1, stride):
        end = start + window_size
        windows.append(features[start:end].T)
        window_labels.append(_majority_label(labels[start:end]))

    stacked = np.stack(windows).astype(np.float32)
    normalized, _, _ = normalize_windows(
        stacked,
        mean=normalization_mean.astype(np.float32),
        std=normalization_std.astype(np.float32),
        epsilon=epsilon,
    )
    encoded_labels = np.array(window_labels, dtype=np.int64)
    return PhoneDatasetBundle(windows=normalized, labels=encoded_labels, num_windows=len(encoded_labels))
