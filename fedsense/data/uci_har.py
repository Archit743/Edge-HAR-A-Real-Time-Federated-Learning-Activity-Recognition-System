from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fedsense.config import AppConfig

from .partition import load_or_create_partition_manifest
from .preprocessing import normalize_windows

SIGNAL_FILES = (
    'body_acc_x',
    'body_acc_y',
    'body_acc_z',
    'body_gyro_x',
    'body_gyro_y',
    'body_gyro_z',
)


@dataclass(slots=True)
class ClientPartition:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray


@dataclass(slots=True)
class FederatedDatasetBundle:
    client_partitions: dict[str, ClientPartition]
    test_x: np.ndarray
    test_y: np.ndarray
    normalization_mean: np.ndarray
    normalization_std: np.ndarray


def _load_signal_matrix(dataset_root: Path, split: str, signal_name: str) -> np.ndarray:
    signal_path = dataset_root / split / 'Inertial Signals' / f'{signal_name}_{split}.txt'
    return np.loadtxt(signal_path, dtype=np.float32)


def load_uci_har_split(dataset_root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    if not dataset_root.exists():
        raise FileNotFoundError(f'UCI HAR dataset path not found: {dataset_root}')

    signals = [_load_signal_matrix(dataset_root, split=split, signal_name=name) for name in SIGNAL_FILES]
    features = np.stack(signals, axis=1).astype(np.float32)
    labels_path = dataset_root / split / f'y_{split}.txt'
    labels = np.loadtxt(labels_path, dtype=np.int64) - 1
    return features, labels


def prepare_federated_uci_har(config: AppConfig) -> FederatedDatasetBundle:
    train_x, train_y = load_uci_har_split(config.data.dataset_root, split='train')
    test_x, test_y = load_uci_har_split(config.data.dataset_root, split='test')

    normalized_train, mean, std = normalize_windows(train_x, epsilon=config.data.normalization_epsilon)
    normalized_test, _, _ = normalize_windows(
        test_x,
        mean=mean,
        std=std,
        epsilon=config.data.normalization_epsilon,
    )

    manifest = load_or_create_partition_manifest(
        manifest_path=config.data.partition_manifest,
        labels=train_y,
        num_clients=config.data.num_clients,
        alpha=config.data.dirichlet_alpha,
        validation_fraction=config.data.validation_fraction,
        seed=config.training.seed,
        normalization_mean=mean,
        normalization_std=std,
    )

    mean = np.array(manifest.normalization_mean, dtype=np.float32)
    std = np.array(manifest.normalization_std, dtype=np.float32)
    client_partitions: dict[str, ClientPartition] = {}

    for client_id, partition in manifest.partitions.items():
        train_indices = np.array(partition['train'], dtype=np.int64)
        val_indices = np.array(partition['val'], dtype=np.int64)

        client_partitions[client_id] = ClientPartition(
            train_x=normalized_train[train_indices],
            train_y=train_y[train_indices],
            val_x=normalized_train[val_indices] if len(val_indices) else normalized_train[train_indices],
            val_y=train_y[val_indices] if len(val_indices) else train_y[train_indices],
        )

    return FederatedDatasetBundle(
        client_partitions=client_partitions,
        test_x=normalized_test,
        test_y=test_y,
        normalization_mean=mean,
        normalization_std=std,
    )
