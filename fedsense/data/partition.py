from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class PartitionManifest:
    seed: int
    alpha: float
    num_clients: int
    validation_fraction: float
    normalization_mean: list[float]
    normalization_std: list[float]
    partitions: dict[str, dict[str, list[int]]]


def load_partition_manifest(manifest_path: Path) -> PartitionManifest:
    if not manifest_path.exists():
        raise FileNotFoundError(f'Partition manifest not found: {manifest_path}')
    with manifest_path.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    return PartitionManifest(**payload)


def _is_manifest_compatible(
    manifest: PartitionManifest,
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    validation_fraction: float,
    seed: int,
) -> bool:
    if manifest.seed != seed:
        return False
    if manifest.num_clients != num_clients:
        return False
    if not np.isclose(manifest.alpha, alpha):
        return False
    if not np.isclose(manifest.validation_fraction, validation_fraction):
        return False
    if len(manifest.partitions) != num_clients:
        return False

    max_index = len(labels) - 1
    for partition in manifest.partitions.values():
        for split_key in ('train', 'val'):
            indices = partition.get(split_key, [])
            if any((index < 0 or index > max_index) for index in indices):
                return False

    return True


def _round_allocation(proportions: np.ndarray, total: int) -> np.ndarray:
    counts = np.floor(proportions * total).astype(int)
    remainder = total - int(counts.sum())
    if remainder > 0:
        for index in np.argsort(proportions)[-remainder:]:
            counts[index] += 1
    return counts


def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float, seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    partitions: list[list[int]] = [[] for _ in range(num_clients)]

    for class_id in np.unique(labels):
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        counts = _round_allocation(proportions, len(class_indices))

        start = 0
        for client_id, count in enumerate(counts):
            end = start + count
            partitions[client_id].extend(class_indices[start:end].tolist())
            start = end

    for client_id, indices in enumerate(partitions):
        if indices:
            rng.shuffle(indices)
            continue
        donor = max(range(num_clients), key=lambda idx: len(partitions[idx]))
        partitions[client_id].append(partitions[donor].pop())

    return {str(client_id): sorted(indices) for client_id, indices in enumerate(partitions)}


def split_train_validation(indices: list[int], validation_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    shuffled = np.array(indices, dtype=np.int64)
    rng.shuffle(shuffled)

    if len(shuffled) < 2:
        return shuffled.tolist(), []

    validation_size = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_size = min(validation_size, len(shuffled) - 1)

    val_indices = shuffled[:validation_size].tolist()
    train_indices = shuffled[validation_size:].tolist()
    return train_indices, val_indices


def create_partition_manifest(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    validation_fraction: float,
    seed: int,
    normalization_mean: np.ndarray,
    normalization_std: np.ndarray,
) -> PartitionManifest:
    coarse = dirichlet_partition(labels=labels, num_clients=num_clients, alpha=alpha, seed=seed)
    partitions: dict[str, dict[str, list[int]]] = {}

    for client_id, indices in coarse.items():
        train_indices, val_indices = split_train_validation(indices, validation_fraction=validation_fraction, seed=seed + int(client_id))
        partitions[client_id] = {'train': train_indices, 'val': val_indices}

    return PartitionManifest(
        seed=seed,
        alpha=alpha,
        num_clients=num_clients,
        validation_fraction=validation_fraction,
        normalization_mean=normalization_mean.astype(float).tolist(),
        normalization_std=normalization_std.astype(float).tolist(),
        partitions=partitions,
    )


def load_or_create_partition_manifest(
    manifest_path: Path,
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    validation_fraction: float,
    seed: int,
    normalization_mean: np.ndarray,
    normalization_std: np.ndarray,
) -> PartitionManifest:
    if manifest_path.exists():
        manifest = load_partition_manifest(manifest_path)
        if _is_manifest_compatible(
            manifest=manifest,
            labels=labels,
            num_clients=num_clients,
            alpha=alpha,
            validation_fraction=validation_fraction,
            seed=seed,
        ):
            return manifest

    manifest = create_partition_manifest(
        labels=labels,
        num_clients=num_clients,
        alpha=alpha,
        validation_fraction=validation_fraction,
        seed=seed,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open('w', encoding='utf-8') as handle:
        json.dump(asdict(manifest), handle, indent=2)
    return manifest
