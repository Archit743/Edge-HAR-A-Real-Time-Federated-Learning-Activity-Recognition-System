from __future__ import annotations

import csv

import numpy as np

from fedsense.config import AppConfig
from fedsense.data.uci_har import FederatedDatasetBundle
from fedsense.model import build_model
from fedsense.training import evaluate_model, train_local_model


def run_centralized_baseline(config: AppConfig, dataset: FederatedDatasetBundle):
    train_x = [partition.train_x for partition in dataset.client_partitions.values()]
    train_y = [partition.train_y for partition in dataset.client_partitions.values()]

    merged_x = train_x[0] if len(train_x) == 1 else np.concatenate(train_x, axis=0)
    merged_y = train_y[0] if len(train_y) == 1 else np.concatenate(train_y, axis=0)

    model = build_model(seed=config.training.seed)
    train_result = train_local_model(model, merged_x, merged_y, config, local_epochs=max(1, config.training.local_epochs * 2))
    test_result = evaluate_model(model, dataset.test_x, dataset.test_y, config)

    baseline_path = config.output.baseline_csv
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with baseline_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['train_loss', 'train_accuracy', 'test_loss', 'test_accuracy', 'num_train_examples', 'num_test_examples'],
        )
        writer.writeheader()
        writer.writerow(
            {
                'train_loss': f'{train_result.loss:.6f}',
                'train_accuracy': f'{train_result.accuracy:.6f}',
                'test_loss': f'{test_result.loss:.6f}',
                'test_accuracy': f'{test_result.accuracy:.6f}',
                'num_train_examples': train_result.num_examples,
                'num_test_examples': test_result.num_examples,
            }
        )

    return baseline_path
