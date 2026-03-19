from __future__ import annotations

import argparse

import flwr as fl
import numpy as np

from fedsense.clients.phone import PhoneHarClient
from fedsense.config import load_config
from fedsense.data.partition import load_partition_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the FedSense phone client.')
    parser.add_argument('--config', default='configs/default.toml')
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = load_partition_manifest(config.data.partition_manifest)
    client = PhoneHarClient(
        config=config,
        normalization_mean=np.array(manifest.normalization_mean, dtype=np.float32),
        normalization_std=np.array(manifest.normalization_std, dtype=np.float32),
    )
    fl.client.start_numpy_client(server_address=f'{config.server.host}:{config.server.port}', client=client)


if __name__ == '__main__':
    main()
