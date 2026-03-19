from __future__ import annotations

import argparse

from fedsense.baseline import run_centralized_baseline
from fedsense.config import ensure_runtime_directories, load_config
from fedsense.data.uci_har import prepare_federated_uci_har


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the centralized FedSense baseline.')
    parser.add_argument('--config', default='configs/default.toml')
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_runtime_directories(config)
    dataset = prepare_federated_uci_har(config)
    run_centralized_baseline(config, dataset)


if __name__ == '__main__':
    main()
