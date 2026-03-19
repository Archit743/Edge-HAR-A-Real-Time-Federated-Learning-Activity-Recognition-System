from __future__ import annotations

import argparse

import flwr as fl

from fedsense.clients.simulated import SimulatedHarClient
from fedsense.config import load_config
from fedsense.data.uci_har import prepare_federated_uci_har


def main() -> None:
    parser = argparse.ArgumentParser(description='Run one networked simulated FedSense client.')
    parser.add_argument('--config', default='configs/default.toml')
    parser.add_argument('--client-id', required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = prepare_federated_uci_har(config)
    partition = dataset.client_partitions[args.client_id]
    client = SimulatedHarClient(client_id=args.client_id, data=partition, config=config)
    fl.client.start_numpy_client(server_address=f'{config.server.host}:{config.server.port}', client=client)


if __name__ == '__main__':
    main()
