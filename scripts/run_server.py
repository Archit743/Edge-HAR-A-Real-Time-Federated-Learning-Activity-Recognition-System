from __future__ import annotations

import argparse

import flwr as fl

from fedsense.baseline import run_centralized_baseline
from fedsense.config import ensure_runtime_directories, load_config
from fedsense.data.uci_har import prepare_federated_uci_har
from fedsense.runtime.metrics import MetricsRecorder
from fedsense.runtime.privacy import write_privacy_report
from fedsense.server import create_strategy


def main() -> None:
    parser = argparse.ArgumentParser(description='Start the FedSense Flower server.')
    parser.add_argument('--config', default='configs/default.toml')
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_runtime_directories(config)

    dataset = prepare_federated_uci_har(config)
    run_centralized_baseline(config, dataset)

    recorder = MetricsRecorder(
        metrics_csv=config.output.metrics_csv,
        plot_path=config.output.plot_path,
        strategy_name=config.training.strategy,
    )
    strategy = create_strategy(dataset=dataset, config=config, recorder=recorder)
    server_address = f'{config.server.host}:{config.server.port}'

    try:
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=config.training.rounds),
            strategy=strategy,
        )
    finally:
        recorder.render_plot()
        write_privacy_report(output_root=config.output.root, report_path=config.output.privacy_report)


if __name__ == '__main__':
    main()
