from __future__ import annotations

import argparse

from fedsense.config import load_config
from fedsense.dashboard import create_dashboard_app


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the FedSense dashboard.')
    parser.add_argument('--config', default='configs/default.toml')
    args = parser.parse_args()

    config = load_config(args.config)
    app, socketio = create_dashboard_app(config.output.metrics_csv)
    socketio.run(app, host=config.extensions.dashboard_host, port=config.extensions.dashboard_port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()
