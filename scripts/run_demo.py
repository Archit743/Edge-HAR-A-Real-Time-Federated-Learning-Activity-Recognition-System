from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

from fedsense.config import load_config


def _wait_for_server(host: str, port: int, timeout_seconds: float) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the FedSense server plus two local simulated clients.')
    parser.add_argument('--config', default='configs/default.toml')
    parser.add_argument('--warmup-seconds', type=float, default=15.0)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    python_executable = sys.executable
    config = load_config(project_root / args.config)

    server_cmd = [python_executable, str(project_root / 'scripts' / 'run_server.py'), '--config', args.config]
    client_zero_cmd = [
        python_executable,
        str(project_root / 'scripts' / 'run_simulated_client.py'),
        '--config',
        args.config,
        '--client-id',
        '0',
    ]
    client_one_cmd = [
        python_executable,
        str(project_root / 'scripts' / 'run_simulated_client.py'),
        '--config',
        args.config,
        '--client-id',
        '1',
    ]

    server = subprocess.Popen(server_cmd, cwd=project_root)
    try:
        if not _wait_for_server(config.server.host, config.server.port, timeout_seconds=args.warmup_seconds):
            raise RuntimeError(
                f'Server did not become ready at {config.server.host}:{config.server.port} '
                f'within {args.warmup_seconds:.1f}s.'
            )
        clients = [
            subprocess.Popen(client_zero_cmd, cwd=project_root),
            subprocess.Popen(client_one_cmd, cwd=project_root),
        ]
        for client in clients:
            client.wait()
        server.wait()
    finally:
        if server.poll() is None:
            server.terminate()
            server.wait(timeout=5)


if __name__ == '__main__':
    main()
