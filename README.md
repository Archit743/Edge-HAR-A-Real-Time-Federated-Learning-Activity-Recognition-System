# Edge-HAR

Edge-HAR is a phased federated learning demo for human activity recognition on local WiFi. The laptop hosts the Flower server plus two simulated UCI HAR clients, and an Android Termux client can join as a real edge device without sending raw sensor data off-device.

## Phase-Oriented Build

The repo is structured to match the implementation phases from `implementation-plan.md`:

1. Shared contracts and model definition.
2. UCI HAR and phone data pipelines.
3. Baseline federated runtime with metrics logging.
4. Real phone client with graceful skip behavior.
5. Privacy validation and reporting.
6. Optional switches for FedProx, DP-SGD, and dashboard.

## Project Layout

```text
fedsense/
  clients/          # Flower clients for simulated and phone devices
  data/             # UCI HAR loading, partitioning, phone CSV windowing
  runtime/          # Metrics logging and privacy auditing
  baseline.py       # Centralized comparison run
  config.py         # Shared TOML config loader
  contracts.py      # Payload validators and network boundary rules
  dashboard.py      # Optional Flask dashboard
  model.py          # Shared 1D-CNN and parameter serialization helpers
  server.py         # Flower strategy and server-side evaluation
  training.py       # Local training/evaluation loops
configs/
  default.toml
scripts/
  run_baseline.py
  run_demo.py
  run_phone_client.py
  run_server.py
  run_simulated_client.py
  run_simulation.py
```

## Setup

```powershell
venv\Scripts\pip.exe install -r requirements.txt
```

## Main Commands

Simulated-only baseline:

```powershell
venv\Scripts\python.exe scripts\run_simulation.py
```

Server for hybrid WiFi demo:

```powershell
venv\Scripts\python.exe scripts\run_server.py
```

One-command local demo with server plus two laptop clients:

```powershell
venv\Scripts\python.exe scripts\run_demo.py
```

Phone client:

```powershell
venv\Scripts\python.exe scripts\run_phone_client.py
```

Note: the phone client loads normalization statistics from `artifacts/partitions/uci_har_partitions.json`.
Run `scripts/run_server.py` or `scripts/run_simulation.py` at least once to generate this manifest before starting the phone client.

## Expected Local Inputs

UCI HAR should live under `data/UCI HAR Dataset` by default. The phone CSV should be written to `phone/sensor_log.csv` by default and include:

- `timestamp`
- `label`
- `acc_x`, `acc_y`, `acc_z`
- `gyro_x`, `gyro_y`, `gyro_z`

Phone labels must match the UCI HAR activity set:

- `WALKING`
- `WALKING_UPSTAIRS`
- `WALKING_DOWNSTAIRS`
- `SITTING`
- `STANDING`
- `LAYING`

## Artifacts

Generated artifacts are written to `artifacts/`:

- `round_metrics.csv`
- `convergence_fedavg_vs_fedprox.png`
- `privacy_audit_report.md`
- `centralized_baseline_metrics.csv`
- `partitions/uci_har_partitions.json`
