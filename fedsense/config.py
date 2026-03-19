from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainingConfig:
    rounds: int
    local_epochs: int
    learning_rate: float
    batch_size: int
    strategy: str
    proximal_mu: float
    seed: int
    device: str
    min_fit_clients: int
    min_available_clients: int
    min_phone_windows_per_round: int


@dataclass(slots=True)
class ServerConfig:
    host: str
    port: int


@dataclass(slots=True)
class DataConfig:
    dataset_root: Path
    partition_manifest: Path
    dirichlet_alpha: float
    num_clients: int
    validation_fraction: float
    window_size: int
    stride: int
    normalization_epsilon: float


@dataclass(slots=True)
class OutputConfig:
    root: Path
    metrics_csv: Path
    plot_path: Path
    privacy_report: Path
    baseline_csv: Path


@dataclass(slots=True)
class PhoneConfig:
    csv_path: Path
    timestamp_column: str
    label_column: str


@dataclass(slots=True)
class ExtensionConfig:
    enable_dashboard: bool
    dashboard_host: str
    dashboard_port: int
    enable_dp: bool
    dp_noise_multiplier: float
    dp_max_grad_norm: float


@dataclass(slots=True)
class AppConfig:
    training: TrainingConfig
    server: ServerConfig
    data: DataConfig
    output: OutputConfig
    phone: PhoneConfig
    extensions: ExtensionConfig


VALID_STRATEGIES = {'fedavg', 'fedprox'}


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def load_config(config_path: str | Path = 'configs/default.toml') -> AppConfig:
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent

    with config_path.open('rb') as handle:
        payload = tomllib.load(handle)

    training = payload['training']
    server = payload['server']
    data = payload['data']
    output = payload['output']
    phone = payload['phone']
    extensions = payload['extensions']

    app_config = AppConfig(
        training=TrainingConfig(
            rounds=int(training['rounds']),
            local_epochs=int(training['local_epochs']),
            learning_rate=float(training['learning_rate']),
            batch_size=int(training['batch_size']),
            strategy=str(training['strategy']).lower(),
            proximal_mu=float(training['proximal_mu']),
            seed=int(training['seed']),
            device=str(training['device']),
            min_fit_clients=int(training['min_fit_clients']),
            min_available_clients=int(training['min_available_clients']),
            min_phone_windows_per_round=int(training['min_phone_windows_per_round']),
        ),
        server=ServerConfig(
            host=str(server['host']),
            port=int(server['port']),
        ),
        data=DataConfig(
            dataset_root=_resolve_path(project_root, data['dataset_root']),
            partition_manifest=_resolve_path(project_root, data['partition_manifest']),
            dirichlet_alpha=float(data['dirichlet_alpha']),
            num_clients=int(data['num_clients']),
            validation_fraction=float(data['validation_fraction']),
            window_size=int(data['window_size']),
            stride=int(data['stride']),
            normalization_epsilon=float(data['normalization_epsilon']),
        ),
        output=OutputConfig(
            root=_resolve_path(project_root, output['root']),
            metrics_csv=_resolve_path(project_root, output['metrics_csv']),
            plot_path=_resolve_path(project_root, output['plot_path']),
            privacy_report=_resolve_path(project_root, output['privacy_report']),
            baseline_csv=_resolve_path(project_root, output['baseline_csv']),
        ),
        phone=PhoneConfig(
            csv_path=_resolve_path(project_root, phone['csv_path']),
            timestamp_column=str(phone['timestamp_column']),
            label_column=str(phone['label_column']),
        ),
        extensions=ExtensionConfig(
            enable_dashboard=bool(extensions['enable_dashboard']),
            dashboard_host=str(extensions['dashboard_host']),
            dashboard_port=int(extensions['dashboard_port']),
            enable_dp=bool(extensions['enable_dp']),
            dp_noise_multiplier=float(extensions['dp_noise_multiplier']),
            dp_max_grad_norm=float(extensions['dp_max_grad_norm']),
        ),
    )
    validate_config(app_config)
    return app_config


def validate_config(config: AppConfig) -> None:
    if config.training.strategy not in VALID_STRATEGIES:
        allowed = ', '.join(sorted(VALID_STRATEGIES))
        raise ValueError(f"Unsupported training.strategy '{config.training.strategy}'. Allowed values: {allowed}")

    if config.training.rounds <= 0:
        raise ValueError('training.rounds must be greater than 0.')
    if config.training.local_epochs <= 0:
        raise ValueError('training.local_epochs must be greater than 0.')
    if config.training.batch_size <= 0:
        raise ValueError('training.batch_size must be greater than 0.')
    if config.training.learning_rate <= 0:
        raise ValueError('training.learning_rate must be greater than 0.')

    if config.training.min_fit_clients <= 0:
        raise ValueError('training.min_fit_clients must be greater than 0.')
    if config.training.min_available_clients < config.training.min_fit_clients:
        raise ValueError('training.min_available_clients must be >= training.min_fit_clients.')

    if config.data.num_clients <= 0:
        raise ValueError('data.num_clients must be greater than 0.')
    if not (0 < config.data.validation_fraction < 1):
        raise ValueError('data.validation_fraction must be between 0 and 1.')
    if config.data.dirichlet_alpha <= 0:
        raise ValueError('data.dirichlet_alpha must be greater than 0.')


def ensure_runtime_directories(config: AppConfig) -> None:
    config.output.root.mkdir(parents=True, exist_ok=True)
    config.data.partition_manifest.parent.mkdir(parents=True, exist_ok=True)
    config.phone.csv_path.parent.mkdir(parents=True, exist_ok=True)
