from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
from torch.utils.tensorboard import SummaryWriter

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class MetricsRecorder:
    def __init__(self, metrics_csv: Path, plot_path: Path, strategy_name: str) -> None:
        self.metrics_csv = metrics_csv
        
        if self.metrics_csv.exists():
            backup_path = self.metrics_csv.parent / f"{self.metrics_csv.stem}_previous{self.metrics_csv.suffix}"
            try:
                if backup_path.exists():
                    backup_path.unlink()
                self.metrics_csv.rename(backup_path)
            except Exception:
                pass
                
        self.plot_path = plot_path
        self.strategy_name = strategy_name
        self.rows: dict[int, dict[str, object]] = {}

        run_dir = self.metrics_csv.parent / "runs" / self.strategy_name
        self.writer = SummaryWriter(log_dir=str(run_dir))

    def record_fit_round(self, server_round: int, results: list[tuple[int, dict[str, object]]], security_rejections: int = 0) -> None:
        row = self.rows.setdefault(
            server_round,
            {
                'round': server_round,
                'strategy': self.strategy_name,
                'participating_clients': 0,
                'total_examples': 0,
                'skipped_clients': 0,
                'phone_participated': 0,
                'train_loss': 0.0,
                'train_accuracy': 0.0,
                'server_loss': '',
                'server_accuracy': '',
                'security_rejections': 0,
            },
        )

        total_examples = sum(num_examples for num_examples, _ in results if num_examples > 0)
        weighted_loss = 0.0
        weighted_accuracy = 0.0

        row['participating_clients'] = len(results)
        row['total_examples'] = total_examples
        row['security_rejections'] = security_rejections
        row['skipped_clients'] = sum(
            1 for num_examples, metrics in results if int(metrics.get('skipped', 0)) == 1 or num_examples == 0
        )
        row['phone_participated'] = int(
            any(metrics.get('client_type') == 'phone' and num_examples > 0 for num_examples, metrics in results)
        )

        if total_examples > 0:
            for num_examples, metrics in results:
                if num_examples <= 0:
                    continue
                weighted_loss += float(metrics.get('train_loss', 0.0)) * num_examples
                weighted_accuracy += float(metrics.get('train_accuracy', 0.0)) * num_examples
            row['train_loss'] = weighted_loss / total_examples
            row['train_accuracy'] = weighted_accuracy / total_examples

            self.writer.add_scalar("Train/Loss", row['train_loss'], server_round)
            self.writer.add_scalar("Train/Accuracy", row['train_accuracy'], server_round)

        self._write_csv()

    def record_server_evaluation(self, server_round: int, loss: float, accuracy: float) -> None:
        row = self.rows.setdefault(server_round, {'round': server_round, 'strategy': self.strategy_name})
        row['server_loss'] = float(loss)
        row['server_accuracy'] = float(accuracy)

        self.writer.add_scalar("Server/Loss", loss, server_round)
        self.writer.add_scalar("Server/Accuracy", accuracy, server_round)

        self._write_csv()
        self.render_plot()

    def _write_csv(self) -> None:
        self.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            'round',
            'strategy',
            'participating_clients',
            'total_examples',
            'skipped_clients',
            'phone_participated',
            'train_loss',
            'train_accuracy',
            'server_loss',
            'server_accuracy',
            'security_rejections',
        ]
        with self.metrics_csv.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for round_number in sorted(self.rows):
                writer.writerow(self.rows[round_number])

    def render_plot(self) -> None:
        rounds = sorted(self.rows)
        if not rounds:
            return

        plotted_rounds = [round_number for round_number in rounds if self.rows[round_number].get('server_loss', '') != '']
        if not plotted_rounds:
            return

        loss_values = [float(self.rows[round_number]['server_loss']) for round_number in plotted_rounds]
        accuracy_values = [float(self.rows[round_number]['server_accuracy']) for round_number in plotted_rounds]

        figure, left_axis = plt.subplots(figsize=(8, 4.5))
        right_axis = left_axis.twinx()

        left_axis.plot(plotted_rounds, loss_values, label='Server Loss', color='#1f77b4', linewidth=2)
        right_axis.plot(plotted_rounds, accuracy_values, label='Server Accuracy', color='#d62728', linewidth=2)

        left_axis.set_xlabel('Round')
        left_axis.set_ylabel('Loss')
        right_axis.set_ylabel('Accuracy')
        left_axis.set_title(f'FedSense Convergence ({self.strategy_name.upper()})')
        left_axis.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(self.plot_path, dpi=150)
        plt.close(figure)
