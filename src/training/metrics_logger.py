"""
Metrics Logger for SASLM Training

Handles:
- Real-time metrics tracking
- Console output formatting
- Optional Weights & Biases integration
- CSV export for analysis
- Training statistics

Usage:
    logger = MetricsLogger("EXP-A1-prose-only")

    for step in training_loop:
        logger.log(step, {
            'train_loss': loss.item(),
            'learning_rate': scheduler.get_last_lr()[0],
        })

        if step % eval_interval == 0:
            logger.log(step, {'val_loss': val_loss}, eval_metrics=True)

    logger.save_summary()
"""

import json
import csv
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class MetricsLogger:
    """
    Comprehensive metrics logging for training experiments.

    Features:
    - Automatic moving averages
    - Time tracking
    - Console output with progress bars
    - Optional wandb integration
    - CSV export
    """

    experiment_name: str
    log_dir: str = './logs'
    console_interval: int = 100       # Print to console every N steps
    use_wandb: bool = False           # Enable Weights & Biases
    wandb_project: str = 'saslm'

    # Internal state
    _metrics: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    _steps: List[int] = field(default_factory=list)
    _timestamps: List[float] = field(default_factory=list)
    _start_time: float = field(default_factory=time.time)
    _last_log_time: float = field(default_factory=time.time)
    _step_times: List[float] = field(default_factory=list)
    _wandb_run: Any = None

    def __post_init__(self):
        """Initialize logging directory and optional wandb."""
        self.log_path = Path(self.log_dir) / self.experiment_name
        self.log_path.mkdir(parents=True, exist_ok=True)

        self._csv_path = self.log_path / 'metrics.csv'
        self._json_path = self.log_path / 'metrics.json'

        # Initialize wandb if requested
        if self.use_wandb:
            self._init_wandb()

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        try:
            import wandb
            self._wandb_run = wandb.init(
                project=self.wandb_project,
                name=self.experiment_name,
                resume='allow',
            )
            print(f"[Logger] wandb initialized: {wandb.run.url}")
        except ImportError:
            print("[Logger] wandb not installed. Install with: pip install wandb")
            self.use_wandb = False
        except Exception as e:
            print(f"[Logger] wandb init failed: {e}")
            self.use_wandb = False

    def log(
        self,
        step: int,
        metrics: Dict[str, float],
        eval_metrics: bool = False,
        print_now: bool = False,
    ):
        """
        Log metrics for a training step.

        Args:
            step: Current training step
            metrics: Dictionary of metric name -> value
            eval_metrics: If True, this is an evaluation step (always print)
            print_now: Force print to console
        """
        current_time = time.time()

        # Track time between logs
        if self._steps:
            self._step_times.append(current_time - self._last_log_time)
        self._last_log_time = current_time

        # Store metrics
        self._steps.append(step)
        self._timestamps.append(current_time)

        for name, value in metrics.items():
            self._metrics[name].append(value)

        # Log to wandb
        if self.use_wandb and self._wandb_run:
            import wandb
            wandb.log(metrics, step=step)

        # Console output
        should_print = (
            print_now or
            eval_metrics or
            (step % self.console_interval == 0)
        )

        if should_print:
            self._print_metrics(step, metrics, eval_metrics)

    def _print_metrics(self, step: int, metrics: Dict[str, float], is_eval: bool):
        """Print formatted metrics to console."""
        elapsed = time.time() - self._start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        # Calculate steps per second
        if len(self._step_times) > 10:
            recent_times = self._step_times[-10:]
            steps_per_sec = 1.0 / (sum(recent_times) / len(recent_times))
        else:
            steps_per_sec = step / max(elapsed, 1)

        # Build output line
        prefix = "[EVAL]" if is_eval else "[TRAIN]"
        parts = [
            f"{prefix}",
            f"step={step:,}",
            f"time={elapsed_str}",
            f"steps/s={steps_per_sec:.1f}",
        ]

        # Add metrics
        for name, value in metrics.items():
            if isinstance(value, float):
                parts.append(f"{name}={value:.4f}")
            else:
                parts.append(f"{name}={value}")

        # Add moving average for loss
        if 'train_loss' in metrics and len(self._metrics['train_loss']) > 100:
            avg = sum(self._metrics['train_loss'][-100:]) / 100
            parts.append(f"loss_avg100={avg:.4f}")

        print(" | ".join(parts))

    def log_config(self, config: Dict):
        """Log experiment configuration."""
        config_path = self.log_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)

        if self.use_wandb and self._wandb_run:
            import wandb
            wandb.config.update(config)

        print(f"[Logger] Config saved to {config_path}")

    def log_model_summary(self, model, input_size: Optional[tuple] = None):
        """Log model architecture summary."""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        summary = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_class': model.__class__.__name__,
        }

        # Try to get layer info
        try:
            layers = []
            for name, module in model.named_modules():
                if len(list(module.children())) == 0:  # Leaf module
                    params = sum(p.numel() for p in module.parameters())
                    if params > 0:
                        layers.append({
                            'name': name,
                            'type': module.__class__.__name__,
                            'params': params,
                        })
            summary['layers'] = layers[:20]  # First 20 layers
        except:
            pass

        summary_path = self.log_path / 'model_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"[Logger] Model: {total_params:,} params ({trainable_params:,} trainable)")

        if self.use_wandb and self._wandb_run:
            import wandb
            wandb.config.update(summary)

    def get_metric(self, name: str) -> List[float]:
        """Get history of a specific metric."""
        return self._metrics.get(name, [])

    def get_best(self, metric: str, mode: str = 'min') -> tuple:
        """Get best value and step for a metric."""
        values = self._metrics.get(metric, [])
        if not values:
            return None, None

        if mode == 'min':
            idx = min(range(len(values)), key=lambda i: values[i])
        else:
            idx = max(range(len(values)), key=lambda i: values[i])

        return values[idx], self._steps[idx] if idx < len(self._steps) else None

    def get_statistics(self) -> Dict:
        """Get summary statistics for all metrics."""
        stats = {
            'experiment': self.experiment_name,
            'total_steps': self._steps[-1] if self._steps else 0,
            'total_time': time.time() - self._start_time,
            'metrics': {},
        }

        for name, values in self._metrics.items():
            if values:
                import numpy as np
                stats['metrics'][name] = {
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'final': values[-1],
                    'count': len(values),
                }

        return stats

    def save_csv(self):
        """Save metrics to CSV file."""
        if not self._steps:
            return

        # Get all metric names
        metric_names = sorted(self._metrics.keys())

        with open(self._csv_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(['step', 'timestamp'] + metric_names)

            # Data rows
            for i, step in enumerate(self._steps):
                row = [step, self._timestamps[i]]
                for name in metric_names:
                    values = self._metrics[name]
                    row.append(values[i] if i < len(values) else '')
                writer.writerow(row)

        print(f"[Logger] Metrics saved to {self._csv_path}")

    def save_summary(self):
        """Save summary statistics and close."""
        stats = self.get_statistics()

        with open(self._json_path, 'w') as f:
            json.dump(stats, f, indent=2)

        self.save_csv()

        print(f"[Logger] Summary saved to {self._json_path}")

        # Close wandb
        if self.use_wandb and self._wandb_run:
            import wandb
            wandb.finish()

    def print_summary(self):
        """Print summary to console."""
        stats = self.get_statistics()

        print("\n" + "="*60)
        print(f"Experiment: {stats['experiment']}")
        print(f"Total steps: {stats['total_steps']:,}")
        print(f"Total time: {timedelta(seconds=int(stats['total_time']))}")
        print("-"*60)

        for name, metric_stats in stats['metrics'].items():
            print(f"{name}:")
            print(f"  final={metric_stats['final']:.4f}, "
                  f"min={metric_stats['min']:.4f}, "
                  f"max={metric_stats['max']:.4f}, "
                  f"mean={metric_stats['mean']:.4f}")

        print("="*60 + "\n")


class ProgressTracker:
    """Simple progress tracker with ETA estimation."""

    def __init__(self, total_steps: int, desc: str = "Training"):
        self.total_steps = total_steps
        self.desc = desc
        self.start_time = time.time()
        self.current_step = 0

    def update(self, step: int):
        """Update progress."""
        self.current_step = step
        elapsed = time.time() - self.start_time

        if step > 0:
            steps_per_sec = step / elapsed
            remaining_steps = self.total_steps - step
            eta_seconds = remaining_steps / steps_per_sec
            eta = timedelta(seconds=int(eta_seconds))
        else:
            eta = "N/A"

        progress = step / self.total_steps * 100

        print(f"\r{self.desc}: {step:,}/{self.total_steps:,} ({progress:.1f}%) | ETA: {eta}",
              end='', flush=True)

    def finish(self):
        """Mark as complete."""
        elapsed = time.time() - self.start_time
        print(f"\r{self.desc}: Complete! {self.total_steps:,} steps in {timedelta(seconds=int(elapsed))}")
