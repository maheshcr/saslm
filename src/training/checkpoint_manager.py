"""
Checkpoint Manager for SASLM Training

Handles:
- Saving/loading checkpoints to Google Drive (survives Colab disconnects)
- Atomic saves (write to temp, then rename - prevents corruption)
- Auto-resume from existing checkpoints
- Best model tracking
- Metrics history persistence (JSONL format)

Usage:
    manager = CheckpointManager("EXP-A1-prose-only")

    # In Colab, mount drive first:
    # from google.colab import drive
    # drive.mount('/content/drive')

    # Resume if checkpoint exists
    start_step = manager.load(model, optimizer, scheduler)

    # During training
    manager.save(model, optimizer, scheduler, step, metrics)

    # Save best model when validation improves
    if val_loss < best_val_loss:
        manager.save_best(model, step, val_loss)
"""

import os
import json
import torch
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple


class CheckpointManager:
    """Robust checkpoint management for long-running training on Colab."""

    def __init__(
        self,
        experiment_name: str,
        base_path: str = '/content/drive/MyDrive/saslm/experiments',
        local_cache: str = '/content/checkpoints',
        save_every_n_steps: int = 1000,
        keep_last_n_checkpoints: int = 3,
    ):
        """
        Initialize checkpoint manager.

        Args:
            experiment_name: Unique name for this experiment (e.g., "EXP-A1-prose-only")
            base_path: Base path on Google Drive for persistence
            local_cache: Local path for faster saves during training
            save_every_n_steps: How often to save checkpoints
            keep_last_n_checkpoints: Number of recent checkpoints to keep (saves space)
        """
        self.experiment_name = experiment_name
        self.save_every_n_steps = save_every_n_steps
        self.keep_last_n_checkpoints = keep_last_n_checkpoints

        # Paths
        self.drive_path = Path(base_path) / experiment_name
        self.local_path = Path(local_cache) / experiment_name

        # File names
        self.checkpoint_file = 'checkpoint_latest.pt'
        self.best_file = 'checkpoint_best.pt'
        self.config_file = 'config.json'
        self.metrics_file = 'metrics.jsonl'
        self.samples_dir = 'samples'

        # State tracking
        self.best_val_loss = float('inf')
        self.metrics_history: List[Dict] = []

        # Create directories
        self._setup_directories()

    def _setup_directories(self):
        """Create necessary directories."""
        # Try to create drive path (may fail if drive not mounted)
        try:
            self.drive_path.mkdir(parents=True, exist_ok=True)
            (self.drive_path / self.samples_dir).mkdir(exist_ok=True)
            self._drive_available = True
        except Exception as e:
            print(f"Warning: Could not create drive path: {e}")
            print("Will save to local path only. Mount Google Drive to persist checkpoints.")
            self._drive_available = False

        # Always create local path
        self.local_path.mkdir(parents=True, exist_ok=True)
        (self.local_path / self.samples_dir).mkdir(exist_ok=True)

    def _get_save_path(self) -> Path:
        """Get the path to save to (drive if available, else local)."""
        return self.drive_path if self._drive_available else self.local_path

    def _atomic_save(self, data: Dict, filepath: Path):
        """
        Save data atomically: write to temp file, then rename.
        This prevents corruption if the process is interrupted mid-write.
        """
        temp_path = filepath.with_suffix('.tmp')
        torch.save(data, temp_path)
        temp_path.rename(filepath)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        metrics: Dict[str, float],
        scheduler: Optional[Any] = None,
        config: Optional[Dict] = None,
        extra_state: Optional[Dict] = None,
    ) -> str:
        """
        Save training checkpoint.

        Args:
            model: The model to save
            optimizer: Optimizer state
            step: Current training step
            metrics: Current metrics (train_loss, val_loss, etc.)
            scheduler: Optional learning rate scheduler
            config: Optional config to save (saved once)
            extra_state: Any additional state to preserve

        Returns:
            Path where checkpoint was saved
        """
        save_path = self._get_save_path()

        # Build checkpoint
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'best_val_loss': self.best_val_loss,
            'timestamp': datetime.now().isoformat(),
            'experiment_name': self.experiment_name,
        }

        if extra_state:
            checkpoint['extra_state'] = extra_state

        # Save checkpoint atomically
        checkpoint_path = save_path / self.checkpoint_file
        self._atomic_save(checkpoint, checkpoint_path)

        # Also save a numbered checkpoint for history
        step_checkpoint_path = save_path / f'checkpoint_step_{step:07d}.pt'
        shutil.copy(checkpoint_path, step_checkpoint_path)

        # Clean up old numbered checkpoints
        self._cleanup_old_checkpoints(save_path)

        # Save config once
        if config:
            config_path = save_path / self.config_file
            if not config_path.exists():
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2, default=str)

        # Append metrics to JSONL
        self._append_metrics(save_path, step, metrics)

        # Also save to local if we saved to drive (for faster access)
        if self._drive_available and save_path != self.local_path:
            local_checkpoint_path = self.local_path / self.checkpoint_file
            self._atomic_save(checkpoint, local_checkpoint_path)

        print(f"[Checkpoint] Saved at step {step:,} to {save_path}")
        return str(checkpoint_path)

    def _cleanup_old_checkpoints(self, save_path: Path):
        """Remove old numbered checkpoints, keeping only the most recent N."""
        checkpoints = sorted(save_path.glob('checkpoint_step_*.pt'))

        if len(checkpoints) > self.keep_last_n_checkpoints:
            for old_ckpt in checkpoints[:-self.keep_last_n_checkpoints]:
                old_ckpt.unlink()
                print(f"[Checkpoint] Removed old checkpoint: {old_ckpt.name}")

    def _append_metrics(self, save_path: Path, step: int, metrics: Dict):
        """Append metrics to JSONL file (append-only, survives crashes)."""
        metrics_path = save_path / self.metrics_file

        entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }

        self.metrics_history.append(entry)

        with open(metrics_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def save_best(
        self,
        model: torch.nn.Module,
        step: int,
        val_loss: float,
        metrics: Optional[Dict] = None,
    ):
        """
        Save best model checkpoint (based on validation loss).

        Args:
            model: The model to save
            step: Current training step
            val_loss: Validation loss that triggered this save
            metrics: Optional additional metrics
        """
        if val_loss >= self.best_val_loss:
            return  # Not the best

        self.best_val_loss = val_loss
        save_path = self._get_save_path()

        best_checkpoint = {
            'step': step,
            'val_loss': val_loss,
            'model_state_dict': model.state_dict(),
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics or {},
        }

        best_path = save_path / self.best_file
        self._atomic_save(best_checkpoint, best_path)

        print(f"[Checkpoint] New best model at step {step:,} (val_loss: {val_loss:.4f})")

    def load(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = 'cuda',
    ) -> Tuple[int, Dict]:
        """
        Load checkpoint if exists.

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Device to load tensors to

        Returns:
            Tuple of (starting_step, last_metrics)
            Returns (0, {}) if no checkpoint found
        """
        # Try drive first, then local
        for path in [self.drive_path, self.local_path]:
            checkpoint_path = path / self.checkpoint_file
            if checkpoint_path.exists():
                return self._load_from_path(
                    checkpoint_path, model, optimizer, scheduler, device
                )

        print("[Checkpoint] No existing checkpoint found, starting fresh")
        return 0, {}

    def _load_from_path(
        self,
        checkpoint_path: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        device: str,
    ) -> Tuple[int, Dict]:
        """Load checkpoint from specific path."""
        print(f"[Checkpoint] Loading from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        step = checkpoint['step']
        metrics = checkpoint.get('metrics', {})
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        # Load metrics history
        self._load_metrics_history()

        print(f"[Checkpoint] Resumed from step {step:,}")
        print(f"[Checkpoint] Best val_loss so far: {self.best_val_loss:.4f}")

        return step, metrics

    def load_best(
        self,
        model: torch.nn.Module,
        device: str = 'cuda',
    ) -> Tuple[int, float]:
        """
        Load best model checkpoint.

        Args:
            model: Model to load state into
            device: Device to load tensors to

        Returns:
            Tuple of (step, val_loss) for the best model
        """
        for path in [self.drive_path, self.local_path]:
            best_path = path / self.best_file
            if best_path.exists():
                checkpoint = torch.load(best_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"[Checkpoint] Loaded best model from step {checkpoint['step']:,}")
                return checkpoint['step'], checkpoint['val_loss']

        raise FileNotFoundError("No best model checkpoint found")

    def _load_metrics_history(self):
        """Load metrics history from JSONL file."""
        for path in [self.drive_path, self.local_path]:
            metrics_path = path / self.metrics_file
            if metrics_path.exists():
                self.metrics_history = []
                with open(metrics_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            self.metrics_history.append(json.loads(line))
                print(f"[Checkpoint] Loaded {len(self.metrics_history)} metrics entries")
                return

    def get_metrics_history(self) -> List[Dict]:
        """Get all recorded metrics."""
        if not self.metrics_history:
            self._load_metrics_history()
        return self.metrics_history

    def get_val_losses(self) -> List[float]:
        """Get validation loss history (for grokking detection)."""
        history = self.get_metrics_history()
        return [m['val_loss'] for m in history if 'val_loss' in m]

    def save_samples(self, step: int, samples: List[str], prompts: List[str]):
        """
        Save generated samples for qualitative inspection.

        Args:
            step: Current training step
            samples: List of generated text samples
            prompts: List of prompts used to generate samples
        """
        save_path = self._get_save_path()
        samples_path = save_path / self.samples_dir / f'samples_step_{step:07d}.txt'

        with open(samples_path, 'w') as f:
            f.write(f"# Samples at step {step:,}\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n\n")

            for i, (prompt, sample) in enumerate(zip(prompts, samples)):
                f.write(f"{'='*60}\n")
                f.write(f"## Sample {i+1}\n")
                f.write(f"**Prompt:** {prompt}\n\n")
                f.write(f"**Generated:**\n{sample}\n\n")

        print(f"[Checkpoint] Saved {len(samples)} samples at step {step:,}")

    def load_config(self) -> Optional[Dict]:
        """Load experiment config if exists."""
        for path in [self.drive_path, self.local_path]:
            config_path = path / self.config_file
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
        return None

    def get_experiment_summary(self) -> Dict:
        """Get summary of experiment state."""
        history = self.get_metrics_history()

        summary = {
            'experiment_name': self.experiment_name,
            'total_steps': history[-1]['step'] if history else 0,
            'total_entries': len(history),
            'best_val_loss': self.best_val_loss,
            'drive_available': self._drive_available,
            'drive_path': str(self.drive_path),
            'local_path': str(self.local_path),
        }

        if history:
            summary['latest_metrics'] = history[-1]
            summary['first_entry'] = history[0].get('timestamp')
            summary['last_entry'] = history[-1].get('timestamp')

        return summary

    def should_save(self, step: int) -> bool:
        """Check if we should save a checkpoint at this step."""
        return step > 0 and step % self.save_every_n_steps == 0


# Convenience function for quick setup
def setup_checkpoint_manager(
    experiment_name: str,
    colab_mode: bool = True,
) -> CheckpointManager:
    """
    Quick setup for checkpoint manager.

    Args:
        experiment_name: Name of the experiment
        colab_mode: If True, uses Colab paths. If False, uses local paths.

    Returns:
        Configured CheckpointManager
    """
    if colab_mode:
        return CheckpointManager(
            experiment_name=experiment_name,
            base_path='/content/drive/MyDrive/saslm/experiments',
            local_cache='/content/checkpoints',
        )
    else:
        # Local development mode
        return CheckpointManager(
            experiment_name=experiment_name,
            base_path='./experiments',
            local_cache='./checkpoints',
        )
