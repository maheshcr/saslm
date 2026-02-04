"""
Grokking Detector for SASLM Training

Detects the "grokking" phenomenon where a model suddenly generalizes
long after achieving low training loss.

Grokking signature:
1. Training loss goes low early (memorization)
2. Validation loss stays high (no generalization)
3. Training continues...
4. Suddenly validation loss drops (GROK!)

This is more likely with:
- Small datasets (like ours!)
- Higher weight decay
- Extended training past convergence

Usage:
    detector = GrokkingDetector()

    for step in training_loop:
        # ... training ...
        detector.update(step, train_loss, val_loss)

        if detector.grokking_detected:
            print(f"Grokking at step {detector.grokking_step}!")
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class GrokkingEvent:
    """Record of a detected grokking event."""
    step: int
    val_loss_before: float
    val_loss_after: float
    relative_drop: float
    train_loss_at_grok: float
    plateau_duration: int  # Steps spent in plateau before grok


@dataclass
class GrokkingDetector:
    """
    Detects grokking by monitoring train/val loss divergence and sudden convergence.

    Grokking detection criteria:
    1. Val loss has been plateaued (low variance) for at least `min_plateau_steps`
    2. Val loss drops by at least `drop_threshold` relative to plateau mean
    3. Train loss was already low before the drop
    """

    # Detection parameters
    window_size: int = 500           # Steps to average for plateau detection
    min_plateau_steps: int = 1000    # Minimum steps in plateau before grok can be detected
    drop_threshold: float = 0.15     # Minimum relative drop to count as grok (15%)
    plateau_variance_threshold: float = 0.02  # Max variance to count as plateau
    train_val_gap_threshold: float = 0.5      # Min train/val gap before grok

    # State
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)

    # Detection results
    grokking_detected: bool = False
    grokking_step: Optional[int] = None
    grokking_event: Optional[GrokkingEvent] = None

    # Tracking
    in_plateau: bool = False
    plateau_start_step: Optional[int] = None
    plateau_mean: Optional[float] = None

    def update(
        self,
        step: int,
        train_loss: float,
        val_loss: float,
    ) -> Optional[GrokkingEvent]:
        """
        Update detector with new loss values.

        Args:
            step: Current training step
            train_loss: Current training loss
            val_loss: Current validation loss

        Returns:
            GrokkingEvent if grokking detected at this step, else None
        """
        self.steps.append(step)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        # Need enough history
        if len(self.val_losses) < self.window_size * 2:
            return None

        # Already detected
        if self.grokking_detected:
            return None

        # Check for grokking
        event = self._check_grokking()
        if event:
            self.grokking_detected = True
            self.grokking_step = event.step
            self.grokking_event = event
            return event

        return None

    def _check_grokking(self) -> Optional[GrokkingEvent]:
        """Check if grokking has occurred."""
        n = len(self.val_losses)
        window = self.window_size

        # Get recent history
        recent_val = self.val_losses[-window:]
        older_val = self.val_losses[-2*window:-window]
        recent_train = self.train_losses[-window:]

        # Calculate statistics
        recent_mean = np.mean(recent_val)
        older_mean = np.mean(older_val)
        older_variance = np.var(older_val)
        recent_train_mean = np.mean(recent_train)

        # Check if older window was a plateau
        is_plateau = older_variance < self.plateau_variance_threshold

        # Check if there's a significant drop
        if older_mean > 0:
            relative_drop = (older_mean - recent_mean) / older_mean
        else:
            relative_drop = 0

        # Check train/val gap before drop (indicates memorization without generalization)
        train_val_gap = older_mean - np.mean(self.train_losses[-2*window:-window])
        had_gap = train_val_gap > self.train_val_gap_threshold

        # Grokking criteria
        if (is_plateau and
            relative_drop > self.drop_threshold and
            had_gap):

            # Find the approximate step where the drop happened
            grok_step = self._find_grok_step()

            return GrokkingEvent(
                step=grok_step,
                val_loss_before=older_mean,
                val_loss_after=recent_mean,
                relative_drop=relative_drop,
                train_loss_at_grok=recent_train_mean,
                plateau_duration=self._estimate_plateau_duration(),
            )

        return None

    def _find_grok_step(self) -> int:
        """Find the step where the steepest drop occurred."""
        n = len(self.val_losses)
        window = min(self.window_size, n // 4)

        # Look for the point of steepest descent
        max_drop = 0
        grok_idx = n - 1

        for i in range(window, n - window):
            before = np.mean(self.val_losses[i-window:i])
            after = np.mean(self.val_losses[i:i+window])
            drop = before - after

            if drop > max_drop:
                max_drop = drop
                grok_idx = i

        return self.steps[grok_idx]

    def _estimate_plateau_duration(self) -> int:
        """Estimate how long the plateau lasted before grokking."""
        if len(self.val_losses) < self.window_size:
            return 0

        # Walk backwards to find when plateau started
        n = len(self.val_losses)
        plateau_end = n - self.window_size
        window = self.window_size // 2

        for i in range(plateau_end, window, -window):
            segment = self.val_losses[i-window:i]
            variance = np.var(segment)

            if variance > self.plateau_variance_threshold:
                plateau_start = i
                return self.steps[plateau_end] - self.steps[plateau_start]

        return self.steps[plateau_end] - self.steps[0]

    def get_analysis(self) -> Dict:
        """Get comprehensive analysis of training dynamics."""
        if len(self.val_losses) < 10:
            return {'status': 'insufficient_data'}

        n = len(self.val_losses)
        window = min(self.window_size, n // 4)

        # Current state
        recent_val_mean = np.mean(self.val_losses[-window:])
        recent_val_var = np.var(self.val_losses[-window:])
        recent_train_mean = np.mean(self.train_losses[-window:])

        # Overall trends
        early_val = np.mean(self.val_losses[:window]) if n > window else self.val_losses[0]
        val_improvement = (early_val - recent_val_mean) / early_val if early_val > 0 else 0

        analysis = {
            'status': 'monitoring',
            'total_steps': self.steps[-1] if self.steps else 0,
            'current_train_loss': self.train_losses[-1],
            'current_val_loss': self.val_losses[-1],
            'recent_train_mean': recent_train_mean,
            'recent_val_mean': recent_val_mean,
            'recent_val_variance': recent_val_var,
            'train_val_gap': recent_val_mean - recent_train_mean,
            'total_val_improvement': val_improvement,
            'in_plateau': recent_val_var < self.plateau_variance_threshold,
        }

        if self.grokking_detected:
            analysis['status'] = 'grokking_detected'
            analysis['grokking_step'] = self.grokking_step
            analysis['grokking_event'] = {
                'step': self.grokking_event.step,
                'val_drop': self.grokking_event.relative_drop,
                'plateau_duration': self.grokking_event.plateau_duration,
            }
        elif analysis['in_plateau'] and analysis['train_val_gap'] > self.train_val_gap_threshold:
            analysis['status'] = 'potential_grokking_setup'
            analysis['message'] = 'Model is memorizing (low train, high val). Grokking may occur if training continues.'

        return analysis

    def get_plot_data(self) -> Dict:
        """Get data formatted for plotting."""
        return {
            'steps': self.steps,
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'grokking_step': self.grokking_step,
        }

    def reset(self):
        """Reset detector state."""
        self.train_losses = []
        self.val_losses = []
        self.steps = []
        self.grokking_detected = False
        self.grokking_step = None
        self.grokking_event = None
        self.in_plateau = False
        self.plateau_start_step = None
        self.plateau_mean = None


def plot_grokking_analysis(detector: GrokkingDetector, save_path: Optional[str] = None):
    """
    Plot training curves with grokking analysis.

    Requires matplotlib (optional dependency).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    data = detector.get_plot_data()
    analysis = detector.get_analysis()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Loss curves
    ax1.plot(data['steps'], data['train_loss'], label='Train Loss', alpha=0.7)
    ax1.plot(data['steps'], data['val_loss'], label='Val Loss', alpha=0.7)

    if data['grokking_step']:
        ax1.axvline(x=data['grokking_step'], color='red', linestyle='--',
                   label=f'Grokking @ step {data["grokking_step"]:,}')

    ax1.set_ylabel('Loss')
    ax1.set_title('Training Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Train-Val gap
    gap = [v - t for t, v in zip(data['train_loss'], data['val_loss'])]
    ax2.plot(data['steps'], gap, label='Val - Train Gap', color='purple', alpha=0.7)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    if data['grokking_step']:
        ax2.axvline(x=data['grokking_step'], color='red', linestyle='--')

    ax2.set_xlabel('Step')
    ax2.set_ylabel('Gap')
    ax2.set_title('Generalization Gap (Val - Train)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


# Quick test
if __name__ == '__main__':
    # Simulate grokking scenario
    detector = GrokkingDetector(window_size=50, min_plateau_steps=100)

    np.random.seed(42)

    # Phase 1: Training loss drops, val stays high (memorization)
    for step in range(0, 1000, 10):
        train = 2.0 * np.exp(-step / 200) + 0.1 + np.random.normal(0, 0.02)
        val = 2.0 + np.random.normal(0, 0.05)  # Stays high
        detector.update(step, train, val)

    # Phase 2: Plateau (both stable)
    for step in range(1000, 3000, 10):
        train = 0.15 + np.random.normal(0, 0.02)
        val = 2.0 + np.random.normal(0, 0.05)  # Still high
        detector.update(step, train, val)

    # Phase 3: GROK! Val suddenly drops
    for step in range(3000, 5000, 10):
        progress = (step - 3000) / 500
        train = 0.15 + np.random.normal(0, 0.02)
        val = 2.0 - min(progress, 1.0) * 1.7 + np.random.normal(0, 0.05)  # Drops!
        event = detector.update(step, train, val)
        if event:
            print(f"\nGROKKING DETECTED!")
            print(f"  Step: {event.step}")
            print(f"  Val loss: {event.val_loss_before:.3f} -> {event.val_loss_after:.3f}")
            print(f"  Drop: {event.relative_drop:.1%}")
            print(f"  Plateau duration: {event.plateau_duration} steps")

    print("\nFinal Analysis:")
    for k, v in detector.get_analysis().items():
        print(f"  {k}: {v}")
