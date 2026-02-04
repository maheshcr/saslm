# SASLM Training Module
# Provides checkpoint management, grokking detection, and training utilities

from .checkpoint_manager import CheckpointManager
from .grokking_detector import GrokkingDetector
from .metrics_logger import MetricsLogger

__all__ = ['CheckpointManager', 'GrokkingDetector', 'MetricsLogger']

# Lazy import for train module (requires torch)
def get_trainer():
    from .train import train, GPT
    return train, GPT
