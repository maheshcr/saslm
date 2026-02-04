"""
Configuration Management for SASLM Experiments

Handles loading, validating, and saving experiment configurations.

Usage:
    from src.utils.config import load_config, ExperimentConfig

    config = load_config('configs/exp_a1.yaml')
    print(config.model.n_layers)
"""

import yaml
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


@dataclass
class DataConfig:
    """Data configuration."""
    corpus_path: str = './data/clean_prose'
    include_types: List[str] = field(default_factory=lambda: ['essay', 'letter', 'commentary'])
    exclude_types: List[str] = field(default_factory=lambda: ['poetry', 'drama'])

    # Period weighting (for weighted sampling)
    period_weights: Dict[str, float] = field(default_factory=lambda: {
        'mature': 3.0,   # 1926-1950
        'middle': 2.0,   # 1910-1926
        'early': 0.5,    # 1893-1910
    })

    # Content type weighting
    content_weights: Dict[str, float] = field(default_factory=lambda: {
        'essay': 2.0,
        'letter': 1.5,
        'commentary': 1.5,
        'poetry': 0.5,
        'drama': 0.3,
    })

    # Priority works (load these first, sample more frequently)
    priority_works: List[str] = field(default_factory=lambda: [
        '21-22TheLifeDivine',
        '23-24TheSynthesisOfYoga',
        '28-31LettersOnYoga',
        '19EssaysOnTheGita',
    ])

    # Train/val split
    train_split: float = 0.95
    val_split: float = 0.05
    seed: int = 42


@dataclass
class TokenizerConfig:
    """Tokenizer configuration."""
    vocab_size: int = 16384
    min_frequency: int = 2
    tokenizer_path: str = './tokenizers/tokenizer_16k'

    # Special tokens
    special_tokens: List[str] = field(default_factory=lambda: [
        '[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]', '<|endoftext|>'
    ])

    # For fine-tuning: extend base tokenizer
    extend_base: bool = False
    base_tokenizer: str = 'gpt2'
    additional_tokens: int = 1000  # Sanskrit/philosophical terms to add


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    architecture: str = 'gpt2-custom'  # 'gpt2-custom' or 'gpt2-finetune'

    # For custom model (Option A)
    vocab_size: int = 16384
    n_layers: int = 6
    n_heads: int = 6
    n_embd: int = 384
    block_size: int = 512
    dropout: float = 0.1
    bias: bool = False  # No bias in linear layers (like LLaMA)

    # For fine-tuning (Option B)
    base_model: str = 'distilgpt2'  # or 'gpt2', 'gpt2-medium'
    freeze_layers: int = 0  # Number of layers to freeze (0 = train all)

    @property
    def estimated_params(self) -> int:
        """Estimate parameter count for custom model."""
        # Rough estimate based on GPT-2 formula
        # params ≈ 12 * n_layers * n_embd^2
        return 12 * self.n_layers * self.n_embd ** 2


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 32
    gradient_accumulation: int = 4
    effective_batch_size: int = 128  # batch_size * gradient_accumulation

    learning_rate: float = 3e-4
    weight_decay: float = 0.1  # Higher for grokking
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    lr_scheduler: str = 'cosine'  # 'cosine', 'linear', 'constant'
    warmup_steps: int = 500
    lr_decay_steps: int = 100000  # For cosine schedule

    max_steps: int = 100000
    eval_interval: int = 500
    save_interval: int = 1000
    sample_interval: int = 2000

    # For fine-tuning (typically lower LR)
    finetune_lr: float = 5e-5


@dataclass
class GrokkingConfig:
    """Grokking detection configuration."""
    enabled: bool = True
    detection_window: int = 500
    detection_threshold: float = 0.15
    plateau_variance_threshold: float = 0.02
    train_val_gap_threshold: float = 0.5


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    device: str = 'auto'  # 'auto', 'cuda', 'mps', 'cpu'
    precision: str = 'fp16'  # 'fp32', 'fp16', 'bf16'
    num_workers: int = 4
    pin_memory: bool = True

    # Colab-specific
    mount_drive: bool = True
    drive_path: str = '/content/drive/MyDrive/saslm/experiments'


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Experiment metadata
    name: str = 'EXP-A1-prose-only'
    approach: str = 'from_scratch'  # 'from_scratch' or 'finetune'
    content: str = 'prose_only'  # 'prose_only' or 'prose_and_poetry'
    description: str = ''

    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    grokking: GrokkingConfig = field(default_factory=GrokkingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def save(self, path: str):
        """Save config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, d: Dict) -> 'ExperimentConfig':
        """Create from dictionary."""
        return cls(
            name=d.get('name', 'unnamed'),
            approach=d.get('approach', 'from_scratch'),
            content=d.get('content', 'prose_only'),
            description=d.get('description', ''),
            data=DataConfig(**d.get('data', {})),
            tokenizer=TokenizerConfig(**d.get('tokenizer', {})),
            model=ModelConfig(**d.get('model', {})),
            training=TrainingConfig(**d.get('training', {})),
            grokking=GrokkingConfig(**d.get('grokking', {})),
            hardware=HardwareConfig(**d.get('hardware', {})),
        )


def load_config(path: str) -> ExperimentConfig:
    """
    Load configuration from YAML file.

    Args:
        path: Path to YAML config file

    Returns:
        ExperimentConfig object
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            data = yaml.safe_load(f)
        elif path.suffix == '.json':
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    return ExperimentConfig.from_dict(data)


def save_config(config: ExperimentConfig, path: str):
    """
    Save configuration to file.

    Args:
        config: ExperimentConfig to save
        path: Output path (YAML or JSON based on extension)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.to_dict()

    with open(path, 'w') as f:
        if path.suffix in ['.yaml', '.yml']:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        elif path.suffix == '.json':
            json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def get_device(config: HardwareConfig) -> str:
    """Determine the best device to use."""
    import torch

    if config.device != 'auto':
        return config.device

    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def print_config(config: ExperimentConfig):
    """Pretty print configuration."""
    print("\n" + "="*60)
    print(f"Experiment: {config.name}")
    print(f"Approach: {config.approach}")
    print(f"Content: {config.content}")
    print("-"*60)

    print("\nModel:")
    print(f"  Architecture: {config.model.architecture}")
    print(f"  Layers: {config.model.n_layers}")
    print(f"  Heads: {config.model.n_heads}")
    print(f"  Embedding: {config.model.n_embd}")
    print(f"  Estimated params: {config.model.estimated_params:,}")

    print("\nTraining:")
    print(f"  Batch size: {config.training.batch_size} x {config.training.gradient_accumulation} = {config.training.effective_batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Weight decay: {config.training.weight_decay}")
    print(f"  Max steps: {config.training.max_steps:,}")

    print("\nData:")
    print(f"  Include types: {config.data.include_types}")
    print(f"  Priority works: {config.data.priority_works[:3]}...")

    print("\nGrokking:")
    print(f"  Detection enabled: {config.grokking.enabled}")
    print(f"  Window size: {config.grokking.detection_window}")

    print("="*60 + "\n")
