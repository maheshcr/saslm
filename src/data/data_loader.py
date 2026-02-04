"""
PyTorch Data Loader for SASLM Training

Creates datasets and dataloaders from the corpus with:
- Tokenization
- Sequence padding/truncation
- Train/val splits
- Weighted sampling

Usage:
    from src.data import create_dataloaders

    train_loader, val_loader = create_dataloaders(
        corpus_dir='./processed_text',
        tokenizer_path='./tokenizers/tokenizer_16k',
        batch_size=32,
        seq_length=512,
    )

    for batch in train_loader:
        input_ids = batch['input_ids']  # (batch_size, seq_length)
        # ... training step ...
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

from .weighted_sampler import WeightedCorpusSampler


class SASLMDataset(Dataset):
    """
    PyTorch Dataset for SASLM training.

    Stores pre-tokenized chunks for efficient loading.
    """

    def __init__(
        self,
        chunks: List[str],
        tokenizer,
        seq_length: int = 512,
        stride: int = None,
    ):
        """
        Initialize dataset.

        Args:
            chunks: List of text chunks
            tokenizer: HuggingFace tokenizer
            seq_length: Sequence length for training
            stride: Stride for sliding window (default: seq_length // 2)
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride or seq_length // 2

        # Tokenize all chunks and create training sequences
        self.sequences = []
        self._tokenize_chunks(chunks)

    def _tokenize_chunks(self, chunks: List[str]):
        """Tokenize chunks and create fixed-length sequences."""
        for chunk in chunks:
            # Tokenize - get the token IDs from the Encoding object
            encoding = self.tokenizer.encode(chunk)
            tokens = encoding.ids  # Extract the list of token IDs

            # Create sequences with sliding window
            for i in range(0, len(tokens) - self.seq_length, self.stride):
                seq = tokens[i:i + self.seq_length + 1]  # +1 for target
                if len(seq) == self.seq_length + 1:
                    self.sequences.append(seq)

            # Handle remaining tokens if they're long enough
            if len(tokens) >= self.seq_length + 1:
                # Already handled by sliding window
                pass
            elif len(tokens) > self.seq_length // 2:
                # Pad shorter sequences
                pad_id = self.tokenizer.token_to_id('[PAD]') or 0
                padded = tokens + [pad_id] * (self.seq_length + 1 - len(tokens))
                self.sequences.append(padded)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]

        # Input is all but last token, target is all but first
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        labels = torch.tensor(seq[1:], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': labels,
        }


class StreamingSASLMDataset(IterableDataset):
    """
    Streaming dataset that samples from weighted corpus.

    For very large corpora or when you want weighted sampling
    during training (not just at data loading time).
    """

    def __init__(
        self,
        sampler: WeightedCorpusSampler,
        tokenizer,
        seq_length: int = 512,
        samples_per_epoch: int = 10000,
    ):
        """
        Initialize streaming dataset.

        Args:
            sampler: Weighted corpus sampler
            tokenizer: HuggingFace tokenizer
            seq_length: Sequence length
            samples_per_epoch: Number of samples per "epoch"
        """
        self.sampler = sampler
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.samples_per_epoch = samples_per_epoch

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for _ in range(self.samples_per_epoch):
            # Sample a chunk
            chunk, _ = self.sampler.sample_chunk()

            # Tokenize - get the token IDs from the Encoding object
            encoding = self.tokenizer.encode(chunk)
            tokens = encoding.ids

            # If too short, sample another
            if len(tokens) < self.seq_length + 1:
                continue

            # Random start position for variety
            max_start = len(tokens) - self.seq_length - 1
            start = random.randint(0, max(0, max_start))

            seq = tokens[start:start + self.seq_length + 1]

            input_ids = torch.tensor(seq[:-1], dtype=torch.long)
            labels = torch.tensor(seq[1:], dtype=torch.long)

            yield {
                'input_ids': input_ids,
                'labels': labels,
            }

    def __len__(self) -> int:
        return self.samples_per_epoch


def load_tokenizer(tokenizer_path: str):
    """
    Load tokenizer from path.

    Supports both custom tokenizers (tokenizers library) and
    HuggingFace transformers tokenizers.
    """
    tokenizer_path = Path(tokenizer_path)

    # Try loading as tokenizers library tokenizer
    if (tokenizer_path / 'tokenizer.json').exists() or tokenizer_path.suffix == '.json':
        from tokenizers import Tokenizer
        if tokenizer_path.is_dir():
            return Tokenizer.from_file(str(tokenizer_path / 'tokenizer.json'))
        else:
            return Tokenizer.from_file(str(tokenizer_path))

    # Try loading as HuggingFace tokenizer
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(str(tokenizer_path))
    except Exception:
        pass

    raise ValueError(f"Could not load tokenizer from {tokenizer_path}")


def create_dataloaders(
    corpus_dir: str,
    tokenizer_path: str,
    batch_size: int = 32,
    seq_length: int = 512,
    val_ratio: float = 0.05,
    num_workers: int = 4,
    include_prose: bool = True,
    include_poetry: bool = False,
    period_weights: Dict[str, float] = None,
    seed: int = 42,
    streaming: bool = False,
    samples_per_epoch: int = 10000,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        corpus_dir: Directory containing text files
        tokenizer_path: Path to tokenizer
        batch_size: Batch size
        seq_length: Sequence length
        val_ratio: Validation split ratio
        num_workers: Number of data loading workers
        include_prose: Include prose books
        include_poetry: Include poetry books
        period_weights: Custom period weights
        seed: Random seed
        streaming: Use streaming dataset with weighted sampling
        samples_per_epoch: Samples per epoch (for streaming)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)

    # Create sampler
    sampler = WeightedCorpusSampler(
        corpus_dir=corpus_dir,
        period_weights=period_weights or {
            'mature': 3.0,
            'middle': 2.0,
            'early': 0.5,
        },
    )

    sampler.load(
        include_prose=include_prose,
        include_poetry=include_poetry,
        verbose=True,
    )

    if streaming:
        # Streaming datasets with weighted sampling
        train_dataset = StreamingSASLMDataset(
            sampler=sampler,
            tokenizer=tokenizer,
            seq_length=seq_length,
            samples_per_epoch=samples_per_epoch,
        )

        # For validation, use a fixed subset
        _, val_chunks = sampler.get_train_val_split(val_ratio, seed)
        val_dataset = SASLMDataset(
            chunks=val_chunks,
            tokenizer=tokenizer,
            seq_length=seq_length,
        )
    else:
        # Static datasets
        train_chunks, val_chunks = sampler.get_train_val_split(val_ratio, seed)

        train_dataset = SASLMDataset(
            chunks=train_chunks,
            tokenizer=tokenizer,
            seq_length=seq_length,
        )

        val_dataset = SASLMDataset(
            chunks=val_chunks,
            tokenizer=tokenizer,
            seq_length=seq_length,
        )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset):,} sequences")
    print(f"  Val:   {len(val_dataset):,} sequences")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not streaming,  # Don't shuffle streaming dataset
        num_workers=num_workers if not streaming else 0,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def create_dataloader_from_config(config) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders from experiment config.

    Args:
        config: ExperimentConfig object

    Returns:
        Tuple of (train_loader, val_loader)
    """
    return create_dataloaders(
        corpus_dir=config.data.corpus_path,
        tokenizer_path=config.tokenizer.tokenizer_path,
        batch_size=config.training.batch_size,
        seq_length=config.model.block_size,
        val_ratio=config.data.val_split,
        include_prose='prose' in config.content,
        include_poetry='poetry' in config.content,
        period_weights=config.data.period_weights,
        seed=config.data.seed,
    )


# Quick test
if __name__ == '__main__':
    print("DataLoader module loaded successfully")
    print("Run with actual tokenizer and corpus to test")
