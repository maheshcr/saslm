"""
Weighted Corpus Sampler for SASLM Training

Implements weighted sampling based on:
- Period (mature works weighted higher)
- Importance (core works weighted higher)
- Content type (essays weighted higher than drama)

This ensures the model sees more of Sri Aurobindo's mature,
systematic philosophical works while still learning from the
broader corpus.

Usage:
    sampler = WeightedCorpusSampler(
        corpus_dir='./data/clean_prose',
        period_weights={'mature': 3.0, 'middle': 2.0, 'early': 0.5},
    )

    # Get a training batch
    batch = sampler.sample(batch_size=32, seq_length=512)
"""

import os
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .corpus_metadata import (
    CORPUS_METADATA,
    BookMetadata,
    get_books_by_filter,
)
from .text_cleaner import TextCleaner, ChunkCleaner


@dataclass
class BookData:
    """Loaded and processed book data."""
    metadata: BookMetadata
    text: str
    chunks: List[str]
    weight: float
    char_count: int
    token_count_estimate: int  # Rough estimate: chars / 4


@dataclass
class WeightedCorpusSampler:
    """
    Samples text from corpus with configurable weights.

    Higher weights = more likely to be sampled.
    """

    corpus_dir: str
    period_weights: Dict[str, float] = field(default_factory=lambda: {
        'mature': 3.0,
        'middle': 2.0,
        'early': 0.5,
    })
    importance_weights: Dict[str, float] = field(default_factory=lambda: {
        'core': 3.0,
        'supplementary': 1.0,
        'reference': 0.3,
    })
    content_weights: Dict[str, float] = field(default_factory=lambda: {
        'essay': 2.0,
        'letter': 1.5,
        'commentary': 1.5,
        'record': 1.5,
        'poetry': 1.0,
        'translation': 0.8,
        'drama': 0.3,
    })

    # Processing options
    chunk_size: int = 2048
    chunk_overlap: int = 128
    clean_text: bool = True

    # Internal state
    books: Dict[str, BookData] = field(default_factory=dict, repr=False)
    total_weight: float = 0.0
    total_chars: int = 0
    total_chunks: int = 0
    _loaded: bool = False

    def __post_init__(self):
        """Initialize but don't load yet."""
        self.corpus_dir = Path(self.corpus_dir)

    def load(
        self,
        include_prose: bool = True,
        include_poetry: bool = False,
        max_books: int = None,
        verbose: bool = True,
    ):
        """
        Load books from corpus.

        Args:
            include_prose: Include prose books
            include_poetry: Include poetry books
            max_books: Limit number of books (for testing)
            verbose: Print progress
        """
        if verbose:
            print(f"Loading corpus from {self.corpus_dir}...")

        cleaner = TextCleaner() if self.clean_text else None
        chunk_cleaner = ChunkCleaner(
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )

        # Get books to load
        books_to_load = []
        for key, meta in CORPUS_METADATA.items():
            if include_prose and meta.include_in_prose:
                books_to_load.append((key, meta))
            elif include_poetry and meta.include_in_poetry:
                if (key, meta) not in books_to_load:
                    books_to_load.append((key, meta))

        if max_books:
            books_to_load = books_to_load[:max_books]

        # Load each book
        for key, meta in books_to_load:
            filepath = self.corpus_dir / meta.filename

            if not filepath.exists():
                if verbose:
                    print(f"  [SKIP] {meta.filename} not found")
                continue

            # Load text
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

            # Clean if requested
            if cleaner:
                text = cleaner.clean(text, meta.title)

            # Calculate weight
            weight = self._calculate_weight(meta)

            # Split into chunks
            chunks = chunk_cleaner._split_into_chunks(text)

            # Create book data
            book_data = BookData(
                metadata=meta,
                text=text,
                chunks=chunks,
                weight=weight,
                char_count=len(text),
                token_count_estimate=len(text) // 4,
            )

            self.books[key] = book_data
            self.total_weight += weight * len(chunks)
            self.total_chars += len(text)
            self.total_chunks += len(chunks)

            if verbose:
                print(f"  [OK] {meta.title}: {len(chunks)} chunks, weight={weight:.2f}")

        self._loaded = True

        if verbose:
            print(f"\nLoaded {len(self.books)} books")
            print(f"Total chunks: {self.total_chunks:,}")
            print(f"Total chars: {self.total_chars:,}")
            print(f"Estimated tokens: {self.total_chars // 4:,}")

    def _calculate_weight(self, meta: BookMetadata) -> float:
        """Calculate sampling weight for a book."""
        period_w = self.period_weights.get(meta.period, 1.0)
        importance_w = self.importance_weights.get(meta.importance, 1.0)
        content_w = self.content_weights.get(meta.content_type, 1.0)

        return period_w * importance_w * content_w

    def sample_chunk(self) -> Tuple[str, BookMetadata]:
        """
        Sample a single chunk based on weights.

        Returns:
            Tuple of (chunk_text, book_metadata)
        """
        if not self._loaded:
            raise RuntimeError("Call load() first")

        # Build weighted list
        weighted_chunks = []
        for key, book in self.books.items():
            for chunk in book.chunks:
                weighted_chunks.append((chunk, book.metadata, book.weight))

        # Sample based on weight
        weights = [w for _, _, w in weighted_chunks]
        total = sum(weights)
        probs = [w / total for w in weights]

        idx = np.random.choice(len(weighted_chunks), p=probs)
        chunk, meta, _ = weighted_chunks[idx]

        return chunk, meta

    def sample_batch(
        self,
        batch_size: int,
        return_metadata: bool = False,
    ) -> List[str]:
        """
        Sample a batch of chunks.

        Args:
            batch_size: Number of chunks to sample
            return_metadata: If True, return (chunks, metadata) tuple

        Returns:
            List of text chunks (or tuple with metadata)
        """
        chunks = []
        metadata = []

        for _ in range(batch_size):
            chunk, meta = self.sample_chunk()
            chunks.append(chunk)
            metadata.append(meta)

        if return_metadata:
            return chunks, metadata
        return chunks

    def get_all_text(self) -> str:
        """Get all text concatenated (for tokenizer training)."""
        if not self._loaded:
            raise RuntimeError("Call load() first")

        texts = []
        for book in self.books.values():
            texts.append(book.text)

        return '\n\n'.join(texts)

    def get_train_val_split(
        self,
        val_ratio: float = 0.05,
        seed: int = 42,
    ) -> Tuple[List[str], List[str]]:
        """
        Split chunks into train and validation sets.

        Args:
            val_ratio: Fraction for validation
            seed: Random seed

        Returns:
            Tuple of (train_chunks, val_chunks)
        """
        if not self._loaded:
            raise RuntimeError("Call load() first")

        random.seed(seed)

        all_chunks = []
        for book in self.books.values():
            all_chunks.extend(book.chunks)

        random.shuffle(all_chunks)

        val_size = int(len(all_chunks) * val_ratio)
        val_chunks = all_chunks[:val_size]
        train_chunks = all_chunks[val_size:]

        return train_chunks, val_chunks

    def get_statistics(self) -> Dict:
        """Get corpus statistics."""
        if not self._loaded:
            return {'loaded': False}

        by_period = {'early': 0, 'middle': 0, 'mature': 0}
        by_importance = {'core': 0, 'supplementary': 0, 'reference': 0}
        by_type = {}

        for book in self.books.values():
            by_period[book.metadata.period] += book.char_count
            by_importance[book.metadata.importance] += book.char_count
            ct = book.metadata.content_type
            by_type[ct] = by_type.get(ct, 0) + book.char_count

        return {
            'loaded': True,
            'num_books': len(self.books),
            'total_chunks': self.total_chunks,
            'total_chars': self.total_chars,
            'estimated_tokens': self.total_chars // 4,
            'by_period': by_period,
            'by_importance': by_importance,
            'by_type': by_type,
        }

    def print_statistics(self):
        """Print formatted statistics."""
        stats = self.get_statistics()

        print("\n" + "="*60)
        print("Corpus Statistics")
        print("="*60)

        print(f"\nBooks loaded: {stats['num_books']}")
        print(f"Total chunks: {stats['total_chunks']:,}")
        print(f"Total characters: {stats['total_chars']:,}")
        print(f"Estimated tokens: {stats['estimated_tokens']:,}")

        print("\nBy Period:")
        for period, chars in stats['by_period'].items():
            pct = chars / stats['total_chars'] * 100 if stats['total_chars'] > 0 else 0
            print(f"  {period:10}: {chars:>10,} chars ({pct:5.1f}%)")

        print("\nBy Importance:")
        for imp, chars in stats['by_importance'].items():
            pct = chars / stats['total_chars'] * 100 if stats['total_chars'] > 0 else 0
            print(f"  {imp:12}: {chars:>10,} chars ({pct:5.1f}%)")

        print("\nBy Content Type:")
        for ct, chars in sorted(stats['by_type'].items(), key=lambda x: -x[1]):
            pct = chars / stats['total_chars'] * 100 if stats['total_chars'] > 0 else 0
            print(f"  {ct:12}: {chars:>10,} chars ({pct:5.1f}%)")

        print("\nEffective Weights (after applying multipliers):")
        for key, book in sorted(self.books.items(), key=lambda x: -x[1].weight):
            print(f"  {book.metadata.title[:40]:40}: {book.weight:.2f}")


class IterableCorpusSampler:
    """
    Iterator that yields weighted samples indefinitely.

    For use with PyTorch DataLoader.
    """

    def __init__(self, sampler: WeightedCorpusSampler):
        self.sampler = sampler

    def __iter__(self):
        return self

    def __next__(self) -> str:
        chunk, _ = self.sampler.sample_chunk()
        return chunk


# Quick test
if __name__ == '__main__':
    import sys

    # Test with actual corpus
    corpus_dir = './processed_text'

    if not os.path.exists(corpus_dir):
        print(f"Corpus directory not found: {corpus_dir}")
        print("Run from project root directory")
        sys.exit(1)

    sampler = WeightedCorpusSampler(
        corpus_dir=corpus_dir,
        period_weights={'mature': 3.0, 'middle': 2.0, 'early': 0.5},
    )

    sampler.load(include_prose=True, max_books=5)
    sampler.print_statistics()

    print("\n" + "="*60)
    print("Sample chunks:")
    print("="*60)

    for i, (chunk, meta) in enumerate([sampler.sample_chunk() for _ in range(3)]):
        print(f"\n[{i+1}] From: {meta.title}")
        print(f"    Period: {meta.period}, Importance: {meta.importance}")
        print(f"    Preview: {chunk[:200]}...")
