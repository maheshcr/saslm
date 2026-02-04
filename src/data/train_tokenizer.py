#!/usr/bin/env python3
"""
Train Custom Tokenizer for SASLM

Trains a BPE tokenizer optimized for Sri Aurobindo's works:
- 16K vocabulary (smaller = higher token frequency)
- Byte-level fallback (preserves Sanskrit diacritics)
- Special tokens for training
- Saves statistics and vocabulary analysis

Usage:
    # Train on prose corpus
    python src/data/train_tokenizer.py --corpus ./data/clean_prose --vocab-size 16384

    # Train on full corpus
    python src/data/train_tokenizer.py --corpus ./data/clean_all --vocab-size 16384

    # Analyze existing tokenizer
    python src/data/train_tokenizer.py --analyze ./tokenizers/tokenizer_16k

Output:
    tokenizers/tokenizer_16k/
    ├── tokenizer.json      # Main tokenizer file
    ├── vocab.json          # Vocabulary mapping
    ├── merges.txt          # BPE merges
    ├── config.json         # Training config
    └── stats.json          # Token statistics
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_corpus_text(corpus_dir: str, verbose: bool = True) -> str:
    """Load all text files from corpus directory."""
    corpus_dir = Path(corpus_dir)

    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    texts = []
    files_loaded = 0

    for filepath in sorted(corpus_dir.glob("*.txt")):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            texts.append(text)
            files_loaded += 1

            if verbose:
                print(f"  Loaded: {filepath.name} ({len(text):,} chars)")

    combined = '\n\n'.join(texts)

    if verbose:
        print(f"\nTotal: {files_loaded} files, {len(combined):,} characters")

    return combined


def train_tokenizer(
    corpus_text: str,
    vocab_size: int = 16384,
    min_frequency: int = 2,
    output_dir: str = './tokenizers/tokenizer_16k',
    verbose: bool = True,
) -> 'Tokenizer':
    """
    Train a BPE tokenizer on the corpus.

    Args:
        corpus_text: Combined text from corpus
        vocab_size: Target vocabulary size
        min_frequency: Minimum token frequency
        output_dir: Where to save the tokenizer
        verbose: Print progress

    Returns:
        Trained tokenizer
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

    if verbose:
        print(f"\nTraining tokenizer...")
        print(f"  Vocab size: {vocab_size:,}")
        print(f"  Min frequency: {min_frequency}")

    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # Pre-tokenizer: ByteLevel for handling all Unicode (including Sanskrit diacritics)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder: ByteLevel to properly decode
    tokenizer.decoder = decoders.ByteLevel()

    # Post-processor for special tokens
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Special tokens
    special_tokens = [
        "[UNK]",      # Unknown token
        "[PAD]",      # Padding
        "[CLS]",      # Classification (not used but standard)
        "[SEP]",      # Separator (not used but standard)
        "[MASK]",     # Mask (for potential MLM)
        "<|endoftext|>",  # End of text (GPT-style)
        "<|startoftext|>",  # Start of text
    ]

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=verbose,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Train from text (using iterator for memory efficiency)
    def text_iterator():
        # Split into chunks to avoid memory issues
        chunk_size = 1_000_000  # 1MB chunks
        for i in range(0, len(corpus_text), chunk_size):
            yield corpus_text[i:i + chunk_size]

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)

    if verbose:
        print(f"  Training complete!")
        print(f"  Final vocab size: {tokenizer.get_vocab_size():,}")

    # Save tokenizer
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save main tokenizer file
    tokenizer.save(str(output_dir / "tokenizer.json"))

    # Save vocab separately for inspection
    vocab = tokenizer.get_vocab()
    with open(output_dir / "vocab.json", 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # Save config
    config = {
        "vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "special_tokens": special_tokens,
        "trained_on": datetime.now().isoformat(),
        "corpus_chars": len(corpus_text),
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    if verbose:
        print(f"\n  Saved to: {output_dir}")

    return tokenizer


def analyze_tokenizer(
    tokenizer_path: str,
    corpus_text: str = None,
    verbose: bool = True,
) -> Dict:
    """
    Analyze a trained tokenizer.

    Args:
        tokenizer_path: Path to tokenizer directory or file
        corpus_text: Optional corpus text for frequency analysis
        verbose: Print analysis

    Returns:
        Analysis dictionary
    """
    from tokenizers import Tokenizer

    tokenizer_path = Path(tokenizer_path)

    # Load tokenizer
    if tokenizer_path.is_dir():
        tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer.json"))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    vocab = tokenizer.get_vocab()
    vocab_size = tokenizer.get_vocab_size()

    # Basic stats
    stats = {
        "vocab_size": vocab_size,
        "special_tokens": [],
        "token_length_distribution": {},
        "sanskrit_tokens": [],
        "common_philosophical_terms": {},
    }

    # Identify special tokens
    special_prefixes = ["[", "<|"]
    for token in vocab:
        if any(token.startswith(p) for p in special_prefixes):
            stats["special_tokens"].append(token)

    # Token length distribution
    lengths = Counter()
    for token in vocab:
        # Decode to get actual length (ByteLevel encoding)
        try:
            decoded = tokenizer.decode([vocab[token]])
            lengths[len(decoded)] += 1
        except:
            lengths[len(token)] += 1

    stats["token_length_distribution"] = dict(sorted(lengths.items())[:20])

    # Sanskrit diacritics check
    sanskrit_chars = set('āīūṛṝḷḹēōṃḥñṅṇṭḍśṣĀĪŪṚṜḶḸĒŌṂḤÑṄṆṬḌŚṢ')
    for token in vocab:
        try:
            decoded = tokenizer.decode([vocab[token]])
            if any(c in sanskrit_chars for c in decoded):
                stats["sanskrit_tokens"].append(decoded)
        except:
            pass

    # Check common philosophical terms
    philosophical_terms = [
        "Supermind", "supermind", "Brahman", "brahman", "Atman", "atman",
        "consciousness", "Consciousness", "Divine", "divine",
        "psychic", "Psychic", "spiritual", "Spiritual",
        "transformation", "Transformation", "yoga", "Yoga",
        "Sachchidananda", "sachchidananda", "Purusha", "purusha",
        "Prakriti", "prakriti", "Shakti", "shakti",
        "supramental", "Supramental", "overmind", "Overmind",
        "integral", "Integral", "evolution", "Evolution",
    ]

    for term in philosophical_terms:
        encoded = tokenizer.encode(term)
        tokens = [tokenizer.decode([t]) for t in encoded.ids]
        stats["common_philosophical_terms"][term] = {
            "num_tokens": len(encoded.ids),
            "tokens": tokens,
        }

    # Corpus frequency analysis
    if corpus_text:
        encoded = tokenizer.encode(corpus_text)
        token_counts = Counter(encoded.ids)

        stats["corpus_analysis"] = {
            "total_tokens": len(encoded.ids),
            "unique_tokens": len(token_counts),
            "avg_token_frequency": len(encoded.ids) / len(token_counts) if token_counts else 0,
            "coverage": len(token_counts) / vocab_size * 100,
        }

        # Most common tokens
        most_common = token_counts.most_common(50)
        stats["most_common_tokens"] = [
            {"token": tokenizer.decode([tid]), "count": count}
            for tid, count in most_common
        ]

        # Least common (used) tokens
        least_common = token_counts.most_common()[-50:]
        stats["least_common_tokens"] = [
            {"token": tokenizer.decode([tid]), "count": count}
            for tid, count in least_common
        ]

    if verbose:
        print("\n" + "="*60)
        print("Tokenizer Analysis")
        print("="*60)

        print(f"\nVocab size: {vocab_size:,}")
        print(f"Special tokens: {len(stats['special_tokens'])}")
        print(f"Sanskrit-containing tokens: {len(stats['sanskrit_tokens'])}")

        print("\nToken length distribution:")
        for length, count in list(stats["token_length_distribution"].items())[:10]:
            print(f"  {length} chars: {count:,} tokens")

        print("\nPhilosophical term tokenization:")
        for term, info in list(stats["common_philosophical_terms"].items())[:15]:
            tokens_str = " | ".join(info["tokens"])
            print(f"  {term:20} -> {info['num_tokens']} tokens: [{tokens_str}]")

        if "corpus_analysis" in stats:
            ca = stats["corpus_analysis"]
            print(f"\nCorpus analysis:")
            print(f"  Total tokens: {ca['total_tokens']:,}")
            print(f"  Unique tokens used: {ca['unique_tokens']:,}")
            print(f"  Avg token frequency: {ca['avg_token_frequency']:.1f}")
            print(f"  Vocab coverage: {ca['coverage']:.1f}%")

            print("\n  Most common tokens:")
            for item in stats["most_common_tokens"][:10]:
                print(f"    '{item['token']}': {item['count']:,}")

        print("\nSample Sanskrit tokens:")
        for token in stats["sanskrit_tokens"][:20]:
            print(f"  {token}")

    return stats


def test_tokenizer(tokenizer_path: str, test_texts: List[str] = None):
    """Test tokenizer with sample texts."""
    from tokenizers import Tokenizer

    tokenizer_path = Path(tokenizer_path)
    if tokenizer_path.is_dir():
        tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer.json"))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    if test_texts is None:
        test_texts = [
            "The Supermind is the Truth-Consciousness.",
            "The psychic being is the soul in evolution.",
            "Sachchidananda is Existence-Consciousness-Bliss.",
            "The ātman is the self, the Brahman is the Absolute.",
            "In the process of involution, consciousness hides itself in Matter.",
            "The transformation of human nature requires the descent of the supramental Force.",
            "Śakti is the Divine Mother, the executive Power of the Divine.",
        ]

    print("\n" + "="*60)
    print("Tokenizer Test")
    print("="*60)

    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        tokens = [tokenizer.decode([t]) for t in encoded.ids]

        print(f"\nOriginal: {text}")
        print(f"Tokens ({len(encoded.ids)}): {tokens}")
        print(f"Decoded: {decoded}")

        # Check roundtrip
        if decoded.strip() != text.strip():
            print("  WARNING: Roundtrip mismatch!")


def compare_tokenizers(path1: str, path2: str, test_text: str):
    """Compare two tokenizers on the same text."""
    from tokenizers import Tokenizer

    def load(path):
        path = Path(path)
        if path.is_dir():
            return Tokenizer.from_file(str(path / "tokenizer.json"))
        return Tokenizer.from_file(str(path))

    tok1 = load(path1)
    tok2 = load(path2)

    enc1 = tok1.encode(test_text)
    enc2 = tok2.encode(test_text)

    print("\n" + "="*60)
    print("Tokenizer Comparison")
    print("="*60)

    print(f"\nText: {test_text[:100]}...")
    print(f"\nTokenizer 1 ({path1}):")
    print(f"  Vocab size: {tok1.get_vocab_size():,}")
    print(f"  Tokens: {len(enc1.ids)}")

    print(f"\nTokenizer 2 ({path2}):")
    print(f"  Vocab size: {tok2.get_vocab_size():,}")
    print(f"  Tokens: {len(enc2.ids)}")

    print(f"\nEfficiency ratio: {len(enc1.ids) / len(enc2.ids):.2f}x")


def main():
    parser = argparse.ArgumentParser(description='Train SASLM tokenizer')
    parser.add_argument('--corpus', default='./data/clean_prose',
                       help='Corpus directory')
    parser.add_argument('--output', default='./tokenizers/tokenizer_16k',
                       help='Output directory')
    parser.add_argument('--vocab-size', type=int, default=16384,
                       help='Vocabulary size')
    parser.add_argument('--min-frequency', type=int, default=2,
                       help='Minimum token frequency')
    parser.add_argument('--analyze', type=str, default=None,
                       help='Analyze existing tokenizer instead of training')
    parser.add_argument('--test', type=str, default=None,
                       help='Test existing tokenizer')
    parser.add_argument('--compare', nargs=2, default=None,
                       help='Compare two tokenizers')
    parser.add_argument('--quiet', action='store_true',
                       help='Quiet mode')

    args = parser.parse_args()
    verbose = not args.quiet

    # Handle analysis mode
    if args.analyze:
        corpus_text = None
        if os.path.exists(args.corpus):
            if verbose:
                print(f"Loading corpus for analysis...")
            corpus_text = load_corpus_text(args.corpus, verbose=verbose)

        stats = analyze_tokenizer(args.analyze, corpus_text, verbose=verbose)

        # Save stats
        output_path = Path(args.analyze)
        if output_path.is_dir():
            stats_path = output_path / "stats.json"
        else:
            stats_path = output_path.parent / "stats.json"

        with open(stats_path, 'w', encoding='utf-8') as f:
            # Convert sets to lists for JSON
            stats_json = json.loads(json.dumps(stats, default=list))
            json.dump(stats_json, f, indent=2, ensure_ascii=False)

        if verbose:
            print(f"\nStats saved to: {stats_path}")

        return 0

    # Handle test mode
    if args.test:
        test_tokenizer(args.test)
        return 0

    # Handle compare mode
    if args.compare:
        corpus_text = load_corpus_text(args.corpus, verbose=False)
        compare_tokenizers(args.compare[0], args.compare[1], corpus_text[:10000])
        return 0

    # Training mode
    if verbose:
        print("="*60)
        print("SASLM Tokenizer Training")
        print("="*60)

    # Load corpus
    if verbose:
        print(f"\nLoading corpus from {args.corpus}...")

    corpus_text = load_corpus_text(args.corpus, verbose=verbose)

    # Train tokenizer
    tokenizer = train_tokenizer(
        corpus_text=corpus_text,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        output_dir=args.output,
        verbose=verbose,
    )

    # Analyze the trained tokenizer
    if verbose:
        print("\nAnalyzing trained tokenizer...")

    stats = analyze_tokenizer(args.output, corpus_text, verbose=verbose)

    # Save stats
    stats_path = Path(args.output) / "stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        stats_json = json.loads(json.dumps(stats, default=list))
        json.dump(stats_json, f, indent=2, ensure_ascii=False)

    # Test tokenizer
    if verbose:
        test_tokenizer(args.output)

    if verbose:
        print("\n" + "="*60)
        print("Training complete!")
        print("="*60)
        print(f"\nTokenizer saved to: {args.output}")
        print(f"Vocab size: {tokenizer.get_vocab_size():,}")

        if "corpus_analysis" in stats:
            ca = stats["corpus_analysis"]
            print(f"Corpus tokens: {ca['total_tokens']:,}")
            print(f"Avg token frequency: {ca['avg_token_frequency']:.1f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
