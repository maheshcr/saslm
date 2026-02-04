#!/usr/bin/env python3
"""
Build Clean Corpus for SASLM Training

This script:
1. Loads raw processed text files
2. Cleans them rigorously (removes artifacts)
3. Classifies and tags them
4. Creates train/val splits
5. Saves to clean_corpus directory
6. Generates corpus statistics

Usage:
    # Build prose-only corpus
    python src/data/build_corpus.py --mode prose

    # Build prose + poetry corpus
    python src/data/build_corpus.py --mode all

    # Verify existing corpus
    python src/data/build_corpus.py --verify

Output:
    data/clean_prose/          # Cleaned prose texts
    data/clean_all/            # Cleaned prose + poetry
    data/corpus_stats.json     # Statistics
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.corpus_metadata import (
    CORPUS_METADATA,
    BookMetadata,
    get_books_by_filter,
    print_corpus_summary,
)
from src.data.text_cleaner import TextCleaner, verify_diacritics


def build_corpus(
    source_dir: str,
    output_dir: str,
    include_prose: bool = True,
    include_poetry: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Build clean corpus from source files.

    Args:
        source_dir: Directory with raw processed text
        output_dir: Output directory for clean text
        include_prose: Include prose books
        include_poetry: Include poetry books
        verbose: Print progress

    Returns:
        Statistics dictionary
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaner = TextCleaner(aggressive_mode=False)

    stats = {
        'build_time': datetime.now().isoformat(),
        'source_dir': str(source_dir),
        'output_dir': str(output_dir),
        'include_prose': include_prose,
        'include_poetry': include_poetry,
        'books': [],
        'totals': {
            'original_chars': 0,
            'cleaned_chars': 0,
            'books_processed': 0,
            'books_skipped': 0,
        },
        'by_period': {'early': 0, 'middle': 0, 'mature': 0},
        'by_importance': {'core': 0, 'supplementary': 0, 'reference': 0},
        'diacritics': {},
    }

    if verbose:
        print(f"\nBuilding corpus from {source_dir}")
        print(f"Output: {output_dir}")
        print(f"Mode: prose={'yes' if include_prose else 'no'}, poetry={'yes' if include_poetry else 'no'}")
        print("-" * 60)

    for key, meta in CORPUS_METADATA.items():
        # Check if book should be included
        should_include = False
        if include_prose and meta.include_in_prose:
            should_include = True
        if include_poetry and meta.include_in_poetry:
            should_include = True

        if not should_include:
            continue

        source_file = source_dir / meta.filename
        if not source_file.exists():
            if verbose:
                print(f"  [SKIP] {meta.filename} - not found")
            stats['totals']['books_skipped'] += 1
            continue

        # Load original text
        with open(source_file, 'r', encoding='utf-8') as f:
            original_text = f.read()

        # Clean text
        cleaned_text = cleaner.clean(original_text, meta.title)

        # Get statistics
        clean_stats = cleaner.get_statistics(original_text, cleaned_text)
        diacritic_stats = verify_diacritics(cleaned_text)

        # Save cleaned text
        output_file = output_dir / meta.filename
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

        # Update stats
        book_stats = {
            'key': key,
            'filename': meta.filename,
            'title': meta.title,
            'period': meta.period,
            'importance': meta.importance,
            'content_type': meta.content_type,
            **clean_stats,
            'diacritics': diacritic_stats['total_diacritics'],
        }
        stats['books'].append(book_stats)

        stats['totals']['original_chars'] += clean_stats['original_chars']
        stats['totals']['cleaned_chars'] += clean_stats['cleaned_chars']
        stats['totals']['books_processed'] += 1

        stats['by_period'][meta.period] += clean_stats['cleaned_chars']
        stats['by_importance'][meta.importance] += clean_stats['cleaned_chars']

        # Aggregate diacritics
        for char, count in diacritic_stats['diacritic_counts'].items():
            stats['diacritics'][char] = stats['diacritics'].get(char, 0) + count

        if verbose:
            reduction = clean_stats['reduction_percent']
            print(f"  [OK] {meta.title[:40]:40} | "
                  f"{clean_stats['cleaned_chars']:>8,} chars | "
                  f"-{reduction:4.1f}% | "
                  f"{diacritic_stats['total_diacritics']:>4} diacritics")

    # Calculate totals
    total_orig = stats['totals']['original_chars']
    total_clean = stats['totals']['cleaned_chars']
    if total_orig > 0:
        stats['totals']['total_reduction_percent'] = (1 - total_clean / total_orig) * 100
    else:
        stats['totals']['total_reduction_percent'] = 0

    stats['totals']['estimated_tokens'] = total_clean // 4
    stats['totals']['total_diacritics'] = sum(stats['diacritics'].values())

    # Save stats
    stats_file = output_dir / 'corpus_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    if verbose:
        print("-" * 60)
        print(f"\nCorpus built successfully!")
        print(f"  Books processed: {stats['totals']['books_processed']}")
        print(f"  Books skipped:   {stats['totals']['books_skipped']}")
        print(f"  Total chars:     {total_clean:,}")
        print(f"  Est. tokens:     {total_clean // 4:,}")
        print(f"  Reduction:       {stats['totals']['total_reduction_percent']:.1f}%")
        print(f"  Diacritics:      {stats['totals']['total_diacritics']:,}")
        print(f"\n  Stats saved to: {stats_file}")

    return stats


def verify_corpus(corpus_dir: str, verbose: bool = True) -> bool:
    """
    Verify corpus integrity.

    Checks:
    - All expected files exist
    - Files are non-empty
    - Diacritics are preserved
    - No obvious artifacts remain

    Returns:
        True if verification passes
    """
    corpus_dir = Path(corpus_dir)
    stats_file = corpus_dir / 'corpus_stats.json'

    if not stats_file.exists():
        print(f"Stats file not found: {stats_file}")
        return False

    with open(stats_file, 'r') as f:
        stats = json.load(f)

    if verbose:
        print(f"\nVerifying corpus at {corpus_dir}")
        print("-" * 60)

    issues = []

    for book in stats['books']:
        filepath = corpus_dir / book['filename']

        # Check file exists
        if not filepath.exists():
            issues.append(f"Missing: {book['filename']}")
            continue

        # Load and check
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        # Check non-empty
        if len(text) < 1000:
            issues.append(f"Too small: {book['filename']} ({len(text)} chars)")

        # Check diacritics preserved
        diacritics = verify_diacritics(text)
        if book['diacritics'] > 0 and diacritics['total_diacritics'] == 0:
            issues.append(f"Lost diacritics: {book['filename']}")

        # Check for artifact patterns
        artifact_count = 0
        for line in text.split('\n')[:100]:  # Check first 100 lines
            line = line.strip()
            if line.isdigit() and len(line) < 5:  # Standalone page number
                artifact_count += 1

        if artifact_count > 5:
            issues.append(f"Possible artifacts: {book['filename']} ({artifact_count} suspicious lines)")

        if verbose:
            status = "[OK]" if not any(book['filename'] in i for i in issues) else "[ISSUE]"
            print(f"  {status} {book['filename']}")

    if verbose:
        print("-" * 60)
        if issues:
            print(f"\nFound {len(issues)} issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\nAll checks passed!")

    return len(issues) == 0


def print_corpus_info(corpus_dir: str):
    """Print detailed corpus information."""
    corpus_dir = Path(corpus_dir)
    stats_file = corpus_dir / 'corpus_stats.json'

    if not stats_file.exists():
        print(f"Stats file not found: {stats_file}")
        return

    with open(stats_file, 'r') as f:
        stats = json.load(f)

    print("\n" + "="*70)
    print("SASLM Corpus Information")
    print("="*70)

    print(f"\nBuild time: {stats['build_time']}")
    print(f"Source: {stats['source_dir']}")
    print(f"Output: {stats['output_dir']}")

    print(f"\n--- Totals ---")
    print(f"Books processed: {stats['totals']['books_processed']}")
    print(f"Total characters: {stats['totals']['cleaned_chars']:,}")
    print(f"Estimated tokens: {stats['totals']['estimated_tokens']:,}")
    print(f"Total diacritics: {stats['totals']['total_diacritics']:,}")

    print(f"\n--- By Period ---")
    total = stats['totals']['cleaned_chars']
    for period, chars in stats['by_period'].items():
        pct = chars / total * 100 if total > 0 else 0
        print(f"  {period:10}: {chars:>10,} chars ({pct:5.1f}%)")

    print(f"\n--- By Importance ---")
    for imp, chars in stats['by_importance'].items():
        pct = chars / total * 100 if total > 0 else 0
        print(f"  {imp:12}: {chars:>10,} chars ({pct:5.1f}%)")

    print(f"\n--- Top Diacritics ---")
    sorted_diacritics = sorted(stats['diacritics'].items(), key=lambda x: -x[1])
    for char, count in sorted_diacritics[:15]:
        print(f"  {char}: {count:,}")

    print(f"\n--- Books (sorted by size) ---")
    sorted_books = sorted(stats['books'], key=lambda x: -x['cleaned_chars'])
    for book in sorted_books[:15]:
        print(f"  {book['title'][:40]:40}: {book['cleaned_chars']:>10,} chars")


def main():
    parser = argparse.ArgumentParser(description='Build SASLM corpus')
    parser.add_argument('--mode', choices=['prose', 'poetry', 'all'],
                       default='prose', help='What to include')
    parser.add_argument('--source', default='./processed_text',
                       help='Source directory')
    parser.add_argument('--output', default=None,
                       help='Output directory (auto-generated if not specified)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify existing corpus instead of building')
    parser.add_argument('--info', action='store_true',
                       help='Print corpus information')
    parser.add_argument('--quiet', action='store_true',
                       help='Quiet mode')

    args = parser.parse_args()

    # Determine output directory
    if args.output:
        output_dir = args.output
    elif args.mode == 'prose':
        output_dir = './data/clean_prose'
    elif args.mode == 'poetry':
        output_dir = './data/clean_poetry'
    else:
        output_dir = './data/clean_all'

    # Handle commands
    if args.info:
        print_corpus_info(output_dir)
        return 0

    if args.verify:
        success = verify_corpus(output_dir, verbose=not args.quiet)
        return 0 if success else 1

    # Build corpus
    include_prose = args.mode in ['prose', 'all']
    include_poetry = args.mode in ['poetry', 'all']

    stats = build_corpus(
        source_dir=args.source,
        output_dir=output_dir,
        include_prose=include_prose,
        include_poetry=include_poetry,
        verbose=not args.quiet,
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
