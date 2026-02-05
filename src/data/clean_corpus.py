#!/usr/bin/env python3
"""
Corpus Cleaner for SASLM

Removes editorial artifacts that contaminate training:
- Page numbers embedded in text ("248 Letters on Yoga - I")
- Editorial notes ("Written by Sri Aurobindo to his secretary...")
- Section markers ("— Ed.", "— Editor")
- Volume/page references ("[Vol. X, p. Y]")
- Footnote references
- Publisher boilerplate

Usage:
    python src/data/clean_corpus.py --input data/clean_prose --output data/clean_prose_v2

    # Preview changes without writing
    python src/data/clean_corpus.py --input data/clean_prose --preview
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Tuple


class CorpusCleaner:
    """Removes editorial artifacts from corpus text."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.stats = {
            'lines_removed': 0,
            'patterns_matched': {},
            'files_processed': 0,
            'chars_before': 0,
            'chars_after': 0,
        }

        # Patterns to remove (compiled for efficiency)
        self.removal_patterns = [
            # Page/volume references
            (r'^\s*\d+\s+Letters on Yoga\s*[-–—]\s*[IVX]+\s*$', 'page_header'),
            (r'^\s*Letters on Yoga\s*[-–—]\s*[IVX]+\s+\d+\s*$', 'page_header'),
            (r'^\s*\d+\s+The Life Divine\s*$', 'page_header'),
            (r'^\s*The Life Divine\s+\d+\s*$', 'page_header'),
            (r'^\s*\d+\s+The Synthesis of Yoga\s*$', 'page_header'),
            (r'^\s*\d+\s+Essays on the Gita\s*$', 'page_header'),
            (r'^\s*\d+\s+Record of Yoga\s*$', 'page_header'),
            (r'^\s*\d+\s+[A-Z][a-z]+\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)?\s*$', 'page_header_generic'),

            # Editorial notes
            (r'^\s*[-–—]\s*Ed\.?\s*$', 'editor_note'),
            (r'^\s*[-–—]\s*Editor\.?\s*$', 'editor_note'),
            (r'^\s*\[Editor\'?s?\s+[Nn]ote.*?\]', 'editor_note'),
            (r'Written by Sri Aurobindo to his secretary.*$', 'editorial_comment'),
            (r'who revised the third of the third person.*$', 'editorial_comment'),

            # Footnote markers and references
            (r'^\s*\d+\s+Written\s+(by|to)\s+', 'footnote'),
            (r'^\s*\[\d+\]\s*$', 'footnote_marker'),
            (r'^\s*\*\s*\*\s*\*\s*$', 'section_break'),

            # Volume/page citations
            (r'\[Vol\.\s*\d+,?\s*p\.?\s*\d+\]', 'citation'),
            (r'\(p\.\s*\d+\)', 'page_ref'),
            (r'\[p\.\s*\d+\]', 'page_ref'),

            # Publisher/copyright
            (r'^\s*©.*$', 'copyright'),
            (r'^\s*Sri Aurobindo Ashram\s*$', 'publisher'),
            (r'^\s*Pondicherry\s*$', 'publisher'),
            (r'^\s*All rights reserved\.?\s*$', 'copyright'),

            # Table of contents style entries
            (r'^\s*Chapter\s+[IVXLC]+\s*\.?\s*$', 'toc_entry'),
            (r'^\s*Part\s+[IVXLC]+\s*\.?\s*$', 'toc_entry'),
            (r'^\s*Section\s+[IVXLC]+\s*\.?\s*$', 'toc_entry'),

            # Standalone numbers (page numbers)
            (r'^\s*\d{1,4}\s*$', 'page_number'),

            # Multiple dots (ellipsis placeholders or TOC dots)
            (r'^\s*\.{4,}\s*$', 'dots'),
            (r'\s*\.{4,}\s*\d+\s*$', 'toc_dots'),
        ]

        # Compile patterns
        self.compiled_patterns = [
            (re.compile(p, re.IGNORECASE | re.MULTILINE), name)
            for p, name in self.removal_patterns
        ]

        # Inline patterns to clean (not remove whole line)
        self.inline_patterns = [
            (r'\[\d+\]', ''),  # Footnote numbers like [1], [23]
            (r'\s*[-–—]\s*Ed\.\s*', ' '),  # Inline editor marks
            (r'\s{3,}', ' '),  # Multiple spaces
        ]

        self.compiled_inline = [
            (re.compile(p), repl) for p, repl in self.inline_patterns
        ]

    def clean_line(self, line: str) -> Tuple[str, List[str]]:
        """
        Clean a single line.

        Returns:
            Tuple of (cleaned_line, list_of_matched_patterns)
            If line should be removed entirely, returns ('', patterns)
        """
        matched = []

        # Check removal patterns
        for pattern, name in self.compiled_patterns:
            if pattern.search(line):
                matched.append(name)
                self.stats['patterns_matched'][name] = \
                    self.stats['patterns_matched'].get(name, 0) + 1
                return '', matched

        # Apply inline cleaning
        cleaned = line
        for pattern, replacement in self.compiled_inline:
            cleaned = pattern.sub(replacement, cleaned)

        return cleaned, matched

    def clean_text(self, text: str) -> str:
        """Clean entire text, removing artifacts."""
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            cleaned, _ = self.clean_line(line)
            if cleaned or cleaned == '':  # Keep empty lines for paragraph structure
                if cleaned.strip() or (cleaned_lines and cleaned_lines[-1].strip()):
                    # Keep line if it has content, or if previous line had content (preserve paragraph breaks)
                    cleaned_lines.append(cleaned)

        # Join and post-process
        result = '\n'.join(cleaned_lines)

        # Remove excessive blank lines
        result = re.sub(r'\n{4,}', '\n\n\n', result)

        # Remove blank lines at start/end
        result = result.strip()

        return result

    def clean_file(self, input_path: Path, output_path: Path = None, preview: bool = False) -> dict:
        """
        Clean a single file.

        Args:
            input_path: Path to input file
            output_path: Path to output file (if None, modifies in place)
            preview: If True, don't write, just report changes

        Returns:
            Dict with cleaning statistics
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            original = f.read()

        cleaned = self.clean_text(original)

        stats = {
            'file': str(input_path),
            'chars_before': len(original),
            'chars_after': len(cleaned),
            'chars_removed': len(original) - len(cleaned),
            'reduction_pct': (1 - len(cleaned) / len(original)) * 100 if original else 0,
        }

        self.stats['chars_before'] += stats['chars_before']
        self.stats['chars_after'] += stats['chars_after']
        self.stats['files_processed'] += 1

        if self.verbose:
            print(f"  {input_path.name}: {stats['chars_removed']:,} chars removed ({stats['reduction_pct']:.1f}%)")

        if not preview and output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned)

        return stats

    def clean_directory(self, input_dir: Path, output_dir: Path = None, preview: bool = False) -> dict:
        """
        Clean all .txt files in a directory.

        Args:
            input_dir: Directory with input files
            output_dir: Directory for output files (if None, uses input_dir)
            preview: If True, don't write, just report changes

        Returns:
            Dict with overall statistics
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir) if output_dir else input_dir

        txt_files = sorted(input_dir.glob('*.txt'))

        if not txt_files:
            print(f"No .txt files found in {input_dir}")
            return self.stats

        print(f"Cleaning {len(txt_files)} files...")
        if preview:
            print("(Preview mode - no files will be written)")
        print()

        for txt_file in txt_files:
            if output_dir == input_dir:
                out_file = txt_file
            else:
                out_file = output_dir / txt_file.name

            self.clean_file(txt_file, out_file, preview)

        return self.stats

    def print_summary(self):
        """Print cleaning summary."""
        print("\n" + "=" * 60)
        print("Cleaning Summary")
        print("=" * 60)
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Characters before: {self.stats['chars_before']:,}")
        print(f"Characters after: {self.stats['chars_after']:,}")
        chars_removed = self.stats['chars_before'] - self.stats['chars_after']
        reduction = (chars_removed / self.stats['chars_before'] * 100) if self.stats['chars_before'] else 0
        print(f"Characters removed: {chars_removed:,} ({reduction:.1f}%)")

        if self.stats['patterns_matched']:
            print("\nPatterns matched:")
            for pattern, count in sorted(self.stats['patterns_matched'].items(), key=lambda x: -x[1]):
                print(f"  {pattern}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Clean corpus text files')
    parser.add_argument('--input', '-i', required=True, help='Input directory')
    parser.add_argument('--output', '-o', help='Output directory (default: overwrite input)')
    parser.add_argument('--preview', '-p', action='store_true', help='Preview changes without writing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    cleaner = CorpusCleaner(verbose=args.verbose or args.preview)
    cleaner.clean_directory(
        input_dir=Path(args.input),
        output_dir=Path(args.output) if args.output else None,
        preview=args.preview,
    )
    cleaner.print_summary()


if __name__ == '__main__':
    main()
