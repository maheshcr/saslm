"""
Text Cleaner for SASLM Corpus

Rigorously removes artifacts from OCR-extracted text:
- Page numbers (standalone or with surrounding text)
- Running headers/footers (book titles, chapter names)
- Table of contents lines
- Index entries
- Footnote markers
- Publisher information
- Excessive whitespace

Preserves:
- Sanskrit diacritics (ā, ī, ū, ṛ, ṅ, ñ, ś, ṣ, ṭ, ḍ, ṇ, ḥ, ṃ)
- Paragraph structure
- Meaningful punctuation

Usage:
    cleaner = TextCleaner()
    clean_text = cleaner.clean(raw_text)

    # Or use the simple function
    from src.data.text_cleaner import clean_text
    result = clean_text(raw_text)
"""

import re
from typing import List, Set, Optional, Tuple
from dataclasses import dataclass, field


# Sanskrit diacritics to preserve
SANSKRIT_DIACRITICS = set('āīūṛṝḷḹēōṃḥñṅṇṭḍśṣĀĪŪṚṜḶḸĒŌṂḤÑṄṆṬḌŚṢ')

# Common header patterns to remove
HEADER_PATTERNS = [
    r'^the\s+life\s+divine\s*$',
    r'^the\s+synthesis\s+of\s+yoga\s*$',
    r'^letters\s+on\s+yoga\s*$',
    r'^essays\s+on\s+the\s+gita\s*$',
    r'^savitri\s*$',
    r'^record\s+of\s+yoga\s*$',
    r'^the\s+secret\s+of\s+the\s+veda\s*$',
    r'^isha\s+upanishad\s*$',
    r'^kena\s+upanishad\s*$',
    r'^book\s+\w+\s*$',
    r'^part\s+\w+\s*$',
    r'^chapter\s+\w+\s*$',
    r'^section\s+\w+\s*$',
    r'^canto\s+\w+\s*$',
]

# Patterns that indicate artifact lines
ARTIFACT_PATTERNS = [
    r'^\s*\d+\s*$',                    # Standalone page number
    r'^\s*-\s*\d+\s*-\s*$',            # Page number with dashes
    r'^\s*\[\s*\d+\s*\]\s*$',          # Bracketed page number
    r'^\s*page\s+\d+\s*$',             # "Page X"
    r'^\s*p\.\s*\d+\s*$',              # "p. X"
    r'^\s*\d+\s+\w+\s*$',              # Page number followed by single word (header)
    r'^\s*\w+\s+\d+\s*$',              # Single word followed by page number
    r'^\s*contents\s*$',               # TOC header
    r'^\s*table\s+of\s+contents\s*$',
    r'^\s*index\s*$',
    r'^\s*appendix\s*$',
    r'^\s*bibliography\s*$',
    r'^\s*glossary\s*$',
    r'^\s*footnotes?\s*$',
    r'^\s*endnotes?\s*$',
    r'^\s*\*\s*\*\s*\*\s*$',           # Section break markers
    r'^\s*\.\s*\.\s*\.\s*$',           # Ellipsis lines
    r'^\s*_{3,}\s*$',                  # Underscores
    r'^\s*-{3,}\s*$',                  # Dashes
    r'^\s*={3,}\s*$',                  # Equals
    r'^\s*©.*$',                       # Copyright
    r'^\s*isbn\s*:?\s*[\d-]+\s*$',     # ISBN
    r'^\s*printed\s+(in|by)\s+',       # Printer info
    r'^\s*published\s+by\s+',          # Publisher info
    r'^\s*sri\s+aurobindo\s+ashram\s*$',  # Publisher
    r'^\s*pondicherry\s*$',            # Location
    r'^\s*all\s+rights\s+reserved\s*$',
]

# Index entry patterns (word followed by page numbers)
INDEX_PATTERNS = [
    r'^\s*\w[\w\s,]+\d+(?:,\s*\d+)*\s*$',  # "Topic 123, 456"
    r'^\s*\w[\w\s,]+\d+-\d+\s*$',          # "Topic 123-456"
]


@dataclass
class TextCleaner:
    """
    Cleans OCR-extracted text by removing artifacts.

    Attributes:
        min_line_length: Lines shorter than this may be headers/artifacts
        max_consecutive_short: Max consecutive short lines before flagging
        preserve_paragraphs: Keep paragraph structure
        aggressive_mode: More aggressive artifact removal
    """

    min_line_length: int = 20
    max_consecutive_short: int = 3
    preserve_paragraphs: bool = True
    aggressive_mode: bool = False

    # Compiled patterns (lazily initialized)
    _header_patterns: List[re.Pattern] = field(default_factory=list, repr=False)
    _artifact_patterns: List[re.Pattern] = field(default_factory=list, repr=False)
    _index_patterns: List[re.Pattern] = field(default_factory=list, repr=False)

    def __post_init__(self):
        """Compile regex patterns."""
        self._header_patterns = [
            re.compile(p, re.IGNORECASE) for p in HEADER_PATTERNS
        ]
        self._artifact_patterns = [
            re.compile(p, re.IGNORECASE) for p in ARTIFACT_PATTERNS
        ]
        self._index_patterns = [
            re.compile(p, re.IGNORECASE) for p in INDEX_PATTERNS
        ]

    def clean(self, text: str, book_title: str = None) -> str:
        """
        Clean text by removing artifacts.

        Args:
            text: Raw OCR text
            book_title: Optional book title for header detection

        Returns:
            Cleaned text
        """
        # Split into lines
        lines = text.split('\n')

        # Process lines
        cleaned_lines = []
        consecutive_short = 0

        for i, line in enumerate(lines):
            # Check if line is an artifact
            is_artifact, reason = self._is_artifact(line, book_title)

            if is_artifact:
                # Track consecutive short/artifact lines
                if len(line.strip()) < self.min_line_length:
                    consecutive_short += 1
                continue

            # Reset counter for valid lines
            if len(line.strip()) >= self.min_line_length:
                consecutive_short = 0

            # Skip if too many consecutive short lines (probably headers/footers)
            if consecutive_short > self.max_consecutive_short:
                continue

            # Clean the line
            cleaned = self._clean_line(line)
            if cleaned:
                cleaned_lines.append(cleaned)

        # Join and post-process
        text = '\n'.join(cleaned_lines)
        text = self._post_process(text)

        return text

    def _is_artifact(self, line: str, book_title: str = None) -> Tuple[bool, str]:
        """
        Check if a line is an artifact.

        Returns:
            Tuple of (is_artifact, reason)
        """
        stripped = line.strip()

        # Empty lines are kept for paragraph structure
        if not stripped:
            return False, ""

        # Check artifact patterns
        for pattern in self._artifact_patterns:
            if pattern.match(stripped):
                return True, "artifact_pattern"

        # Check header patterns
        for pattern in self._header_patterns:
            if pattern.match(stripped):
                return True, "header_pattern"

        # Check book title as header
        if book_title:
            title_lower = book_title.lower()
            if stripped.lower() == title_lower or \
               stripped.lower().startswith(title_lower[:20]):
                return True, "book_title"

        # Check index patterns (aggressive mode only)
        if self.aggressive_mode:
            for pattern in self._index_patterns:
                if pattern.match(stripped):
                    return True, "index_entry"

        # Very short lines that are all caps (likely headers)
        if len(stripped) < 30 and stripped.isupper():
            return True, "all_caps_header"

        # Lines that are just numbers and punctuation
        if re.match(r'^[\d\s\.\-,;:]+$', stripped):
            return True, "numeric_only"

        return False, ""

    def _clean_line(self, line: str) -> str:
        """Clean a single line."""
        # Remove leading/trailing whitespace but preserve indentation structure
        line = line.rstrip()

        # Remove inline page numbers like "[123]" or "(p. 45)"
        line = re.sub(r'\[\d+\]', '', line)
        line = re.sub(r'\(p\.\s*\d+\)', '', line)

        # Remove footnote markers like "¹" or "*" at word boundaries
        line = re.sub(r'[¹²³⁴⁵⁶⁷⁸⁹⁰\*†‡§]+(?=\s|$)', '', line)

        # Normalize whitespace (but keep single spaces)
        line = re.sub(r'  +', ' ', line)

        # Remove OCR artifacts like "l" for "I" at start of sentences
        # (careful not to over-correct)

        return line

    def _post_process(self, text: str) -> str:
        """Post-process the entire text."""
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Remove excessive blank lines (more than 2 consecutive)
        text = re.sub(r'\n{4,}', '\n\n\n', text)

        # Remove blank lines at start/end
        text = text.strip()

        # Ensure paragraphs are separated by blank lines
        if self.preserve_paragraphs:
            # This is a heuristic: lines ending with period followed by
            # line starting with capital letter likely indicate paragraph break
            pass  # Keep existing paragraph structure

        return text

    def get_statistics(self, original: str, cleaned: str) -> dict:
        """Get cleaning statistics."""
        orig_lines = original.split('\n')
        clean_lines = cleaned.split('\n')

        orig_chars = len(original)
        clean_chars = len(cleaned)

        return {
            'original_lines': len(orig_lines),
            'cleaned_lines': len(clean_lines),
            'lines_removed': len(orig_lines) - len(clean_lines),
            'original_chars': orig_chars,
            'cleaned_chars': clean_chars,
            'chars_removed': orig_chars - clean_chars,
            'reduction_percent': (1 - clean_chars / orig_chars) * 100 if orig_chars > 0 else 0,
        }


class ChunkCleaner:
    """
    Cleans text and splits into training chunks.

    Creates overlapping chunks suitable for language model training.
    """

    def __init__(
        self,
        chunk_size: int = 2048,
        overlap: int = 128,
        min_chunk_size: int = 256,
    ):
        """
        Initialize chunk cleaner.

        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size to keep
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.cleaner = TextCleaner()

    def process(self, text: str, book_title: str = None) -> List[str]:
        """
        Clean text and split into chunks.

        Args:
            text: Raw text
            book_title: Optional book title

        Returns:
            List of text chunks
        """
        # Clean first
        cleaned = self.cleaner.clean(text, book_title)

        # Split into chunks
        chunks = self._split_into_chunks(cleaned)

        return chunks

    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []

        # Try to split at paragraph boundaries
        paragraphs = text.split('\n\n')

        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) > self.chunk_size:
                # Save current chunk if large enough
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())

                # Start new chunk with overlap
                if self.overlap > 0 and len(current_chunk) > self.overlap:
                    # Take last `overlap` characters
                    current_chunk = current_chunk[-self.overlap:] + '\n\n' + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    current_chunk += '\n\n' + para
                else:
                    current_chunk = para

        # Don't forget the last chunk
        if len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())

        return chunks


def clean_text(text: str, book_title: str = None, aggressive: bool = False) -> str:
    """
    Convenience function to clean text.

    Args:
        text: Raw text to clean
        book_title: Optional book title for header detection
        aggressive: Use aggressive cleaning mode

    Returns:
        Cleaned text
    """
    cleaner = TextCleaner(aggressive_mode=aggressive)
    return cleaner.clean(text, book_title)


def verify_diacritics(text: str) -> dict:
    """
    Verify Sanskrit diacritics are preserved.

    Returns:
        Dict with diacritic counts
    """
    counts = {}
    for char in SANSKRIT_DIACRITICS:
        count = text.count(char)
        if count > 0:
            counts[char] = count

    return {
        'total_diacritics': sum(counts.values()),
        'unique_diacritics': len(counts),
        'diacritic_counts': counts,
    }


# Quick test
if __name__ == '__main__':
    sample = """
    THE LIFE DIVINE

    123

    Book Two: The Knowledge and the Ignorance

    Chapter I

    The Destiny of the Individual

    124

    An omnipresent Reality is the truth of all life and existence whether
    absolute or relative, whether corporeal or incorporeal, whether animate
    or inanimate, whether intelligent or unintelligent; and in all its
    infinitely varying and even constantly opposed self-expressions, from
    the contradictions nearest to our ordinary experience to those remotest
    antinomies which lose themselves on the verges of the Ineffable, the
    Reality is one and not a sum or concourse.

    THE LIFE DIVINE                                                     125

    This is the truth that the Upanishads teach us, the truth of the Brahman,
    the One without a second, ātman, Self of all that is.

    © Sri Aurobindo Ashram
    """

    cleaner = TextCleaner()
    cleaned = cleaner.clean(sample, "The Life Divine")

    print("ORIGINAL:")
    print(sample)
    print("\nCLEANED:")
    print(cleaned)
    print("\nSTATISTICS:")
    stats = cleaner.get_statistics(sample, cleaned)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nDIACRITICS:")
    diacritics = verify_diacritics(cleaned)
    print(f"  Found: {diacritics['total_diacritics']} diacritics")
    print(f"  Types: {diacritics['diacritic_counts']}")
