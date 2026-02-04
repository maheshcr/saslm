"""
Corpus Metadata for Sri Aurobindo's Complete Works

Defines metadata for each volume:
- content_type: essay, letter, poetry, drama, commentary, record
- period: early (1893-1910), middle (1910-1926), mature (1926-1950)
- importance: core, supplementary, reference
- include_in_prose: whether to include in prose-only training
- include_in_poetry: whether to include in poetry training
- exclude_pages: page ranges to skip (cover, index, appendix, etc.)

Usage:
    from src.data.corpus_metadata import CORPUS_METADATA, get_book_metadata

    meta = get_book_metadata('21-22TheLifeDivine')
    print(meta['content_type'])  # 'essay'
    print(meta['period'])        # 'mature'
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class BookMetadata:
    """Metadata for a single book/volume."""
    filename: str
    title: str
    content_type: str  # essay, letter, poetry, drama, commentary, record, translation
    period: str  # early, middle, mature
    importance: str  # core, supplementary, reference
    include_in_prose: bool
    include_in_poetry: bool
    year_start: int
    year_end: int
    description: str
    exclude_pages: List[tuple] = None  # [(start, end), ...] pages to skip
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            'filename': self.filename,
            'title': self.title,
            'content_type': self.content_type,
            'period': self.period,
            'importance': self.importance,
            'include_in_prose': self.include_in_prose,
            'include_in_poetry': self.include_in_poetry,
            'year_start': self.year_start,
            'year_end': self.year_end,
            'description': self.description,
            'exclude_pages': self.exclude_pages,
            'notes': self.notes,
        }


# Complete Works of Sri Aurobindo (CWSA) metadata
CORPUS_METADATA: Dict[str, BookMetadata] = {

    # =========================================================================
    # CORE PROSE WORKS - Highest Priority
    # =========================================================================

    '21-22TheLifeDivine': BookMetadata(
        filename='21-22TheLifeDivine.txt',
        title='The Life Divine',
        content_type='essay',
        period='mature',
        importance='core',
        include_in_prose=True,
        include_in_poetry=False,
        year_start=1914,
        year_end=1950,
        description='Central philosophical work on Integral Yoga metaphysics',
        exclude_pages=[(1, 10), (-20, -1)],  # Cover, TOC, Index
        notes='Most important work. Revised multiple times. Final version 1939-1940.',
    ),

    '23-24TheSynthesisofYoga': BookMetadata(
        filename='23-24TheSynthesisofYoga.txt',
        title='The Synthesis of Yoga',
        content_type='essay',
        period='mature',
        importance='core',
        include_in_prose=True,
        include_in_poetry=False,
        year_start=1914,
        year_end=1921,
        description='Comprehensive treatise on yoga methodology',
        exclude_pages=[(1, 8), (-15, -1)],
        notes='Practical counterpart to Life Divine.',
    ),

    '28LettersOnYoga-I': BookMetadata(
        filename='28LettersOnYoga-I.txt',
        title='Letters on Yoga - I',
        content_type='letter',
        period='mature',
        importance='core',
        include_in_prose=True,
        include_in_poetry=False,
        year_start=1927,
        year_end=1950,
        description='Letters to disciples on yoga practice',
        exclude_pages=[(1, 6)],
        notes='Direct teaching in Q&A format. Very clear explanations.',
    ),

    '29LettersOnYoga-II': BookMetadata(
        filename='29LettersOnYoga-II.txt',
        title='Letters on Yoga - II',
        content_type='letter',
        period='mature',
        importance='core',
        include_in_prose=True,
        include_in_poetry=False,
        year_start=1927,
        year_end=1950,
        description='Letters on planes and parts of being',
        exclude_pages=[(1, 6)],
    ),

    '30LettersOnYoga-III': BookMetadata(
        filename='30LettersOnYoga-III.txt',
        title='Letters on Yoga - III',
        content_type='letter',
        period='mature',
        importance='core',
        include_in_prose=True,
        include_in_poetry=False,
        year_start=1927,
        year_end=1950,
        description='Letters on experiences and transformation',
        exclude_pages=[(1, 6)],
    ),

    '31LettersOnYoga-IV': BookMetadata(
        filename='31LettersOnYoga-IV.txt',
        title='Letters on Yoga - IV',
        content_type='letter',
        period='mature',
        importance='core',
        include_in_prose=True,
        include_in_poetry=False,
        year_start=1927,
        year_end=1950,
        description='Letters on obstacles and transformation',
        exclude_pages=[(1, 6)],
    ),

    '19EssaysOnTheGita': BookMetadata(
        filename='19EssaysOnTheGita.txt',
        title='Essays on the Gita',
        content_type='commentary',
        period='middle',
        importance='core',
        include_in_prose=True,
        include_in_poetry=False,
        year_start=1916,
        year_end=1920,
        description='Philosophical commentary on the Bhagavad Gita',
        exclude_pages=[(1, 8), (-10, -1)],
        notes='Systematic interpretation. Important bridge work.',
    ),

    # =========================================================================
    # ADDITIONAL CORE PROSE - User requested
    # =========================================================================

    '10-11RecordOfYoga': BookMetadata(
        filename='10-11RecordOfYoga.txt',
        title='Record of Yoga',
        content_type='record',
        period='middle',
        importance='core',
        include_in_prose=True,
        include_in_poetry=False,
        year_start=1909,
        year_end=1927,
        description='Personal diary of yogic experiences and siddhis',
        exclude_pages=[(1, 15), (-20, -1)],
        notes='Intimate spiritual diary. Unique vocabulary for experiences.',
    ),

    '17IshaUpanishad': BookMetadata(
        filename='17IshaUpanishad.txt',
        title='Isha Upanishad',
        content_type='commentary',
        period='middle',
        importance='core',
        include_in_prose=True,
        include_in_poetry=False,
        year_start=1914,
        year_end=1915,
        description='Commentary on the Isha Upanishad',
        exclude_pages=[(1, 6)],
        notes='Deep Upanishadic interpretation.',
    ),

    '15TheSecretOfTheVeda': BookMetadata(
        filename='15TheSecretOfTheVeda.txt',
        title='The Secret of the Veda',
        content_type='commentary',
        period='middle',
        importance='core',
        include_in_prose=True,
        include_in_poetry=False,
        year_start=1914,
        year_end=1916,
        description='Psychological interpretation of Vedic hymns',
        exclude_pages=[(1, 10), (-15, -1)],
        notes='Revolutionary Vedic interpretation. Rich Sanskrit vocabulary.',
    ),

    '18KenaAndOtherUpanishads': BookMetadata(
        filename='18KenaAndOtherUpanishads.txt',
        title='Kena and Other Upanishads',
        content_type='commentary',
        period='middle',
        importance='core',
        include_in_prose=True,
        include_in_poetry=False,
        year_start=1909,
        year_end=1914,
        description='Commentaries on Kena, Mundaka, and other Upanishads',
        exclude_pages=[(1, 6)],
        notes='Essential Upanishadic philosophy.',
    ),

    '12EssaysDivineAndHuman': BookMetadata(
        filename='12EssaysDivineAndHuman.txt',
        title='Essays Divine and Human',
        content_type='essay',
        period='middle',
        importance='core',
        include_in_prose=True,
        include_in_poetry=False,
        year_start=1910,
        year_end=1920,
        description='Philosophical essays on human and divine nature',
        exclude_pages=[(1, 8)],
        notes='Transitional philosophical writings.',
    ),

    # =========================================================================
    # SUPPLEMENTARY PROSE WORKS
    # =========================================================================

    '25TheHumanCycle': BookMetadata(
        filename='25TheHumanCycle.txt',
        title='The Human Cycle',
        content_type='essay',
        period='middle',
        importance='supplementary',
        include_in_prose=True,
        include_in_poetry=False,
        year_start=1916,
        year_end=1918,
        description='Social philosophy and human evolution',
        exclude_pages=[(1, 6), (-10, -1)],
    ),

    '20TheRenaissanceInIndia': BookMetadata(
        filename='20TheRenaissanceInIndia.txt',
        title='The Renaissance in India',
        content_type='essay',
        period='middle',
        importance='supplementary',
        include_in_prose=True,
        include_in_poetry=False,
        year_start=1918,
        year_end=1921,
        description='Essays on Indian culture and spirituality',
        exclude_pages=[(1, 8)],
    ),

    '13EssaysInPhilosophyAndYoga': BookMetadata(
        filename='13EssaysInPhilosophyAndYoga.txt',
        title='Essays in Philosophy and Yoga',
        content_type='essay',
        period='middle',
        importance='supplementary',
        include_in_prose=True,
        include_in_poetry=False,
        year_start=1909,
        year_end=1921,
        description='Miscellaneous philosophical and yogic essays',
        exclude_pages=[(1, 6)],
    ),

    '32TheMotherWithLettersOnTheMother': BookMetadata(
        filename='32TheMotherWithLettersOnTheMother.txt',
        title='The Mother with Letters on The Mother',
        content_type='essay',
        period='mature',
        importance='supplementary',
        include_in_prose=True,
        include_in_poetry=False,
        year_start=1927,
        year_end=1950,
        description='Essays on The Mother and letters about her',
        exclude_pages=[(1, 8)],
    ),

    '35LettersOnHimselfAndTheAshram': BookMetadata(
        filename='35LettersOnHimselfAndTheAshram.txt',
        title='Letters on Himself and the Ashram',
        content_type='letter',
        period='mature',
        importance='supplementary',
        include_in_prose=True,
        include_in_poetry=False,
        year_start=1926,
        year_end=1950,
        description='Autobiographical letters and ashram matters',
        exclude_pages=[(1, 8)],
    ),

    '36AutobiographicalNotes': BookMetadata(
        filename='36AutobiographicalNotes.txt',
        title='Autobiographical Notes',
        content_type='letter',
        period='mature',
        importance='supplementary',
        include_in_prose=True,
        include_in_poetry=False,
        year_start=1926,
        year_end=1950,
        description='Autobiographical fragments and notes',
        exclude_pages=[(1, 6)],
    ),

    # =========================================================================
    # POETRY WORKS
    # =========================================================================

    '33-34Savitri': BookMetadata(
        filename='33-34Savitri.txt',
        title='Savitri',
        content_type='poetry',
        period='mature',
        importance='core',
        include_in_prose=False,
        include_in_poetry=True,
        year_start=1916,
        year_end=1950,
        description='Epic poem - Legend and Symbol',
        exclude_pages=[(1, 15), (-20, -1)],
        notes='Magnum opus. Revised continuously until death.',
    ),

    '02CollectedPoems': BookMetadata(
        filename='02CollectedPoems.txt',
        title='Collected Poems',
        content_type='poetry',
        period='early',  # spans all periods
        importance='supplementary',
        include_in_prose=False,
        include_in_poetry=True,
        year_start=1890,
        year_end=1950,
        description='Shorter poems from all periods',
        exclude_pages=[(1, 10), (-15, -1)],
    ),

    '27LettersOnPoetryAndArt': BookMetadata(
        filename='27LettersOnPoetryAndArt.txt',
        title='Letters on Poetry and Art',
        content_type='letter',
        period='mature',
        importance='supplementary',
        include_in_prose=True,
        include_in_poetry=True,
        year_start=1930,
        year_end=1950,
        description='Letters on poetic and artistic theory',
        exclude_pages=[(1, 8)],
    ),

    '26TheFuturePoetry': BookMetadata(
        filename='26TheFuturePoetry.txt',
        title='The Future Poetry',
        content_type='essay',
        period='middle',
        importance='supplementary',
        include_in_prose=True,
        include_in_poetry=True,
        year_start=1917,
        year_end=1920,
        description='Essays on the nature and future of poetry',
        exclude_pages=[(1, 6)],
    ),

    # =========================================================================
    # REFERENCE WORKS (lower weight)
    # =========================================================================

    '01EarlyCulturalWritings': BookMetadata(
        filename='01EarlyCulturalWritings.txt',
        title='Early Cultural Writings',
        content_type='essay',
        period='early',
        importance='reference',
        include_in_prose=True,
        include_in_poetry=False,
        year_start=1890,
        year_end=1910,
        description='Early essays on culture and literature',
        exclude_pages=[(1, 10)],
        notes='Very early writings. Different style.',
    ),

    '06-07BandeMataram': BookMetadata(
        filename='06-07BandeMataram.txt',
        title='Bande Mataram',
        content_type='essay',
        period='early',
        importance='reference',
        include_in_prose=False,  # Political, not spiritual
        include_in_poetry=False,
        year_start=1906,
        year_end=1908,
        description='Political writings from nationalist period',
        notes='Political phase. Exclude from spiritual training.',
    ),

    '08Karmayogin': BookMetadata(
        filename='08Karmayogin.txt',
        title='Karmayogin',
        content_type='essay',
        period='early',
        importance='reference',
        include_in_prose=False,  # Political
        include_in_poetry=False,
        year_start=1909,
        year_end=1910,
        description='Political journal writings',
        notes='Political phase. Exclude from spiritual training.',
    ),

    '14VedicAndPhilologicalStudies': BookMetadata(
        filename='14VedicAndPhilologicalStudies.txt',
        title='Vedic and Philological Studies',
        content_type='commentary',
        period='middle',
        importance='reference',
        include_in_prose=True,
        include_in_poetry=False,
        year_start=1912,
        year_end=1914,
        description='Scholarly Vedic studies and philology',
        exclude_pages=[(1, 8)],
        notes='Technical linguistic content.',
    ),

    '16HymnsToTheMysticFire': BookMetadata(
        filename='16HymnsToTheMysticFire.txt',
        title='Hymns to the Mystic Fire',
        content_type='translation',
        period='middle',
        importance='reference',
        include_in_prose=True,
        include_in_poetry=True,
        year_start=1914,
        year_end=1946,
        description='Translations of Rig Veda hymns to Agni',
        exclude_pages=[(1, 10)],
    ),

    '03-04CollectedPlaysAndStories': BookMetadata(
        filename='03-04CollectedPlaysAndStories.txt',
        title='Collected Plays and Stories',
        content_type='drama',
        period='early',
        importance='reference',
        include_in_prose=False,  # Drama format
        include_in_poetry=False,
        year_start=1890,
        year_end=1910,
        description='Early dramatic works and short stories',
        notes='Exclude - different format.',
    ),

    '05Translations': BookMetadata(
        filename='05Translations.txt',
        title='Translations',
        content_type='translation',
        period='early',
        importance='reference',
        include_in_prose=False,
        include_in_poetry=False,
        year_start=1893,
        year_end=1912,
        description='Translations from Sanskrit, Bengali, Tamil',
        notes='Translations, not original work.',
    ),

    '09WritingsInBengaliAndSanskrit': BookMetadata(
        filename='09WritingsInBengaliAndSanskrit.txt',
        title='Writings in Bengali and Sanskrit',
        content_type='essay',
        period='early',
        importance='reference',
        include_in_prose=False,  # Non-English
        include_in_poetry=False,
        year_start=1893,
        year_end=1920,
        description='Writings in Bengali and Sanskrit',
        notes='Non-English texts. Exclude.',
    ),
}


def get_book_metadata(book_key: str) -> Optional[BookMetadata]:
    """Get metadata for a specific book."""
    return CORPUS_METADATA.get(book_key)


def get_books_by_filter(
    content_types: List[str] = None,
    periods: List[str] = None,
    importance_levels: List[str] = None,
    include_prose: bool = None,
    include_poetry: bool = None,
) -> List[BookMetadata]:
    """
    Filter books by criteria.

    Args:
        content_types: List of types to include (essay, letter, poetry, etc.)
        periods: List of periods to include (early, middle, mature)
        importance_levels: List of importance levels (core, supplementary, reference)
        include_prose: If True, only books marked for prose training
        include_poetry: If True, only books marked for poetry training

    Returns:
        List of matching BookMetadata objects
    """
    results = []

    for key, meta in CORPUS_METADATA.items():
        # Apply filters
        if content_types and meta.content_type not in content_types:
            continue
        if periods and meta.period not in periods:
            continue
        if importance_levels and meta.importance not in importance_levels:
            continue
        if include_prose is True and not meta.include_in_prose:
            continue
        if include_poetry is True and not meta.include_in_poetry:
            continue

        results.append(meta)

    return results


def get_prose_core_books() -> List[BookMetadata]:
    """Get core prose books for training."""
    return get_books_by_filter(
        importance_levels=['core'],
        include_prose=True,
    )


def get_prose_all_books() -> List[BookMetadata]:
    """Get all prose books (core + supplementary)."""
    return get_books_by_filter(
        importance_levels=['core', 'supplementary'],
        include_prose=True,
    )


def get_poetry_books() -> List[BookMetadata]:
    """Get poetry books for training."""
    return get_books_by_filter(include_poetry=True)


def print_corpus_summary():
    """Print summary of corpus metadata."""
    print("\n" + "="*70)
    print("SASLM Corpus Summary")
    print("="*70)

    # By importance
    for importance in ['core', 'supplementary', 'reference']:
        books = get_books_by_filter(importance_levels=[importance])
        print(f"\n{importance.upper()} ({len(books)} books):")
        for book in books:
            prose = "P" if book.include_in_prose else "-"
            poetry = "V" if book.include_in_poetry else "-"
            print(f"  [{prose}{poetry}] {book.period:6} | {book.content_type:12} | {book.title}")

    # Counts
    print("\n" + "-"*70)
    prose_core = len(get_prose_core_books())
    prose_all = len(get_prose_all_books())
    poetry = len(get_poetry_books())
    print(f"Prose Core: {prose_core} books")
    print(f"Prose All:  {prose_all} books")
    print(f"Poetry:     {poetry} books")


if __name__ == '__main__':
    print_corpus_summary()
