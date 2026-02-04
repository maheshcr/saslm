# SASLM Data Module
# Provides data loading, cleaning, and weighted sampling

from .corpus_metadata import CORPUS_METADATA, get_book_metadata, get_books_by_filter
from .text_cleaner import TextCleaner, clean_text
from .weighted_sampler import WeightedCorpusSampler

# Lazy imports for torch-dependent modules
def _get_data_loader():
    from .data_loader import SASLMDataset, create_dataloaders
    return SASLMDataset, create_dataloaders

__all__ = [
    'CORPUS_METADATA',
    'get_book_metadata',
    'get_books_by_filter',
    'TextCleaner',
    'clean_text',
    'WeightedCorpusSampler',
]

# Only add data_loader exports if torch is available
try:
    from .data_loader import SASLMDataset, create_dataloaders
    __all__.extend(['SASLMDataset', 'create_dataloaders'])
except ImportError:
    pass  # torch not installed, skip data_loader
