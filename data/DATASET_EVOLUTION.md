# SASLM Dataset Evolution & Statistics

**Document Version**: 1.0
**Last Updated**: 2026-02-04

This document tracks the evolution of the SASLM training dataset, documenting improvements and their rationale.

---

## Executive Summary

| Metric | v1 (Original) | v2 (Current) | Improvement |
|--------|---------------|--------------|-------------|
| Text Size | ~12 MB | ~30 MB | **2.5x larger** |
| Estimated Tokens | ~3M | ~7.5M | **2.5x more** |
| Vocabulary Size | 32K | 16K (target) | **2x smaller** |
| Avg Token Frequency | ~94 | ~468 | **5x higher** |
| Core Books | 6 | 12 | **2x more** |
| Sanskrit Diacritics | Unknown | 7,091 verified | Preserved |

---

## The Problem (v1)

### TinyStories vs SASLM Comparison

The TinyStories paper achieved coherent text generation with a small model because of favorable data characteristics:

| Metric | TinyStories | SASLM v1 | Ratio |
|--------|-------------|----------|-------|
| Training Data | ~1.2 GB | ~12 MB | **100x less** |
| Vocabulary Complexity | 4-6 year old | PhD + Sanskrit | **10x harder** |
| Estimated Tokens | ~200M+ | ~3M | **66x less** |
| Vocab Size | ~10K effective | 32K | **3x more** |
| Avg Token Frequency | ~20,000 | ~94 | **200x less** |

### Why v1 Failed

**The Math**: With 32K vocabulary and ~3M tokens, each token appeared on average only **94 times**. Neural networks need **thousands** of repetitions to learn patterns. The model literally couldn't learn because it never saw the same tokens enough times.

### Additional v1 Issues

1. **Artifact contamination**: Page numbers, headers, footers in training data
2. **No content weighting**: Early teenage writings weighted same as mature philosophy
3. **No period awareness**: All periods treated equally
4. **Missing core texts**: Several important works not included

---

## The Solution (v2)

### Strategy

1. **Expand core corpus**: Add more of Sri Aurobindo's essential works
2. **Reduce vocabulary**: 16K instead of 32K (doubles token frequency)
3. **Clean rigorously**: Remove all OCR artifacts
4. **Weight by importance**: Core works sampled 3x more
5. **Weight by period**: Mature works sampled 3x more than early works

### Books Added to Core Dataset

| Book | Volume | Period | Why Important |
|------|--------|--------|---------------|
| Record of Yoga | 10-11 | Middle | Personal yogic diary, unique vocabulary |
| Isha Upanishad | 17 | Middle | Deep Upanishadic philosophy |
| The Secret of the Veda | 15 | Middle | Vedic interpretation, rich Sanskrit |
| Kena and Other Upanishads | 18 | Middle | Essential Upanishadic philosophy |
| Essays Divine and Human | 12 | Middle | Transitional philosophical essays |

### Complete Core Prose Dataset (v2)

| # | Title | Chars | Period | Importance |
|---|-------|-------|--------|------------|
| 1 | The Life Divine | 2,491,361 | Mature | Core |
| 2 | Record of Yoga | 2,624,618 | Middle | Core |
| 3 | The Synthesis of Yoga | 2,001,282 | Mature | Core |
| 4 | The Human Cycle | 1,474,393 | Middle | Supplementary |
| 5 | Letters on Himself and Ashram | 1,482,867 | Mature | Supplementary |
| 6 | Letters on Yoga - IV | 1,438,890 | Mature | Core |
| 7 | Early Cultural Writings | 1,433,917 | Early | Reference |
| 8 | Letters on Poetry and Art | 1,337,149 | Mature | Supplementary |
| 9 | Essays on the Gita | 1,297,264 | Middle | Core |
| 10 | Isha Upanishad | 1,252,817 | Middle | Core |
| 11 | Vedic and Philological Studies | 1,231,001 | Middle | Reference |
| 12 | Autobiographical Notes | 1,157,535 | Mature | Supplementary |
| 13 | The Mother with Letters | 1,156,974 | Mature | Supplementary |
| 14 | Essays in Philosophy and Yoga | 1,125,900 | Middle | Supplementary |
| 15 | Hymns to the Mystic Fire | 1,123,610 | Middle | Reference |
| 16 | The Secret of the Veda | 1,095,299 | Middle | Core |
| 17 | Letters on Yoga - I | 1,043,215 | Mature | Core |
| 18 | The Renaissance in India | 996,302 | Middle | Supplementary |
| 19 | Letters on Yoga - II | 911,367 | Mature | Core |
| 20 | Essays Divine and Human | 884,104 | Middle | Core |
| 21 | Letters on Yoga - III | 811,080 | Mature | Core |
| 22 | The Future Poetry | 806,377 | Middle | Supplementary |
| 23 | Kena and Other Upanishads | 738,916 | Middle | Core |

**Total: 29,916,238 characters (~30 MB)**

---

## Corpus Statistics (v2)

### Prose-Only Corpus

```
Location: data/clean_prose/
Books: 23
Total Characters: 29,916,238
Estimated Tokens: 7,479,059
Sanskrit Diacritics: 7,091
```

#### Distribution by Period
| Period | Characters | Percentage | Weight |
|--------|------------|------------|--------|
| Mature (1926-1950) | 13,831,720 | 46.2% | 3.0x |
| Middle (1910-1926) | 14,650,601 | 49.0% | 2.0x |
| Early (1893-1910) | 1,433,917 | 4.8% | 0.5x |

#### Distribution by Importance
| Importance | Characters | Percentage | Weight |
|------------|------------|------------|--------|
| Core | 16,590,213 | 55.5% | 3.0x |
| Supplementary | 9,537,497 | 31.9% | 1.0x |
| Reference | 3,788,528 | 12.7% | 0.3x |

#### Effective Sampling Distribution

After applying weights, the model will see:

| Category | Raw % | Weighted % | Effective Ratio |
|----------|-------|------------|-----------------|
| Core Mature | ~25% | ~45% | 1.8x over-sampled |
| Core Middle | ~30% | ~36% | 1.2x over-sampled |
| Supplementary | ~32% | ~16% | 0.5x under-sampled |
| Early/Reference | ~13% | ~3% | 0.2x under-sampled |

### Prose + Poetry Corpus

```
Location: data/clean_all/
Books: 25 (adds Savitri + Collected Poems)
Total Characters: 32,007,761
Estimated Tokens: 8,001,940
Sanskrit Diacritics: 7,113
```

---

## Sanskrit Diacritics Preserved

The cleaning process verified preservation of Sanskrit diacritics:

| Character | Name | Count |
|-----------|------|-------|
| ā | a-macron | 4,509 |
| ś | s-acute | 842 |
| ī | i-macron | 697 |
| ū | u-macron | 419 |
| ñ | n-tilde | 322 |
| Ñ | N-tilde (caps) | 160 |
| ē | e-macron | 62 |
| ṣ | s-underdot | 32 |
| ō | o-macron | 27 |
| Others | Various | 21 |

**Total: 7,091 diacritical characters**

These are critical for proper Sanskrit terminology (e.g., "ātman", "Brahman", "sādhanā", "Śakti").

---

## Token Frequency Analysis

### The Key Metric

For a language model to learn patterns, it needs to see each token many times.

| Scenario | Tokens | Vocab | Avg Frequency | Learning Quality |
|----------|--------|-------|---------------|------------------|
| TinyStories | 200M | 10K | 20,000 | Excellent |
| SASLM v1 | 3M | 32K | 94 | Poor |
| SASLM v2 | 7.5M | 16K | 468 | Fair |
| SASLM v2 (after weighting) | 7.5M | 16K | ~600* | Improved |

*Effective frequency increases because common philosophical terms appear more often in core works.

### Improvement Factor

```
v1 frequency: 3M / 32K = 94
v2 frequency: 7.5M / 16K = 468

Improvement: 468 / 94 = 5x
```

This 5x improvement in token frequency should significantly help the model learn patterns.

---

## What's Still Different from TinyStories

Even with improvements, SASLM faces challenges TinyStories didn't:

| Challenge | TinyStories | SASLM v2 |
|-----------|-------------|----------|
| Vocabulary complexity | Simple | Complex philosophical terms |
| Sentence structure | Short, simple | Long, nested, complex |
| Concept density | Low | High |
| Domain specificity | General stories | Specialized philosophy |
| Language mixing | English only | English + Sanskrit |

### Mitigation Strategies

1. **Transfer Learning (Option B)**: Fine-tune GPT-2 which already knows English
2. **Smaller Model (Option A)**: Use ~25M params to match data size
3. **Extended Training**: Train longer to allow grokking
4. **Domain Evaluation**: Custom eval prompts for philosophical coherence

---

## Files Reference

### Corpus Files
```
data/
├── clean_prose/              # Prose-only corpus
│   ├── 21-22TheLifeDivine.txt
│   ├── 23-24TheSynthesisofYoga.txt
│   ├── ... (23 files)
│   └── corpus_stats.json
├── clean_all/                # Prose + Poetry corpus
│   ├── ... (25 files)
│   └── corpus_stats.json
└── DATASET_EVOLUTION.md      # This document
```

### Metadata Files
```
src/data/corpus_metadata.py   # Book metadata definitions
configs/exp_*.yaml            # Experiment configurations
```

---

## Changelog

### v2.0 (2026-02-04)
- Added 5 core books: Record of Yoga, Isha Upanishad, Secret of Veda, Kena Upanishad, Essays Divine and Human
- Built clean_prose corpus (23 books, 30MB, 7.5M tokens)
- Built clean_all corpus (25 books, 32MB, 8M tokens)
- Implemented period and importance weighting
- Verified Sanskrit diacritic preservation (7,091 characters)
- Reduced target vocabulary from 32K to 16K
- Documented token frequency improvement (5x)

### v1.0 (Original)
- Initial extraction of ~12MB text
- 32K vocabulary
- No weighting or curation
- Limited core texts

---

## References

1. TinyStories Paper: "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?"
2. Sri Aurobindo Ashram: Complete Works of Sri Aurobindo (CWSA)
3. Project Documentation: `EXECUTION_PLAN_V2.md`, `EXPERIMENT_FRAMEWORK.md`
