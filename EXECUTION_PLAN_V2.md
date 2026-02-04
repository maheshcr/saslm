# SASLM v2: Execution Plan & Technical Analysis

**Project**: Sri Aurobindo Small Language Model
**Date**: February 4, 2026
**Status**: Fresh Start with Lessons Learned

---

## 1. Problem Statement

Build a specialized 50-60M parameter Small Language Model trained on Sri Aurobindo's Complete Works (CWSA) that captures his unique philosophical ontology, vocabulary, and writing style.

**Inspiration**: TinyStories paper - demonstrating that small, focused models can achieve coherent generation within a constrained domain.

---

## 2. What Went Wrong: Post-Mortem Analysis

### 2.1 The Data-to-Vocabulary Ratio Problem (Critical)

| Metric | TinyStories | SASLM v1 | Ratio |
|--------|-------------|----------|-------|
| Training Data | ~1.2 GB | ~12 MB | **100x less** |
| Vocabulary Complexity | 4-6 year old level | PhD-level + Sanskrit | **10x+ harder** |
| Estimated Tokens | ~200M+ | ~3M | **66x less** |
| Vocab Size | ~10K effective | 32K | **3x more** |
| Avg Token Frequency | ~20,000 | ~94 | **200x less** |

**The Math**: With 32K vocabulary and ~3M tokens, each token appears on average only **94 times**. For a neural network to learn meaningful associations, tokens need to appear **thousands** of times. TinyStories succeeds because each token in their limited vocabulary appears ~20,000 times on average.

**Diagnosis**: The model couldn't learn patterns because it never saw the same tokens enough times to form associations.

### 2.2 Data Quality Issues

| Issue | Impact | Status in v1 |
|-------|--------|--------------|
| Page numbers in text | Noise tokens, wasted capacity | Partially cleaned |
| Headers/footers | Repetitive artifacts | Cropped via OCR |
| Index pages | List-like noise | Some excluded |
| Cover pages | Title noise | Some excluded |
| Appendices | Reference material, not prose | Not excluded |
| Table of Contents | Structural noise | Not excluded |

**Contrast with TinyStories**: Their synthetic dataset has ZERO artifacts. Every token is meaningful story content.

### 2.3 Content Heterogeneity Problem

Sri Aurobindo's works span radically different forms:

| Content Type | Characteristics | % of Corpus (est.) |
|--------------|-----------------|-------------------|
| Essays/Prose | Structured arguments, philosophical | 40% |
| Poetry | Meter, imagery, compressed meaning | 25% |
| Drama | Dialogue, stage directions | 10% |
| Letters | Conversational, Q&A format | 15% |
| Translations/Commentaries | Mixed Sanskrit, interpretive | 10% |

**Problem**: Training on all simultaneously confuses the model about what "good output" looks like.

### 2.4 Temporal Evolution Problem

Sri Aurobindo's thought evolved dramatically:

| Period | Works | Spiritual Development |
|--------|-------|----------------------|
| 1893-1910 (Age 21-38) | Early poetry, political writings | Pre-yogic, nationalist phase |
| 1910-1914 (Age 38-42) | Pondicherry arrival, early yoga | Transitional, experimental |
| 1914-1926 (Age 42-54) | Arya journal, Life Divine draft | Major realizations, systematic philosophy |
| 1926-1950 (Age 54-78) | Letters, Savitri, revised Life Divine | Mature integral yoga |

**Problem**: Giving equal weight to teenage essays and mature philosophical works dilutes the "authentic voice" we want to capture.

### 2.5 Adding More Data Didn't Help Much

You tried adding:
- Complete Works of the Mother
- Complete Works of Swami Vivekananda

**Why it only moderately helped**:
- Added more vocabulary diversity (different authors have different term usage)
- Didn't fix the fundamental data/vocab ratio
- May have diluted Sri Aurobindo's unique voice

---

## 3. The Path Forward: SASLM v2 Architecture

### 3.1 Strategy Options

| Approach | Pros | Cons | Feasibility |
|----------|------|------|-------------|
| **A. Smaller Model** | Matches data size | Limited capacity | Medium |
| **B. Smaller Vocab** | Higher token frequency | Longer sequences | High |
| **C. Transfer Learning** | Leverages pretrained knowledge | May override unique style | High |
| **D. Data Augmentation** | More training signal | May dilute authenticity | Low |
| **E. Curriculum Learning** | Focused training | Complex implementation | Medium |

**Recommended**: Combine B + C + E

### 3.2 Proposed Architecture

```
SASLM v2 Configuration
======================

Base: GPT-2 Small (124M params) OR custom ~30M model
Approach: Fine-tune pretrained model on curated corpus

Tokenizer:
- Vocab Size: 16,384 (down from 32K)
- Special handling for Sanskrit diacritics
- Byte-fallback for rare terms

Model (if training from scratch):
- n_layers: 6 (down from 12)
- n_heads: 6 (down from 12)
- n_embd: 384 (down from 768)
- block_size: 512
- Parameters: ~25M

OR

Model (if fine-tuning):
- Base: GPT-2 Small (124M) or DistilGPT-2 (82M)
- Fine-tune all layers on curated corpus
- Learning rate: 1e-5 to 5e-5
```

### 3.3 Data Curation Strategy

**Phase 1: Rigorous Cleaning**
```
For each PDF:
├── Exclude: Cover, TOC, Index, Appendix, Bibliography
├── Exclude: Publisher notes, Editor notes
├── Remove: Page numbers, running headers
├── Flag: Footnotes (separate or exclude)
└── Verify: Sanskrit diacritics preserved
```

**Phase 2: Content Classification**
```
Tag each text chunk with:
├── source_file: Original PDF name
├── content_type: essay | poetry | letter | drama | commentary
├── period: early (1893-1910) | middle (1910-1926) | mature (1926-1950)
├── importance: core | supplementary | reference
└── quality_score: 1-5 (manual or heuristic)
```

**Phase 3: Weighted Sampling**
```python
sampling_weights = {
    'period': {
        'mature': 3.0,    # 3x more likely to sample
        'middle': 2.0,
        'early': 0.5
    },
    'content_type': {
        'essay': 2.0,     # Prioritize prose
        'letter': 1.5,
        'poetry': 0.5,    # Reduce initially
        'drama': 0.3
    },
    'importance': {
        'core': 3.0,      # Life Divine, Synthesis of Yoga, Savitri
        'supplementary': 1.0,
        'reference': 0.3
    }
}
```

### 3.4 Core Works to Prioritize

| Title | Volumes | Period | Why Prioritize |
|-------|---------|--------|----------------|
| The Life Divine | 21-22 | Mature | Central philosophical work |
| The Synthesis of Yoga | 23-24 | Mature | Practical yoga methodology |
| Letters on Yoga | 28-31 | Mature | Direct teaching, Q&A clarity |
| Savitri | 33-34 | Mature | Poetic masterwork (train separately?) |
| Essays on the Gita | 19 | Middle | Systematic commentary |
| The Human Cycle | 25 | Middle | Social philosophy |

---

## 4. Implementation Plan: Week-by-Week

### Week 1: Foundation & Data

#### Day 1-2: Audit & Clean Data
- [ ] Create `data_audit.py` script to analyze current processed_text/
- [ ] Identify remaining artifacts (page numbers, headers, etc.)
- [ ] Create exclusion lists for each PDF (pages to skip)
- [ ] Document data quality issues found

#### Day 3-4: Re-extract Priority Texts
- [ ] Start with Life Divine (volumes 21-22) as clean reference
- [ ] Manual verification of 10 random pages per volume
- [ ] Create `clean_corpus/` directory with verified texts
- [ ] Log all cleaning decisions

#### Day 5: Content Classification
- [ ] Create `corpus_metadata.json` with tags for each file
- [ ] Classify by: period, content_type, importance
- [ ] Calculate corpus statistics (tokens per category)

#### Day 6-7: Tokenizer v2
- [ ] Train new 16K vocab tokenizer on clean corpus
- [ ] Verify Sanskrit diacritic handling
- [ ] Compare token frequencies vs v1 tokenizer
- [ ] Document tokenizer statistics

**Deliverables Week 1**:
- [ ] `data_audit_report.md`
- [ ] `clean_corpus/` directory
- [ ] `corpus_metadata.json`
- [ ] `tokenizer_v2/` with stats

---

### Week 2: Training Infrastructure

#### Day 1-2: Experiment Tracking Setup
- [ ] Set up Weights & Biases (wandb) project
- [ ] Create training config schema (YAML)
- [ ] Implement checkpoint management with metadata
- [ ] Create evaluation dataset (held-out test set)

#### Day 3-4: Training Script v2
- [ ] Refactor `train_saslm.py` with:
  - Config file support
  - wandb integration
  - Weighted sampling dataloader
  - Validation loss tracking
  - Perplexity calculation
- [ ] Add gradient norm logging
- [ ] Add sample generation during training

#### Day 5-6: Baseline Experiments
- [ ] Train tiny model (5M params) for 1 hour as sanity check
- [ ] Verify loss decreases, samples improve
- [ ] Debug any data loading issues
- [ ] Document baseline results

#### Day 7: Fine-tuning Setup (Alternative Path)
- [ ] Download GPT-2 Small or DistilGPT-2
- [ ] Adapt tokenizer (extend with Sanskrit terms)
- [ ] Create fine-tuning script
- [ ] Test loading and inference

**Deliverables Week 2**:
- [ ] `configs/` directory with experiment configs
- [ ] `train_v2.py` with full logging
- [ ] wandb project with baseline runs
- [ ] `finetune.py` script (if using transfer learning)

---

### Week 3: Training & Iteration

#### Day 1-3: Full Training Run
- [ ] Launch full training on Colab Pro (GPU)
- [ ] Monitor via wandb dashboard
- [ ] Generate samples at checkpoints
- [ ] Log training time, GPU utilization

#### Day 4-5: Evaluation
- [ ] Run perplexity on held-out test set
- [ ] Run LLM-judge evaluation (need API key)
- [ ] Create human evaluation rubric
- [ ] Manually evaluate 20 samples

#### Day 6-7: Iteration
- [ ] Analyze what's working/not working
- [ ] Adjust hyperparameters if needed
- [ ] Second training run with improvements
- [ ] Document learnings

**Deliverables Week 3**:
- [ ] Trained model checkpoint with full metadata
- [ ] Training curves (wandb export)
- [ ] Evaluation results CSV
- [ ] `iteration_notes.md`

---

### Week 4: Polish & Publish

#### Day 1-2: Model Selection
- [ ] Compare all trained models
- [ ] Select best checkpoint
- [ ] Final evaluation pass

#### Day 3-4: HuggingFace Upload
- [ ] Create model card (README.md)
- [ ] Upload model and tokenizer
- [ ] Test download and inference
- [ ] Add usage examples

#### Day 5-6: Documentation
- [ ] Write methodology section (for paper)
- [ ] Create reproducibility guide
- [ ] Document all hyperparameters
- [ ] Archive all experiment configs

#### Day 7: Demo & Share
- [ ] Create simple demo script
- [ ] Optional: Gradio/Streamlit demo
- [ ] Share on HuggingFace

**Deliverables Week 4**:
- [ ] HuggingFace model page (public)
- [ ] `METHODOLOGY.md`
- [ ] `demo.py` or web demo
- [ ] All experiment artifacts archived

---

## 5. Logging & Audit Framework

### 5.1 What to Log

```yaml
# experiment_config.yaml template
experiment:
  name: "saslm_v2_exp001"
  date: "2026-02-XX"
  hypothesis: "Smaller vocab + weighted sampling improves coherence"

data:
  corpus_version: "clean_v1"
  total_tokens: 0  # Fill after processing
  vocab_size: 16384
  train_split: 0.95
  val_split: 0.05
  sampling_weights: {...}

model:
  architecture: "gpt2-custom"  # or "gpt2-small-finetune"
  n_layers: 6
  n_heads: 6
  n_embd: 384
  block_size: 512
  dropout: 0.1
  total_params: 0  # Calculate

training:
  batch_size: 32
  gradient_accumulation: 4
  learning_rate: 3e-4
  lr_scheduler: "cosine"
  warmup_steps: 100
  max_steps: 10000
  eval_interval: 500
  save_interval: 1000

hardware:
  device: "cuda"  # or "mps"
  gpu_model: "T4"  # or "A100", "V100"
  precision: "fp16"
```

### 5.2 Metrics to Track

| Metric | Frequency | Purpose |
|--------|-----------|---------|
| Training Loss | Every step | Convergence |
| Validation Loss | Every eval_interval | Overfitting detection |
| Perplexity | Every eval_interval | Language modeling quality |
| Gradient Norm | Every step | Training stability |
| Learning Rate | Every step | Scheduler verification |
| Sample Outputs | Every eval_interval | Qualitative assessment |
| Token/sec | Every step | Performance |
| GPU Memory | Every step | Resource usage |

### 5.3 Artifact Storage

```
experiments/
├── exp001_baseline/
│   ├── config.yaml
│   ├── training_log.csv
│   ├── checkpoints/
│   │   ├── step_1000.pt
│   │   ├── step_2000.pt
│   │   └── best.pt
│   ├── samples/
│   │   ├── step_1000_samples.txt
│   │   └── step_2000_samples.txt
│   ├── eval_results.json
│   └── notes.md
├── exp002_smaller_vocab/
│   └── ...
└── comparison_report.md
```

---

## 6. Success Criteria

### 6.1 Quantitative

| Metric | Target | Stretch |
|--------|--------|---------|
| Validation Perplexity | < 50 | < 30 |
| LLM Judge (Ontology) | > 6/10 | > 8/10 |
| LLM Judge (Style) | > 5/10 | > 7/10 |
| LLM Judge (Coherence) | > 6/10 | > 8/10 |
| Training Stability | No loss spikes | Smooth curve |

### 6.2 Qualitative

- [ ] Model can complete philosophical sentences coherently
- [ ] Model uses Sanskrit terms appropriately
- [ ] Model captures Sri Aurobindo's explanatory style
- [ ] Model doesn't generate obvious artifacts (page numbers, etc.)
- [ ] Generated text "feels" like Sri Aurobindo to someone familiar with his work

---

## 7. Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Still not enough data | High | Use transfer learning as fallback |
| Overfitting | Medium | Early stopping, dropout, regularization |
| Loss of Sanskrit terms | Medium | Verify tokenizer, byte-fallback |
| Colab disconnects | High | Frequent checkpoints, resume logic |
| API costs for eval | Low | Use free tiers, batch requests |

---

## 8. Resources

- **Compute**: Google Colab Pro (GPU hours)
- **Storage**: HuggingFace Pro (model hosting)
- **Tracking**: Weights & Biases (free tier)
- **Evaluation**: Anthropic/OpenAI API (pay per use)

---

## 9. Open Questions

1. **Fine-tune vs from-scratch?**
   - Fine-tuning GPT-2 might give coherence faster
   - From-scratch is more "pure" for the paper
   - **Decision needed**

2. **Include poetry or not?**
   - Poetry has different structure, may confuse model
   - Could train separate poetry model
   - **Decision needed**

3. **How much data augmentation is acceptable?**
   - Back-translation?
   - Paraphrasing?
   - **Decision: Probably none, authenticity matters**

4. **Evaluation methodology for paper**
   - Need human evaluators familiar with Sri Aurobindo?
   - How many samples?
   - **To be designed**

---

## Appendix A: Key Files Reference

| File | Purpose | Status |
|------|---------|--------|
| `src/train_saslm.py` | Main training script | Needs refactor |
| `src/train_tokenizer.py` | Tokenizer training | Works, needs config |
| `src/inference.py` | Text generation | Has vocab mismatch bug |
| `src/evaluate.py` | LLM judge evaluation | Framework ready |
| `src/pipeline.py` | PDF extraction | Works |
| `book_config.json` | Per-book extraction settings | Complete |
| `processed_text/` | Extracted text files | Needs re-audit |

---

## Appendix B: TinyStories Reference

From the paper "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?"

Key insights:
- 33M parameter model achieved coherent story generation
- Dataset: 1.2GB of synthetic stories
- Vocabulary: Limited to 4-6 year old level
- **Critical factor**: High token frequency due to limited vocab + large dataset

What they did right:
- Matched model capacity to data size
- Ensured clean, consistent data
- Simple, repetitive patterns for model to learn

---

*Document Version: 2.0*
*Last Updated: 2026-02-04*
