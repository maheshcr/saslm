# SASLM v2: Experimental Framework

**Objective**: Systematically compare training approaches and content types while observing emergent phenomena (including grokking).

---

## 1. Experimental Matrix

```
                    ┌─────────────────┬─────────────────┐
                    │   Prose Only    │ Prose + Poetry  │
┌───────────────────┼─────────────────┼─────────────────┤
│ Option A          │                 │                 │
│ (From Scratch)    │    EXP-A1       │    EXP-A2       │
│ ~25M params       │                 │                 │
├───────────────────┼─────────────────┼─────────────────┤
│ Option B          │                 │                 │
│ (Fine-tune GPT2)  │    EXP-B1       │    EXP-B2       │
│ ~82M params       │                 │                 │
└───────────────────┴─────────────────┴─────────────────┘
```

### Execution Order
1. **EXP-A1**: From scratch, prose only (baseline)
2. **EXP-B1**: Fine-tune, prose only (compare approach)
3. **EXP-A2**: From scratch, prose + poetry (compare content)
4. **EXP-B2**: Fine-tune, prose + poetry (full comparison)

---

## 2. Grokking Observation Protocol

### What is Grokking?
A phenomenon where a model suddenly generalizes long after achieving low training loss. The model first memorizes, plateaus, then "groks" the underlying pattern.

```
Loss
│
│ ████
│     ████  Training Loss       ──── Training
│         ████████████████████  ---- Validation
│
│ ----
│     ----
│         ----
│             ----    ┌─── GROK!
│                 ----▼----
│                          ----
└─────────────────────────────── Steps
      ▲              ▲
      │              │
  Memorization   Generalization
```

### Why SASLM is a Good Candidate for Grokking
- Small dataset (limited memorization ceiling)
- Complex patterns (rich philosophical structure to "grok")
- Training beyond convergence is feasible

### Grokking Detection Setup

```python
# Key hyperparameters for grokking
grokking_config = {
    'weight_decay': 0.1,          # Higher than usual - encourages grokking
    'train_steps': 100_000,       # Train 10x longer than convergence
    'eval_interval': 100,         # Frequent eval to catch the moment
    'patience': None,             # NO early stopping
    'lr_schedule': 'constant',    # Or very slow decay
}

# Metrics to track
grokking_metrics = {
    'train_loss': [],             # Should go low quickly
    'val_loss': [],               # Watch for sudden drop
    'train_accuracy': [],         # Optional: next-token accuracy
    'val_accuracy': [],           # Key grokking indicator
    'weight_norm': [],            # Often decreases before grokking
    'gradient_norm': [],          # May spike at grokking
}
```

### Grokking Detection Algorithm

```python
def detect_grokking(val_losses, window=500, threshold=0.15):
    """
    Detect grokking: sudden drop in val_loss after plateau.

    Returns step where grokking occurred, or None.
    """
    if len(val_losses) < window * 2:
        return None

    for i in range(window, len(val_losses) - window):
        before = np.mean(val_losses[i-window:i])
        after = np.mean(val_losses[i:i+window])

        # Plateau before, drop after
        variance_before = np.var(val_losses[i-window:i])
        relative_drop = (before - after) / before

        if variance_before < 0.01 and relative_drop > threshold:
            return i

    return None
```

---

## 3. Checkpointing Strategy for Colab

### The Problem
- Colab Pro: ~24 hour sessions, random disconnects
- Colab Pro+: Longer, but still not infinite
- Training for grokking may require 50-100+ hours

### The Solution: Robust Resume System

```python
# checkpoint_manager.py

import os
import torch
from pathlib import Path
from google.colab import drive
import json
from datetime import datetime

class CheckpointManager:
    def __init__(self, experiment_name, drive_path='/content/drive/MyDrive/saslm'):
        self.experiment_name = experiment_name
        self.drive_path = Path(drive_path) / experiment_name
        self.drive_path.mkdir(parents=True, exist_ok=True)

        self.checkpoint_file = self.drive_path / 'checkpoint.pt'
        self.config_file = self.drive_path / 'config.json'
        self.metrics_file = self.drive_path / 'metrics.jsonl'
        self.best_file = self.drive_path / 'best_model.pt'

    def save(self, model, optimizer, scheduler, step, metrics, config=None):
        """Atomic save to Google Drive."""
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }

        # Atomic save: write to temp, then rename
        temp_file = self.checkpoint_file.with_suffix('.tmp')
        torch.save(checkpoint, temp_file)
        temp_file.rename(self.checkpoint_file)

        # Save config once
        if config and not self.config_file.exists():
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)

        # Append metrics
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps({'step': step, **metrics}) + '\n')

        print(f"✓ Checkpoint saved at step {step}")

    def save_best(self, model, step, val_loss):
        """Save best model separately."""
        torch.save({
            'step': step,
            'val_loss': val_loss,
            'model_state_dict': model.state_dict(),
        }, self.best_file)
        print(f"✓ New best model at step {step} (val_loss: {val_loss:.4f})")

    def load(self, model, optimizer, scheduler=None):
        """Load checkpoint if exists, return starting step."""
        if not self.checkpoint_file.exists():
            print("No checkpoint found, starting fresh")
            return 0, {}

        checkpoint = torch.load(self.checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        step = checkpoint['step']
        metrics = checkpoint.get('metrics', {})

        print(f"✓ Resumed from step {step}")
        return step, metrics

    def load_metrics_history(self):
        """Load all historical metrics for plotting."""
        if not self.metrics_file.exists():
            return []

        metrics = []
        with open(self.metrics_file, 'r') as f:
            for line in f:
                metrics.append(json.loads(line))
        return metrics
```

### Training Loop with Resume

```python
def train_with_resume(config):
    # Mount Google Drive
    drive.mount('/content/drive')

    # Initialize
    checkpoint_mgr = CheckpointManager(config['experiment_name'])
    model = create_model(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']  # Important for grokking!
    )
    scheduler = get_scheduler(optimizer, config)

    # Resume if checkpoint exists
    start_step, last_metrics = checkpoint_mgr.load(model, optimizer, scheduler)
    best_val_loss = last_metrics.get('best_val_loss', float('inf'))

    # Load metrics history for grokking detection
    metrics_history = checkpoint_mgr.load_metrics_history()
    val_losses = [m['val_loss'] for m in metrics_history if 'val_loss' in m]

    # Training loop
    model.train()
    for step in range(start_step, config['max_steps']):
        # ... training step ...

        # Evaluation
        if step % config['eval_interval'] == 0:
            val_loss = evaluate(model, val_loader)
            train_loss = current_loss

            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': scheduler.get_last_lr()[0],
                'weight_norm': get_weight_norm(model),
            }

            # Check for grokking
            val_losses.append(val_loss)
            grok_step = detect_grokking(val_losses)
            if grok_step:
                print(f"🎉 GROKKING DETECTED at step {grok_step}!")
                metrics['grokking_detected'] = True

            # Save checkpoint
            checkpoint_mgr.save(model, optimizer, scheduler, step, metrics, config)

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_mgr.save_best(model, step, val_loss)

        # More frequent lightweight saves (every 1000 steps)
        if step % 1000 == 0 and step % config['eval_interval'] != 0:
            checkpoint_mgr.save(model, optimizer, scheduler, step,
                              {'train_loss': current_loss}, config)
```

### Colab Session Management

```python
# At the start of each Colab session:

# 1. Mount drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Check for existing experiment
experiment_name = "EXP-A1-prose-only"
checkpoint_path = f'/content/drive/MyDrive/saslm/{experiment_name}/checkpoint.pt'

import os
if os.path.exists(checkpoint_path):
    print(f"Found existing experiment: {experiment_name}")
    print("Will resume from last checkpoint")
else:
    print(f"Starting new experiment: {experiment_name}")

# 3. Run training (will auto-resume)
train_with_resume(config)
```

---

## 4. Experiment Configurations

### EXP-A1: From Scratch, Prose Only

```yaml
experiment:
  name: "EXP-A1-prose-only"
  approach: "from_scratch"
  content: "prose_only"

data:
  include_types: ["essay", "letter", "commentary"]
  exclude_types: ["poetry", "drama"]
  period_weights:
    mature: 3.0
    middle: 2.0
    early: 0.5
  priority_works:
    - "21-22TheLifeDivine"
    - "23-24TheSynthesisOfYoga"
    - "28-31LettersOnYoga"
    - "19EssaysOnTheGita"

tokenizer:
  vocab_size: 16384
  min_frequency: 2

model:
  architecture: "gpt2-custom"
  n_layers: 6
  n_heads: 6
  n_embd: 384
  block_size: 512
  dropout: 0.1
  # Estimated params: ~25M

training:
  batch_size: 32
  gradient_accumulation: 4
  effective_batch_size: 128
  learning_rate: 3e-4
  weight_decay: 0.1           # Higher for grokking
  lr_schedule: "cosine_with_restarts"
  warmup_steps: 500
  max_steps: 100000           # Long for grokking observation
  eval_interval: 200
  checkpoint_interval: 1000

grokking:
  enabled: true
  detection_window: 500
  detection_threshold: 0.15
```

### EXP-B1: Fine-tune GPT-2, Prose Only

```yaml
experiment:
  name: "EXP-B1-prose-only-finetune"
  approach: "finetune"
  content: "prose_only"
  base_model: "distilgpt2"    # 82M params, faster than gpt2-small

data:
  # Same as EXP-A1
  include_types: ["essay", "letter", "commentary"]
  exclude_types: ["poetry", "drama"]
  period_weights:
    mature: 3.0
    middle: 2.0
    early: 0.5

tokenizer:
  strategy: "extend_base"      # Add Sanskrit tokens to GPT-2 tokenizer
  additional_tokens: 1000      # Sanskrit/philosophical terms

model:
  base: "distilgpt2"
  freeze_layers: 0            # Fine-tune all layers

training:
  batch_size: 16              # Smaller due to larger model
  gradient_accumulation: 8
  learning_rate: 5e-5         # Lower for fine-tuning
  weight_decay: 0.01
  max_steps: 50000            # May converge faster
  eval_interval: 200
```

### EXP-A2: From Scratch, Prose + Poetry

```yaml
experiment:
  name: "EXP-A2-prose-poetry"
  approach: "from_scratch"
  content: "prose_and_poetry"

data:
  include_types: ["essay", "letter", "commentary", "poetry"]
  exclude_types: ["drama"]
  period_weights:
    mature: 3.0
    middle: 2.0
    early: 0.5
  content_weights:
    essay: 2.0
    letter: 1.5
    poetry: 1.0               # Equal weight now
    commentary: 1.5
  priority_works:
    - "21-22TheLifeDivine"
    - "33-34Savitri"          # Add Savitri
    - "02CollectedPoems"
    - "23-24TheSynthesisOfYoga"

# Model and training same as EXP-A1
```

### EXP-B2: Fine-tune GPT-2, Prose + Poetry

```yaml
experiment:
  name: "EXP-B2-prose-poetry-finetune"
  approach: "finetune"
  content: "prose_and_poetry"

# Combines EXP-B1 training config with EXP-A2 data config
```

---

## 5. Evaluation Framework

### Metrics Per Experiment

#### Quantitative
| Metric | Tool | When |
|--------|------|------|
| Perplexity (val set) | Built-in | Every eval_interval |
| Train/Val Loss Gap | Built-in | Every eval_interval |
| Grokking Detection | Custom | Continuous |
| Token-level Accuracy | Built-in | Every eval_interval |

#### Qualitative (LLM Judge)
| Dimension | Description | Scale |
|-----------|-------------|-------|
| Ontological Accuracy | Correct use of Aurobindo's concepts | 1-10 |
| Stylistic Fidelity | Sentence structure, vocabulary | 1-10 |
| Coherence | Grammar, logical flow | 1-10 |
| Authenticity | "Feels like" Sri Aurobindo | 1-10 |

### Evaluation Prompts

#### Prose Prompts (for EXP-A1, EXP-B1)
```json
[
  {"prompt": "The Supermind is not merely a higher mind but", "type": "ontology"},
  {"prompt": "The psychic being differs from the soul in that", "type": "ontology"},
  {"prompt": "In the process of spiritual evolution, matter must", "type": "cosmology"},
  {"prompt": "The goal of Integral Yoga is not merely liberation but", "type": "practice"},
  {"prompt": "The Divine Mother manifests in four primary aspects:", "type": "theology"},
  {"prompt": "The vital being resists transformation because", "type": "psychology"},
  {"prompt": "Surrender in yoga does not mean", "type": "practice"},
  {"prompt": "The difference between the Overmind and Supermind is", "type": "ontology"},
  {"prompt": "The three modes of Nature, sattwa, rajas, and tamas,", "type": "philosophy"},
  {"prompt": "The hostile forces attack the sadhak by", "type": "practice"}
]
```

#### Poetry Prompts (additional for EXP-A2, EXP-B2)
```json
[
  {"prompt": "O Sun-Word, thou sole wide world-interpreter,", "type": "invocation", "source": "Savitri"},
  {"prompt": "A lonely soul passioning for the Infinite,", "type": "imagery", "source": "Savitri"},
  {"prompt": "The golden bridals of the earth and sun", "type": "metaphor"},
  {"prompt": "In the silence of the spirit's sky", "type": "meditation"},
  {"prompt": "Heaven's call is rare, rarer the heart that heeds;", "type": "aphorism"}
]
```

### Comparison Report Template

```markdown
# Experiment Comparison Report

## Summary Table
| Metric | EXP-A1 | EXP-B1 | EXP-A2 | EXP-B2 |
|--------|--------|--------|--------|--------|
| Final Val Loss | | | | |
| Final Perplexity | | | | |
| Grokking Observed | | | | |
| Grokking Step | | | | |
| LLM Judge (Avg) | | | | |
| Training Time | | | | |

## Key Findings
1. ...
2. ...

## Sample Outputs Comparison
[Side-by-side generations from each model]

## Recommendations
...
```

---

## 6. Timeline

### Week 1: Data & Infrastructure
- Days 1-2: Clean data, separate prose/poetry
- Days 3-4: Build tokenizers (16K for A, extended for B)
- Days 5-6: Build training scripts with checkpointing
- Day 7: Test on tiny subset, verify resume works

### Week 2: Experiments A1 & B1 (Prose Only)
- Days 1-3: Run EXP-A1 (from scratch, prose)
- Days 4-6: Run EXP-B1 (fine-tune, prose)
- Day 7: Evaluate both, document results

### Week 3: Experiments A2 & B2 (Prose + Poetry)
- Days 1-3: Run EXP-A2 (from scratch, prose+poetry)
- Days 4-6: Run EXP-B2 (fine-tune, prose+poetry)
- Day 7: Evaluate both, document results

### Week 4: Analysis & Publication
- Days 1-2: Comparative analysis, grokking review
- Days 3-4: Write methodology, results
- Days 5-6: HuggingFace upload (best models)
- Day 7: Final documentation

---

## 7. File Structure

```
saslm/
├── EXPERIMENT_FRAMEWORK.md      # This document
├── EXECUTION_PLAN_V2.md         # Overall project plan
│
├── data/
│   ├── raw_pdfs/                # Original PDFs
│   ├── processed_v1/            # Old extraction (reference)
│   ├── clean_prose/             # Cleaned prose texts
│   ├── clean_poetry/            # Cleaned poetry texts
│   └── corpus_metadata.json     # Tags for all texts
│
├── tokenizers/
│   ├── tokenizer_16k/           # For Option A
│   └── tokenizer_extended/      # For Option B
│
├── configs/
│   ├── exp_a1.yaml
│   ├── exp_b1.yaml
│   ├── exp_a2.yaml
│   └── exp_b2.yaml
│
├── src/
│   ├── data/
│   │   ├── clean_corpus.py
│   │   ├── classify_content.py
│   │   └── weighted_sampler.py
│   ├── model/
│   │   ├── gpt_from_scratch.py
│   │   └── finetune_gpt2.py
│   ├── training/
│   │   ├── train.py
│   │   ├── checkpoint_manager.py
│   │   └── grokking_detector.py
│   ├── eval/
│   │   ├── evaluate.py
│   │   ├── llm_judge.py
│   │   └── prompts.json
│   └── utils/
│       └── logging.py
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_train_exp_a1.ipynb
│   ├── 03_train_exp_b1.ipynb
│   ├── 04_train_exp_a2.ipynb
│   ├── 05_train_exp_b2.ipynb
│   └── 06_comparison_analysis.ipynb
│
├── experiments/                  # On Google Drive
│   ├── EXP-A1-prose-only/
│   │   ├── config.yaml
│   │   ├── checkpoint.pt
│   │   ├── best_model.pt
│   │   ├── metrics.jsonl
│   │   └── samples/
│   ├── EXP-B1-prose-only-finetune/
│   ├── EXP-A2-prose-poetry/
│   └── EXP-B2-prose-poetry-finetune/
│
└── results/
    ├── comparison_report.md
    ├── figures/
    └── model_cards/
```

---

## 8. Quick Start Commands

```bash
# 1. Setup environment
pip install torch transformers tokenizers wandb datasets tqdm

# 2. Mount drive (in Colab)
from google.colab import drive
drive.mount('/content/drive')

# 3. Run experiment (will auto-resume if checkpoint exists)
python src/training/train.py --config configs/exp_a1.yaml

# 4. Monitor (in separate cell)
python src/training/monitor.py --experiment EXP-A1-prose-only

# 5. Evaluate
python src/eval/evaluate.py --experiment EXP-A1-prose-only --judge claude
```

---

*Framework Version: 1.0*
*Created: 2026-02-04*
