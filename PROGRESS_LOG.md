# SASLM Progress Log

## Current Status (Feb 2025)

### Completed
1. **Infrastructure built** - Full training pipeline with checkpointing, grokking detection, weighted sampling
2. **Corpus expanded** - 23 books, 30MB, ~7.5M tokens (up from 12MB in v1)
3. **Tokenizer trained** - 16K vocab BPE tokenizer optimized for Sanskrit diacritics
4. **EXP-A1 trained** - 100K steps, from-scratch 17M param model
5. **EXP-A1 evaluated** - LLM Judge score: 3.26/10 (up from 2.80 with broken chunking)
6. **EXP-B1 attempted** - DistilGPT-2 fine-tuning failed (tokenizer mismatch, garbage output)

### Key Findings

#### EXP-A1 (From Scratch) - Score 3.26/10
- Model learned vocabulary (Supermind, psychic being, Sachchidananda)
- Coherence still weak (scores 1-3)
- **Root cause**: Data artifacts contaminating training
  - Page headers: "248 Letters on Yoga - I"
  - Editorial notes: "Written by Sri Aurobindo to his secretary..."
  - Footnote markers
- **Overfitting**: Train loss 1.73 vs Val loss 3.95 (gap of 2.2)

#### EXP-B1 (GPT-2 Fine-tune) - Failed
- GPT-2's BPE tokenizer fragments Sanskrit terms badly
- "Sachchidananda" → broken subwords → garbage output
- Output was nonsensical fragments: "ledge", "ita", "ani"
- **Conclusion**: GPT-2 tokenizer incompatible with this corpus

### Bugs Fixed
1. **Chunking bug** - Books weren't being split into chunks (23 chunks total instead of 15,591)
   - Fixed in `src/data/text_cleaner.py` with sliding window fallback
2. **HuggingFace model call** - Wrong function signature for fine-tuning
   - Fixed with `compute_loss()` helper in `train.py`
3. **Tokenizer mismatch** - Fine-tuning used wrong tokenizer
   - Fixed with `HFTokenizerWrapper` class in `train.py`

---

## Next Steps (In Order)

### Step 1: Clean the Corpus
Remove editorial artifacts that are contaminating training.

```bash
# In Colab, after git pull:

# Preview what will be cleaned
!python src/data/clean_corpus.py --input {CORPUS_PATH} --preview --verbose

# Backup and clean
!cp -r {CORPUS_PATH} {CORPUS_PATH}_backup
!python src/data/clean_corpus.py --input {CORPUS_PATH} --verbose
```

### Step 2: Retrain EXP-A1 with Clean Corpus
```python
# Delete old checkpoint
import shutil
checkpoint_dir = f'{EXPERIMENTS_PATH}/EXP-A1-prose-only'
if os.path.exists(checkpoint_dir):
    shutil.rmtree(checkpoint_dir)

EXPERIMENT = "EXP-A1"
# Update config cells (11-12)
# Run training cell (16)
!python src/training/train.py --config {updated_config_path}
```

**Expected improvement**: Coherence scores should improve without artifact noise.

### Step 3: Try Pythia Fine-tuning (EXP-C1)
Pythia-70M has a better tokenizer for diverse text.

```python
EXPERIMENT = "EXP-C1"
config_map["EXP-C1"] = "configs/exp_c1_pythia_finetune.yaml"
# Update config with paths
# Run training
!python src/training/train.py --config {updated_config_path}
```

### Step 4: Compare Results
Run LLM Judge on both models:

| Experiment | Model | Expected Score | Notes |
|------------|-------|----------------|-------|
| EXP-A1 v2 | 17M from-scratch | 4-5/10 | With clean corpus |
| EXP-C1 | 70M Pythia fine-tune | 5-6/10 | Better tokenizer |

---

## File Locations

### On Google Drive
```
/content/drive/MyDrive/saslm/
├── clean_prose/              # Corpus (23 text files)
├── clean_prose_backup/       # Backup before cleaning (create this)
├── tokenizers/
│   └── tokenizer_16k/        # Custom tokenizer
└── experiments/
    ├── EXP-A1-prose-only/    # From-scratch checkpoints
    └── EXP-C1-pythia-finetune/  # Pythia checkpoints (future)
```

### In Repo
```
configs/
├── exp_a1_prose_only.yaml      # From-scratch config
├── exp_b1_prose_only_finetune.yaml  # GPT-2 fine-tune (failed)
└── exp_c1_pythia_finetune.yaml # Pythia fine-tune (new)

src/data/
├── clean_corpus.py             # Corpus cleaning script (new)
├── text_cleaner.py             # Chunking with sliding window
└── weighted_sampler.py         # Weighted sampling by period/importance

src/training/
├── train.py                    # Main training script
├── checkpoint_manager.py       # Google Drive checkpointing
└── grokking_detector.py        # Grokking phenomenon detection
```

---

## LLM Judge Evaluation Code

Add this cell in Colab after loading a model:

```python
import os
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

test_prompts = [
    {"id": "ytt_001", "category": "Ontology", "prompt": "The distinction between the psychic being and the spiritual self is"},
    {"id": "ytt_002", "category": "Ontology", "prompt": "The three lower planes of existence are Matter, Life, and"},
    {"id": "ytt_003", "category": "Cosmology", "prompt": "In the process of involution, the Divine Consciousness hides itself in"},
    {"id": "ytt_004", "category": "Yoga Practice", "prompt": "The goal of Integral Yoga is not Nirvana, but"},
    {"id": "ytt_005", "category": "Psychology", "prompt": "The role of the ego in the early stages of evolution is to"},
]

results = []

for item in test_prompts:
    prompt = item['prompt']

    # Generate with model
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], device='cuda')
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=100, temperature=0.8, top_k=50)
    generated = tokenizer.decode(output[0].tolist())
    completion = generated[len(prompt):]

    # Judge with Claude
    judge_prompt = f'''You are an expert scholar in Sri Aurobindo's philosophy.
Evaluate this completion:

PROMPT: "{prompt}"
COMPLETION: "{completion}"

Grade 1-10 on:
1. Ontological Accuracy: Correct use of terms like Supermind, Psychic Being
2. Stylistic Fidelity: Captures Aurobindo's style
3. Coherence: Grammatically/logically consistent

Return JSON only:
{{"ontological_accuracy": <1-10>, "stylistic_fidelity": <1-10>, "coherence": <1-10>, "overall_score": <avg>, "reasoning": "<brief>"}}'''

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{"role": "user", "content": judge_prompt}]
    )

    import json
    content = response.content[0].text
    scores = json.loads(content[content.find('{'):content.rfind('}')+1])

    print(f"\n{'='*60}")
    print(f"[{item['id']}] {item['category']}")
    print(f"Prompt: {prompt}")
    print(f"Generated: {completion[:200]}...")
    print(f"Scores: O={scores.get('ontological_accuracy')}, S={scores.get('stylistic_fidelity')}, C={scores.get('coherence')}, Overall={scores.get('overall_score')}")

    results.append({**item, 'completion': completion, **scores})

avg_score = sum(r['overall_score'] for r in results) / len(results)
print(f"\n{'='*60}")
print(f"AVERAGE OVERALL SCORE: {avg_score:.2f} / 10")
```

---

## Historical Scores

| Version | Date | Chunks | Score | Notes |
|---------|------|--------|-------|-------|
| EXP-A1 (broken) | Feb 2025 | 23 | 2.80 | Chunking bug |
| EXP-A1 (fixed) | Feb 2025 | 15,591 | 3.26 | Still has data artifacts |
| EXP-A1 (clean) | TBD | 15,591 | ? | After corpus cleaning |
| EXP-C1 (Pythia) | TBD | 15,591 | ? | Different tokenizer |
