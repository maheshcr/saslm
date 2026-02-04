# Research Paper Guide: SASLM

## Title Idea
**SASLM: A Domain-Specific Small Language Model for the Works of Sri Aurobindo**

## 1. When to Read the Papers
*   **Now (Planning Phase)**: Read the **Introduction** and **Methodology** of `TinyStories.pdf`.
    *   *Why*: To understand how they simplified the dataset. We are doing the opposite (training on a complex, high-register dataset), but the *scale* (SLM) is similar. Note their architecture choices (depth vs width).
*   **During Data Prep**: Read `RegionalTinyStories.pdf` if it discusses handling non-English languages or specific tokenization issues for regional scripts. Use their tokenization insights.
*   **During Writing**: Read the **Evaluation** sections of both to see what tables and graphs they included. You want to replicate a similar "Comparison to Baseline" table.

## 2. Structure of the Paper
1.  **Abstract**: 200 words. "We present SASLM, a 60M parameter model trained exclusively on the 37-volume Complete Works of Sri Aurobindo... demonstrating that small models can capture complex philosophical coherence..."
2.  **Introduction**:
    *   The rise of SLMs (cite TinyStories).
    *   The challenge of domain-specific philosophical texts.
    *   Why Sri Aurobindo? (Unique vocabulary, mixed English/Sanskrit, complex sentence structures).
3.  **Dataset Construction** (Critically important):
    *   Describe the source (CWSA).
    *   **Detailed steps**: How you extracted text, filtered languages, and preserved diacritics. (Ref: `data_preparation_guide.md`).
    *   Stats: Final token count, vocab size.
4.  **Methodology**:
    *   Model Architecture: GPT-style, params listing.
    *   Tokenizer: Custom BPE vs Standard GPT-2. Show examples of Sanskrit words tokenized by both to show efficiency gains.
5.  **Experiments & Results**:
    *   **Perplexity**: Report training and validation loss.
    *   **Generation Quality**: Show side-by-side examples.
        *   Prompt: "The nature of the Supermind is..."
    *   **Benchmarks**:
        *   Standard: BLiMP (if applicable), or simplified GLUE (might be too hard/irrelevant).
        *   Custom: "SASLM-Eval" - a set of 50 cloze tests (fill in the blank) created from the text itself.
6.  **Conclusion & Future Work**:
    *   Scaling up.
    *   Fine-tuning for specific aspects (Poetry vs Prose).

## 3. Data to Capture for the Paper
*   **Training Logs**: Keep a CSV of `step`, `train_loss`, `val_loss`, `tokens_seen`. Plots of Loss Curves are mandatory.
*   **Tokenizer Stats**: "Average tokens per word". Compare standard English tokenizer vs your Custom Tokenizer on a sample of Sri Aurobindo's text. Lower is better/more efficient.
*   **Hardware Usage**: "Trained on 1x A100 for X hours".

## 4. Writing Schedule
*   **Week 1**: Data Prep & Tokenizer Paper Section (Write while you code).
*   **Week 2**: Methodology Section (Write while training runs).
*   **Week 3**: Experiments & Results (After training).
*   **Week 4**: Abstract & Intro (Write last to match the final narrative).
*   **Week 5**: Review & Formatting (LaTeX/Overleaf).
