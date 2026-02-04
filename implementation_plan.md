# Sri Aurobindo Small Language Model (SASLM) - Implementation Plan

# Goal Description
The goal is to build a Small Language Model (SLM) trained on the complete works of Sri Aurobindo (SASLM). The project involves:
1.  **Data Extraction & Cleaning**: Extracting text from PDFs, handling mixed languages (English, Sanskrit, etc.), pruning metadata, and preserving structural nuances (chapters, accents).
2.  **Tokenizer Training**: creating a custom tokenizer optimized for the specific vocabulary of the corpus.
3.  **Model Training**: Training a Transformer-based SLM (approx. 50-60M parameters) on this refined dataset.
4.  **Evaluation**: Testing the model on relevant benchmarks.
5.  **Research & Publication**: Publishing the model on HuggingFace and writing a research paper.

## User Review Required
> [!IMPORTANT]
> **PDF Extraction Strategy**: We need to confirm if we should use OCR tools (like Tesseract) or text extraction libraries (like `pymupdf` or `pdfplumber`). Given the "mixed text" and "accents" requirement, simple text extraction might fail on complex layouts or non-standard fonts. Deep learning based layout analysis models (like LayoutParser) might be needed for high quality.

> [!NOTE]
> **Compute Resources**: Training even a small model requires GPU. The plan assumes use of the user's RunPod or Colab Pro accounts.

## Proposed Changes

### Data Preparation (Local & Colab)

#### [NEW] [data_prep_pipeline.ipynb](file:///Users/maheshcr/projects/saslm/data_prep_pipeline.ipynb)
A new notebook/script to handle the extraction pipeline:
1.  **Extraction**: Unzip `cwsa.zip`. Iterate through PDFs.
2.  **Text Extraction**: Use `pdfplumber` or `pymupdf` to extract text while preserving layout info where possible to identify headers/footers.
3.  **Filtering**: 
    - Detect language per block/paragraph using `langdetect` or `fasttext`.
    - Keep English and Sanskrit. 
    - Filter out Bengali/Tamil blocks.
4.  **Cleaning**: 
    - Regex patterns to remove headers (e.g., "Volume X", "Chapter Y" repeated at top of pages) and footers.
    - Heuristics to fix broken hyphenation at line ends.
    - Preserve accent marks (diacritics).
5.  **Structuring**:
    - Insert special tokens or markers for Chapter titles vs Body text if we want the model to learn structure (optional, but good for "progression").
    - Save as `raw_text.txt` or JSONL format.

#### [NEW] [tokenizer_training.ipynb](file:///Users/maheshcr/projects/saslm/tokenizer_training.ipynb)
1.  Train a BPE tokenizer (using HuggingFace `tokenizers` library) on the cleaned corpus.
2.  Vocab size: Recommendations usually ~32k-50k for this size of model, but given the Sanskrit/English mix, we might want to ensure Sanskrit characters are not overly fragmented.
3.  Save tokenizer.

### Model Training (RunPod/Colab)

#### [MODIFY] [Sri_Aurobindo_Small_Language_Model.ipynb](file:///Users/maheshcr/projects/saslm/Sri_Aurobindo_Small_Language_Model.ipynb)
Refactor the existing notebook:
1.  **Dataset Loading**: Replace TinyStories loading with loading our custom `raw_text.txt`.
2.  **Tokenization**: Load our custom trained tokenizer instead of `gpt2`.
3.  **Config**: Adjust `GPTConfig` for ~60M params.
    - `n_layer=8`, `n_head=8`, `n_embd=512` (Example values, to be tuned).
4.  **Training Loop**: Ensure checkpoints are saved to Drive/Persistent storage.

### Research & Benchmarking

#### [NEW] [research_paper_guide.md](file:///Users/maheshcr/projects/saslm/research_paper_guide.md)
A guide with:
1.  **Abstract**: Drafting the core promise.
2.  **Methodology**: Documenting the data cleaning steps (crucial for the paper).
3.  **Benchmarks**:
    - Perplexity on held-out SASLM test set.
    - "TinyStories" style generation capabilities (coherence, grammar).
    - Specific "Sri Aurobindo" queries (philosophical completion).
4.  **Ablation Studies**: (Optional) Compare tokenizer performance (Standard English vs Custom).

## Verification Plan

### Automated Tests
- **Data Cleaning**: create a small `test_sample.pdf` (or use a page from the corpus) and verify:
    - No header/footers remains.
    - Sanskrit accents are preserved.
    - Non-target languages are removed.
- **Tokenizer**: Verify it correctly encodes/decodes a mixed English/Sanskrit sentence without losing characters.
- **Overfitting Check**: Train on a very small subset (1 book) and check if loss drops to near 0.

### Manual Verification
- Inspect `cleaned_text_sample.txt` to visually confirm quality.
- Generate samples from the trained model and qualitatively assess "voice" and coherence.
