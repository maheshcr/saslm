# Data Preparation Guide for SASLM

## Overview
This document details the strategy for converting the PDF volumes of Sri Aurobindo's works into a clean, tokenizable text dataset. The primary challenges are the mixed language content (English/Sanskrit), complex layout (headers, footers, poetry vs prose), and the need to preserve diacritics.

## 1. Initial Exploration (Local)
**Goal**: Identify the structure, fonts, and common artifacts in the PDFs.

*   **Tools**: `pdfplumber` (Python), `mutool` (CLI).
*   **Action**:
    *   Extract the first 5 pages of "The Life Divine" (Prose) and "Savitri" (Poetry) and "Record of Yoga" (Mixed).
    *   Inspect `page.extract_text()` output vs `page.extract_words()` (which gives coordinates).
    *   **Heuristic Definition**: Measure the Y-coordinate of standard headers and footers. Usually, the top 5-10% of a page contains the "Volume Title" or "Chapter Title".
    *   **Font Analysis**: Check if Sanskrit text uses a specific embedded font or if it's Unicode. If it's old legacy fonts (non-Unicode), we might need an OCR mapping step. However, CWSA PDFs are usually modern and Unicode-compliant.

## 2. Extraction Pipeline (Local or Colab)
Since the dataset is ~60MB zipped, extracting text is CPU-bound but not heavy. Can be done locally or on Colab.

### Step-by-Step Logic
1.  **Iterate Volumes**: Process `01EarlyCulturalWritings.pdf` through `37...`.
2.  **Page Processing**:
    *   **Crop Headers/Footers**: Based on the heuristics defined in step 1, crop the bounding box of the page to exclude the top header and bottom footer (e.g., page number).
    *   **Text Extraction**: Use `pdfplumber` with `x_tolerance=1` (strict horizontal grouping) to preserve word spacing.
3.  **Language Filtering**:
    *   The user wants to ignore Bengali/Tamil but keep Sanskrit/English.
    *   **Strategy**:
        *   Split text into paragraphs.
        *   Use `langdetect` or `fasttext` on each paragraph.
        *   If `lang` is `en` or `sa` (Sanskrit) or contains a high threshold of Diacritics (common in transliterated Sanskrit), KEEP it.
        *   If `lang` is `bn` (Bengali) or `ta` (Tamil) AND script is non-Latin, DISCARD it.
        *   *Note*: Often Bengali/Tamil in these books might be in original script. We can filter by Unicode ranges for Bengali/Tamil scripts.
4.  **Structure Preservation**:
    *   **Chapter Headings**: If a line is centered and all caps (detectable via `chars[0]['fontname']` or properties in `pdfplumber`), format it as `\n\n# <TITLE>\n\n`.
    *   **Poetry**: Detect indentation. If a block of lines has substantial left margin variance compared to prose, keep the newlines. For prose, we might want to unwrap lines (join lines that don't end in punctuation).
5.  **Normalization**:
    *   Fix hyphenation: `re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)` (Basic rule, needs refinement).
    *   **Crucial**: Do NOT normalize unicode to ASCII. We must preserve `ā`, `ī`, `ū`, `ṛ`, `ñ`, `ṅ`, `ṣ`, `ś` etc.

## 3. Output Format
Save the output in two formats:
1.  **`raw_corpus.txt`**: All text concatenated (separated by `<|endoftext|>` or specific markers like `<|book_start|>` if we use custom tokens).
2.  **`corpus_metadata.jsonl`**:
    ```json
    {"text": "Paragraph content...", "meta": {"volume": "21_TheLifeDivine", "page": 45, "is_poetry": false}}
    ```
    This allows us to filter or re-weight data later.

## 4. Tokenizer Training
*   **Library**: `tokenizers` (HuggingFace).
*   **Type**: BPE (Byte Pair Encoding).
*   **Vocab Size**: 30,000 - 50,000.
*   **Special Instructions**:
    *   Train on `raw_corpus.txt`.
    *   Ensure the tokenizer separates Sanskrit compounds correctly (BPE usually finds subwords effectively).
    *   Check coverage: `tokenizer.encode("Auṃ tat sat")` should not yield `[UNK]`.

## 5. Verification
*   **Visual Diff**: Compare `extracted_sample.txt` vs the PDF page side-by-side.
*   **Search**: Grep for common known phrases (e.g., "Supramental", "Overmind") to ensure they aren't mangled (e.g., "Su pra mental").
