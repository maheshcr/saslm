import fitz
import os
import re

def check_secret_of_veda_diacritics():
    """
    Specifically checks 'The Secret of the Veda' (pages 50-75) for Sanskrit diacritics.
    """
    pdf_path = "temp_data/15TheSecretOfTheVeda.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    print(f"--- Diacritic Scan: {os.path.basename(pdf_path)} (Pages 50-75) ---")
    
    doc = fitz.open(pdf_path)
    found_diacritics = set()
    sample_sanskrit_words = []
    
    # Check pages 50 to 75 (0-indexed: 49 to 74)
    start_page = 49
    end_page = 75
    
    for i in range(start_page, end_page):
        if i >= len(doc): break
        
        page = doc[i]
        text = page.get_text()
        
        # 1. Capture non-ascii
        non_ascii = [c for c in text if ord(c) > 127]
        found_diacritics.update(non_ascii)
        
        # 2. Look for expected Sanskrit terms to see IT they are missing diacritics
        # "Ritam", "Satyam", "Agni", "Veda" often appear with marks like "Ṛtam", "Satyam" (no marks usually), 
        # "Agni" (no marks), "Vedā" (maybe).
        # Let's look for common transliteration markers: ā, ī, ū, ṛ, ṭ, ḍ, ṇ, ś, ṣ, ṃ, ḥ
        
        # Regex for words containing non-ascii
        words = text.split()
        for w in words:
            if any(ord(c) > 127 for c in w):
                # Clean punctuation mostly
                w_clean = w.strip('.,;:"')
                if len(w_clean) > 3: # Ignore simple quotes
                    sample_sanskrit_words.append(w_clean)

    print("\n>>> Non-ASCII Characters Found <<<")
    print(sorted(list(found_diacritics)))
    
    print("\n>>> Sample Mixed Words (Context Check) <<<")
    # Show unique words found
    unique_words = sorted(list(set(sample_sanskrit_words)))
    for w in unique_words[:20]: # Print first 20
        print(w)

    if not unique_words:
        print("WARNING: No complex words found. Check for normalization (e.g., 'Ritam' instead of 'Ṛtam').")

    # Double check for Specific expected chars
    expected = ['ā', 'ī', 'ū', 'ṛ', 'ś', 'ṣ']
    missing = [c for c in expected if c not in found_diacritics]
    if missing:
        print(f"\n[CRITICAL] Missing standard Sanskrit diacritics: {missing}")
        print("This implies the PDF text layer might be simple ASCII.")

if __name__ == "__main__":
    check_secret_of_veda_diacritics()
