import logging
import os
import pytesseract
from pdf2image import convert_from_path
from pipeline import load_book_config, generate_prune_phrases
from extraction_utils import crop_image_header_footer, clean_ocr_text_content

# Setup logging
logging.basicConfig(level=logging.INFO)

TARGET_PDF = "raw_data/35LettersOnHimselfAndTheAshram.pdf"
DPI = 300
LANG = 'script/Latin'

def test_extraction():
    if not os.path.exists(TARGET_PDF):
        print(f"Error: {TARGET_PDF} not found.")
        return

    filename = os.path.basename(TARGET_PDF)
    config = load_book_config()
    book_conf = config.get(filename, {})
    header_pct = book_conf.get("header_percent", 0.08)
    footer_pct = book_conf.get("footer_percent", 0.08)
    
    prune_phrases = generate_prune_phrases(filename)
    print(f"Prune phrases: {prune_phrases}")
    
    # Process just first 20 pages
    images = convert_from_path(TARGET_PDF, first_page=1, last_page=20, dpi=DPI)
    
    full_text = []
    for i, img in enumerate(images):
        cropped = crop_image_header_footer(img, header_pct, footer_pct)
        text = pytesseract.image_to_string(cropped, lang=LANG)
        cleaned = clean_ocr_text_content(text, prune_phrases=prune_phrases)
        full_text.append(cleaned)
        
    final_output = "\n".join(full_text)
    
    # Check for artifacts
    artifacts = [
        "256 Letters on Himself",
        "Letters on Himself and the Ashram"
    ]
    
    print("\n--- Extraction Result Preview ---")
    print(final_output[:2000]) # First 2000 chars
    print("\n-------------------------------\n")
    
    for art in artifacts:
        if art.lower() in final_output.lower():
            print(f"[FAIL] Found artifact: '{art}'")
        else:
            print(f"[PASS] Artifact '{art}' NOT found.")

if __name__ == "__main__":
    test_extraction()
