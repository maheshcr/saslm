import fitz
import os
import re

def check_diacritics(pdf_path, search_limit_pages=100):
    """
    Scans the first N pages for non-ASCII characters to verify diacritic preservation.
    """
    print(f"--- Diacritic Scan: {os.path.basename(pdf_path)} ---")
    
    doc = fitz.open(pdf_path)
    found_diacritics = []
    
    for i in range(min(len(doc), search_limit_pages)):
        page = doc[i]
        text = page.get_text()
        
        # Find non-ascii characters
        # explicit check for characters > 127
        non_ascii = [c for c in text if ord(c) > 127]
        
        if non_ascii:
            # Group them to see unique ones
            unique = set(non_ascii)
            # Filter out common punctuation like quotes, dashes, bullets if possible, 
            # but for now just show what we found.
            found_diacritics.extend(list(unique))
            
            # Print a snippet context for the first few findings
            for char in list(unique)[:3]:
                idx = text.find(char)
                start = max(0, idx - 10)
                end = min(len(text), idx + 10)
                snippet = text[start:end].replace('\n', ' ')
                print(f"Page {i+1}: Found '{char}' (U+{ord(char):04X}) in context: ...{snippet}...")
            
            # If we found good evidence, break early
            if len(set(found_diacritics)) > 5:
                break
                
    if not found_diacritics:
        print("WARNING: No non-ASCII characters found. Text might be normalized or Font encoding is custom.")
    else:
        print(f"\nUnique Non-ASCII chars found: {sorted(list(set(found_diacritics)))}")

if __name__ == "__main__":
    check_diacritics("temp_data/10-11RecordOfYoga.pdf")
    check_diacritics("temp_data/21-22TheLifeDivine.pdf")
