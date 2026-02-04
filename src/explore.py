import pdfplumber
import sys
import os

def inspect_page_layout(pdf_path, page_number):
    """
    Extracts words from a specific page along with their bounding boxes.
    Useful for determining header/footer cutoffs.
    
    Args:
        pdf_path (str): Path to the PDF file.
        page_number (int): Page number to inspect (1-based index).
    """
    print(f"--- Inspecting Layout: {os.path.basename(pdf_path)} (Page {page_number}) ---")
    
    with pdfplumber.open(pdf_path) as pdf:
        # pdfplumber pages are 0-indexed
        if page_number < 1 or page_number > len(pdf.pages):
            print(f"Error: Page {page_number} is out of range (1-{len(pdf.pages)})")
            return

        page = pdf.pages[page_number - 1]
        words = page.extract_words()
        
        # Sort by 'top' (vertical position) to see what comes first (likely headers)
        sorted_words = sorted(words, key=lambda x: x['top'])
        
        print(f"{'Text':<20} | {'Top':<10} | {'Bottom':<10} | {'Font':<10}")
        print("-" * 60)
        
        # Print first 10 items (likely header/top metadata)
        for w in sorted_words[:10]:
            print(f"{w['text'][:20]:<20} | {w['top']:<10.2f} | {w['bottom']:<10.2f} | {w.get('fontname', 'N/A')}")
            
        print("...")
        
        # Print last 10 items (likely footer/page numbers)
        for w in sorted_words[-10:]:
            print(f"{w['text'][:20]:<20} | {w['top']:<10.2f} | {w['bottom']:<10.2f} | {w.get('fontname', 'N/A')}")
            
    print("\n")

def check_text_quality(pdf_path, page_number):
    """
    Extracts raw text to verify diacritics and overall readability.
    
    Args:
        pdf_path (str): Path to the PDF file.
        page_number (int): Page number to inspect (1-based index).
    """
    print(f"--- Checking Text Quality: {os.path.basename(pdf_path)} (Page {page_number}) ---")
    
    with pdfplumber.open(pdf_path) as pdf:
        if page_number < 1 or page_number > len(pdf.pages):
            print(f"Error: Page {page_number} is out of range.")
            return

        page = pdf.pages[page_number - 1]
        text = page.extract_text()
        
        print(">>> START TEXT SNIPPET <<<")
        # Print first 500 chars to avoid flooding
        print(text[:500] if text else "[No text extracted]")
        print(">>> END TEXT SNIPPET <<<\n")

def main():
    """
    Main execution function.
    """
    base_dir = "temp_data"
    
    # Define test cases: (Filename, Page Number)
    # Page 50 is usually safe to be regular content (skipping ToC/Preface)
    test_cases = [
        ("21-22TheLifeDivine.pdf", 50),     # Prose
        ("33-34Savitri.pdf", 50),           # Poetry
        ("10-11RecordOfYoga.pdf", 50)       # potentially Mixed
    ]
    
    for filename, page_num in test_cases:
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            inspect_page_layout(filepath, page_num)
            check_text_quality(filepath, page_num)
        else:
            print(f"File not found: {filepath}")

if __name__ == "__main__":
    main()
