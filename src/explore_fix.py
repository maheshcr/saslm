import pdfplumber
import fitz  # PyMuPDF
import sys
import os

def test_spacing_fix(pdf_path, page_number):
    """
    Test pdfplumber with different x_tolerance to fix run-on words.
    """
    print(f"--- Fiddling with Spacing: {os.path.basename(pdf_path)} (Page {page_number}) ---")
    
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        
        # Default is usually around 3. Let's try larger values to separate words, 
        # OR smaller values? 
        # Run-together words usually mean x_tolerance is too LARGE (grouping separate letters as one word)? 
        # No, run-together means it thinks distinct words are one word.
        # Actually pdfplumber gathers characters. If the gap < x_tolerance, it joins them.
        # So we probably need to REDUCE x_tolerance or check how it handles spaces.
        
        # Let's try explicit extraction
        text_default = page.extract_text()
        print(f"Default: {text_default[:100]}...")
        
        # Try adjusting x_tolerance
        # Only layout=True sometimes helps.
        text_layout = page.extract_text(layout=True)
        print(f"Layout=True: {text_layout[:100]}...")
        
        # Try specific tolerance
        text_tol = page.extract_text(x_tolerance=1) # Default is 3
        print(f"x_tolerance=1: {text_tol[:100]}...")


def test_fitz_decoding(pdf_path, page_number):
    """
    Test if PyMuPDF (fitz) can decode the CID characters better than pdfplumber.
    """
    print(f"--- PyMuPDF Decoding: {os.path.basename(pdf_path)} (Page {page_number}) ---")
    
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number - 1)
        text = page.get_text("text")
        print(">>> FITZ TEXT START <<<")
        print(text[:500])
        print(">>> FITZ TEXT END <<<")
    except Exception as e:
        print(f"Fitz failed: {e}")

def main():
    base_dir = "temp_data"
    
    # Files
    prose_path = os.path.join(base_dir, "21-22TheLifeDivine.pdf")
    poetry_path = os.path.join(base_dir, "33-34Savitri.pdf")
    mixed_path = os.path.join(base_dir, "10-11RecordOfYoga.pdf")
    
    if os.path.exists(prose_path):
        test_spacing_fix(prose_path, 50)
        
    if os.path.exists(mixed_path):
        test_spacing_fix(mixed_path, 50)

    if os.path.exists(poetry_path):
        # First check if Fitz handles the CID mess
        test_fitz_decoding(poetry_path, 50)

if __name__ == "__main__":
    main()
