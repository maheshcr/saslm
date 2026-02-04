import pytesseract
from pdf2image import convert_from_path
import os

def test_ocr(pdf_path, page_number):
    """
    Runs Tesseract OCR on a specific page of a PDF.
    
    Args:
        pdf_path (str): Path to PDF.
        page_number (int): Page number (1-based).
    """
    print(f"--- Running OCR on: {os.path.basename(pdf_path)} (Page {page_number}) ---")
    
    if not os.path.exists(pdf_path):
        print("File not found.")
        return

    try:
        # Convert PDF page to image
        # first_page -> 1-based index? No, it takes no page arg, it returns list.
        # We use first_page and last_page parameters to limit processing
        images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
        
        if not images:
            print("Failed to convert PDF to image.")
            return

        image = images[0]
        
        # Run OCR
        # We need to specify lang='eng' (and maybe 'san' if installed, but 'eng' handles diacritics reasonably well usually? 
        # actually standard 'eng' model might fail on 'ā', 'ī'. 
        # Ideally we need 'eng' or 'eng+san'.)
        # Let's try default first.
        text = pytesseract.image_to_string(image, lang='script/Latin')
        
        print(">>> OCR OUTPUT START <<<")
        print(text[:1000])
        print(">>> OCR OUTPUT END <<<")
        
        # Check for specific diacritics
        print("\nChecking for diacritics in OCR output...")
        diacritics = [c for c in text if ord(c) > 127]
        unique_dia = sorted(list(set(diacritics)))
        print(f"Found non-ASCII: {unique_dia}")
        
    except Exception as e:
        print(f"OCR Failed: {e}")
        print("Ensure 'poppler' is installed (brew install poppler) for pdf2image.")

if __name__ == "__main__":
    test_ocr("temp_data/15TheSecretOfTheVeda.pdf", 62) # Page 62 usually has content
