import fitz
import os
import logging
import re
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_pdf_metadata(pdf_path):
    """
    Retrieves metadata from a PDF file.
    """
    try:
        doc = fitz.open(pdf_path)
        return doc.metadata
    except Exception as e:
        logging.error(f"Error reading metadata from {pdf_path}: {e}")
        return {}

def crop_image_header_footer(image, header_percent=0.08, footer_percent=0.08):
    """
    Crops header and footer from a PIL Image.
    
    Args:
        image (PIL.Image): Input image.
        header_percent (float): Fraction to cut from top.
        footer_percent (float): Fraction to cut from bottom.
        
    Returns:
        PIL.Image: Cropped image.
    """
    w, h = image.size
    top = int(h * header_percent)
    bottom = int(h * (1 - footer_percent))
    return image.crop((0, top, w, bottom))

def clean_ocr_text_content(text, prune_phrases=None):
    """
    Filtering of OCR noise and specific artifacts.
    
    Args:
        text (str): Raw OCR text.
        prune_phrases (list): List of strings (e.g. book titles) to remove if they appear as standalone lines.
    """
    if not text: return ""
    lines = text.split('\n')
    cleaned_lines = []
    
    # Common artifacts regex
    # 1. Page numbers like "123", "Page 123", "12 | Book Title"
    re_page_num = re.compile(r'^(page\s*)?\d{1,4}(\s*[:|-]\s*)?$', re.IGNORECASE)
    
    for line in lines:
        s = line.strip()
        if not s:
            continue
            
        # 1. Filter Page Numbers (more aggressive)
        if re_page_num.match(s):
            continue
            
        if s.isdigit() and len(s) < 5:
            continue

        # 2. Filter known headers/titles
        should_prune = False
        if prune_phrases:
            for phrase in prune_phrases:
                # If the line is exactly the phrase (ignoring case/punct) or very close
                # Simple case-insensitive exact match for now
                if phrase and phrase.lower() in s.lower():
                     # If the line is SHORT (mostly just the title), prune it.
                     # If it's a long sentence containing the title, keep it (might be content).
                     if len(s) < len(phrase) + 10:
                         should_prune = True
                         break
        
        if should_prune:
            continue

        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines)
