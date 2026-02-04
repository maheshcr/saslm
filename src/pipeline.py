import os
import glob
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
import pytesseract
from pdf2image import convert_from_path
from tqdm import tqdm
import fitz

# Import shared utilities
from extraction_utils import crop_image_header_footer, clean_ocr_text_content

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("extraction.log"),
                        logging.StreamHandler()
                    ])

# Configuration
RAW_DATA_DIR = "raw_data/"
OUTPUT_DIR = "processed_text"
CONFIG_FILE = "book_config.json"
DPI = 300
LANG = 'script/Latin'
DEFAULT_HEADER_PERCENT = 0.08
DEFAULT_FOOTER_PERCENT = 0.08

def setup_directories():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def load_book_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load config file: {e}")
    return {}

def generate_prune_phrases(filename):
    """
    Generates potential header/title phrases from filename.
    e.g. "35LettersOnHimselfAndTheAshram.pdf" -> 
         ["Letters On Himself And The Ashram", "LettersonHimselfAndTheAshram", "Letters on Himself and the Ashram"]
    """
    base = os.path.splitext(filename)[0]
    
    # 1. Remove leading digits, hyphens, etc (e.g. "35-", "01")
    # match patterns like "35", "01-02", "33-34"
    clean_base = re.sub(r'^[\d\-\.]+', '', base).strip()
    
    # 2. Split camelCase
    # Insert space before capital letters
    spaced = ''.join(' ' + c if c.isupper() else c for c in clean_base).strip()
    
    candidates = [
        clean_base, 
        spaced, 
        spaced.replace("  ", " "),
        # Common lowercase variant often found in headers
        spaced.title(),
        # Also just the raw base just in case
        base 
    ]
    return list(set(candidates)) # Dedupe

def process_pdf(args):
    pdf_path, config = args
    filename = os.path.basename(pdf_path)
    item_name = os.path.splitext(filename)[0]
    output_path = os.path.join(OUTPUT_DIR, f"{item_name}.txt")
    
    if os.path.exists(output_path):
        logging.info(f"Skipping {filename}, already processed.")
        return

    # Determine config for this book
    book_conf = config.get(filename, {})
    header_pct = book_conf.get("header_percent", DEFAULT_HEADER_PERCENT)
    footer_pct = book_conf.get("footer_percent", DEFAULT_FOOTER_PERCENT)
    exclude_ranges = book_conf.get("exclude_pages", []) # List of [start, end] inclusive
    
    prune_phrases = generate_prune_phrases(filename)

    logging.info(f"Processing {filename} (Crop: H={header_pct}, F={footer_pct}, Excludes: {exclude_ranges})...")
    
    try:
        full_text = []
        
        # Get page count
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        doc.close()

        # Batch process pages
        BATCH_SIZE = 15
        for i in range(1, num_pages + 1, BATCH_SIZE):
            last = min(i + BATCH_SIZE - 1, num_pages)
            
            # Check if this entire batch is excluded? No, checks per page usually safer but slower.
            # Let's filter ranges.
            
            # Simple optimization: if batch is completely inside an excluded range, skip it.
            # But converting to images is the slow part.
            
            pages_to_process = []
            for p in range(i, last + 1):
                is_excluded = False
                for start, end in exclude_ranges:
                    if start <= p <= end:
                        is_excluded = True
                        break
                if not is_excluded:
                    pages_to_process.append(p)
            
            if not pages_to_process:
                logging.info(f"[{filename}] Skipping pages {i}-{last} (Excluded).")
                continue

            # Log progress
            logging.info(f"[{filename}] Processing pages {i} to {last} of {num_pages}...")
            
            # convert_from_path can take a list of pages? No, it takes first/last.
            # So we must fetch the chunk and then discard images for excluded pages.
            # Or fetch one by one? Fetching one by one is slow.
            # Fetching batch is fast.
            
            images = convert_from_path(pdf_path, first_page=i, last_page=last, dpi=DPI)
            
            # Map images back to page numbers
            # image[0] corresponds to page 'i', image[1] to 'i+1', etc.
            
            for idx, img in enumerate(images):
                page_num = i + idx
                
                # Check exclusion again
                is_excluded = False
                for start, end in exclude_ranges:
                    if start <= page_num <= end:
                        is_excluded = True
                        break
                
                if is_excluded:
                    continue

                # Use shared utility for cropping
                cropped_img = crop_image_header_footer(img, header_pct, footer_pct)
                
                # OCR
                text = pytesseract.image_to_string(cropped_img, lang=LANG)
                
                # Use shared utility for cleaning
                cleaned = clean_ocr_text_content(text, prune_phrases=prune_phrases)
                full_text.append(cleaned)
                
    except Exception as e:
        logging.error(f"Failed to process {filename}: {e}")
        return

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(full_text))
        
    logging.info(f"Finished {filename}. Saved to {output_path}")

def main():
    setup_directories()
    config = load_book_config()
    
    # Process all PDFs in raw_data (or temp_data for testing)
    pdf_files = glob.glob("temp_data/*.pdf") + glob.glob("raw_data/*.pdf")
    
    if not pdf_files:
        logging.warning("No PDFs found to process.")
        return
        
    logging.info(f"Found {len(pdf_files)} PDFs to process.")
    
    # Prepare arguments (path, config) for each file
    tasks = [(p, config) for p in pdf_files]

    # Process sequentially or in parallel
    # Sequential for now to monitor logs easily, or small pool
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(tqdm(executor.map(process_pdf, tasks), total=len(tasks)))

if __name__ == "__main__":
    main()
