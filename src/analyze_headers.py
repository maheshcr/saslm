import os
import glob
import json
import logging
import numpy as np
from pdf2image import convert_from_path
import pytesseract
from tqdm import tqdm
import fitz  # PyMuPDF

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

RAW_DATA_DIR = "raw_data"
CONFIG_FILE = "book_config.json"
SAMPLE_PAGES = [50, 100, 150]  # Pages to sample (avoid covers)
DPI = 300

def get_crop_suggestions(pdf_path):
    """
    Analyzes sample pages of a PDF to suggest header and footer crop percentages.
    """
    filename = os.path.basename(pdf_path)
    
    # Check page count
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        doc.close()
    except Exception as e:
        logging.error(f"Could not open {filename}: {e}")
        return None

    # Adjust sample pages if book is short
    samples = [p for p in SAMPLE_PAGES if p < num_pages]
    if not samples:
        samples = [num_pages // 2] if num_pages > 1 else [1]

    header_percents = []
    footer_percents = []

    try:
        # distinct pages to sample
        for page_num in samples:
            # Convert to image (1-indexed for pdf2image if strictly following, but fitz is 0-indexed. 
            # convert_from_path `first_page` is 1-indexed.)
            images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=DPI)
            if not images:
                continue
            
            img = images[0]
            width, height = img.size
            
            # OCR with bounding boxes
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
            # Filter for high confidence text
            n_boxes = len(data['text'])
            
            # Collect valid bounding boxes (normalized y-coordinates)
            valid_boxes = []
            for i in range(n_boxes):
                if int(data['conf'][i]) > 60 and data['text'][i].strip():
                    y = data['top'][i]
                    h = data['height'][i]
                    # Normalize
                    y_start = y / height
                    y_end = (y + h) / height
                    valid_boxes.append((y_start, y_end))
            
            if not valid_boxes:
                continue
                
            # Sort by Y position
            valid_boxes.sort(key=lambda x: x[0])
            
            # Heuristic: 
            # Header is typically the first isolated line(s).
            # We look for a significant vertical gap between the first few lines and the main body.
            
            # Find largest gaps in the top 15% and bottom 15%
            
            # Top Analysis (Header)
            suggested_header = 0.05 # Default min
            
            # Look at the first 5 boxes
            # If there is a big gap (> 1.5% of page height) between box i and i+1, that might be the header break
            for i in range(min(5, len(valid_boxes) - 1)):
                current_box = valid_boxes[i]
                next_box = valid_boxes[i+1]
                
                gap = next_box[0] - current_box[1]
                
                # If we are still in the top 15% of the page
                if current_box[1] < 0.15:
                    # If gap is substantial (e.g. > 1% of height), assume break
                    if gap > 0.01: 
                        suggested_header = next_box[0] - 0.005 # Cut just before next line
                    else:
                        # Continue consuming lines as part of header? 
                        # Or maybe the header is just very close.
                        # Conservative: if we are very high up (<5%), it's definitely header info usually.
                        pass
                else:
                    break
            
            # Enforce a safe maximum for header (don't cut more than 15% automatically)
            suggested_header = min(suggested_header, 0.15)
            # Enforce a minimum
            suggested_header = max(suggested_header, 0.08) # Our old default was 0.08, keep it as baseline?
                                                           # Actually, if we detect NOTHING, maybe 0.05 is better?
                                                           # Let's stick to detecting.
            
            # Bottom Analysis (Footer)
            suggested_footer = 0.05
            
            # Look at last 5 boxes reversed
            for i in range(len(valid_boxes) - 1, max(len(valid_boxes) - 6, 0), -1):
                current_box = valid_boxes[i]
                prev_box = valid_boxes[i-1]
                
                gap = current_box[0] - prev_box[1]
                
                # If we are in bottom 15%
                if current_box[0] > 0.85:
                    if gap > 0.01:
                        # current_box is footer, cut above it
                        # footer percent is distance from bottom
                        # crop amount = 1.0 - (current_box[0] - margin)
                        suggested_footer = 1.0 - (current_box[0] - 0.005)
                        break
                else:
                    break

            suggested_footer = min(suggested_footer, 0.15)
            suggested_footer = max(suggested_footer, 0.08)
            
            header_percents.append(suggested_header)
            footer_percents.append(suggested_footer)

    except Exception as e:
        logging.warning(f"Error processing {filename} pages: {e}")

    # Average the findings
    if header_percents:
        avg_header = sum(header_percents) / len(header_percents)
        avg_footer = sum(footer_percents) / len(footer_percents)
        
        # Round to 3 decimals
        return {
            "header_percent": round(avg_header, 3),
            "footer_percent": round(avg_footer, 3),
            "notes": f"Analyzed {len(samples)} pages"
        }
    else:
        # Fallback
        return {
            "header_percent": 0.08,
            "footer_percent": 0.08,
            "notes": "Fallback default, analysis failed or empty"
        }

def main():
    # Recursive glob
    pdf_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.pdf"), recursive=True))
    if not pdf_files:
        print(f"No PDFs found in {RAW_DATA_DIR}")
        return

    config = {}
    
    # Load existing if present
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            try:
                config = json.load(f)
            except:
                pass

    print(f"Analyzing {len(pdf_files)} PDFs...")
    
    for pdf_path in tqdm(pdf_files):
        filename = os.path.basename(pdf_path)
        
        # Skip if already in config (unless custom override needs update, but generally preserve)
        if filename in config:
            logging.info(f"Skipping {filename} (Already in config).")
            continue
        
        logging.info(f"Analyzing {filename}...")
        results = get_crop_suggestions(pdf_path)
        
        if results:
            # Preserve existing specific settings if we're just updating crops? 
            # Actually, simply merging is safer.
            existing = config.get(filename, {})
            existing.update(results)
            config[filename] = existing
            
            # Save incrementally
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
                
    print(f"Analysis complete. Config saved to {CONFIG_FILE}")

if __name__ == "__main__":
    main()
