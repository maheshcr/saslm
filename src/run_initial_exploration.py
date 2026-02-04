import os
import glob
from extraction_utils import extract_text_with_layout, get_pdf_metadata

def run_exploration():
    """
    Runs the initial exploration on the sample PDFs in temp_data.
    """
    print("Beginning Initial Data Exploration...")
    
    # Locate PDFs
    pdf_files = glob.glob("temp_data/*.pdf")
    
    if not pdf_files:
        print("No PDFs found in temp_data. Please extract samples first.")
        return

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"\n{'='*40}")
        print(f"Analyzing: {filename}")
        print(f"{'='*40}")
        
        # 1. Metadata
        meta = get_pdf_metadata(pdf_path)
        print(f"Title: {meta.get('title', 'Unknown')}")
        print(f"Author: {meta.get('author', 'Unknown')}")
        
        # 2. Extract Sample Page (Page 50) with Header Filtering
        # Note: We use page 49 for 0-based index
        print("\n--- Extracted Text (Page 50) with Header/Footer Filter ---")
        pages = extract_text_with_layout(pdf_path, pages=[49], header_cutoff=130, footer_cutoff=750)
        
        if pages:
            text = pages[0]
            print(text[:1000]) # First 1000 chars
            
            # 3. Check for specific artifacts
            if "(cid:" in text:
                print("\n[ALERT] CID Encoding artifacts found! Text extraction may be compromised.")
            else:
                print("\n[OK] No CID artifacts detected.")
                
        else:
            print("Failed to extract text from page 50.")

if __name__ == "__main__":
    run_exploration()
