# process_data.py

import fitz  # PyMuPDF
import os
import json
import re
from tqdm import tqdm
import logging

# --- Configuration ---
INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
CHUNK_SIZE = 512  # Number of tokens (approximated by words for simplicity)
CHUNK_OVERLAP = 50 # Number of tokens to overlap between chunks

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_text(text):
    """
    Performs basic text cleaning.
    - Removes excessive newlines and whitespace.
    - Merges hyphenated words.
    """
    text = re.sub(r'\s*\n\s*', ' ', text)  # Replace newlines with spaces
    text = re.sub(r'(\w)-\s*(\w)', r'\1\2', text) # Merge hyphenated words
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    return text

def process_and_chunk_pdfs():
    """
    Extracts text from PDFs, cleans it, and splits it into structured,
    overlapping chunks.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logging.info(f"Created directory: {OUTPUT_DIR}")

    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]
    
    logging.info(f"Found {len(pdf_files)} PDFs to process.")

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        file_path = os.path.join(INPUT_DIR, pdf_file)
        
        try:
            doc = fitz.open(file_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            
            cleaned_text = clean_text(full_text)
            words = cleaned_text.split()
            
            chunks = []
            for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk_words = words[i:i + CHUNK_SIZE]
                chunk_text = " ".join(chunk_words)
                
                # Create a structured chunk
                chunk_data = {
                    'source': pdf_file,
                    'chunk_id': len(chunks) + 1,
                    'text': chunk_text
                }
                chunks.append(chunk_data)

            # Save the chunks to a JSON file
            output_filename = os.path.splitext(pdf_file)[0] + ".json"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=4)
            
            logging.info(f"Processed {pdf_file} into {len(chunks)} chunks.")

        except Exception as e:
            logging.error(f"Failed to process {pdf_file}. Error: {e}")

if __name__ == "__main__":
    process_and_chunk_pdfs()
    logging.info("--- Data processing and chunking finished! ---")