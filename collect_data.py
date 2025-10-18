# collect_data.py

import arxiv
import os
import logging
from tqdm import tqdm

# --- Configuration ---
SEARCH_QUERY = "Large Language Models" # Example search query
MAX_RESULTS = 100 # Let's start with 100 papers
OUTPUT_DIR = "data/raw"

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_papers():
    """
    Searches for papers on ArXiv based on a query and downloads them.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logging.info(f"Created directory: {OUTPUT_DIR}")

    logging.info(f"Searching for '{SEARCH_QUERY}' on ArXiv...")

    # Construct the search object
    search = arxiv.Search(
        query=SEARCH_QUERY,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate # Get the latest papers
    )

    results = list(search.results()) # Convert generator to list to get a total for tqdm
    logging.info(f"Found {len(results)} papers. Starting download...")

    # Use tqdm for a progress bar
    for paper in tqdm(results, desc="Downloading Papers"):
        try:
            # Download the PDF to the specified directory
            paper.download_pdf(dirpath=OUTPUT_DIR)
            logging.info(f"Successfully downloaded: {paper.title}")
        except Exception as e:
            logging.error(f"Failed to download {paper.title}. Error: {e}")

if __name__ == "__main__":
    download_papers()
    logging.info("--- Data collection finished! ---")