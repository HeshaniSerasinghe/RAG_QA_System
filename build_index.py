# build_index.py

import os
import json
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

# --- Configuration ---
PROCESSED_DATA_DIR = "data/processed"
DB_PATH = "chroma_db"
COLLECTION_NAME = "arxiv_papers"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # A fast and effective model

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_index():
    """
    Builds the ChromaDB vector index from processed text data.
    """
    logging.info("--- Starting Index Building ---")

    # 1. Initialize ChromaDB client and collection
    # Using a PersistentClient saves the DB to disk at the specified path.
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # get_or_create_collection is idempotent: it will create the collection
    # if it doesn't exist, or load it if it does.
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    logging.info(f"ChromaDB collection '{COLLECTION_NAME}' loaded/created.")

    # 2. Load the embedding model
    logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    # The device='cpu' is a good default. If you have a CUDA-enabled GPU, 
    # you can change this to 'cuda' for a significant speed-up.
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
    logging.info("Embedding model loaded successfully.")

    # 3. Process files and add to ChromaDB
    processed_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith(".json")]
    
    for filename in tqdm(processed_files, desc="Indexing Documents"):
        filepath = os.path.join(PROCESSED_DATA_DIR, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        # Prepare data for ChromaDB batch insertion
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [{'source': chunk['source']} for chunk in chunks]
        # Create a unique ID for each chunk
        ids = [f"{chunk['source']}_{chunk['chunk_id']}" for chunk in chunks]

        # Skip if no documents to process
        if not documents:
            continue
            
        # 4. Generate embeddings
        embeddings = model.encode(documents, show_progress_bar=False).tolist()

        # 5. Add to the collection
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    logging.info(f"--- Index Building Finished ---")
    logging.info(f"Total documents in collection: {collection.count()}")

if __name__ == "__main__":
    build_index()