# qa_pipeline.py

import chromadb
import ollama
from sentence_transformers import SentenceTransformer
import logging

# --- Configuration ---
DB_PATH = "chroma_db"
COLLECTION_NAME = "arxiv_papers"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "phi3"  # LLM_MODEL_NAME = "llama3" - when more ram available

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    The main function to run the RAG Q&A loop.
    """
    # 1. Initialize all components
    logging.info("Initializing RAG components...")
    
    # ChromaDB client
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    
    # Embedding model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
    
    logging.info("--- RAG System Ready. Ask a question! ---")
    print("\n(Type 'exit' to quit)")

    # 2. Start the interactive Q&A loop
    while True:
        query = input("\nYour Question: ")
        if query.lower() == 'exit':
            break

        # 3. Embed the user's query
        query_embedding = embedding_model.encode(query).tolist()

        # 4. Retrieve relevant context from ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5  # Retrieve the top 5 most relevant chunks
        )
        
        retrieved_documents = results['documents'][0]
        context_str = "\n\n---\n\n".join(retrieved_documents)
        
        # 5. Build the prompt for the LLM
        prompt_template = f"""
        SYSTEM: You are an expert AI assistant for answering questions about scientific research.
        Use only the context provided below to answer the user's question.
        If the context does not contain the answer, state clearly that you cannot answer based on the provided documents.
        
        CONTEXT:
        {context_str}
        
        USER QUESTION:
        {query}
        """

        # 6. Generate the answer using the LLM
        print("\nGenerating answer...")
        response = ollama.chat(
            model=LLM_MODEL_NAME,
            messages=[
                {'role': 'user', 'content': prompt_template}
            ]
        )
        
        print("\nAnswer:")
        print(response['message']['content'])

if __name__ == "__main__":
    main()