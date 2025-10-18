# qa_pipeline.py

import chromadb
import ollama
from sentence_transformers import SentenceTransformer
import logging
import mlflow
import os

# --- Configuration ---
DB_PATH = "chroma_db"
COLLECTION_NAME = "arxiv_papers"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "phi3"
RELEVANCE_THRESHOLD = 1.3  # <<< --- NEW GUARD RAIL ---
                           # (Tune this value. 1.0 is a good start)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Set up MLflow ---
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
mlflow.set_experiment("RAG Q&A System")

def main():
    """
    The main function to run the RAG Q&A loop.
    """
    logging.info("Initializing RAG components...")
    
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
    
    logging.info("--- RAG System Ready. Ask a question! ---")
    print("\n(Type 'exit' to quit)")

    while True:
        query = input("\nYour Question: ")
        if query.lower() == 'exit':
            break
        
        # 1. Embed the query
        query_embedding = embedding_model.encode(query).tolist()

        # 2. Retrieve context (NOW INCLUDES DISTANCES)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=['documents', 'distances']  # <<< --- ADD 'distances' ---
        )
        
        best_distance = results['distances'][0][0]
        logging.info(f"Best retrieval distance: {best_distance}")

        # === NEW GUARD RAIL LOGIC ===
        if best_distance > RELEVANCE_THRESHOLD:
            print("\nAnswer:")
            print("I am sorry, but I cannot find any relevant information in the provided documents to answer that question.")
            
            # Log this "failed" retrieval to MLflow
            with mlflow.start_run():
                mlflow.log_param("user_query", query)
                mlflow.log_param("llm_model", LLM_MODEL_NAME)
                mlflow.log_param("embedding_model", EMBEDDING_MODEL_NAME)
                mlflow.log_param("status", "Rejected by relevance threshold")
                mlflow.log_param("best_distance", best_distance)
                mlflow.log_text("", "retrieved_context.txt")
                mlflow.log_text("Rejected by relevance threshold.", "model_answer.txt")
            continue # Skip to the next loop iteration
        # === END OF GUARD RAIL ===

        # If we are here, the context is RELEVANT.
        retrieved_documents = results['documents'][0]
        context_str = "\n\n---\n\n".join(retrieved_documents)
        
        # 3. Build the prompt for the LLM
        prompt_template = f"""
        SYSTEM: You are an expert AI assistant for answering questions about scientific research.
        Use only the context provided below to answer the user's question.
        If the context does not contain the answer, state clearly that you cannot answer based on the provided documents.
        
        CONTEXT:
        {context_str}
        
        USER QUESTION:
        {query}
        """

        # === MLFLOW START RUN (for a successful generation) ===
        with mlflow.start_run() as run:
            print("\nGenerating answer...")
            
            # 4. Log parameters
            mlflow.log_param("user_query", query)
            mlflow.log_param("llm_model", LLM_MODEL_NAME)
            mlflow.log_param("embedding_model", EMBEDDING_MODEL_NAME)
            mlflow.log_param("status", "Answer generated")
            mlflow.log_param("best_distance", best_distance)

            # 5. Generate the answer
            response = ollama.chat(
                model=LLM_MODEL_NAME,
                messages=[
                    {'role': 'user', 'content': prompt_template}
                ]
            )
            
            answer = response['message']['content']
            print("\nAnswer:")
            print(answer)

            # 6. Log results
            mlflow.log_text(context_str, "retrieved_context.txt")
            mlflow.log_text(answer, "model_answer.txt")
            
            print(f"--- Run logged with ID: {run.info.run_id} ---")
        # === MLFLOW END RUN ===

if __name__ == "__main__":
    main()