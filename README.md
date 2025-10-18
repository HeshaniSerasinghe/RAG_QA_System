# End-to-End RAG Q&A System for Scientific Research

This project is a complete, end-to-end Question-Answering (Q&A) system built to solve the primary weaknesses of Large Language Models (LLMs): **hallucination** and **outdated knowledge**.

Instead of relying on a general-purpose LLM, this system uses **Retrieval-Augmented Generation (RAG)**. It first retrieves relevant, factual context from a private library of documents (in this case, ArXiv research papers) and then uses a local LLM to generate an answer _based only on that context_.

This project is built with a full **MLOps (Machine Learning Operations)** workflow, using **DVC** for data versioning and **MLflow** for experiment tracking, ensuring the entire process is professional, reproducible, and robust.

---

## ✨ Key Features

- **Modular Pipeline:** Separate, easy-to-run scripts for data collection, processing, and indexing.
- **Factual & Private:** All data and models (Ollama, ChromaDB) run 100% locally. Your documents and queries never leave your machine.
- **Anti-Hallucination Guard Rail:** A custom relevance threshold prevents the LLM from answering irrelevant questions, solving the "puppy" and "cricket" problem.
- **Data Version Control:** Uses **DVC** to track large data files (PDFs, JSONs) and the vector database, keeping the Git repository clean and lightweight.
- **Experiment Tracking:** Uses **MLflow** to log every single question, the retrieved context, the generated answer, and performance metrics in a clean web dashboard.

---

## 🛠️ Tech Stack (The "What" and "Why")

This project uses a modern, open-source stack.

| Technology                     | Purpose                    | Why We Used It                                                                                                                  |
| :----------------------------- | :------------------------- | :------------------------------------------------------------------------------------------------------------------------------ |
| **Python**                     | Core Language              | The universal language for data science and AI development.                                                                     |
| **Ollama**                     | Local LLM Server           | Provides easy, free access to run powerful LLMs (like `phi3` or `tinyllama`) privately on our own machine.                      |
| **Sentence-Transformers**      | Embedding Model            | The best-in-class library for converting text chunks into meaningful numerical vectors (embeddings).                            |
| **ChromaDB**                   | Vector Database            | A high-performance, open-source database specifically designed to store and search billions of vectors for semantic similarity. |
| **DVC (Data Version Control)** | MLOps: Data Versioning     | Git is for code, DVC is for data. It versions our large `data/` and `chroma_db/` folders without bloating the Git history.      |
| **MLflow**                     | MLOps: Experiment Tracking | Creates a scientific logbook for every experiment. This lets us track, compare, and debug every question we ask the system.     |

---

## 🔧 Project Workflow & Usage

Follow these steps to build and run the pipeline from scratch.

### 1. Installation

1.  **Clone the Repository**

    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd RAG-QA-System
    ```

2.  **Create a Virtual Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    .\venv\Scripts\activate   # On Windows
    ```

3.  **Install & Pull an LLM**

    - [Download and install Ollama](https://ollama.com).
    - Pull a small model to run locally (we use `tinyllama` for low-RAM systems):
      ```bash
      ollama pull tinyllama
      ```

4.  **Install Python Dependencies**
    - First, generate the `requirements.txt` file from your environment (if it's not already in the repo):
      ```bash
      pip freeze > requirements.txt
      ```
    - Then, install all required libraries:
      ```bash
      pip install -r requirements.txt
      ```

### 2. Running the Pipeline

**Phase 1: Data Collection & Processing**

- This downloads the PDFs from ArXiv and processes them into clean JSON chunks.
  ```bash
  python collect_data.py
  python process_data.py
  ```

**Phase 2: Indexing**

- This creates the vector embeddings and builds the ChromaDB vector database.
  ```bash
  python build_index.py
  ```

**Phase 3: Running the Q&A System**

- This starts the interactive chat session.
  ```bash
  python qa_pipeline.py
  ```

**Phase 4: Tracking Experiments**

- While the Q&A system is running, open a **second terminal** (with the `venv` activated) and run:
  ```bash
  mlflow ui
  ```
- Open `http://127.0.0.1:5000` in your browser to see the live MLflow dashboard.

---

## ⚙️ Configuration & Customization

This project is designed to be easily modified. You can change key parameters in the following files:

### `collect_data.py`

- `SEARCH_QUERY = "Large Language Models"`
  - Change this string to any topic you want (e.g., `"Reinforcement Learning"`, `"Computer Vision"`).
- `MAX_RESULTS = 100`
  - Change this to download more or fewer papers.

### `process_data.py`

- `CHUNK_SIZE = 512`
  - The number of words in each text chunk. Smaller chunks are more precise but create a larger database.
- `CHUNK_OVERLAP = 50`
  - The number of words to overlap between chunks to ensure no context is lost.

### `build_index.py`

- `EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"`
  - You can swap this with any other model from the [Sentence-Transformers library](https://www.sbert.net/docs/pretrained_models.html).

### `qa_pipeline.py`

- `LLM_MODEL_NAME = "tinyllama"`
  - Change this to any model you have downloaded with Ollama (e.g., `"phi3"`, `"llama3"`).
- `RELEVANCE_THRESHOLD = 1.3`
  - **This is the most important guard rail.** A _lower_ number makes it _stricter_. If the best-found document has a distance score _higher_ than this, the system will refuse to answer. Tune this based on your own experiments (we found `1.3` works well).
- `n_results=5`
  - Find this inside the `collection.query()` function. It controls how many context chunks are sent to the LLM.

---

## 📈 Future Improvements

This project is a strong foundation. The next steps to productionize it would be:

- **Phase 5: Deployment:** Wrap the Q&A pipeline in a **FastAPI** server and package the entire application with **Docker** for easy, reproducible deployment.
- **Web Interface:** Build a simple **Streamlit** or **Gradio** web app to provide a user-friendly chat interface.
- **Advanced Evaluation:** Use a framework like **RAGAs** to formally evaluate the quality of the retrieval and generation, allowing for more scientific tuning of the parameters.
- **Cloud Remote:** Push the DVC-tracked data to a cloud remote (like Google Drive or S3) to make the project fully shareable and reproducible by anyone, anywhere.

---

## 📂 Project Structure

```
RAG-QA-System/
├── .dvc/                   # DVC internal files
├── .git/                   # Git internal files
├── venv/                   # Python virtual environment (Ignored by Git)
├── data/                   # Raw PDFs and processed JSONs (Tracked by DVC)
├── chroma_db/              # The Vector Database (Tracked by DVC)
├── mlruns/                 # MLflow experiment logs (Ignored by Git)
├── .gitignore              # Tells Git what to ignore
├── data.dvc                # DVC pointer file for the 'data' folder
├── chroma_db.dvc           # DVC pointer file for the 'chroma_db' folder
├── collect_data.py         # Script to download ArXiv papers
├── process_data.py         # Script to clean and chunk PDFs
├── build_index.py          # Script to create the vector database
├── qa_pipeline.py          # The main Q&A application
├── requirements.txt        # A list of all Python libraries
└── README.md               # This file
```
