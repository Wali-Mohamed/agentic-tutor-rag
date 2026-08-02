# agentic-tutor-rag
An Agentic RAG virtual assistant for GCSE math tutoring, featuring tool calling, evaluation, and monitoring. Built for LLM Zoomcamp 2026.



## 🎯 1. Problem Description
Navigating independent tutoring services can be confusing for parents and students. They frequently ask the same questions: *"Do you cover Higher Tier AQA?"*, *"How much is a 1-hour session?"*, or *"Do you provide homework?"* 
Wali solves this by acting as a 24/7 automated assistant, answering logistical and academic FAQ questions instantly by retrieving facts from a structured knowledge base, saving the tutor hours of administrative work.

## 🧠 2. RAG Flow
The application uses an advanced Retrieval-Augmented Generation (RAG) architecture:
* **Knowledge Base:** JSON FAQ data.
* **Retrieval Engine:** Vector Search using `FastEmbed` (BAAI/bge-small-en-v1.5) to capture semantic intent, overcoming the vocabulary gaps of traditional Keyword Search.
* **LLM Generation:** Powered by `gpt-4o-mini` using a strictly evaluated prompt template to prevent hallucination.

## 🧠 Architecture & Retrieval Strategy

This project uses a custom-built, native RAG (Retrieval-Augmented Generation) pipeline without relying on heavy frameworks like LangChain. 

### The Engine
* **Embeddings:** ONNX-based embedding models for lightweight, fast vectorization.
* **Vector Search:** Custom-built NumPy matrix dot-product search (`VectorIndex`) for exact control over the retrieval math.
* **Generation:** OpenAI API (`gpt-4o-mini`) strictly prompted to answer using only provided context.
* **Telemetry:** SQLite database integration for logging user queries, LLM responses, and capturing user feedback (thumbs up/down).

---

## 📊 Evaluation & Data-Driven Architecture

During development, I built an evaluation pipeline to test three distinct retrieval strategies against a ground-truth dataset to measure **Hit Rate** and **Mean Reciprocal Rank (MRR)**.

1. **Baseline (Vector Only):** Direct semantic search using the ONNX vector index.
2. **Standard Advanced RAG:** LLM query rewriting + Hybrid Search (Vector + Minsearch keyword matching) + Cross-Encoder Reranking.
3. **Agentic RAG:** An Agentic Query Planner (native OpenAI JSON mode) routing exact keywords to Minsearch and semantic intent to the Vector Index, followed by Cross-Encoder Reranking.

### Evaluation Leaderboard:

| System | Hit Rate | MRR |
| :--- | :--- | :--- |
| 🏆 **1. Baseline (Vector Only)** | **77.59%** | **0.5188** |
| 3. New Agentic RAG | 68.97% | 0.3536 |
| 2. Old Advanced RAG | 38.79% | 0.2464 |

### The Engineering Decision

Despite building a fully functional Agentic Hybrid RAG pipeline, the data proved that the **Baseline Vector Search** outperformed the more complex architectures for this specific math-tutoring dataset. 

Following the engineering principle of deploying the simplest, fastest, and most cost-effective model that meets requirements, the final Streamlit application is powered purely by the optimized `VectorRAG` pipeline. This avoids LLM semantic drift, eliminates extra API costs for query planning, and significantly reduces user latency.

## ⚖️ 4. LLM Generation Evaluation
The LLM generation phase was evaluated using the **LLM-as-a-Judge** standard (powered by `gpt-4o-mini` and Pydantic structured outputs) to measure semantic equivalence to the ground truth.
* **Strict Prompt (Brief & Factual):** **83.3%** Accuracy
* **Friendly Prompt (Warm & Polite):** 76.7% Accuracy
* **Conclusion:** The Strict prompt was chosen for the final application to maximize factual reliability. (See notebook: `evaluation.ipynb`).

## 🖥️ 5. Interface
The interface is built with **Streamlit**, featuring:
* A conversational chat UI using `st.chat_message`.
* Session state memory to preserve chat history.
* A multi-page layout containing the main bot and a dedicated analytics dashboard.

## 🚰 6. Ingestion Pipeline
Data ingestion is fully automated using **dlt (Data Load Tool)**.
Instead of reading raw JSON files directly in the app, the `ingest_pipeline.py` script automatically extracts the data, infers the schema, and loads it into a local **DuckDB** data warehouse. The Streamlit app then queries this DuckDB database to build its vector index.

## 📈 7. Monitoring
The application monitors usage and collects user feedback:
* **Feedback:** A thumbs-up/thumbs-down widget captures user satisfaction.
* **Storage:** Chats and feedback are logged locally using **SQLite** (`data/chat_logs.db`).
* **Dashboard:** A Streamlit dashboard (`pages/dashboard.py`) utilizes `Plotly` to display 5 distinct charts: Daily Traffic, User Satisfaction (Pie), Peak Usage Hours, Query Length Distribution, and Feedback Trends over time.

## 🐳 8. Reproducibility & Containerization
The project is fully reproducible and containerized.
* Dependency management is strictly handled by **`uv`**.
* A `Dockerfile` and `docker-compose.yaml` are provided.
* Because the architecture utilizes local file-based databases (DuckDB & SQLite), no heavy external database containers are required. Docker Compose seamlessly mounts the `./data` volume so all databases and logs persist across container restarts.

---

## 🚀 Quick Start / Reproducibility Steps


### Prerequisites
1. **Docker Desktop:** [Install Docker Desktop](https://docs.docker.com/get-docker/) (Required for Docker option)
2. **uv:** [Install uv](https://docs.astral.sh/uv/getting-started/installation/) (Required for local ingestion and local run)
3. **Make (Optional but recommended):** We use a `Makefile` to simplify commands. 
   * **Mac:** `xcode-select --install` or `brew install make`
   * **Linux:** `sudo apt install make`
   * **Windows:** Use [Scoop](https://scoop.sh/) (`scoop install make`) or run via WSL/Git Bash.

   *(Note: If you do not have `make` installed, you can run the raw commands shown in the notes below).*

4. **Environment Configuration:** Create a `.env` file in the root directory and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=sk-your-api-key-here

###  🚀 Quick Start / Reproducibility Steps
Step 1: Ingest the Knowledge Base (Required First)
Before starting the Streamlit application, run the ingestion pipeline to build the local DuckDB database (data/tutor_pipeline.duckdb):

Bash
# 1. Install dependencies
make setup

# 2. Run dlt ingestion pipeline into DuckDB
make ingest

### Without make: 


```bash
uv sync 

```
followed by 

```bash
uv run python modules/ingest_pipeline.py
```

### Step 2: Run the Application
Option A: Run via Docker (Recommended)
Build and launch the containerized app (which mounts the ./data folder created in Step 1):

```bash
make up
```
Without make: run

```bash
 docker compose up --build -d
```
Access App: http://localhost:8501

Stop Container:
```bash

 make down
 
 ```
 or 
 ```bash
 docker compose down
 ```

### Option B: Run Locally (using uv)
If you prefer to run it locally without Docker:

# 1. Install dependencies
```bash
make setup

```

# 2. Run the dlt ingestion pipeline (creates the DuckDB database)

```bash
make ingest
```

# 3. Launch the Streamlit app
```bash
make run

```
or without make
```bash
uv run streamlit run app.py
```
Access App: http://localhost:8501


Stop App: Press Ctrl + C in your terminal.
