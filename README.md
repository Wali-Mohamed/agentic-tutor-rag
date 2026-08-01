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

## 📊 3. Retrieval Evaluation
Multiple retrieval approaches were evaluated using a synthetic ground-truth dataset. 
* **Keyword Search (minsearch):** Hit Rate @ 5: 60.00% | MRR: 0.3752
* **Vector Search (FastEmbed):** Hit Rate @ 5: **71.30%** | MRR: **0.4277**
* **Conclusion:** Vector Search outperformed Keyword search and was integrated into the final app. (See notebook: `evaluation.ipynb`).

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
1. Install [Docker](https://docs.docker.com/get-docker/) or [uv](https://github.com/astral-sh/uv).
2. Create a `.env` file in the root directory and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=sk-your-api-key-here


### Option A: Run via Docker (Recommended)
You can build and run the entire application using the provided `Makefile`:

```bash
make up

```

Access the app at: http://localhost:8501

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
