from pathlib import Path
import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Monitoring Dashboard", layout="wide")
st.title("📊 RAG Monitoring Dashboard")

# Connect to database and load data
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "chat_logs.db"

try:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM logs", conn)
    conn.close()
except Exception as e:
    st.error("No database found. Chat with the bot first to generate logs!")
    st.stop()

if df.empty:
    st.warning("No chat data available yet. Please interact with the bot.")
    st.stop()

# Data Preprocessing
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.date

# --- FALLBACK LOGIC ---
# If your database doesn't have these columns yet, this prevents the app from crashing.
if "total_tokens" not in df.columns:
    # Rough estimation: 1 token ≈ 4 characters
    df["total_tokens"] = ((df["user_query"].apply(len) + df["bot_response"].apply(len)) / 4).astype(int)

if "llm_judge_rating" not in df.columns:
    df["llm_judge_rating"] = "Not Evaluated"
# ----------------------

# Map feedback integers to labels
feedback_map = {1: "Thumbs Up", -1: "Thumbs Down", 0: "No Feedback"}
if "feedback" in df.columns:
    df["feedback_label"] = df["feedback"].map(feedback_map)
else:
    df["feedback_label"] = "No Feedback"

# Layout for charts
col1, col2 = st.columns(2)

with col1:
    # CHART 1: Total Interactions Over Time (Bar Chart)
    st.subheader("1. Daily Traffic")
    daily_traffic = df.groupby("date").size().reset_index(name="chats")
    fig1 = px.bar(daily_traffic, x="date", y="chats", text_auto=True)
    st.plotly_chart(fig1, use_container_width=True)

    # CHART 2: Feedback Distribution (Pie Chart)
    st.subheader("2. User Satisfaction")
    feedback_counts = (
        df[df["feedback"] != 0]["feedback_label"].value_counts().reset_index()
    )
    if not feedback_counts.empty:
        fig2 = px.pie(
            feedback_counts,
            names="feedback_label",
            values="count",
            color="feedback_label",
            color_discrete_map={"Thumbs Up": "green", "Thumbs Down": "red"},
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No user feedback submitted yet.")

with col2:
    # CHART 3: LLM-as-a-Judge (Pie Chart) replacing Peak Usage
    st.subheader("3. LLM-as-a-Judge Evaluation")
    judge_counts = df["llm_judge_rating"].value_counts().reset_index()
    judge_counts.columns = ["Rating", "Count"]
    
    fig3 = px.pie(
        judge_counts, 
        names="Rating", 
        values="Count", 
        color="Rating",
        # You can adjust these colors based on the exact strings your DB uses
        color_discrete_map={"Good": "green", "Bad": "red", "Not Evaluated": "gray"}
    )
    st.plotly_chart(fig3, use_container_width=True)

    # CHART 4: Total Number of Tokens replacing Query Length
    st.subheader("4. Daily Token Usage")
    daily_tokens = df.groupby("date")["total_tokens"].sum().reset_index()
    fig4 = px.bar(daily_tokens, x="date", y="total_tokens", text_auto=True)
    fig4.update_traces(marker_color='#ff7f0e')
    st.plotly_chart(fig4, use_container_width=True)

# Display Raw Data Table at the bottom
st.subheader("Recent Chat Logs")

# Dynamically select columns so the LLM Judge rating and tokens appear cleanly
display_cols = ["timestamp", "user_query", "bot_response", "feedback_label", "llm_judge_rating", "total_tokens"]
display_cols = [col for col in display_cols if col in df.columns] # Ensure columns exist

st.dataframe(df.sort_values(by="timestamp", ascending=False)[display_cols].head(10))

# ---------------------------------------------------------
# ⚖️ HARDCODED PIPELINE EVALUATION REPORT
# ---------------------------------------------------------
st.divider()
st.header("⚖️ Offline Retrieval Evaluation")
st.write(
    "These metrics represent the offline evaluation conducted using a synthetic ground-truth dataset to optimize the pipeline."
)

col_eval1, col_eval2 = st.columns(2)

with col_eval1:
    st.subheader("1. LLM Generation Accuracy")
    st.caption("Evaluated using LLM-as-a-Judge (gpt-4o-mini)")

    prompt_data = pd.DataFrame(
        {
            "Prompt Template": ["Strict (Brief & Factual)", "Friendly (Warm)"],
            "Accuracy (%)": [83.3, 76.7],
        }
    )

    fig_prompt = px.bar(
        prompt_data,
        x="Prompt Template",
        y="Accuracy (%)",
        text="Accuracy (%)",
        color="Prompt Template",
        color_discrete_sequence=["#2ca02c", "#1f77b4"],
    )
    fig_prompt.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_prompt.update_layout(
        showlegend=False, yaxis_range=[0, 100], margin=dict(t=30)
    )
    st.plotly_chart(fig_prompt, use_container_width=True)

with col_eval2:
    st.subheader("2. Retrieval Hit Rate @ 5")
    st.caption("Vector Search vs Keyword Search")

    retrieval_data = pd.DataFrame(
        {
            "Search Engine": ["Keyword (minsearch)", "Vector (FastEmbed)"],
            "Hit Rate (%)": [60.0, 71.3],
        }
    )

    fig_retrieval = px.bar(
        retrieval_data,
        x="Search Engine",
        y="Hit Rate (%)",
        text="Hit Rate (%)",
        color="Search Engine",
        color_discrete_sequence=["#ff7f0e", "#9467bd"],
    )
    fig_retrieval.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_retrieval.update_layout(
        showlegend=False, yaxis_range=[0, 100], margin=dict(t=30)
    )
    st.plotly_chart(fig_retrieval, use_container_width=True)

# Advanced RAG Metrics Banner
st.subheader("✨ Active Architecture")
met1, met2, met3 = st.columns(3)
met1.metric(
    label="Search Engine",
    value="Vector (NumPy)",
    help="FastEmbed ONNX Vector search",
)
met2.metric(
    label="Retrieval Strategy",
    value="Baseline",
    help="Direct semantic search without reranking.",
)
met3.metric(
    label="Generation Model",
    value="gpt-4o-mini",
    help="Strict context-only generation.",
)