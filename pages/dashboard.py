import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from pathlib import Path

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
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour
df['query_length'] = df['user_query'].apply(len)

# Map feedback integers to labels
feedback_map = {1: "Thumbs Up", -1: "Thumbs Down", 0: "No Feedback"}
df['feedback_label'] = df['feedback'].map(feedback_map)

# Layout for charts
col1, col2 = st.columns(2)

with col1:
    # CHART 1: Total Interactions Over Time (Line Chart)
    st.subheader("1. Daily Traffic")
    daily_traffic = df.groupby('date').size().reset_index(name='chats')
    fig1 = px.line(daily_traffic, x='date', y='chats', markers=True)
    st.plotly_chart(fig1, use_container_width=True)

    # CHART 2: Feedback Distribution (Pie Chart)
    st.subheader("2. User Satisfaction")
    feedback_counts = df[df['feedback'] != 0]['feedback_label'].value_counts().reset_index()
    if not feedback_counts.empty:
        fig2 = px.pie(feedback_counts, names='feedback_label', values='count', color='feedback_label',
                      color_discrete_map={"Thumbs Up": "green", "Thumbs Down": "red"})
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No feedback submitted yet.")

    # CHART 3: Most Active Hours (Bar Chart)
    st.subheader("3. Peak Usage Hours")
    hourly_traffic = df.groupby('hour').size().reset_index(name='chats')
    fig3 = px.bar(hourly_traffic, x='hour', y='chats')
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    # CHART 4: Query Length Distribution (Histogram)
    st.subheader("4. User Query Length (Characters)")
    fig4 = px.histogram(df, x='query_length', nbins=20)
    st.plotly_chart(fig4, use_container_width=True)

    # CHART 5: Feedback Over Time (Stacked Bar)
    st.subheader("5. Feedback Trends Over Time")
    feedback_time = df[df['feedback'] != 0].groupby(['date', 'feedback_label']).size().reset_index(name='count')
    if not feedback_time.empty:
        fig5 = px.bar(feedback_time, x='date', y='count', color='feedback_label', barmode='stack',
                      color_discrete_map={"Thumbs Up": "green", "Thumbs Down": "red"})
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Not enough feedback data over time yet.")

# Display Raw Data Table at the bottom
st.subheader("Recent Chat Logs")
st.dataframe(df.sort_values(by="timestamp", ascending=False).head(10))