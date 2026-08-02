import os
import sys
import streamlit as st
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()


# Ensure modules can be imported
sys.path.append(os.path.abspath('.'))

from modules.ingest import load_faq_data, build_indices
from modules.rag_helper import VectorRAG
from modules.db import init_db, log_chat, update_feedback  # <-- NEW IMPORT

# Initialize the SQLite DB when the app starts
init_db()  # <-- NEW INIT

# Page Configuration
st.set_page_config(
    page_title="Wali Math Tutor AI",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Wali - GCSE & KS3 Math Tutor Assistant")
st.caption("Ask questions about online math lessons, pricing, exam boards (AQA, Edexcel, OCR), and trial sessions.")
def evaluate_response_with_llm(user_query, bot_response):
    """Uses gpt-4o-mini as an expert evaluator to classify response relevance."""
    prompt = f"""
You are an expert evaluator for a RAG system.
Analyze the relevance of the generated answer to the given question.

Question: "{user_query}"
Answer: "{bot_response}"

Classify the answer as:
- RELEVANT: the answer addresses the question
- PARTLY_RELEVANT: the answer partially addresses the question
- NON_RELEVANT: the answer does not address the question

Return ONLY one of these exact terms: RELEVANT, PARTLY_RELEVANT, or NON_RELEVANT.
    """.strip()
    
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        rating = response.choices[0].message.content.strip()
        
        # Parse output securely
        if "PARTLY_RELEVANT" in rating:
            return "PARTLY_RELEVANT"
        elif "RELEVANT" in rating:
            return "RELEVANT"
        else:
            return "NON_RELEVANT"
    except Exception:
        return "NOT_EVALUATED"

# Inside your init_rag_system() function:
@st.cache_resource
def init_rag_system():
    documents = load_faq_data()
    # Unpack both indices
    keyword_index, vector_index = build_indices(documents)
    
    client = OpenAI()
    PROMPT_STRICT = "QUESTION: {question}\nCONTEXT: {context}\nAnswer ONLY using facts in the context. Be brief."
    
    # Pass both indices to the AdvancedRAG class
    rag_system = VectorRAG(
        index=vector_index, 
        llm_client=client, 
        prompt_template=PROMPT_STRICT
    )
    return rag_system

# Load RAG system (runs once)
with st.spinner("Loading knowledge base and vector index..."):
    rag = init_rag_system()

# 2. Manage Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render past chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Handle User Input (UPDATED WITH LOGGING)
# 3. Handle User Input
if prompt := st.chat_input("e.g., How much do lessons cost?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Searching knowledge base..."):
        try:
            response = rag.rag(prompt)
        except Exception as e:
            response = f"Sorry, an error occurred: {e}"

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    # 1. Run LLM-as-a-Judge evaluation automatically
    judge_rating = evaluate_response_with_llm(prompt, response)

    # 2. Approximate total tokens (or use 0 if you prefer)
    estimated_tokens = int((len(prompt) + len(response)) / 4)

    # 3. Log to Database with the judge rating and token count
    log_id = log_chat(prompt, response, total_tokens=estimated_tokens, llm_judge_rating=judge_rating)
    st.session_state["last_log_id"] = log_id

# 4. Show Feedback widget for the most recent answer (NEW)
if "last_log_id" in st.session_state:
    def handle_feedback():
        # st.session_state.fb returns 1 for Thumbs Up, 0 for Thumbs Down
        fb_val = 1 if st.session_state.fb == 1 else -1
        update_feedback(st.session_state.last_log_id, fb_val)
        st.toast("Thanks for your feedback! 📝")

    st.feedback("thumbs", key="fb", on_change=handle_feedback)
