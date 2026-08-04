import time
import random
from modules.db import init_db, log_chat, update_feedback, update_judge_rating

# Grouping questions, realistic answers, and their appropriate ratings together
Q_AND_A_PAIRS = [
    {
        "question": "Do you cover Higher Tier AQA?",
        "answer": "Yes, I cover Higher Tier for AQA, Edexcel, and OCR.",
        "rating": "RELEVANT"
    },
    {
        "question": "How much is a 1-hour session?",
        "answer": "A standard 1-hour session is £35.",
        "rating": "RELEVANT"
    },
    {
        "question": "Can you explain the quadratic formula?",
        "answer": "The quadratic formula is x = (-b ± √(b² - 4ac)) / 2a.",
        "rating": "RELEVANT"
    },
    {
        "question": "What topics are included in GCSE Edexcel Math?",
        "answer": "Edexcel includes Number, Algebra, Ratio & Proportion, Geometry, Probability, and Statistics.",
        "rating": "RELEVANT"
    },
    # Intentionally vague answer to simulate a partial hit
    {
        "question": "Do you provide homework?",
        "answer": "I provide a lot of math resources and worksheets for students.",
        "rating": "PARTLY_RELEVANT"
    },
    # Intentionally bad answer to simulate a failure
    {
        "question": "How do I book a tutoring session with Wali?",
        "answer": "Wali has been teaching math for 5 years.", 
        "rating": "NON_RELEVANT"
    }
]

def logical_user_feedback(rating):
    """Makes user feedback match the quality of the answer."""
    if rating == "RELEVANT":
        return random.choice([1, 1, 1, 1, -1])  # 80% thumbs up
    elif rating == "PARTLY_RELEVANT":
        return random.choice([1, -1])           # 50/50 chance
    else:
        return -1                               # Always thumbs down if non-relevant

def generate_one():
    # Pick one logical Q&A pair
    pair = random.choice(Q_AND_A_PAIRS)
    
    user_query = pair["question"]
    bot_response = pair["answer"]
    expected_rating = pair["rating"]
    total_tokens = random.randint(60, 350)

    # 1. Log the chat into SQLite
    log_id = log_chat(
        user_query=user_query,
        bot_response=bot_response,
        total_tokens=total_tokens,
        llm_judge_rating="Not Evaluated"
    )

    # 2. 70% chance to simulate LLM-as-a-Judge evaluation
    if random.random() < 0.7:
        update_judge_rating(log_id, expected_rating)

    # 3. 50% chance to simulate a user clicking Thumbs Up or Down
    if random.random() < 0.5:
        feedback_val = logical_user_feedback(expected_rating)
        update_feedback(log_id, feedback_val)

def generate_live():
    init_db()
    print("🚀 Starting logical data generation (Ctrl+C to stop)...", flush=True)
    
    while True:
        try:
            generate_one()
            print("Pushed 1 logical Q&A pair to chat_logs.db", flush=True)
            time.sleep(2) # 30 questions per minute
        except Exception as e:
            print(f"Error generating data: {e}")
            time.sleep(2)

if __name__ == "__main__":
    try:
        generate_live()
    except KeyboardInterrupt:
        print("\nStopped synthetic data generator.")