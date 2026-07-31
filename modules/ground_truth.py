import json
from openai import OpenAI
from tqdm.auto import tqdm

def generate_ground_truth_dataset(input_filepath, output_filepath, model="gpt-4o-mini"):
    """
    Reads a knowledge base JSON, generates synthetic questions using an LLM, 
    and saves a ground truth dataset for RAG evaluation.
    """
    # 1. Load your knowledge base documents
    with open(input_filepath, 'r') as f:
        documents = json.load(f)

    client = OpenAI()

    PROMPT_TEMPLATE = """
You are creating a test dataset to evaluate a RAG assistant for Wali, a GCSE and KS3 Math tutor.

Based on the following FAQ context, write THREE realistic, natural questions that a parent or student might ask Wali.

FAQ Context:
{context}

Original Answer:
{answer}

Return EXACTLY 3 questions, with each question on its own separate line. 
Do not include numbers, bullet points, quotes, or any extra text.
""".strip()

    ground_truth = []
    print(f"Generating questions for {len(documents)} documents...")

    # 2. Iterate and generate questions
    for idx, doc in enumerate(tqdm(documents)):
        context = doc.get('mapped_context', '')
        expected_answer = doc.get('expected_answer', '')
        chunk_id = doc.get('chunk_id', f"chunk_{idx}")
        
        # Skip empty items
        if not context:
            continue

        prompt = PROMPT_TEMPLATE.format(context=context, answer=expected_answer)
        
        try:
            # Generate question using the LLM
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            
            generated_question = response.choices[0].message.content.strip().strip('"')
            
            # Build the record
            ground_truth.append({
                "question": generated_question,
                "user_query": generated_question, 
                "chunk_id": chunk_id,
                "answer_orig": expected_answer,
                "category": doc.get("category", "General"),
                "subcategory": doc.get("subcategory", "General")
            })
        except Exception as e:
            print(f"Warning: Failed to generate question for chunk {chunk_id}. Error: {e}")

    # 3. Save the results
    with open(output_filepath, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\nDone! Saved {len(ground_truth)} records to {output_filepath}")