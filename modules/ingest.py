import json
from minsearch import Index as KeywordIndex
from fastembed import TextEmbedding
from modules.rag_helper import VectorIndex  

def load_faq_data():
    with open('data/knowledge-base.json', 'r') as f:
        return json.load(f)

def build_indices(documents):
    # 1. Build Keyword Index (minsearch)
    print("Building Keyword Index...")
    keyword_index = KeywordIndex(
        text_fields=['user_query', 'mapped_context', 'expected_answer'],
        keyword_fields=['chunk_id', 'category', 'subcategory']
    )
    keyword_index.fit(documents)
    
    # 2. Build Vector Index (FastEmbed)
    print("Building Vector Index...")
    texts_to_embed = [
        f"Question: {doc['user_query']} | Context: {doc['mapped_context']} | Answer: {doc['expected_answer']}"
        for doc in documents
    ]
    embedder = TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')
    vector_index = VectorIndex(embedder=embedder)
    vector_index.fit(documents, texts_to_embed)
    
    print("Both indices built successfully!")
    return keyword_index, vector_index