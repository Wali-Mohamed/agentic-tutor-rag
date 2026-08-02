import json
from minsearch import Index as KeywordIndex
from fastembed import TextEmbedding
from modules.rag_helper import VectorIndex  
import duckdb

def load_faq_data():
    # Simple relative path
    conn = duckdb.connect('data/tutor_pipeline.duckdb')
    
    query = "SELECT * FROM wali_kb.faq_source"
    documents = conn.execute(query).df().to_dict(orient='records')
    
    clean_docs = [
        {k: v for k, v in doc.items() if not k.startswith('_dlt')}
        for doc in documents
    ]
    
    return clean_docs


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
