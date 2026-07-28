import json
import requests
from minsearch import Index

def load_faq_data():
    # Load directly from your local JSON file
    with open('knowledge-base.json', 'r') as f:
        documents = json.load(f)
        
    return documents


def build_index(documents):
    index = Index(
        # text_fields are the natural language columns you want to search against
        text_fields=['user_query', 'mapped_context', 'expected_answer'],
        
        # keyword_fields are used for exact match filtering (e.g., filtering by "pricing")
        keyword_fields=['chunk_id', 'category', 'subcategory']
    )
    
    index.fit(documents)
    return index
